//! Top-level processor: combines text tokenization and audio feature extraction.

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::audio::{input::AudioInput, normalize::normalize_audio_input};
use crate::processor::{chat_template, feat_lengths, feature_extractor, tokenizer::Tokenizer};

const FEATURE_PADDING_MODE_ENV: &str = "VASR_FEATURE_PADDING_MODE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeaturePaddingMode {
    Feature,
    Waveform,
    WaveformFull,
}

fn parse_feature_padding_mode(value: Option<&str>) -> Result<FeaturePaddingMode> {
    match value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("waveform")
        .to_ascii_lowercase()
        .as_str()
    {
        "waveform" | "waveforms" => Ok(FeaturePaddingMode::Waveform),
        "waveform-full" | "waveforms-full" | "waveform_full" | "waveforms_full" => {
            Ok(FeaturePaddingMode::WaveformFull)
        }
        "feature" | "features" => Ok(FeaturePaddingMode::Feature),
        other => bail!(
            "{FEATURE_PADDING_MODE_ENV} must be `feature`, `waveform`, or `waveform-full`, got {other:?}"
        ),
    }
}

fn feature_padding_mode_from_env() -> Result<FeaturePaddingMode> {
    parse_feature_padding_mode(std::env::var(FEATURE_PADDING_MODE_ENV).ok().as_deref())
}

#[derive(Debug, Clone)]
pub struct AsrProcessor {
    pub tokenizer: Tokenizer,
}

impl AsrProcessor {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn build_text_prompt(&self, context: &str, force_language: Option<&str>) -> String {
        chat_template::build_prompt(context, force_language)
    }

    pub fn prepare_one(&self, prompt: &str, audio: &AudioInput<'_>) -> Result<PreparedInputs> {
        let wav = normalize_audio_input(audio)?;
        let feats = feature_extractor::extract_features(&wav)?;

        // Placeholder expansion must match the audio encoder output length.
        let n_frames = feats
            .feature_attention_mask
            .iter()
            .filter(|&&x| x != 0)
            .count();
        let placeholder_len = feat_lengths::feat_extract_output_length(n_frames);

        let audio_pad_id = self.tokenizer.token_to_id(chat_template::AUDIO_PAD)?;
        let base_ids = self.tokenizer.encode(prompt)?;
        let input_ids =
            expand_audio_pad_ids_first(base_ids.as_slice(), audio_pad_id, placeholder_len);
        let attention_mask = vec![1u32; input_ids.len()];

        Ok(PreparedInputs {
            input_ids,
            attention_mask,
            input_features: feats.input_features,
            feature_attention_mask: feats.feature_attention_mask,
        })
    }

    pub fn prepare_batch(&self, items: &[(&str, &AudioInput<'_>)]) -> Result<Vec<PreparedInputs>> {
        self.prepare_batch_inner(items)
            .map(|(prepared, _)| prepared)
    }

    #[cfg(feature = "timing")]
    pub fn prepare_batch_timed(
        &self,
        items: &[(&str, &AudioInput<'_>)],
    ) -> Result<(Vec<PreparedInputs>, PrepareBatchTimings)> {
        let (prepared, timings) = self.prepare_batch_inner(items)?;
        Ok((prepared, timings))
    }

    fn prepare_batch_inner(
        &self,
        items: &[(&str, &AudioInput<'_>)],
    ) -> Result<(Vec<PreparedInputs>, PrepareBatchTimings)> {
        #[cfg(feature = "timing")]
        let mut timings = PrepareBatchTimings::default();
        #[cfg(not(feature = "timing"))]
        let timings = PrepareBatchTimings::default();
        if items.is_empty() {
            return Ok((vec![], timings));
        }

        // Normalize/resample audio first so feature extraction sees the model's native sample rate.
        #[cfg(feature = "timing")]
        let start_normalize = std::time::Instant::now();
        let wavs: Vec<Vec<f32>> = items
            .par_iter()
            .map(|(_, audio)| normalize_audio_input(audio))
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "timing")]
        {
            timings.normalize_us = duration_to_us(start_normalize.elapsed());
        }

        #[cfg(feature = "timing")]
        let start_token_lookup = std::time::Instant::now();
        let pad_id = self.tokenizer.token_to_id("<|endoftext|>")?;
        let audio_pad_id = self.tokenizer.token_to_id(chat_template::AUDIO_PAD)?;
        #[cfg(feature = "timing")]
        {
            timings.token_lookup_us = duration_to_us(start_token_lookup.elapsed());
        }

        // Extract features. The default waveform mode preserves the previous
        // zero-padded waveform boundary behavior; feature mode is an opt-in experiment.
        #[cfg(feature = "timing")]
        let start_feature_extract = std::time::Instant::now();
        let feats = match feature_padding_mode_from_env()? {
            FeaturePaddingMode::Feature => extract_features_with_feature_padding(&wavs)?,
            FeaturePaddingMode::Waveform => extract_features_with_waveform_padding(&wavs)?,
            FeaturePaddingMode::WaveformFull => extract_features_with_full_waveform_padding(&wavs)?,
        };
        #[cfg(feature = "timing")]
        {
            timings.feature_extract_us = duration_to_us(start_feature_extract.elapsed());
        }

        #[cfg(feature = "timing")]
        let start_tokenize_expand = std::time::Instant::now();
        let mut input_ids: Vec<Vec<u32>> = Vec::with_capacity(items.len());

        for ((prompt, _), f) in items.iter().zip(feats.iter()) {
            let n_frames = f.feature_attention_mask.iter().filter(|&&x| x != 0).count();
            let audio_out_len = feat_lengths::feat_extract_output_length(n_frames);

            let base_ids = self.tokenizer.encode(prompt)?;
            let ids = expand_audio_pad_ids_first(base_ids.as_slice(), audio_pad_id, audio_out_len);
            input_ids.push(ids);
        }
        #[cfg(feature = "timing")]
        {
            timings.tokenize_expand_us = duration_to_us(start_tokenize_expand.elapsed());
        }

        // Right-pad batched prompts so paged prefill can consume them directly.
        #[cfg(feature = "timing")]
        let start_pad = std::time::Instant::now();
        let max_tokens = input_ids.iter().map(Vec::len).max().unwrap_or(0);
        let mut out: Vec<PreparedInputs> = Vec::with_capacity(items.len());

        for (ids, f) in input_ids.into_iter().zip(feats) {
            let len = ids.len();
            if len > max_tokens {
                bail!(
                    "internal error: sequence len {} > max_tokens {}",
                    len,
                    max_tokens
                );
            }
            let pad = max_tokens - len;

            let mut padded_ids: Vec<u32> = Vec::with_capacity(max_tokens);
            padded_ids.extend(ids);
            padded_ids.extend(std::iter::repeat_n(pad_id, pad));

            let mut attn: Vec<u32> = Vec::with_capacity(max_tokens);
            attn.extend(std::iter::repeat_n(1u32, len));
            attn.extend(std::iter::repeat_n(0u32, pad));

            out.push(PreparedInputs {
                input_ids: padded_ids,
                attention_mask: attn,
                input_features: f.input_features,
                feature_attention_mask: f.feature_attention_mask,
            });
        }

        #[cfg(feature = "timing")]
        {
            timings.pad_us = duration_to_us(start_pad.elapsed());
        }

        Ok((out, timings))
    }

    pub fn require_ready(&self) -> Result<()> {
        self.tokenizer.require_loaded()
    }
}

fn extract_features_with_feature_padding(
    wavs: &[Vec<f32>],
) -> Result<Vec<feature_extractor::Features>> {
    let mut feats: Vec<feature_extractor::Features> = wavs
        .par_iter()
        .map(|wav| feature_extractor::extract_features(wav))
        .collect::<Result<Vec<_>>>()?;
    pad_feature_batch_to_max_frames(feats.as_mut_slice())?;
    Ok(feats)
}

fn extract_features_with_waveform_padding(
    wavs: &[Vec<f32>],
) -> Result<Vec<feature_extractor::Features>> {
    let max_samples = wavs.iter().map(Vec::len).max().unwrap_or(0);
    let mut feats: Vec<feature_extractor::Features> = wavs
        .par_iter()
        .map(|wav| feature_extractor::extract_features_with_padded_waveform_len(wav, max_samples))
        .collect::<Result<Vec<_>>>()?;
    pad_feature_batch_to_max_frames(feats.as_mut_slice())?;
    Ok(feats)
}

fn extract_features_with_full_waveform_padding(
    wavs: &[Vec<f32>],
) -> Result<Vec<feature_extractor::Features>> {
    // WhisperFeatureExtractor uses hop_length=160 for Qwen3-ASR.
    const HOP_LENGTH: usize = 160;

    let max_samples = wavs.iter().map(Vec::len).max().unwrap_or(0);
    wavs.par_iter()
        .map(|wav| {
            let real_len = wav.len();
            let mut padded = wav.clone();
            if real_len < max_samples {
                padded.resize(max_samples, 0.0);
            }

            let mut f = feature_extractor::extract_features(&padded)?;

            let frames_total = padded.len() / HOP_LENGTH;
            if f.feature_attention_mask.len() != frames_total {
                bail!(
                    "internal error: feature_attention_mask len {} != frames_total {}",
                    f.feature_attention_mask.len(),
                    frames_total
                );
            }

            let mut mask: Vec<u8> = Vec::with_capacity(frames_total);
            for i in 0..frames_total {
                let sample_idx = i.saturating_mul(HOP_LENGTH);
                mask.push(u8::from(sample_idx < real_len));
            }
            f.feature_attention_mask = mask;
            Ok(f)
        })
        .collect()
}

fn pad_feature_batch_to_max_frames(features: &mut [feature_extractor::Features]) -> Result<()> {
    let max_frames = features
        .iter()
        .filter_map(|f| f.input_features.first().map(Vec::len))
        .max()
        .unwrap_or(0);

    for (i, f) in features.iter_mut().enumerate() {
        if f.input_features.is_empty() {
            bail!("features[{i}] has no mel rows");
        }
        let frames = f
            .input_features
            .first()
            .map(Vec::len)
            .ok_or_else(|| anyhow::anyhow!("features[{i}] missing first mel row"))?;
        if f.feature_attention_mask.len() != frames {
            bail!(
                "features[{i}] mask len mismatch: mask={} frames={frames}",
                f.feature_attention_mask.len()
            );
        }
        if frames > max_frames {
            bail!("features[{i}] frames {frames} exceeds max_frames {max_frames}");
        }
        let pad = max_frames - frames;
        if pad == 0 {
            continue;
        }
        for (row_idx, row) in f.input_features.iter_mut().enumerate() {
            if row.len() != frames {
                bail!(
                    "features[{i}].input_features[{row_idx}] len mismatch: expected={frames}, got={}",
                    row.len()
                );
            }
            row.extend(std::iter::repeat_n(0.0f32, pad));
        }
        f.feature_attention_mask
            .extend(std::iter::repeat_n(0u8, pad));
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "timing", derive(serde::Serialize))]
pub struct PrepareBatchTimings {
    pub normalize_us: u64,
    pub token_lookup_us: u64,
    pub feature_extract_us: u64,
    pub tokenize_expand_us: u64,
    pub pad_us: u64,
}

#[cfg(feature = "timing")]
fn duration_to_us(d: std::time::Duration) -> u64 {
    let us = d.as_micros();
    if us > u128::from(u64::MAX) {
        u64::MAX
    } else {
        us as u64
    }
}

#[derive(Debug, Clone)]
pub struct PreparedInputs {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub input_features: Vec<Vec<f32>>,
    pub feature_attention_mask: Vec<u8>,
}

pub(crate) fn expand_audio_pad_ids_first(ids: &[u32], audio_pad_id: u32, n: usize) -> Vec<u32> {
    if n == 1 {
        return ids.to_vec();
    }

    let mut out: Vec<u32> = Vec::with_capacity(ids.len().saturating_add(n.saturating_sub(1)));
    let mut expanded = false;
    for &id in ids {
        if !expanded && id == audio_pad_id {
            out.extend(std::iter::repeat_n(audio_pad_id, n));
            expanded = true;
        } else {
            out.push(id);
        }
    }
    out
}

/// Expand the `<|audio_pad|>` token to the exact number of audio encoder output positions.
///
/// The official processor replaces each occurrence of `<|audio_pad|>` with a placeholder repeated `n` times,
/// then restores the audio token. The placeholder step prevents recursive re-expansion.
pub fn expand_audio_placeholder(prompt: &str, n: usize) -> String {
    let audio_token = chat_template::AUDIO_PAD;
    let placeholder = "<|audio_placeholder|>";
    let expanded_placeholder = placeholder.repeat(n);

    // Replace only the first occurrence per audio input.
    let s = prompt.replacen(audio_token, &expanded_placeholder, 1);
    s.replace(placeholder, audio_token)
}

#[cfg(test)]
mod tests {
    use super::{
        FeaturePaddingMode, expand_audio_pad_ids_first, pad_feature_batch_to_max_frames,
        parse_feature_padding_mode,
    };
    use crate::processor::feature_extractor::Features;

    #[test]
    fn test_expand_audio_pad_ids_first_expands_only_first() -> anyhow::Result<()> {
        let base = vec![1u32, 2, 3, 2, 4];
        let got = expand_audio_pad_ids_first(base.as_slice(), 2, 3);
        let exp = vec![1u32, 2, 2, 2, 3, 2, 4];
        if got != exp {
            anyhow::bail!("expanded mismatch: expected={exp:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_expand_audio_pad_ids_first_n1_is_noop() -> anyhow::Result<()> {
        let base = vec![9u32, 8, 7, 8, 6];
        let got = expand_audio_pad_ids_first(base.as_slice(), 8, 1);
        if got != base {
            anyhow::bail!("expected noop for n=1: base={base:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_expand_audio_pad_ids_first_n0_removes_first() -> anyhow::Result<()> {
        let base = vec![1u32, 2, 3, 2, 4];
        let got = expand_audio_pad_ids_first(base.as_slice(), 2, 0);
        let exp = vec![1u32, 3, 2, 4];
        if got != exp {
            anyhow::bail!("removal mismatch: expected={exp:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_expand_audio_pad_ids_first_missing_token_noop() -> anyhow::Result<()> {
        let base = vec![1u32, 3, 4];
        let got = expand_audio_pad_ids_first(base.as_slice(), 2, 5);
        if got != base {
            anyhow::bail!("expected noop when token missing: base={base:?} got={got:?}");
        }
        Ok(())
    }

    #[test]
    fn test_pad_feature_batch_to_max_frames_marks_padding() -> anyhow::Result<()> {
        let mut feats = vec![
            Features {
                input_features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                feature_attention_mask: vec![1, 1],
            },
            Features {
                input_features: vec![vec![5.0], vec![6.0]],
                feature_attention_mask: vec![1],
            },
        ];

        pad_feature_batch_to_max_frames(feats.as_mut_slice())?;

        assert_eq!(
            feats[0].input_features,
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
        assert_eq!(feats[0].feature_attention_mask, vec![1, 1]);
        assert_eq!(
            feats[1].input_features,
            vec![vec![5.0, 0.0], vec![6.0, 0.0]]
        );
        assert_eq!(feats[1].feature_attention_mask, vec![1, 0]);
        Ok(())
    }

    #[test]
    fn test_parse_feature_padding_mode_accepts_feature_and_waveform() -> anyhow::Result<()> {
        assert_eq!(
            parse_feature_padding_mode(None)?,
            FeaturePaddingMode::Waveform
        );
        assert_eq!(
            parse_feature_padding_mode(Some("feature"))?,
            FeaturePaddingMode::Feature
        );
        assert_eq!(
            parse_feature_padding_mode(Some("waveform"))?,
            FeaturePaddingMode::Waveform
        );
        assert_eq!(
            parse_feature_padding_mode(Some("waveform-full"))?,
            FeaturePaddingMode::WaveformFull
        );
        assert!(parse_feature_padding_mode(Some("surprise")).is_err());
        Ok(())
    }
}
