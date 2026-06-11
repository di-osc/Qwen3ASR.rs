use anyhow::{Context, Result};
use candle_core::Device;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, TextSpan, TimeRange,
    Timeline, Token, Waveform,
};
use vasr_models::inference::utils::continuous_paged_batch_enabled;
#[cfg(feature = "timing")]
use vasr_models::qwen3_asr::TranscribeTimings;
use vasr_models::qwen3_asr::{
    AudioInput, Batch, LoadOptions, Qwen3Asr, StreamOptions as RuntimeStreamOptions,
    TranscribeOptions as RuntimeTranscribeOptions,
};

use crate::model::{AsrModel, AsrOptions, StreamingAsrModel};

pub struct Qwen3AsrModel {
    inner: Qwen3Asr,
}

impl Qwen3AsrModel {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        load_options: &LoadOptions,
    ) -> Result<Self> {
        let inner = Qwen3Asr::from_pretrained(model_id_or_path, device, load_options)
            .with_context(|| format!("failed to load Qwen3-ASR model from {model_id_or_path:?}"))?;
        Ok(Self { inner })
    }

    pub fn inner(&self) -> &Qwen3Asr {
        &self.inner
    }
}

impl AsrModel for Qwen3AsrModel {
    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        let mut timelines = self.transcribe_batch(std::slice::from_ref(waveform), options)?;
        timelines
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR returned no transcription"))
    }

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        let runtime_options = RuntimeTranscribeOptions {
            context: Batch::one(options.context.clone()),
            language: Batch::one(options.language.clone()),
            return_timestamps: false,
            max_new_tokens: options.max_new_tokens,
            max_batch_size: options.max_batch_size,
            max_batch_audio_sec: options.max_batch_audio_sec,
            chunk_max_sec: options.chunk_max_sec,
            bucket_by_length: !continuous_paged_batch_enabled(),
        };
        let audio = waveforms
            .iter()
            .map(|waveform| AudioInput::Waveform {
                samples: &waveform.samples,
                sample_rate: waveform.sample_rate,
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "timing")]
        let outputs = {
            let (outputs, timings) = self.inner.transcribe_timed(audio, runtime_options)?;
            log_transcribe_timings(waveforms.len(), &timings);
            outputs
        };
        #[cfg(not(feature = "timing"))]
        let outputs = self.inner.transcribe(audio, runtime_options)?;
        if outputs.len() != waveforms.len() {
            anyhow::bail!(
                "Qwen3-ASR returned {} transcriptions for {} inputs",
                outputs.len(),
                waveforms.len()
            );
        }
        Ok(outputs
            .into_iter()
            .zip(waveforms)
            .map(|(output, waveform)| {
                let mut timeline = Timeline::new("qwen3_asr_audio");
                let range = TimeRange::new(
                    vasr_data::DurationMs(0),
                    vasr_data::DurationMs(waveform.duration_ms().round() as u64),
                );
                if !output.language.is_empty() {
                    timeline.push(Annotation::new(
                        range,
                        AnnotationPayload::Language(output.language.clone()),
                        AnnotationSource::Model("qwen3_asr".to_string()),
                        AnnotationStatus::Final,
                    ));
                }
                timeline.push(Annotation::new(
                    range,
                    AnnotationPayload::Transcription(TextSpan {
                        text: output.text,
                        tokens: Vec::new(),
                        language: Some(output.language),
                    }),
                    AnnotationSource::Model("qwen3_asr".to_string()),
                    AnnotationStatus::Final,
                ));
                timeline
            })
            .collect())
    }

    fn start_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
        let stream = self.inner.start_stream(RuntimeStreamOptions {
            context: options.context.clone(),
            language: options.language.clone(),
            max_new_tokens: options.max_new_tokens,
            ..RuntimeStreamOptions::default()
        })?;
        Ok(Box::new(Qwen3AsrStreamModel {
            inner: Some(stream),
        }))
    }
}

#[cfg(feature = "timing")]
fn timing_sec(us: u64) -> f64 {
    us as f64 / 1_000_000.0
}

#[cfg(feature = "timing")]
fn tokens_per_sec(tokens: usize, us: u64) -> f64 {
    if us == 0 {
        0.0
    } else {
        tokens as f64 / timing_sec(us)
    }
}

#[cfg(feature = "timing")]
fn log_transcribe_timings(items: usize, t: &TranscribeTimings) {
    tracing::info!(
        target: "vasr_runtime::models::qwen3_asr",
        "qwen3_asr_timing | items={} | chunks={} | batches={} | total={:.3}s | normalize={:.3}s | chunking={:.3}s | prepare={:.3}s | prepare_norm={:.3}s | prepare_tok_lookup={:.3}s | prepare_feat={:.3}s | prepare_tok_expand={:.3}s | prepare_pad={:.3}s | stack={:.3}s | audio_encoder={:.3}s | prefill={:.3}s | prefill_tokens={} | prefill_tok_s={:.1} | prefill_inputs={:.3}s | prefill_rope={:.3}s | prefill_metadata={:.3}s | prefill_mask={:.3}s | prefill_forward={:.3}s | prefill_gather={:.3}s | prefill_decode_setup={:.3}s | prefill_argmax={:.3}s | decode={:.3}s | decode_tok_s={:.1} | decode_token_tensor={:.3}s | decode_embed={:.3}s | decode_position={:.3}s | decode_metadata={:.3}s | decode_graph_replay={:.3}s | decode_forward={:.3}s | decode_pre_argmax_sync={:.3}s | decode_argmax={:.3}s | decode_steps={} | generated_tokens={} | token_decode={:.3}s",
        items,
        t.chunks,
        t.batches,
        timing_sec(t.total_us),
        timing_sec(t.audio_normalize_us),
        timing_sec(t.audio_chunking_us),
        timing_sec(t.processor_prepare_batch_us),
        timing_sec(t.processor_prepare_normalize_us),
        timing_sec(t.processor_prepare_token_lookup_us),
        timing_sec(t.processor_prepare_feature_extract_us),
        timing_sec(t.processor_prepare_tokenize_expand_us),
        timing_sec(t.processor_prepare_pad_us),
        timing_sec(t.stack_features_us),
        timing_sec(t.audio_encoder_us),
        timing_sec(t.generation.prefill_us),
        t.generation.prefill_tokens,
        tokens_per_sec(t.generation.prefill_tokens, t.generation.prefill_us),
        timing_sec(t.generation.prefill_inputs_us),
        timing_sec(t.generation.prefill_rope_us),
        timing_sec(t.generation.prefill_metadata_us),
        timing_sec(t.generation.prefill_mask_us),
        timing_sec(t.generation.prefill_forward_us),
        timing_sec(t.generation.prefill_gather_us),
        timing_sec(t.generation.prefill_decode_setup_us),
        timing_sec(t.generation.prefill_argmax_us),
        timing_sec(t.generation.decode_us),
        tokens_per_sec(t.generation.tokens_generated, t.generation.decode_us),
        timing_sec(t.generation.decode_token_tensor_us),
        timing_sec(t.generation.decode_embed_us),
        timing_sec(t.generation.decode_position_us),
        timing_sec(t.generation.decode_metadata_us),
        timing_sec(t.generation.decode_graph_replay_us),
        timing_sec(t.generation.decode_forward_us),
        timing_sec(t.generation.decode_pre_argmax_sync_us),
        timing_sec(t.generation.decode_argmax_us),
        t.generation.steps,
        t.generation.tokens_generated,
        timing_sec(t.tokenizer_decode_us),
    );
}

struct Qwen3AsrStreamModel {
    inner: Option<vasr_models::qwen3_asr::AsrStream>,
}

impl StreamingAsrModel for Qwen3AsrStreamModel {
    fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<Annotation>> {
        let stream = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.push_audio_chunk(&AudioInput::Waveform {
            samples: &chunk.waveform.samples,
            sample_rate: chunk.waveform.sample_rate,
        })?;
        Ok(output
            .map(|output| {
                vec![Annotation::new(
                    chunk.range,
                    AnnotationPayload::Transcription(TextSpan {
                        text: output.text,
                        tokens: Vec::<Token>::new(),
                        language: Some(output.language),
                    }),
                    AnnotationSource::Model("qwen3_asr".to_string()),
                    AnnotationStatus::Partial,
                )]
            })
            .unwrap_or_default())
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.finish()?;
        Ok(vec![Annotation::new(
            TimeRange::default(),
            AnnotationPayload::Transcription(TextSpan {
                text: output.text,
                tokens: Vec::new(),
                language: Some(output.language),
            }),
            AnnotationSource::Model("qwen3_asr".to_string()),
            AnnotationStatus::Final,
        )])
    }
}
