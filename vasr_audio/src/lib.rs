//! Audio loading for vASR.

use anyhow::Result;
use std::path::Path;
use vasr_data::{AudioSource, Waveform};
use vasr_models::qwen3_asr::AudioInput as RuntimeAudioInput;

#[derive(Debug, Clone)]
pub struct AudioLoadOptions {
    pub target_sample_rate: u32,
    pub target_channels: u16,
    pub normalize: bool,
}

impl Default for AudioLoadOptions {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            target_channels: 1,
            normalize: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct AudioLoader;

impl AudioLoader {
    pub fn load(&self, source: &AudioSource, _options: &AudioLoadOptions) -> Result<Waveform> {
        let samples = match source {
            AudioSource::Path(path) => {
                vasr_models::qwen3_asr::audio::normalize::normalize_audio_input(
                    &RuntimeAudioInput::Path(path),
                )?
            }
            AudioSource::Url(url) => {
                if let Some(path) = local_path_from_urlish(url) {
                    vasr_models::qwen3_asr::audio::normalize::normalize_audio_input(
                        &RuntimeAudioInput::Path(path),
                    )?
                } else {
                    vasr_models::qwen3_asr::audio::normalize::normalize_audio_input(
                        &RuntimeAudioInput::Url(url),
                    )?
                }
            }
            AudioSource::Base64(b64) => {
                vasr_models::qwen3_asr::audio::normalize::normalize_audio_input(
                    &RuntimeAudioInput::Base64(b64),
                )?
            }
            AudioSource::Bytes(bytes) => {
                let mut pcm = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
                return Ok(Waveform::from_i16_pcm(&pcm, 16_000));
            }
            AudioSource::Waveform(waveform) => return Ok(waveform.clone()),
        };
        Ok(Waveform::new(samples, 16_000))
    }
}

fn local_path_from_urlish(value: &str) -> Option<&Path> {
    if let Some(rest) = value.strip_prefix("file://") {
        return Some(Path::new(rest));
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return None;
    }
    Some(Path::new(value))
}

#[cfg(test)]
mod tests {
    use super::local_path_from_urlish;

    #[test]
    fn urlish_local_path_detection_supports_file_urls_and_plain_paths() {
        assert_eq!(
            local_path_from_urlish("file:///tmp/audio.wav").and_then(|path| path.to_str()),
            Some("/tmp/audio.wav")
        );
        assert_eq!(
            local_path_from_urlish("/tmp/audio.wav").and_then(|path| path.to_str()),
            Some("/tmp/audio.wav")
        );
        assert!(local_path_from_urlish("https://example.com/audio.wav").is_none());
    }
}
