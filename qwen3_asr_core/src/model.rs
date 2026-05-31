use anyhow::{Result, bail};

use crate::device::{ResolvedOptions, resolve_options};
use crate::transcribe::{TranscribeOptions, TranscriptionResult};

#[derive(Debug, Clone)]
pub struct Qwen3Asr {
    model_id_or_path: String,
    options: ResolvedOptions,
    use_flash_attn: bool,
}

impl Qwen3Asr {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &str,
        dtype: &str,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let trimmed = model_id_or_path.trim();
        if trimmed.is_empty() {
            bail!("model_id_or_path must not be empty");
        }
        let options = resolve_options(device, dtype)?;
        Ok(Self {
            model_id_or_path: trimmed.to_string(),
            options,
            use_flash_attn,
        })
    }

    pub fn model_id_or_path(&self) -> &str {
        &self.model_id_or_path
    }

    pub fn device_label(&self) -> String {
        self.options.device.to_string()
    }

    pub fn use_flash_attn(&self) -> bool {
        self.use_flash_attn
    }

    pub fn transcribe_path(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
    ) -> Result<TranscriptionResult> {
        if audio_path.trim().is_empty() {
            bail!("audio path must not be empty");
        }
        options.validate()?;
        bail!(
            "Candle Qwen3-ASR inference is not wired yet; model={} device={}",
            self.model_id_or_path,
            self.options.device
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3Asr;
    use crate::transcribe::TranscribeOptions;

    #[test]
    fn constructs_with_resolved_options() -> anyhow::Result<()> {
        let model = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", "cpu", "auto", false)?;
        assert_eq!(model.model_id_or_path(), "Qwen/Qwen3-ASR-0.6B");
        assert_eq!(model.device_label(), "cpu");
        Ok(())
    }

    #[test]
    fn transcription_reports_inference_not_wired_yet() -> anyhow::Result<()> {
        let model = Qwen3Asr::from_pretrained("Qwen/Qwen3-ASR-0.6B", "cpu", "auto", false)?;
        let err = model
            .transcribe_path("audio.wav", TranscribeOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("Candle Qwen3-ASR inference is not wired yet"));
        Ok(())
    }
}
