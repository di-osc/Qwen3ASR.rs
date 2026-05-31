use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, bail};
use qwen3_asr_runtime::{
    AudioInput, Batch, LoadOptions, Qwen3Asr as RuntimeQwen3Asr,
    TranscribeOptions as RuntimeTranscribeOptions,
};

use crate::device::{ResolvedOptions, resolve_options};
use crate::stream::{Qwen3AsrStream, StreamOptions};
use crate::transcribe::{TranscribeOptions, TranscriptionResult};

pub struct Qwen3Asr {
    model_id_or_path: String,
    options: ResolvedOptions,
    use_flash_attn: bool,
    isq: Option<String>,
    inner: RuntimeQwen3Asr,
}

impl fmt::Debug for Qwen3Asr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen3Asr")
            .field("model_id_or_path", &self.model_id_or_path)
            .field("options", &self.options)
            .field("use_flash_attn", &self.use_flash_attn)
            .field("isq", &self.isq)
            .finish_non_exhaustive()
    }
}

impl Qwen3Asr {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &str,
        dtype: &str,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        let trimmed = model_id_or_path.trim();
        if trimmed.is_empty() {
            bail!("model_id_or_path must not be empty");
        }
        let options = resolve_options(device, dtype)?;
        let candle_device = options.to_candle_device()?;
        let load_options = LoadOptions {
            dtype: options.dtype.to_candle(),
            use_flash_attn,
            isq: isq.map(str::to_string),
        };
        let inner = RuntimeQwen3Asr::from_pretrained(trimmed, &candle_device, &load_options)
            .with_context(|| format!("failed to load Qwen3-ASR model from {trimmed:?}"))?;
        Ok(Self {
            model_id_or_path: trimmed.to_string(),
            options,
            use_flash_attn,
            isq: isq.map(str::to_string),
            inner,
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

    pub fn isq(&self) -> Option<&str> {
        self.isq.as_deref()
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
        let runtime_options = RuntimeTranscribeOptions {
            context: Batch::one(options.context),
            language: Batch::one(options.language),
            return_timestamps: false,
            max_new_tokens: options.max_new_tokens,
            max_batch_size: 32,
            chunk_max_sec: None,
            bucket_by_length: false,
        };
        let audio_path = Path::new(audio_path);
        let mut outputs = self
            .inner
            .transcribe(vec![AudioInput::Path(audio_path)], runtime_options)
            .with_context(|| format!("failed to transcribe audio file {audio_path:?}"))?;
        let output = outputs
            .pop()
            .ok_or_else(|| anyhow::anyhow!("runtime returned no transcription results"))?;
        Ok(output.into())
    }

    pub fn start_stream(&self, options: StreamOptions) -> Result<Qwen3AsrStream> {
        let runtime_stream = self
            .inner
            .start_stream(options.to_runtime())
            .context("failed to start Qwen3-ASR stream")?;
        Ok(Qwen3AsrStream::new(runtime_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3Asr;

    #[test]
    fn missing_local_model_dir_fails_during_model_loading() {
        let missing = std::env::temp_dir().join(format!(
            "qwen3_asr_rs_missing_model_dir_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&missing);
        std::fs::create_dir_all(&missing).expect("create temp model dir");
        let err = Qwen3Asr::from_pretrained(
            missing
                .to_str()
                .unwrap_or("/tmp/qwen3_asr_rs_missing_model_dir"),
            "cpu",
            "auto",
            false,
            None,
        )
        .unwrap_err()
        .to_string();
        let _ = std::fs::remove_dir_all(&missing);
        assert!(!err.contains("Candle Qwen3-ASR inference is not wired yet"));
    }
}
