use anyhow::{Result, bail};
use qwen3_asr_runtime::{AudioInput, StreamOptions as RuntimeStreamOptions};

use crate::transcribe::TranscriptionResult;

#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub language: Option<String>,
    pub context: String,
    pub chunk_size_sec: f32,
    pub unfixed_chunk_num: usize,
    pub unfixed_token_num: usize,
    pub max_new_tokens: usize,
    pub audio_window_sec: Option<f32>,
    pub text_window_tokens: Option<usize>,
}

impl Default for StreamOptions {
    fn default() -> Self {
        let opts = RuntimeStreamOptions::default();
        Self {
            language: opts.language,
            context: opts.context,
            chunk_size_sec: opts.chunk_size_sec,
            unfixed_chunk_num: opts.unfixed_chunk_num,
            unfixed_token_num: opts.unfixed_token_num,
            max_new_tokens: opts.max_new_tokens,
            audio_window_sec: opts.audio_window_sec,
            text_window_tokens: opts.text_window_tokens,
        }
    }
}

impl StreamOptions {
    pub fn to_runtime(&self) -> RuntimeStreamOptions {
        RuntimeStreamOptions {
            context: self.context.clone(),
            language: self.language.clone(),
            chunk_size_sec: self.chunk_size_sec,
            unfixed_chunk_num: self.unfixed_chunk_num,
            unfixed_token_num: self.unfixed_token_num,
            max_new_tokens: self.max_new_tokens,
            audio_window_sec: self.audio_window_sec,
            text_window_tokens: self.text_window_tokens,
        }
    }
}

pub struct Qwen3AsrStream {
    inner: Option<qwen3_asr_runtime::AsrStream>,
}

impl Qwen3AsrStream {
    pub(crate) fn new(inner: qwen3_asr_runtime::AsrStream) -> Self {
        Self { inner: Some(inner) }
    }

    pub fn push_audio_chunk(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<TranscriptionResult>> {
        let stream = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("stream is already finished"))?;
        let output = stream.push_audio_chunk(&AudioInput::Waveform {
            samples,
            sample_rate,
        })?;
        Ok(output.map(TranscriptionResult::from))
    }

    pub fn finish(&mut self) -> Result<TranscriptionResult> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| anyhow::anyhow!("stream is already finished"))?;
        Ok(TranscriptionResult::from(stream.finish()?))
    }

    pub fn is_finished(&self) -> bool {
        self.inner.is_none()
    }
}

impl TryFrom<RuntimeStreamOptions> for StreamOptions {
    type Error = anyhow::Error;

    fn try_from(opts: RuntimeStreamOptions) -> Result<Self> {
        if !opts.chunk_size_sec.is_finite() || opts.chunk_size_sec <= 0.0 {
            bail!("chunk_size_sec must be finite and > 0");
        }
        Ok(Self {
            language: opts.language,
            context: opts.context,
            chunk_size_sec: opts.chunk_size_sec,
            unfixed_chunk_num: opts.unfixed_chunk_num,
            unfixed_token_num: opts.unfixed_token_num,
            max_new_tokens: opts.max_new_tokens,
            audio_window_sec: opts.audio_window_sec,
            text_window_tokens: opts.text_window_tokens,
        })
    }
}
