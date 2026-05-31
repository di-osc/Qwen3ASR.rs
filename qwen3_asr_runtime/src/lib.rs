//! Qwen3-ASR implementation for Rust using Candle.

pub mod audio;
pub mod config;
pub mod error;
pub mod inference;
pub mod model;
pub mod processor;

#[cfg(feature = "forced-aligner")]
pub mod forced_aligner;

use anyhow::Result;
use candle_core::Device;

pub use audio::input::AudioInput;
pub use inference::streaming::AsrStream;
#[cfg(feature = "timing")]
pub use inference::transcribe::TranscribeTimings;
pub use inference::types::{AsrTranscription, Batch, StreamOptions, TranscribeOptions};
pub use model::weights::LoadOptions;
pub use processor::AsrProcessor;

#[derive(Debug)]
pub struct Qwen3Asr {
    device: Device,
    config: config::AsrConfig,
    processor: AsrProcessor,
    _model: model::AsrModel,
}

impl Qwen3Asr {
    /// Canonical language names supported by the official Qwen3-ASR stack.
    pub fn supported_languages(&self) -> &'static [&'static str] {
        inference::utils::SUPPORTED_LANGUAGES
    }

    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let (config, model) =
            model::weights::load_model_from_pretrained(model_id_or_path, device, opts)?;
        let thinker_type = config
            .thinker_config
            .model_type
            .as_deref()
            .unwrap_or_default();
        if thinker_type.contains("forced_aligner") {
            anyhow::bail!(
                "loaded a forced aligner checkpoint (thinker_config.model_type={thinker_type:?}); use the forced aligner API instead"
            );
        }
        let tokenizer = processor::tokenizer::Tokenizer::from_pretrained(model_id_or_path)?;
        let processor = AsrProcessor::new(tokenizer);
        Ok(Self {
            device: device.clone(),
            config,
            processor,
            _model: model,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &config::AsrConfig {
        &self.config
    }

    pub fn processor(&self) -> &AsrProcessor {
        &self.processor
    }

    pub fn transcribe(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<Vec<AsrTranscription>> {
        inference::transcribe::transcribe(
            &self._model,
            &self.processor,
            &self.device,
            &audio,
            &opts,
        )
    }

    #[cfg(feature = "timing")]
    pub fn transcribe_timed(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
        inference::transcribe::transcribe_timed(
            &self._model,
            &self.processor,
            &self.device,
            &audio,
            &opts,
        )
    }

    #[cfg(feature = "forced-aligner")]
    pub fn transcribe_with_forced_aligner(
        &self,
        forced_aligner: &forced_aligner::Qwen3ForcedAligner,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<Vec<AsrTranscription>> {
        inference::transcribe::transcribe_with_forced_aligner(
            &self._model,
            &self.processor,
            &self.device,
            forced_aligner,
            &audio,
            &opts,
        )
    }

    #[cfg(all(feature = "forced-aligner", feature = "timing"))]
    pub fn transcribe_with_forced_aligner_timed(
        &self,
        forced_aligner: &forced_aligner::Qwen3ForcedAligner,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
        inference::transcribe::transcribe_with_forced_aligner_timed(
            &self._model,
            &self.processor,
            &self.device,
            forced_aligner,
            &audio,
            &opts,
        )
    }

    pub fn start_stream(&self, opts: StreamOptions) -> Result<AsrStream<'_>> {
        inference::streaming::start_stream(&self._model, &self.processor, &self.device, &opts)
    }

    pub fn require_ready(&self) -> Result<()> {
        self.processor.require_ready()
    }
}
