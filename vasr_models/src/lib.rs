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
use std::sync::Arc;

pub use audio::input::AudioInput;
#[cfg(feature = "paged-attn")]
pub use inference::batch_scheduler::{AsrBatchScheduler, AsrBatchSchedulerConfig};
pub use inference::streaming::AsrStream;
#[cfg(feature = "timing")]
pub use inference::transcribe::TranscribeTimings;
pub use inference::types::{AsrTranscription, Batch, StreamOptions, TranscribeOptions};
#[cfg(feature = "paged-attn")]
pub use model::paged_batch_engine::{PagedBatchConfig, PagedBatchState, PagedDecodeSlot};
#[cfg(feature = "paged-attn")]
pub use model::paged_cache_runtime::{PagedCacheConfig, PagedCacheStats};
pub use model::weights::LoadOptions;
pub use processor::AsrProcessor;

pub mod qwen3_asr {
    #[cfg(feature = "timing")]
    pub use crate::TranscribeTimings;
    pub use crate::audio;
    pub use crate::config;
    pub use crate::error;
    #[cfg(feature = "forced-aligner")]
    pub use crate::forced_aligner;
    pub use crate::inference;
    pub use crate::model;
    pub use crate::processor;
    #[cfg(feature = "paged-attn")]
    pub use crate::{
        AsrBatchScheduler, AsrBatchSchedulerConfig, PagedBatchConfig, PagedBatchState,
        PagedCacheConfig, PagedCacheStats, PagedDecodeSlot,
    };
    pub use crate::{
        AsrProcessor, AsrStream, AsrTranscription, AudioInput, Batch, LoadOptions, Qwen3Asr,
        StreamOptions, TranscribeOptions,
    };
}

#[derive(Debug)]
pub struct Qwen3Asr {
    device: Arc<Device>,
    config: config::AsrConfig,
    processor: Arc<AsrProcessor>,
    _model: Arc<model::AsrModel>,
    #[cfg(feature = "paged-attn")]
    paged_cache: Option<model::paged_cache_runtime::SharedPagedCacheRuntime>,
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
        #[cfg(feature = "paged-attn")]
        let paged_cache = if device.is_metal() || device.is_cuda() {
            let (num_layers, num_kv_heads, head_dim) = model.thinker.paged_cache_config();
            Some(Arc::new(std::sync::Mutex::new(
                model::paged_cache_runtime::PagedCacheRuntime::new(
                    num_layers,
                    num_kv_heads,
                    head_dim,
                    opts.dtype,
                    device,
                    opts.paged_cache.unwrap_or_default(),
                )?,
            )))
        } else {
            None
        };
        Ok(Self {
            device: Arc::new(device.clone()),
            config,
            processor: Arc::new(processor),
            _model: Arc::new(model),
            #[cfg(feature = "paged-attn")]
            paged_cache,
        })
    }

    pub fn device(&self) -> &Device {
        self.device.as_ref()
    }

    pub fn config(&self) -> &config::AsrConfig {
        &self.config
    }

    pub fn processor(&self) -> &AsrProcessor {
        self.processor.as_ref()
    }

    #[cfg(feature = "paged-attn")]
    pub fn paged_cache_stats(&self) -> Option<PagedCacheStats> {
        self.paged_cache
            .as_ref()
            .and_then(|runtime| runtime.lock().ok().map(|runtime| runtime.stats()))
    }

    /// Pre-capture a vLLM-style padded CUDA decode graph at max_batch.
    #[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
    pub fn prewarm_cuda_decode_graphs(&self, max_batch: usize) -> Result<usize> {
        let Some(runtime) = self.paged_cache.as_ref() else {
            return Ok(0);
        };
        let mut guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        Ok(guard.prewarm_cuda_decode_graphs(
            &self._model.thinker,
            self.device.as_ref(),
            max_batch,
        )?)
    }

    pub fn transcribe(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<Vec<AsrTranscription>> {
        inference::transcribe::transcribe(
            self._model.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
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
            self._model.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
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
            self._model.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
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
            self._model.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            forced_aligner,
            &audio,
            &opts,
        )
    }

    pub fn start_stream(&self, opts: StreamOptions) -> Result<AsrStream> {
        inference::streaming::start_stream(
            Arc::clone(&self._model),
            Arc::clone(&self.processor),
            Arc::clone(&self.device),
            &opts,
        )
    }

    pub fn require_ready(&self) -> Result<()> {
        self.processor.require_ready()
    }
}
