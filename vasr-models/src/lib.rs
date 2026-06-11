//! Qwen3-ASR implementation for Rust using Candle.

pub mod audio;
pub mod config;
pub mod download;
pub mod error;
pub mod inference;
pub mod model;
pub mod processor;

#[cfg(feature = "forced-aligner")]
pub mod forced_aligner;

use anyhow::Result;
use candle_core::Device;
use std::sync::Arc;

/// Directory where ModelScope models are cached.
///
/// Respects the `VASR_MODEL_DIR` environment variable. When set, it is used
/// directly (no further path composition). Otherwise defaults to
/// `$HOME/.cache/vasr` (or `/tmp/.cache/vasr` when `$HOME` is unset).
pub fn modelscope_cache_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("VASR_MODEL_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
        .join(".cache")
        .join("vasr")
}

pub use audio::input::AudioInput;
#[cfg(feature = "paged-attn")]
pub use inference::batch_scheduler::{AsrBatchScheduler, AsrBatchSchedulerConfig};
pub use inference::streaming::AsrStream;
#[cfg(feature = "timing")]
pub use inference::transcribe::TranscribeTimings;
pub use inference::types::{AsrTranscription, Batch, StreamOptions, TranscribeOptions};
#[cfg(feature = "timing")]
pub use model::generation::GenerationTimings;
#[cfg(feature = "paged-attn")]
pub use model::paged_batch_engine::{PagedBatchConfig, PagedBatchState, PagedDecodeSlot};
pub use model::weights::LoadOptions;
pub use processor::AsrProcessor;
#[cfg(feature = "paged-attn")]
pub use vasr_paged_attn;
#[cfg(feature = "paged-attn")]
pub use vasr_paged_attn::{PagedCacheConfig, PagedCacheMemory, PagedCacheStats};
pub use vasr_quant;

pub mod qwen3_asr {
    #[cfg(feature = "timing")]
    pub use crate::GenerationTimings;
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
    paged_cache: Option<vasr_paged_attn::SharedPagedCacheRuntime>,
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
            let config = opts
                .paged_cache
                .unwrap_or_else(|| vasr_paged_attn::PagedCacheConfig {
                    block_size: 32,
                    memory: if device.is_cuda() {
                        vasr_paged_attn::PagedCacheMemory::GpuMemoryFraction(0.8)
                    } else {
                        vasr_paged_attn::PagedCacheMemory::ContextSize(100_000)
                    },
                });
            Some(Arc::new(std::sync::Mutex::new(
                vasr_paged_attn::PagedCacheRuntime::new(
                    num_layers,
                    num_kv_heads,
                    head_dim,
                    opts.dtype,
                    device,
                    config,
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

    pub fn inner_model(&self) -> &model::AsrModel {
        &self._model
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

    #[cfg(feature = "timing")]
    pub fn decode_text_timed(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<(String, u64, usize, usize)> {
        let (text, timings) = self.decode_text_timed_with_metrics(prompt, max_new_tokens, true)?;
        Ok((
            text,
            timings.decode_us,
            timings.steps,
            timings.tokens_generated,
        ))
    }

    #[cfg(feature = "timing")]
    pub fn decode_text_timed_with_metrics(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        stop_at_eos: bool,
    ) -> Result<(String, GenerationTimings)> {
        if prompt.is_empty() {
            return Ok((String::new(), GenerationTimings::default()));
        }

        let prompt = if max_new_tokens == 0 {
            format!("{prompt} ")
        } else {
            prompt.to_string()
        };
        let input_ids = self.processor.tokenizer.encode(&prompt)?;
        if input_ids.is_empty() {
            return Ok((String::new(), GenerationTimings::default()));
        }

        let attention_mask = vec![1u32; input_ids.len()];

        let eos_token_ids = if stop_at_eos {
            vec![
                self.processor
                    .tokenizer
                    .token_to_id(crate::processor::chat_template::IM_END)?,
                self.processor.tokenizer.token_to_id("<|endoftext|>")?,
            ]
        } else {
            vec![u32::MAX]
        };
        let eos_token_ids = {
            let mut ids = eos_token_ids;
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        let input_rows: [&[u32]; 1] = [input_ids.as_slice()];
        let attention_rows: [&[u32]; 1] = [attention_mask.as_slice()];

        let (generated, timings) = {
            #[cfg(feature = "paged-attn")]
            {
                crate::model::generation::greedy_generate_cached_batch_timed_with_paged_runtime(
                    &self._model.thinker,
                    self.device.as_ref(),
                    input_rows.as_slice(),
                    attention_rows.as_slice(),
                    None,
                    max_new_tokens,
                    eos_token_ids.as_slice(),
                    self.paged_cache.as_ref(),
                )?
            }
            #[cfg(not(feature = "paged-attn"))]
            {
                crate::model::generation::greedy_generate_cached_batch_timed(
                    &self._model.thinker,
                    self.device.as_ref(),
                    input_rows.as_slice(),
                    attention_rows.as_slice(),
                    None,
                    max_new_tokens,
                    eos_token_ids.as_slice(),
                )?
            }
        };

        let gen_ids = generated.into_iter().next().unwrap_or_default();
        let text = self.processor.tokenizer.decode(gen_ids.as_slice())?;
        Ok((text, timings))
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
