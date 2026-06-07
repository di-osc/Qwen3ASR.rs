//! Candle model implementation (audio tower + thinker LM + generation).

use std::path::PathBuf;

pub mod attention;
pub mod audio_encoder;
pub mod generation;
pub mod kv_cache;
pub mod name_map;
#[cfg(feature = "paged-attn")]
pub mod paged_batch_engine;
pub mod rope;
pub mod thinker;
pub mod thinker_text;
pub mod weights;

pub use vasr_quant::isq_linear;

#[cfg(feature = "paged-attn")]
pub mod paged_kv_cache {
    pub use vasr_paged_attn::paged_kv_cache::*;
}

#[cfg(feature = "paged-attn")]
pub mod paged_cache_runtime {
    pub use vasr_paged_attn::paged_cache_runtime::*;
}

#[derive(Debug)]
pub struct AsrModel {
    pub model_dir: PathBuf,
    pub weights_paths: Vec<PathBuf>,
    pub thinker: thinker::ThinkerForConditionalGeneration,
}
