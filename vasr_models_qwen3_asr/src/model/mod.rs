//! Candle model implementation (audio tower + thinker LM + generation).

use std::path::PathBuf;

pub mod attention;
pub mod audio_encoder;
#[cfg(feature = "cuda-graph")]
pub mod cuda_graph;
pub mod generation;
pub mod isq_linear;
pub mod kv_cache;
#[cfg(feature = "metal-paged-attn")]
mod metal_argmax;
pub mod name_map;
#[cfg(feature = "paged-attn")]
pub mod paged_kv_cache;
#[cfg(feature = "cuda")]
mod q8_mmvq;
pub mod rope;
pub mod thinker;
pub mod thinker_text;
pub mod weights;

#[derive(Debug)]
pub struct AsrModel {
    pub model_dir: PathBuf,
    pub weights_paths: Vec<PathBuf>,
    pub thinker: thinker::ThinkerForConditionalGeneration,
}
