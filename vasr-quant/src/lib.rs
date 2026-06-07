//! In-situ quantization (ISQ/AFQ) and decode matmul fast paths for VASR.

pub mod isq_linear;
#[cfg(feature = "metal-paged-attn")]
mod metal_argmax;
#[cfg(feature = "cuda")]
mod q8_mmvq;

#[cfg(feature = "metal-paged-attn")]
pub use metal_argmax::{MetalArgmaxScratch, argmax_token_id};
