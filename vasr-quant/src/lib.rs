//! In-situ quantization (ISQ/AFQ) and decode matmul fast paths for VASR.

pub mod isq_linear;
#[cfg(feature = "metal-paged-attn")]
mod metal_argmax;
#[cfg(feature = "cuda")]
mod q8_mmvq;

#[cfg(feature = "metal-paged-attn")]
pub use metal_argmax::{MetalArgmaxScratch, argmax_token_id};

// Re-export mistralrs_quant rotary kernels for Metal/CUDA accelerated RoPE
#[cfg(any(
    feature = "metal-paged-attn",
    feature = "cuda-paged-attn",
    feature = "cuda-quant"
))]
pub use mistralrs_quant::rotary::apply_rotary_qk;
