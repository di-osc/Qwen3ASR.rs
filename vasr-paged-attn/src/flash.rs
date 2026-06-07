//! Flash-attention varlen metadata shared by paged prefill.

use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct FlashKMeta {
    pub max: u32,
    pub cumulative_seqlens: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct FlashParams {
    pub max_q: u32,
    pub cumulative_seqlens_q: Option<Tensor>,
    pub logical_k: FlashKMeta,
    pub causal: bool,
}
