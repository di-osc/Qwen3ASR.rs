//! Flash-attention varlen metadata shared by paged prefill.

use candle_core::Tensor;

#[derive(Debug, Clone)]
pub struct FlashKMeta {
    pub max: u32,
    pub cumulative_seqlens: Option<Tensor>,
}

impl FlashKMeta {
    pub fn new(max: u32, cumulative_seqlens: Option<Tensor>) -> Self {
        Self {
            max,
            cumulative_seqlens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlashParams {
    pub max_q: u32,
    pub cumulative_seqlens_q: Option<Tensor>,
    pub logical_k: FlashKMeta,
    pub causal: bool,
}

impl FlashParams {
    pub fn new_prefill(
        max_q: Option<usize>,
        cumulative_seqlens_q: Option<Tensor>,
        max_k: Option<usize>,
        cumulative_seqlens_k: Option<Tensor>,
        causal: bool,
    ) -> Option<Self> {
        Some(Self {
            max_q: max_q?.try_into().ok()?,
            cumulative_seqlens_q,
            logical_k: FlashKMeta::new(max_k?.try_into().ok()?, cumulative_seqlens_k),
            causal,
        })
    }
}
