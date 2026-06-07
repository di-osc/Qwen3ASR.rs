//! Model forward hook used by CUDA decode graph capture/replay.

use anyhow::Result;
use candle_core::Tensor;

use crate::{PagedInputMetadata, PagedKvCache};

pub trait PagedCudaDecodeForward {
    fn forward_input_ids_with_paged_cache(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor>;
}
