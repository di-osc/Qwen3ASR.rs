//! PagedAttention KV cache, block manager, and CUDA decode graph helpers.

#[cfg(feature = "cuda-graph")]
pub mod cuda_graph;
pub mod decode_forward;
pub mod flash;
pub mod paged_cache_runtime;
pub mod paged_kv_cache;

pub use decode_forward::PagedCudaDecodeForward;
pub use flash::{FlashKMeta, FlashParams};
pub use paged_cache_runtime::{
    PagedBlockManager, PagedCacheConfig, PagedCacheMemory, PagedCacheRuntime, PagedCacheStats,
    SharedPagedCacheRuntime, bytes_per_paged_block,
};
pub use paged_kv_cache::{PAD_SLOT_ID, PagedInputMetadata, PagedKvCache};

#[cfg(feature = "paged-attn")]
pub use mistralrs_paged_attn;
