#[cfg(feature = "cuda-graph")]
use std::collections::HashSet;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda-graph")]
use candle_core::Tensor;
use candle_core::{DType, Device, Result};

#[cfg(feature = "cuda-graph")]
use crate::cuda_graph::{
    CUDA_DECODE_GRAPH_MAX_GRAPHS, DecodeCudaGraph, bucket_block_table_cols,
    cuda_graph_batch_bucket, cuda_graph_batch_buckets, cuda_graph_prewarm_batch_limit,
    pad_decode_batch_to_max,
};
#[cfg(feature = "cuda-graph")]
use crate::decode_forward::PagedCudaDecodeForward;
#[cfg(feature = "cuda-graph")]
use crate::paged_kv_cache::PagedInputMetadata;
use crate::paged_kv_cache::PagedKvCache;

#[derive(Debug, Clone, Copy)]
pub enum PagedCacheMemory {
    ContextSize(usize),
    Blocks(usize),
    /// KV pool budget = `fraction * total_vram - already_used` at allocation time (CUDA only).
    GpuMemoryFraction(f32),
}

#[derive(Debug, Clone, Copy)]
pub struct PagedCacheConfig {
    pub block_size: usize,
    pub memory: PagedCacheMemory,
}

impl Default for PagedCacheConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            memory: PagedCacheMemory::GpuMemoryFraction(0.8),
        }
    }
}

/// Estimated KV bytes for one paged block (all transformer layers).
pub fn bytes_per_paged_block(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    dtype: DType,
) -> Result<usize> {
    if num_layers == 0 || num_kv_heads == 0 || head_dim == 0 || block_size == 0 {
        candle_core::bail!("paged block sizing requires non-zero layer/head/block parameters");
    }
    let dtype_bytes = dtype.size_in_bytes();
    if dtype_bytes == 0 || 16 % dtype_bytes != 0 {
        candle_core::bail!("unsupported paged kv cache dtype: {dtype:?}");
    }
    let elems_per_layer = num_kv_heads
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(block_size))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| candle_core::Error::Msg("paged block element count overflow".into()))?;
    let elems = num_layers
        .checked_mul(elems_per_layer)
        .ok_or_else(|| candle_core::Error::Msg("paged block element count overflow".into()))?;
    elems
        .checked_mul(dtype_bytes)
        .ok_or_else(|| candle_core::Error::Msg("paged block byte count overflow".into()))
}

#[cfg(feature = "cuda")]
fn blocks_for_byte_budget(byte_budget: usize, bytes_per_block: usize) -> Result<usize> {
    if bytes_per_block == 0 {
        candle_core::bail!("paged block byte size must be non-zero");
    }
    let usable_blocks = byte_budget / bytes_per_block;
    let num_blocks = usable_blocks.saturating_add(1);
    if num_blocks <= 1 {
        candle_core::bail!(
            "paged KV byte budget {byte_budget} is too small for one block ({bytes_per_block} bytes)"
        );
    }
    Ok(num_blocks)
}

#[cfg(feature = "cuda")]
fn cuda_memory_mib(gpu_id: usize) -> Result<(u64, u64)> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let child = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            &gpu_id.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            candle_core::Error::Msg(format!(
                "failed to launch nvidia-smi for GPU memory query: {err}"
            ))
        })?;
    let stdout = child
        .stdout
        .ok_or_else(|| candle_core::Error::Msg("nvidia-smi produced no stdout".into()))?;
    let mut lines = BufReader::new(stdout).lines();
    let line = lines
        .next()
        .transpose()
        .map_err(|err| candle_core::Error::Msg(format!("failed to read nvidia-smi output: {err}")))?
        .ok_or_else(|| candle_core::Error::Msg("nvidia-smi returned no GPU memory stats".into()))?;
    let mut parts = line.split(',').map(str::trim);
    let total_mib = parts
        .next()
        .ok_or_else(|| candle_core::Error::Msg("nvidia-smi missing memory.total".into()))?
        .parse::<u64>()
        .map_err(|err| {
            candle_core::Error::Msg(format!("invalid nvidia-smi memory.total {line:?}: {err}"))
        })?;
    let used_mib = parts
        .next()
        .ok_or_else(|| candle_core::Error::Msg("nvidia-smi missing memory.used".into()))?
        .parse::<u64>()
        .map_err(|err| {
            candle_core::Error::Msg(format!("invalid nvidia-smi memory.used {line:?}: {err}"))
        })?;
    Ok((total_mib, used_mib))
}

#[cfg(feature = "cuda")]
fn cuda_kv_byte_budget(gpu_id: usize, fraction: f32) -> Result<usize> {
    let (total_mib, used_mib) = cuda_memory_mib(gpu_id)?;
    let target_mib = ((total_mib as f64) * f64::from(fraction)).floor() as u64;
    let budget_mib = target_mib.saturating_sub(used_mib);
    if budget_mib == 0 {
        candle_core::bail!(
            "GPU KV budget exhausted: total_mib={total_mib} used_mib={used_mib} fraction={fraction}"
        );
    }
    usize::try_from(budget_mib)
        .map_err(|_| candle_core::Error::Msg("GPU KV budget MiB overflow".into()))?
        .checked_mul(1024 * 1024)
        .ok_or_else(|| candle_core::Error::Msg("GPU KV budget byte overflow".into()))
}

fn resolve_num_blocks(
    memory: PagedCacheMemory,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<usize> {
    match memory {
        PagedCacheMemory::ContextSize(tokens) => Ok(tokens.div_ceil(block_size) + 1),
        PagedCacheMemory::Blocks(blocks) => Ok(blocks),
        PagedCacheMemory::GpuMemoryFraction(fraction) => {
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    fraction,
                    device,
                    num_layers,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    dtype,
                );
                candle_core::bail!(
                    "GpuMemoryFraction paged cache sizing requires the `cuda` feature"
                );
            }
            #[cfg(feature = "cuda")]
            {
                if !device.is_cuda() {
                    candle_core::bail!(
                        "GpuMemoryFraction paged cache sizing requires a CUDA device; got {device:?}"
                    );
                }
                if !(fraction > 0.0 && fraction <= 1.0) {
                    candle_core::bail!("GPU memory fraction must be in (0.0, 1.0]; got {fraction}");
                }
                let candle_core::DeviceLocation::Cuda { gpu_id } = device.location() else {
                    candle_core::bail!(
                        "GpuMemoryFraction paged cache sizing requires a CUDA device"
                    );
                };
                let byte_budget = cuda_kv_byte_budget(gpu_id, fraction)?;
                let bytes_per_block =
                    bytes_per_paged_block(num_layers, num_kv_heads, head_dim, block_size, dtype)?;
                blocks_for_byte_budget(byte_budget, bytes_per_block)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PagedCacheStats {
    pub block_size: usize,
    pub num_blocks: usize,
    pub free_blocks: usize,
    pub max_context_tokens: usize,
    pub bytes: usize,
}

#[derive(Debug)]
struct RequestBlocks {
    block_ids: Vec<usize>,
}

#[derive(Debug)]
pub struct PagedBlockManager {
    block_size: usize,
    free: VecDeque<usize>,
    request_blocks: HashMap<usize, RequestBlocks>,
}

impl PagedBlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut free = VecDeque::with_capacity(num_blocks.saturating_sub(1));
        for block_id in 1..num_blocks {
            free.push_back(block_id);
        }
        Self {
            block_size,
            free,
            request_blocks: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn slots_available_for(&self, request_id: usize, num_tokens: usize) -> bool {
        let required = num_tokens.div_ceil(self.block_size);
        let existing = self
            .request_blocks
            .get(&request_id)
            .map(|blocks| blocks.block_ids.len())
            .unwrap_or(0);
        let needed = required.saturating_sub(existing);
        needed <= self.free.len()
    }

    pub fn blocks_for_request(&self, request_id: usize) -> usize {
        self.request_blocks
            .get(&request_id)
            .map(|blocks| blocks.block_ids.len())
            .unwrap_or(0)
    }

    pub fn total_allocated_blocks(&self) -> usize {
        self.request_blocks
            .values()
            .map(|blocks| blocks.block_ids.len())
            .sum()
    }

    /// Block ids `1..num_blocks` are allocatable; block `0` is reserved for padded batches.
    pub fn allocatable_block_capacity(num_blocks: usize) -> usize {
        num_blocks.saturating_sub(1)
    }

    pub fn allocate_slots(&mut self, request_id: usize, num_tokens: usize) -> Result<()> {
        if !self.try_allocate_slots(request_id, num_tokens)? {
            candle_core::bail!(
                "paged KV cache exhausted: request_id={request_id} num_tokens={num_tokens} free_blocks={}",
                self.free.len()
            );
        }
        Ok(())
    }

    /// Grow a request allocation to cover `num_tokens`. Returns false when the pool is full.
    pub fn try_allocate_slots(&mut self, request_id: usize, num_tokens: usize) -> Result<bool> {
        let required = num_tokens.div_ceil(self.block_size);
        let existing = self
            .request_blocks
            .get(&request_id)
            .map(|blocks| blocks.block_ids.len())
            .unwrap_or(0);
        let needed = required.saturating_sub(existing);
        if needed == 0 {
            return Ok(true);
        }
        if needed > self.free.len() {
            return Ok(false);
        }
        let entry = self
            .request_blocks
            .entry(request_id)
            .or_insert_with(|| RequestBlocks {
                block_ids: Vec::with_capacity(required),
            });
        for _ in 0..needed {
            let block_id = self
                .free
                .pop_front()
                .ok_or_else(|| candle_core::Error::Msg("paged KV free list underflow".into()))?;
            entry.block_ids.push(block_id);
        }
        Ok(true)
    }

    /// Shrink a request allocation to the minimum blocks needed for `num_tokens`.
    ///
    /// Freed tail blocks are returned to the pool in reverse order (LRU-friendly), matching
    /// mistral.rs `KVCacheManager::trim_request_to_num_tokens`.
    pub fn trim_request_to_num_tokens(&mut self, request_id: usize, num_tokens: usize) {
        let num_required_blocks = num_tokens.div_ceil(self.block_size);
        let Some(entry) = self.request_blocks.get_mut(&request_id) else {
            return;
        };
        if num_required_blocks >= entry.block_ids.len() {
            return;
        }
        let mut removed: Vec<usize> = entry.block_ids.drain(num_required_blocks..).collect();
        removed.reverse();
        for block_id in removed {
            if block_id != 0 {
                self.free.push_back(block_id);
            }
        }
    }

    pub fn block_ids(&self, request_id: usize) -> Result<&[usize]> {
        self.request_blocks
            .get(&request_id)
            .map(|blocks| blocks.block_ids.as_slice())
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged KV request has no allocated blocks: request_id={request_id}"
                ))
            })
    }

    pub fn free_request(&mut self, request_id: usize) {
        if let Some(mut blocks) = self.request_blocks.remove(&request_id) {
            blocks.block_ids.reverse();
            for block_id in blocks.block_ids {
                if block_id != 0 {
                    self.free.push_back(block_id);
                }
            }
        }
    }

    pub fn free_many(&mut self, request_ids: &[usize]) {
        for &request_id in request_ids {
            self.free_request(request_id);
        }
    }

    pub fn block_tables_for(&self, request_ids: &[usize]) -> Result<Vec<Vec<usize>>> {
        request_ids
            .iter()
            .map(|&request_id| self.block_ids(request_id).map(|ids| ids.to_vec()))
            .collect()
    }
}

#[derive(Debug)]
#[cfg(feature = "cuda-graph")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CudaDecodeGraphKey {
    batch: usize,
    block_table_cols: usize,
}

#[derive(Debug)]
pub struct PagedCacheRuntime {
    cache: PagedKvCache,
    manager: PagedBlockManager,
    stats: PagedCacheStats,
    #[cfg(feature = "cuda-graph")]
    cuda_graph_max_batch: usize,
    #[cfg(feature = "cuda-graph")]
    cuda_decode_graphs_disabled: bool,
    #[cfg(feature = "cuda-graph")]
    cuda_decode_graph_disabled_keys: HashSet<CudaDecodeGraphKey>,
    #[cfg(feature = "cuda-graph")]
    cuda_decode_graphs: HashMap<CudaDecodeGraphKey, DecodeCudaGraph>,
}

impl PagedCacheRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
        config: PagedCacheConfig,
    ) -> Result<Self> {
        if !matches!(config.block_size, 8 | 16 | 32) {
            candle_core::bail!(
                "paged cache block size must be 8, 16, or 32; got {}",
                config.block_size
            );
        }
        let num_blocks = resolve_num_blocks(
            config.memory,
            num_layers,
            num_kv_heads,
            head_dim,
            config.block_size,
            dtype,
            device,
        )?;
        if num_blocks <= 1 {
            candle_core::bail!("paged cache requires at least 2 blocks including null block");
        }
        let cache = PagedKvCache::new_pool(
            num_layers,
            num_kv_heads,
            head_dim,
            config.block_size,
            num_blocks,
            dtype,
            device,
        )?;
        let bytes = cache.estimated_bytes();
        let stats = PagedCacheStats {
            block_size: config.block_size,
            num_blocks,
            free_blocks: num_blocks - 1,
            max_context_tokens: (num_blocks - 1) * config.block_size,
            bytes,
        };
        Ok(Self {
            cache,
            manager: PagedBlockManager::new(num_blocks, config.block_size),
            stats,
            #[cfg(feature = "cuda-graph")]
            cuda_graph_max_batch: 0,
            #[cfg(feature = "cuda-graph")]
            cuda_decode_graphs_disabled: false,
            #[cfg(feature = "cuda-graph")]
            cuda_decode_graph_disabled_keys: HashSet::new(),
            #[cfg(feature = "cuda-graph")]
            cuda_decode_graphs: HashMap::new(),
        })
    }

    pub fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    pub fn manager(&self) -> &PagedBlockManager {
        &self.manager
    }

    pub fn manager_mut(&mut self) -> &mut PagedBlockManager {
        &mut self.manager
    }

    pub fn block_tables_for(&self, request_ids: &[usize]) -> Result<Vec<Vec<usize>>> {
        self.manager.block_tables_for(request_ids)
    }

    pub fn stats(&self) -> PagedCacheStats {
        let mut stats = self.stats.clone();
        stats.free_blocks = self.manager.free_blocks();
        stats
    }

    #[cfg(feature = "cuda-graph")]
    pub fn cuda_decode_graph_count(&self) -> usize {
        self.cuda_decode_graphs.len()
    }

    #[cfg(feature = "cuda-graph")]
    pub fn cuda_decode_graphs_enabled(&self) -> bool {
        !self.cuda_decode_graphs_disabled && self.cuda_graph_max_batch > 0
    }

    #[cfg(feature = "cuda-graph")]
    pub fn disable_cuda_decode_graphs(&mut self) {
        self.cuda_decode_graphs_disabled = true;
    }

    #[cfg(all(feature = "cuda-graph", test))]
    fn set_cuda_graph_max_batch_for_test(&mut self, max_batch: usize) {
        self.cuda_graph_max_batch = max_batch;
    }

    #[cfg(feature = "cuda-graph")]
    fn cuda_decode_graph_key(
        &self,
        input_ids: &Tensor,
        metadata: &PagedInputMetadata,
    ) -> Result<CudaDecodeGraphKey> {
        let (actual_batch, q_len) = input_ids.dims2()?;
        if q_len != 1 {
            candle_core::bail!("CUDA decode graph requires single-token decode steps (q_len=1)");
        }
        let max_batch = self.cuda_graph_max_batch;
        if max_batch == 0 {
            candle_core::bail!("CUDA decode graphs are not prewarmed");
        }
        if actual_batch == 0 || actual_batch > max_batch {
            candle_core::bail!(
                "CUDA decode actual batch must be in 1..={max_batch}; got {actual_batch}"
            );
        }

        let meta_cols = metadata.block_tables.dim(1)?;
        let graph_cols = bucket_block_table_cols(meta_cols, self.cache.block_size());
        let graph_batch = cuda_graph_batch_bucket(actual_batch, max_batch).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "CUDA decode graph has no batch bucket for actual_batch={actual_batch} max_batch={max_batch}"
            ))
        })?;
        Ok(CudaDecodeGraphKey {
            batch: graph_batch,
            block_table_cols: graph_cols,
        })
    }

    #[cfg(feature = "cuda-graph")]
    pub fn cuda_decode_graph_enabled_for(
        &self,
        input_ids: &Tensor,
        metadata: &PagedInputMetadata,
    ) -> Result<bool> {
        if !self.cuda_decode_graphs_enabled() {
            return Ok(false);
        }
        let key = self.cuda_decode_graph_key(input_ids, metadata)?;
        Ok(!self.cuda_decode_graph_disabled_keys.contains(&key))
    }

    #[cfg(feature = "cuda-graph")]
    pub fn disable_cuda_decode_graph_for(
        &mut self,
        input_ids: &Tensor,
        metadata: &PagedInputMetadata,
    ) -> Result<()> {
        let key = self.cuda_decode_graph_key(input_ids, metadata)?;
        self.cuda_decode_graph_disabled_keys.insert(key);
        Ok(())
    }

    #[cfg(feature = "cuda-graph")]
    fn insert_cuda_decode_graph(&mut self, key: CudaDecodeGraphKey, graph: DecodeCudaGraph) {
        if self.cuda_decode_graphs.len() >= CUDA_DECODE_GRAPH_MAX_GRAPHS
            && !self.cuda_decode_graphs.contains_key(&key)
        {
            if let Some(oldest) = self.cuda_decode_graphs.keys().next().copied() {
                self.cuda_decode_graphs.remove(&oldest);
            }
        }
        self.cuda_decode_graphs.insert(key, graph);
    }

    #[cfg(feature = "cuda-graph")]
    pub fn cuda_decode_graph<M: PagedCudaDecodeForward>(
        &mut self,
        thinker: &M,
        input_ids: &Tensor,
        position_ids: &Tensor,
        metadata: &PagedInputMetadata,
    ) -> Result<Tensor> {
        if !input_ids.device().is_cuda() {
            candle_core::bail!("CUDA decode graph requires a CUDA device");
        }
        if self.cuda_decode_graphs_disabled {
            candle_core::bail!("CUDA decode graphs are disabled");
        }
        let (actual_batch, _) = input_ids.dims2()?;
        let key = self.cuda_decode_graph_key(input_ids, metadata)?;
        if self.cuda_decode_graph_disabled_keys.contains(&key) {
            candle_core::bail!("CUDA decode graph disabled for key={key:?}");
        }
        let graph_batch = key.batch;
        let graph_cols = key.block_table_cols;
        let device = input_ids.device();

        let run_graph = |graph: &DecodeCudaGraph| -> Result<Tensor> {
            if graph.captured_batch() != graph_batch {
                candle_core::bail!(
                    "CUDA decode graph batch mismatch: captured={} bucket={graph_batch}",
                    graph.captured_batch()
                );
            }
            let (padded_ids, padded_pos, padded_meta) = pad_decode_batch_to_max(
                graph_batch,
                input_ids,
                position_ids,
                metadata,
                graph.block_table_cols(),
                device,
            )
            .map_err(|err| {
                candle_core::Error::Msg(format!("CUDA decode graph batch padding failed: {err}"))
            })?;
            let logits = graph
                .replay_decode(&padded_pos, &padded_ids, &padded_meta)
                .map_err(|err| {
                    candle_core::Error::Msg(format!("CUDA decode graph replay failed: {err}"))
                })?;
            logits
                .narrow(0, 0, actual_batch)
                .map_err(|err| candle_core::Error::from(err))
        };

        if let Some(graph) = self.cuda_decode_graphs.get(&key) {
            return run_graph(graph);
        }

        let (padded_ids, padded_pos, padded_meta) = pad_decode_batch_to_max(
            graph_batch,
            input_ids,
            position_ids,
            metadata,
            graph_cols,
            device,
        )
        .map_err(|err| {
            candle_core::Error::Msg(format!("CUDA decode graph capture padding failed: {err}"))
        })?;
        let (graph, warmup_logits) = DecodeCudaGraph::capture_decode(
            thinker,
            &self.cache,
            &padded_ids,
            &padded_pos,
            &padded_meta,
        )
        .map_err(|err| {
            candle_core::Error::Msg(format!(
                "CUDA decode graph capture failed for batch={graph_batch} block_table_cols={graph_cols}: {err}"
            ))
        })?;
        self.insert_cuda_decode_graph(key, graph);
        tracing::debug!(
            "CUDA decode graph captured batch={graph_batch} block_table_cols={graph_cols}"
        );
        warmup_logits
            .narrow(0, 0, actual_batch)
            .map_err(|err| candle_core::Error::from(err))
    }

    #[cfg(feature = "cuda-graph")]
    pub fn prewarm_cuda_decode_graphs<M: PagedCudaDecodeForward>(
        &mut self,
        thinker: &M,
        device: &Device,
        max_batch: usize,
    ) -> Result<usize> {
        if !device.is_cuda() || max_batch == 0 {
            return Ok(0);
        }

        let block_size = self.cache.block_size();
        let base_block_table_cols = crate::cuda_graph::cuda_graph_block_bucket(block_size);
        let block_table_col_buckets = [base_block_table_cols, base_block_table_cols * 2];
        let limit = cuda_graph_prewarm_batch_limit(max_batch);
        self.cuda_graph_max_batch = limit;
        self.cuda_decode_graphs_disabled = false;
        self.cuda_decode_graph_disabled_keys.clear();
        self.cuda_decode_graphs.clear();

        let mut captured = 0usize;
        for block_table_cols in block_table_col_buckets {
            for batch in cuda_graph_batch_buckets(limit) {
                let graph = DecodeCudaGraph::capture_batch_warmup(
                    thinker,
                    &self.cache,
                    batch,
                    block_table_cols,
                    device,
                )
                .map_err(|err| {
                    candle_core::Error::Msg(format!(
                        "CUDA decode graph prewarm failed for batch={batch} block_table_cols={block_table_cols}: {err}"
                    ))
                })?;
                self.insert_cuda_decode_graph(
                    CudaDecodeGraphKey {
                        batch,
                        block_table_cols,
                    },
                    graph,
                );
                captured = captured.saturating_add(1);
            }
        }

        tracing::info!(
            "CUDA decode graph pre-captured {captured} graph buckets up to max_batch={limit} block_table_cols={block_table_col_buckets:?} (osc-transformers/vLLM-style padded replay)"
        );
        Ok(captured)
    }
}

pub type SharedPagedCacheRuntime = Arc<Mutex<PagedCacheRuntime>>;

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    #[cfg(feature = "cuda")]
    use super::blocks_for_byte_budget;
    use super::{
        PagedBlockManager, PagedCacheConfig, PagedCacheMemory, PagedCacheRuntime,
        bytes_per_paged_block,
    };

    #[test]
    fn bytes_per_paged_block_matches_qwen3_asr_defaults() -> anyhow::Result<()> {
        let bytes = bytes_per_paged_block(28, 8, 128, 32, DType::BF16)?;
        assert_eq!(bytes, 3_670_016);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn blocks_for_byte_budget_reserves_null_block() -> anyhow::Result<()> {
        let bytes_per_block = bytes_per_paged_block(28, 8, 128, 32, DType::BF16)?;
        let num_blocks = blocks_for_byte_budget(bytes_per_block * 10, bytes_per_block)?;
        assert_eq!(num_blocks, 11);
        Ok(())
    }

    #[test]
    fn block_manager_allocates_extends_and_frees_requests() -> anyhow::Result<()> {
        let mut manager = PagedBlockManager::new(8, 4);
        assert_eq!(manager.free_blocks(), 7);

        manager.allocate_slots(10, 5)?;
        assert_eq!(manager.block_ids(10)?.len(), 2);
        assert_eq!(manager.free_blocks(), 5);

        manager.allocate_slots(10, 9)?;
        assert_eq!(manager.block_ids(10)?.len(), 3);
        assert_eq!(manager.free_blocks(), 4);

        manager.free_request(10);
        assert_eq!(manager.free_blocks(), 7);
        Ok(())
    }

    #[test]
    fn block_manager_trim_releases_tail_blocks() -> anyhow::Result<()> {
        let mut manager = PagedBlockManager::new(8, 4);
        manager.allocate_slots(10, 12)?;
        assert_eq!(manager.block_ids(10)?.len(), 3);
        assert_eq!(manager.free_blocks(), 4);

        manager.trim_request_to_num_tokens(10, 5);
        assert_eq!(manager.block_ids(10)?.len(), 2);
        assert_eq!(manager.free_blocks(), 5);

        manager.trim_request_to_num_tokens(10, 1);
        assert_eq!(manager.block_ids(10)?.len(), 1);
        assert_eq!(manager.free_blocks(), 6);
        Ok(())
    }

    #[test]
    fn block_manager_try_allocate_reports_exhaustion() -> anyhow::Result<()> {
        let mut manager = PagedBlockManager::new(4, 4);
        assert!(manager.try_allocate_slots(1, 4)?);
        assert!(!manager.try_allocate_slots(2, 16)?);
        assert_eq!(manager.free_blocks(), 2);
        Ok(())
    }

    #[cfg(feature = "cuda-graph")]
    #[test]
    fn cuda_graph_disable_is_scoped_to_the_padded_key() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut runtime = PagedCacheRuntime::new(
            1,
            1,
            8,
            DType::F32,
            &device,
            PagedCacheConfig {
                block_size: 16,
                memory: PagedCacheMemory::Blocks(8),
            },
        )?;
        runtime.set_cuda_graph_max_batch_for_test(20);

        let input_ids = Tensor::zeros((5usize, 1usize), DType::U32, &device)?;
        let block_tables = Tensor::zeros((5usize, 17usize), DType::U32, &device)?;
        let slot_mapping = Tensor::zeros((5usize,), DType::I64, &device)?;
        let context_lens = Tensor::zeros((5usize,), DType::U32, &device)?;
        let metadata = crate::model::paged_kv_cache::PagedInputMetadata {
            slot_mapping,
            block_tables,
            context_lens,
            max_context_len: 272,
            token_attention_mask: None,
            prefill_attention_mask: None,
            prefill_causal_only: false,
            query_lens: None,
            kv_lens: None,
            cu_seqlens_q: None,
            cu_seqlens_kv: None,
            max_query_len: None,
            max_kv_len: None,
        };

        let key = runtime.cuda_decode_graph_key(&input_ids, &metadata)?;
        assert_eq!(key.batch, 8);
        assert_eq!(key.block_table_cols, 32);
        assert!(runtime.cuda_decode_graph_enabled_for(&input_ids, &metadata)?);

        runtime.disable_cuda_decode_graph_for(&input_ids, &metadata)?;
        assert!(!runtime.cuda_decode_graph_enabled_for(&input_ids, &metadata)?);

        let smaller_input_ids = Tensor::zeros((4usize, 1usize), DType::U32, &device)?;
        assert!(runtime.cuda_decode_graph_enabled_for(&smaller_input_ids, &metadata)?);
        Ok(())
    }
}
