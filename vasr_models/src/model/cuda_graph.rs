//! CUDA graph replay for single-token paged decode.

use crate::model::isq_linear::set_linear_is_prefill;
use crate::model::paged_kv_cache::{PagedInputMetadata, PagedKvCache};
use crate::model::thinker::ThinkerForConditionalGeneration;
use anyhow::{Result, anyhow, bail};
use candle_core::cuda_backend::CudaDType;
use candle_core::cuda_backend::cudarc::driver::sys::{
    self, CUgraphInstantiate_flags, CUmemPool_attribute, CUmemoryPool, CUstreamCaptureMode,
};
use candle_core::cuda_backend::cudarc::driver::{CudaGraph as CudarcGraph, CudaStream, DevicePtr};
use candle_core::{DType, Device, Storage, Tensor};
use std::ptr;
use std::sync::Arc;

/// Context-length bucket for stable CUDA graph keys (matches mistral.rs).
pub const CUDA_GRAPH_CONTEXT_BUCKET_TOKENS: usize = 256;
/// Maximum decode batch width for vLLM-style padded CUDA graph capture/replay.
pub const CUDA_DECODE_GRAPH_MAX_BATCH: usize = 256;
/// Maximum cached graphs keyed by block-table width (context bucket).
pub const CUDA_DECODE_GRAPH_MAX_GRAPHS: usize = 256;

/// Match osc-transformers/vLLM-style decode graph batch buckets.
///
/// Small batches get exact powers of two. Larger batches use multiples of 16 so a
/// 20-way decode replays a 32-wide graph instead of padding all the way to the
/// configured service maximum.
pub fn cuda_graph_batch_bucket(actual_batch: usize, max_batch: usize) -> Option<usize> {
    if actual_batch == 0 || max_batch == 0 || actual_batch > max_batch {
        return None;
    }
    let bucket = match actual_batch {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        5..=8 => 8,
        n => n.div_ceil(16) * 16,
    };
    Some(bucket.min(max_batch))
}

pub fn cuda_graph_batch_buckets(max_batch: usize) -> Vec<usize> {
    let max_batch = cuda_graph_prewarm_batch_limit(max_batch);
    if max_batch == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for bucket in [1usize, 2, 4, 8] {
        if bucket <= max_batch {
            out.push(bucket);
        }
    }
    let mut bucket = 16usize;
    while bucket <= max_batch {
        out.push(bucket);
        bucket = bucket.saturating_add(16);
    }
    if out.last().copied() != Some(max_batch) {
        out.push(max_batch);
    }
    out
}

/// Blocks per 256-token CUDA graph bucket.
pub fn cuda_graph_block_bucket(block_size: usize) -> usize {
    CUDA_GRAPH_CONTEXT_BUCKET_TOKENS.div_ceil(block_size).max(1)
}

/// Pad block-table width to kernel-friendly power-of-two buckets.
pub fn bucket_block_table_cols(blocks: usize, block_size: usize) -> usize {
    let block_bucket = cuda_graph_block_bucket(block_size);
    if blocks == 0 {
        return block_bucket;
    }
    blocks.next_power_of_two().max(block_bucket)
}

/// Clamp configured max batch to the CUDA graph capture limit.
pub fn cuda_graph_prewarm_batch_limit(max_batch: usize) -> usize {
    max_batch.min(CUDA_DECODE_GRAPH_MAX_BATCH)
}

struct CudaGraph {
    graph: CudarcGraph,
    stream: Arc<CudaStream>,
}

impl CudaGraph {
    fn begin(stream: &CudaStream) -> Result<()> {
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| anyhow!("cuStreamBeginCapture failed: {e:?}"))
    }

    fn end(stream: Arc<CudaStream>) -> Result<Self> {
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| candle_core::Error::Msg(format!("cuStreamEndCapture failed: {e:?}")))?
            .ok_or_else(|| candle_core::Error::Msg("CUDA graph capture returned null".into()))?;
        Ok(Self { graph, stream })
    }

    fn launch(&self) -> Result<()> {
        self.graph
            .launch()
            .map_err(|e| anyhow!("cuGraphLaunch failed: {e:?}"))
    }

    fn upload(&self) -> Result<()> {
        self.graph
            .upload()
            .map_err(|e| anyhow!("cuGraphUpload failed: {e:?}"))
    }
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

pub struct DecodeCudaGraph {
    graph: CudaGraph,
    input_ids: Tensor,
    position_ids: Tensor,
    slot_mapping: Tensor,
    context_lens: Tensor,
    _block_tables: Tensor,
    output: Tensor,
}

impl DecodeCudaGraph {
    /// Capture a decode graph for a fixed batch/block-table shape (model-load prewarm).
    pub fn capture_batch_warmup(
        thinker: &ThinkerForConditionalGeneration,
        paged_cache: &PagedKvCache,
        batch: usize,
        block_table_cols: usize,
        device: &Device,
    ) -> Result<Self> {
        if batch == 0 || block_table_cols == 0 {
            bail!("CUDA graph prewarm requires non-zero batch and block table width");
        }
        let max_context_len = block_table_cols.saturating_mul(paged_cache.block_size());
        let mut block_tables = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = 1usize.saturating_add(row.saturating_mul(block_table_cols));
            let table = (0..block_table_cols)
                .map(|col| start.saturating_add(col))
                .collect::<Vec<_>>();
            block_tables.push(table);
        }
        let starts = vec![max_context_len.saturating_sub(1); batch];
        let context_lens = vec![max_context_len; batch];
        let metadata = paged_cache.input_metadata_from_block_tables(
            &block_tables,
            &starts,
            1,
            &context_lens,
            device,
        )?;
        let input_ids = Tensor::zeros((batch, 1usize), DType::U32, device)?;
        let position_ids = Tensor::zeros((3usize, batch, 1usize), DType::I64, device)?;
        let (graph, _warmup) =
            Self::capture_decode(thinker, paged_cache, &input_ids, &position_ids, &metadata)?;
        Ok(graph)
    }

    pub fn capture_decode(
        thinker: &ThinkerForConditionalGeneration,
        paged_cache: &PagedKvCache,
        input_ids: &Tensor,
        position_ids: &Tensor,
        input_metadata: &PagedInputMetadata,
    ) -> Result<(Self, Tensor)> {
        let device = input_ids.device();
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        sync_stream(&stream)?;

        let input_ids_buf = input_ids.contiguous()?.copy()?;
        let position_ids_buf = position_ids.contiguous()?.copy()?;
        let slot_mapping = input_metadata.slot_mapping.contiguous()?.copy()?;
        let context_lens = input_metadata.context_lens.contiguous()?.copy()?;
        let block_tables = input_metadata.block_tables.contiguous()?.copy()?;
        sync_stream(&stream)?;

        let graph_metadata = PagedInputMetadata {
            slot_mapping: slot_mapping.clone(),
            block_tables: block_tables.clone(),
            context_lens: context_lens.clone(),
            max_context_len: input_metadata.max_context_len,
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

        let _htod_cache = cuda.enable_cuda_graph_htod_cache();
        let _linear_decode_guard = set_linear_is_prefill(false);
        let warmup_logits = thinker
            .forward_input_ids_with_paged_cache(
                &input_ids_buf,
                &position_ids_buf,
                paged_cache,
                &graph_metadata,
            )
            .map_err(|err| anyhow!("CUDA graph capture warmup forward failed: {err}"))?;
        sync_stream(&stream)?;
        set_capture_mem_pool(&stream)?;
        CudaGraph::begin(&stream)
            .map_err(|err| anyhow!("CUDA graph capture begin failed: {err}"))?;
        let output = thinker
            .forward_input_ids_with_paged_cache(
                &input_ids_buf,
                &position_ids_buf,
                paged_cache,
                &graph_metadata,
            )
            .map_err(|err| anyhow!("CUDA graph captured forward failed: {err}"))?;
        let graph = CudaGraph::end(stream)
            .map_err(|err| anyhow!("CUDA graph capture end/instantiate failed: {err}"))?;
        graph
            .upload()
            .map_err(|err| anyhow!("CUDA graph upload failed: {err}"))?;
        sync_stream(&graph.stream)?;

        Ok((
            Self {
                graph,
                input_ids: input_ids_buf,
                position_ids: position_ids_buf,
                slot_mapping,
                context_lens,
                _block_tables: block_tables,
                output,
            },
            warmup_logits,
        ))
    }

    pub fn capture(
        thinker: &ThinkerForConditionalGeneration,
        paged_cache: &PagedKvCache,
        device: &Device,
        max_context_len: usize,
    ) -> Result<Self> {
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        sync_stream(&stream)?;
        set_capture_mem_pool(&stream)?;

        let input_ids = Tensor::zeros((1usize, 1usize), DType::U32, device)?;
        let position_ids = Tensor::zeros((3usize, 1usize, 1usize), DType::I64, device)?;
        let slot_mapping = Tensor::from_vec(vec![0i64], (1usize,), device)?;
        let context_lens = Tensor::from_vec(vec![1u32], (1usize,), device)?;
        let block_tables = paged_cache.block_tables_tensor(device)?;
        let input_metadata = PagedInputMetadata {
            slot_mapping: slot_mapping.clone(),
            block_tables: block_tables.clone(),
            context_lens: context_lens.clone(),
            max_context_len,
            token_attention_mask: None,
            prefill_attention_mask: None,
            prefill_causal_only: true,
            query_lens: None,
            kv_lens: None,
            cu_seqlens_q: None,
            cu_seqlens_kv: None,
            max_query_len: None,
            max_kv_len: None,
        };

        let _htod_cache = cuda.enable_cuda_graph_htod_cache();
        {
            let _linear_decode_guard = set_linear_is_prefill(false);
            let _warmup = thinker.forward_input_ids_with_paged_cache(
                &input_ids,
                &position_ids,
                paged_cache,
                &input_metadata,
            )?;
        }
        sync_stream(&stream)?;
        CudaGraph::begin(&stream)?;
        let _linear_decode_guard = set_linear_is_prefill(false);
        let output = thinker.forward_input_ids_with_paged_cache(
            &input_ids,
            &position_ids,
            paged_cache,
            &input_metadata,
        )?;
        let graph = CudaGraph::end(stream)?;
        graph.upload()?;
        sync_stream(&graph.stream)?;

        Ok(Self {
            graph,
            input_ids,
            position_ids,
            slot_mapping,
            context_lens,
            _block_tables: block_tables,
            output,
        })
    }

    pub fn replay_step(
        &self,
        position_ids: &Tensor,
        input_ids: &Tensor,
        slot_pos: usize,
        context_len: usize,
        _device: &Device,
    ) -> Result<Tensor> {
        let slot = i64::try_from(slot_pos)
            .map_err(|_| anyhow!("decode slot position overflows i64: {slot_pos}"))?;
        let context = u32::try_from(context_len)
            .map_err(|_| anyhow!("decode context length overflows u32: {context_len}"))?;
        let input_ids = input_ids.contiguous()?;
        let position_ids = position_ids.contiguous()?;
        copy_tensor_to_tensor_async(&input_ids, &self.input_ids, &self.graph.stream)?;
        copy_tensor_to_tensor_async(&position_ids, &self.position_ids, &self.graph.stream)?;
        copy_i64_to_tensor_async(&self.slot_mapping, slot, &self.graph.stream)?;
        copy_u32_to_tensor_async(&self.context_lens, context, &self.graph.stream)?;
        self.graph.launch()?;
        Ok(self.output.clone())
    }

    pub fn block_table_cols(&self) -> usize {
        self._block_tables.dims().get(1).copied().unwrap_or(0)
    }

    pub fn captured_batch(&self) -> usize {
        self.input_ids.dims().get(0).copied().unwrap_or(0)
    }

    pub fn replay_decode(
        &self,
        position_ids: &Tensor,
        input_ids: &Tensor,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor> {
        let input_ids = input_ids.contiguous()?;
        let position_ids = position_ids.contiguous()?;
        copy_tensor_to_tensor_async(&input_ids, &self.input_ids, &self.graph.stream)?;
        copy_tensor_to_tensor_async(&position_ids, &self.position_ids, &self.graph.stream)?;
        copy_tensor_to_tensor_async(
            &input_metadata.slot_mapping,
            &self.slot_mapping,
            &self.graph.stream,
        )?;
        copy_tensor_to_tensor_async(
            &input_metadata.context_lens,
            &self.context_lens,
            &self.graph.stream,
        )?;
        copy_tensor_to_tensor_async(
            &input_metadata.block_tables,
            &self._block_tables,
            &self.graph.stream,
        )?;
        self.graph.launch()?;
        Ok(self.output.clone())
    }
}

/// Pad an active decode batch to the captured max batch size (vLLM-style).
pub fn pad_decode_batch_to_max(
    max_batch: usize,
    input_ids: &Tensor,
    position_ids: &Tensor,
    metadata: &PagedInputMetadata,
    block_table_cols: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, PagedInputMetadata)> {
    let (actual_batch, _) = input_ids.dims2()?;
    if actual_batch == 0 || actual_batch > max_batch {
        bail!("actual batch {actual_batch} exceeds cuda graph max batch {max_batch}");
    }

    let metadata = pad_metadata_block_tables(metadata, block_table_cols, device)?;
    if actual_batch == max_batch {
        return Ok((
            input_ids.contiguous()?,
            position_ids.contiguous()?,
            metadata,
        ));
    }

    let padded_ids = input_ids.pad_with_zeros(0, 0, max_batch - actual_batch)?;

    let pos_dims = position_ids.dims();
    if pos_dims.len() != 3 || pos_dims[2] != 1 {
        bail!("expected position_ids shape (3, batch, 1), got {pos_dims:?}");
    }
    let padded_position_ids = position_ids.pad_with_zeros(1, 0, max_batch - actual_batch)?;
    let slot_mapping = metadata
        .slot_mapping
        .pad_with_zeros(0, 0, max_batch - actual_batch)?;
    let context_lens = pad_u32_vector_with_value(&metadata.context_lens, max_batch, 1, device)?;
    let block_tables = metadata
        .block_tables
        .pad_with_zeros(0, 0, max_batch - actual_batch)?;

    Ok((
        padded_ids,
        padded_position_ids,
        PagedInputMetadata {
            slot_mapping,
            context_lens,
            block_tables,
            max_context_len: metadata.max_context_len,
            token_attention_mask: None,
            prefill_attention_mask: None,
            prefill_causal_only: false,
            query_lens: None,
            kv_lens: None,
            cu_seqlens_q: None,
            cu_seqlens_kv: None,
            max_query_len: None,
            max_kv_len: None,
        },
    ))
}

pub fn pad_metadata_block_tables(
    metadata: &PagedInputMetadata,
    cols: usize,
    device: &Device,
) -> Result<PagedInputMetadata> {
    let (batch, meta_cols) = metadata.block_tables.dims2()?;
    if meta_cols > cols {
        bail!("cannot pad block tables from {meta_cols} columns down to {cols}");
    }
    if meta_cols == cols {
        return Ok(metadata.clone());
    }
    let padded = pad_block_table_cols(&metadata.block_tables, batch, meta_cols, cols, device)?;
    Ok(PagedInputMetadata {
        block_tables: padded,
        ..metadata.clone()
    })
}

fn pad_u32_vector_with_value(
    tensor: &Tensor,
    target_len: usize,
    value: u32,
    device: &Device,
) -> Result<Tensor> {
    let len = tensor.dim(0)?;
    if len > target_len {
        bail!("cannot pad vector from {len} elements down to {target_len}");
    }
    if len == target_len {
        return Ok(tensor.contiguous()?);
    }
    if value != 1 {
        bail!("only u32 vector padding with value 1 is supported, got {value}");
    }
    let pad = Tensor::ones((target_len - len,), DType::U32, device)?;
    Ok(Tensor::cat(&[tensor, &pad], 0)?)
}

fn pad_block_table_cols(
    block_tables: &Tensor,
    batch: usize,
    meta_cols: usize,
    cols: usize,
    device: &Device,
) -> Result<Tensor> {
    if meta_cols == 0 {
        return Ok(Tensor::zeros((batch, cols), DType::U32, device)?);
    }
    let tail = block_tables
        .narrow(1, meta_cols - 1, 1)?
        .broadcast_as((batch, cols - meta_cols))?
        .contiguous()?;
    Ok(Tensor::cat(&[block_tables, &tail], 1)?)
}

impl std::fmt::Debug for DecodeCudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeCudaGraph")
            .field("input_shape", &self.input_ids.dims())
            .field("position_shape", &self.position_ids.dims())
            .field("block_table_shape", &self._block_tables.dims())
            .finish()
    }
}

fn copy_i64_to_tensor_async(t: &Tensor, value: i64, stream: &CudaStream) -> Result<()> {
    if t.dtype() != DType::I64 || t.elem_count() != 1 || !t.device().is_cuda() {
        return Err(anyhow!("expected one-element CUDA i64 tensor"));
    }
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cuda_storage) = &*storage else {
        return Err(anyhow!("expected CUDA tensor"));
    };
    let slice = cuda_storage.as_cuda_slice::<i64>()?;
    let (ptr, _guard) = slice.device_ptr(stream);
    let ptr = ptr + (layout.start_offset() * std::mem::size_of::<i64>()) as u64;
    unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            ptr,
            &value as *const i64 as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("cuMemcpyHtoDAsync(i64) failed: {e:?}"))
    }
}

fn copy_u32_to_tensor_async(t: &Tensor, value: u32, stream: &CudaStream) -> Result<()> {
    if t.dtype() != DType::U32 || t.elem_count() != 1 || !t.device().is_cuda() {
        return Err(anyhow!("expected one-element CUDA u32 tensor"));
    }
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cuda_storage) = &*storage else {
        return Err(anyhow!("expected CUDA tensor"));
    };
    let slice = cuda_storage.as_cuda_slice::<u32>()?;
    let (ptr, _guard) = slice.device_ptr(stream);
    let ptr = ptr + (layout.start_offset() * std::mem::size_of::<u32>()) as u64;
    unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            ptr,
            &value as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
        .result()
        .map_err(|e| anyhow!("cuMemcpyHtoDAsync(u32) failed: {e:?}"))
    }
}

fn copy_tensor_to_tensor_async(src: &Tensor, dst: &Tensor, stream: &CudaStream) -> Result<()> {
    if src.dtype() != dst.dtype() || src.elem_count() != dst.elem_count() {
        return Err(anyhow!(
            "CUDA graph tensor copy mismatch: src dtype={:?} elems={}, dst dtype={:?} elems={}",
            src.dtype(),
            src.elem_count(),
            dst.dtype(),
            dst.elem_count()
        ));
    }
    let elem_size = src.dtype().size_in_bytes();
    let bytes = src
        .elem_count()
        .checked_mul(elem_size)
        .ok_or_else(|| anyhow!("CUDA graph tensor copy byte length overflow"))?;
    match src.dtype() {
        DType::U32 => copy_tensor_to_tensor_async_typed::<u32>(src, dst, bytes, stream),
        DType::I64 => copy_tensor_to_tensor_async_typed::<i64>(src, dst, bytes, stream),
        dtype => Err(anyhow!(
            "unsupported CUDA graph tensor copy dtype: {dtype:?}"
        )),
    }
}

fn copy_tensor_to_tensor_async_typed<
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + CudaDType,
>(
    src: &Tensor,
    dst: &Tensor,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    let (src_storage, src_layout) = src.storage_and_layout();
    let Storage::Cuda(src_cuda_storage) = &*src_storage else {
        return Err(anyhow!("expected CUDA tensor"));
    };
    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let Storage::Cuda(dst_cuda_storage) = &*dst_storage else {
        return Err(anyhow!("expected CUDA tensor"));
    };
    let src_slice = src_cuda_storage.as_cuda_slice::<T>()?;
    let dst_slice = dst_cuda_storage.as_cuda_slice::<T>()?;
    let (src_ptr, _src_guard) = src_slice.device_ptr(stream);
    let (dst_ptr, _dst_guard) = dst_slice.device_ptr(stream);
    let src_ptr = src_ptr + (src_layout.start_offset() * std::mem::size_of::<T>()) as u64;
    let dst_ptr = dst_ptr + (dst_layout.start_offset() * std::mem::size_of::<T>()) as u64;
    unsafe {
        sys::cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, stream.cu_stream())
            .result()
            .map_err(|e| anyhow!("cuMemcpyDtoDAsync failed: {e:?}"))
    }
}

fn sync_stream(stream: &CudaStream) -> Result<()> {
    unsafe {
        sys::cuStreamSynchronize(stream.cu_stream())
            .result()
            .map_err(|e| anyhow!("cuStreamSynchronize failed: {e:?}"))
    }
}

fn set_capture_mem_pool(stream: &CudaStream) -> Result<()> {
    let mut pool: CUmemoryPool = ptr::null_mut();
    let device = stream.context().cu_device();
    unsafe {
        sys::cuDeviceGetDefaultMemPool(&mut pool, device)
            .result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("cuDeviceGetDefaultMemPool failed: {e:?}"))
            })?;

        let threshold: u64 = u64::MAX;
        sys::cuMemPoolSetAttribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const _ as _,
        )
        .result()
        .map_err(|e| candle_core::Error::Msg(format!("cuMemPoolSetAttribute failed: {e:?}")))?;

        sys::cuDeviceSetMemPool(device, pool)
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use crate::model::paged_kv_cache::PagedInputMetadata;

    use super::{
        bucket_block_table_cols, cuda_graph_batch_bucket, cuda_graph_batch_buckets,
        pad_decode_batch_to_max,
    };

    #[test]
    fn cuda_graph_batch_bucket_matches_osc_transformers_schedule() {
        let cases = [
            (0, 64, None),
            (1, 64, Some(1)),
            (2, 64, Some(2)),
            (3, 64, Some(4)),
            (4, 64, Some(4)),
            (5, 64, Some(8)),
            (8, 64, Some(8)),
            (9, 64, Some(16)),
            (16, 64, Some(16)),
            (17, 64, Some(32)),
            (20, 64, Some(32)),
            (33, 64, Some(48)),
            (64, 64, Some(64)),
            (65, 64, None),
            (20, 20, Some(20)),
        ];

        for (actual, max, expected) in cases {
            assert_eq!(cuda_graph_batch_bucket(actual, max), expected);
        }
    }

    #[test]
    fn cuda_graph_batch_buckets_include_configured_max_batch() {
        assert_eq!(cuda_graph_batch_buckets(0), Vec::<usize>::new());
        assert_eq!(cuda_graph_batch_buckets(8), vec![1, 2, 4, 8]);
        assert_eq!(cuda_graph_batch_buckets(20), vec![1, 2, 4, 8, 16, 20]);
        assert_eq!(
            cuda_graph_batch_buckets(64),
            vec![1, 2, 4, 8, 16, 32, 48, 64]
        );
    }

    #[test]
    fn block_table_bucket_uses_kernel_friendly_power_of_two_widths() {
        assert_eq!(bucket_block_table_cols(0, 32), 8);
        assert_eq!(bucket_block_table_cols(1, 32), 8);
        assert_eq!(bucket_block_table_cols(8, 32), 8);
        assert_eq!(bucket_block_table_cols(9, 32), 16);
        assert_eq!(bucket_block_table_cols(16, 32), 16);
        assert_eq!(bucket_block_table_cols(17, 32), 32);
        assert_eq!(bucket_block_table_cols(24, 32), 32);
        assert_eq!(bucket_block_table_cols(33, 32), 64);
    }

    #[test]
    fn pad_decode_batch_uses_valid_dummy_rows_for_cuda_graph_capture() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let input_ids = Tensor::from_vec(vec![11u32, 22, 33], (3usize, 1usize), &device)?;
        let position_ids = Tensor::zeros((3usize, 3usize, 1usize), DType::I64, &device)?;
        let slot_mapping = Tensor::from_vec(vec![64i64, 96, 128], (3usize,), &device)?;
        let context_lens = Tensor::from_vec(vec![65u32, 97, 129], (3usize,), &device)?;
        let block_tables = Tensor::from_vec(
            vec![
                1u32, 2, 3, 4, //
                5, 6, 7, 8, //
                9, 10, 11, 12,
            ],
            (3usize, 4usize),
            &device,
        )?;
        let metadata = PagedInputMetadata {
            slot_mapping,
            block_tables,
            context_lens,
            max_context_len: 129,
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

        let (ids, pos, padded) =
            pad_decode_batch_to_max(8, &input_ids, &position_ids, &metadata, 8, &device)?;

        assert_eq!(
            ids.to_vec2::<u32>()?,
            vec![
                vec![11],
                vec![22],
                vec![33],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
                vec![0],
            ]
        );
        assert_eq!(pos.dims(), &[3usize, 8, 1]);
        assert_eq!(
            padded.slot_mapping.to_vec1::<i64>()?,
            vec![64, 96, 128, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            padded.context_lens.to_vec1::<u32>()?,
            vec![65, 97, 129, 1, 1, 1, 1, 1]
        );
        assert_eq!(padded.block_tables.dim(1)?, 8);
        assert_eq!(padded.max_context_len, 129);
        assert_eq!(
            padded.block_tables.to_vec2::<u32>()?,
            vec![
                vec![1, 2, 3, 4, 4, 4, 4, 4],
                vec![5, 6, 7, 8, 8, 8, 8, 8],
                vec![9, 10, 11, 12, 12, 12, 12, 12],
                vec![0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0],
            ]
        );
        Ok(())
    }
}
