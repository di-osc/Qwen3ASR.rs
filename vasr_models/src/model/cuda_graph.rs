//! Minimal CUDA graph replay for single-token paged decode.

use crate::model::paged_kv_cache::PagedKvCache;
use crate::model::thinker::ThinkerForConditionalGeneration;
use anyhow::{Result, anyhow};
use candle_core::cuda_backend::CudaDType;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::cudarc::driver::sys::{
    CUgraphInstantiate_flags, CUmemPool_attribute, CUmemoryPool, CUstream, CUstreamCaptureMode, lib,
};
use candle_core::{DType, Device, Storage, Tensor};
use std::mem::MaybeUninit;
use std::ptr;

struct CudaGraph {
    graph: candle_core::cuda_backend::cudarc::driver::sys::CUgraph,
    exec: candle_core::cuda_backend::cudarc::driver::sys::CUgraphExec,
    stream: CUstream,
}

impl CudaGraph {
    fn begin(stream: CUstream) -> Result<()> {
        unsafe {
            lib()
                .cuStreamBeginCapture_v2(
                    stream,
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .result()
                .map_err(|e| anyhow!("cuStreamBeginCapture failed: {e:?}"))
        }
    }

    fn end(stream: CUstream) -> Result<Self> {
        let mut graph = MaybeUninit::uninit();
        let graph = unsafe {
            lib()
                .cuStreamEndCapture(stream, graph.as_mut_ptr())
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuStreamEndCapture failed: {e:?}"))
                })?;
            graph.assume_init()
        };

        let mut exec = MaybeUninit::uninit();
        let exec = unsafe {
            lib()
                .cuGraphInstantiateWithFlags(
                    exec.as_mut_ptr(),
                    graph,
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u32
                        as u64,
                )
                .result()
                .map_err(|e| {
                    candle_core::Error::Msg(format!("cuGraphInstantiateWithFlags failed: {e:?}"))
                })?;
            exec.assume_init()
        };

        Ok(Self {
            graph,
            exec,
            stream,
        })
    }

    fn launch(&self) -> Result<()> {
        unsafe {
            lib()
                .cuGraphLaunch(self.exec, self.stream)
                .result()
                .map_err(|e| anyhow!("cuGraphLaunch failed: {e:?}"))
        }
    }

    fn upload(&self) -> Result<()> {
        unsafe {
            lib()
                .cuGraphUpload(self.exec, self.stream)
                .result()
                .map_err(|e| anyhow!("cuGraphUpload failed: {e:?}"))
        }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = lib().cuGraphExecDestroy(self.exec).result();
            let _ = lib().cuGraphDestroy(self.graph).result();
        }
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
    pub fn capture(
        thinker: &ThinkerForConditionalGeneration,
        paged_cache: &PagedKvCache,
        device: &Device,
        max_context_len: usize,
    ) -> Result<Self> {
        let cuda = device.as_cuda_device()?.cuda_device();
        let stream = *cuda.cu_stream();
        sync_stream(stream)?;
        set_capture_mem_pool(&cuda)?;

        let input_ids = Tensor::zeros((1usize, 1usize), DType::U32, device)?;
        let position_ids = Tensor::zeros((3usize, 1usize, 1usize), DType::I64, device)?;
        let slot_mapping = Tensor::from_vec(vec![0i64], (1usize,), device)?;
        let context_lens = Tensor::from_vec(vec![1u32], (1usize,), device)?;
        let block_tables = paged_cache.block_tables_tensor(device)?;
        let input_metadata = attention_rs::InputMetadata {
            is_prefill: false,
            is_mla: false,
            sequence_ids: None,
            mamba_slot_mapping: None,
            slot_mapping: slot_mapping.clone(),
            block_tables: Some(block_tables.clone()),
            context_lens: Some(context_lens.clone()),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            max_context_len,
            seqlens: None,
            flashinfer_metadata: None,
        };

        let _param_cache = candle_core::cuda_backend::cuda_param_cache_scope(true);
        CudaGraph::begin(stream)?;
        let output = thinker.forward_input_ids_with_paged_cache(
            &input_ids,
            &position_ids,
            paged_cache,
            &input_metadata,
        )?;
        let graph = CudaGraph::end(stream)?;
        graph.upload()?;
        sync_stream(stream)?;

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

    pub fn replay(
        &self,
        position_ids: &Tensor,
        input_ids: &Tensor,
        input_metadata: &attention_rs::InputMetadata,
    ) -> Result<Tensor> {
        self.input_ids.copy_(input_ids, 0)?;
        self.position_ids.copy_(position_ids, 0)?;
        self.slot_mapping.copy_(&input_metadata.slot_mapping, 0)?;
        let context_lens = input_metadata.context_lens.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("CUDA graph replay requires context_lens".into())
        })?;
        self.context_lens.copy_(context_lens, 0)?;
        self.graph.launch()?;
        Ok(self.output.clone())
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
        self.input_ids.copy_(input_ids, 0)?;
        self.position_ids.copy_(position_ids, 0)?;
        copy_i64_to_tensor_async(&self.slot_mapping, slot, self.graph.stream)?;
        copy_u32_to_tensor_async(&self.context_lens, context, self.graph.stream)?;
        self.graph.launch()?;
        Ok(self.output.clone())
    }
}

fn copy_i64_to_tensor_async(t: &Tensor, value: i64, stream: CUstream) -> Result<()> {
    if t.dtype() != DType::I64 || t.elem_count() != 1 || !t.device().is_cuda() {
        return Err(anyhow!("expected one-element CUDA i64 tensor"));
    }
    let ptr = tensor_device_ptr::<i64>(t)?;
    unsafe {
        lib()
            .cuMemcpyHtoDAsync_v2(
                ptr,
                &value as *const i64 as *const std::ffi::c_void,
                std::mem::size_of::<i64>(),
                stream,
            )
            .result()
            .map_err(|e| anyhow!("cuMemcpyHtoDAsync(i64) failed: {e:?}"))
    }
}

fn copy_u32_to_tensor_async(t: &Tensor, value: u32, stream: CUstream) -> Result<()> {
    if t.dtype() != DType::U32 || t.elem_count() != 1 || !t.device().is_cuda() {
        return Err(anyhow!("expected one-element CUDA u32 tensor"));
    }
    let ptr = tensor_device_ptr::<u32>(t)?;
    unsafe {
        lib()
            .cuMemcpyHtoDAsync_v2(
                ptr,
                &value as *const u32 as *const std::ffi::c_void,
                std::mem::size_of::<u32>(),
                stream,
            )
            .result()
            .map_err(|e| anyhow!("cuMemcpyHtoDAsync(u32) failed: {e:?}"))
    }
}

fn tensor_device_ptr<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + CudaDType>(
    t: &Tensor,
) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cuda_storage) = &*storage else {
        return Err(anyhow!("expected CUDA tensor"));
    };
    let slice = cuda_storage.as_cuda_slice::<T>()?;
    Ok(*slice.device_ptr() + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
}

fn sync_stream(stream: CUstream) -> Result<()> {
    unsafe {
        lib()
            .cuStreamSynchronize(stream)
            .result()
            .map_err(|e| anyhow!("cuStreamSynchronize failed: {e:?}"))
    }
}

fn set_capture_mem_pool(
    cuda: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaDevice>,
) -> Result<()> {
    let mut pool: CUmemoryPool = ptr::null_mut();
    unsafe {
        lib()
            .cuDeviceGetDefaultMemPool(&mut pool, *cuda.cu_device())
            .result()
            .map_err(|e| {
                candle_core::Error::Msg(format!("cuDeviceGetDefaultMemPool failed: {e:?}"))
            })?;

        let threshold: u64 = u64::MAX;
        lib()
            .cuMemPoolSetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &threshold as *const _ as _,
            )
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuMemPoolSetAttribute failed: {e:?}")))?;

        lib()
            .cuDeviceSetMemPool(*cuda.cu_device(), pool)
            .result()
            .map_err(|e| candle_core::Error::Msg(format!("cuDeviceSetMemPool failed: {e:?}")))?;
    }
    Ok(())
}
