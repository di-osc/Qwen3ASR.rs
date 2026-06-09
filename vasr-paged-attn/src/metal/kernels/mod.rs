use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{OnceLock, RwLock};

use candle_core::DType;
use candle_metal_kernels::metal::{Buffer, ComputePipeline, ConstantValues, Device, Library};
use objc2_metal::MTLDevice;
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLMathMode, MTLSize};

mod utils;
pub use utils::EncoderProvider;
use utils::RawBytesEncoder;

#[cfg(target_os = "macos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vasr_varlen_prefill.metallib"));

type ComputePipelineState = ComputePipeline;

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("could not lock kernel map: {0}")]
    LockError(String),
    #[error("failed to load metallib: {0}")]
    LoadLibraryError(String),
    #[error("failed to load function: {0}")]
    LoadFunctionError(String),
    #[error("failed to create pipeline: {0}")]
    FailedToCreatePipeline(String),
    #[error("dtype mismatch, got {got:?}, expected {expected:?}")]
    DTypeMismatch { expected: Vec<DType>, got: DType },
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipelineState>;

static LIBRARY: OnceLock<Library> = OnceLock::new();

#[derive(Debug, Default)]
pub struct Kernels {
    pipelines: RwLock<Pipelines>,
}

impl Kernels {
    pub fn new() -> Self {
        Self {
            pipelines: RwLock::new(Pipelines::new()),
        }
    }

    pub fn load_library(&self, device: &Device) -> Result<Library, MetalKernelError> {
        if let Some(lib) = LIBRARY.get() {
            return Ok(lib.clone());
        }
        let lib = if !KERNELS.is_empty() {
            let data = dispatch2::DispatchData::from_static_bytes(KERNELS);
            let raw = device
                .as_ref()
                .newLibraryWithData_error(&data)
                .map_err(|e| {
                    MetalKernelError::LoadLibraryError(format!(
                        "failed to load varlen prefill metallib: {e}"
                    ))
                })?;
            Library::new(raw)
        } else {
            self.compile_at_runtime(device)?
        };
        Ok(LIBRARY.get_or_init(|| lib).clone())
    }

    fn compile_at_runtime(&self, device: &Device) -> Result<Library, MetalKernelError> {
        let source = include_str!("prefill_paged_attn.metal");
        let compile_options = {
            let opts = MTLCompileOptions::new();
            opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
            opts.setMathMode(MTLMathMode::Fast);
            opts
        };
        device
            .new_library_with_source(source, Some(&compile_options))
            .map_err(|e| {
                MetalKernelError::LoadLibraryError(format!(
                    "failed to compile varlen prefill kernels: {e}"
                ))
            })
    }

    fn load_pipeline(
        &self,
        device: &Device,
        name: String,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name.clone(), None);
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let func = self
            .load_library(device)?
            .get_function(&name, None)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
        pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VarlenPrefillDType {
    F16,
    BF16,
    F32,
}

impl VarlenPrefillDType {
    pub(crate) fn from_candle(dtype: DType) -> Result<Self, MetalKernelError> {
        match dtype {
            DType::F16 => Ok(Self::F16),
            DType::BF16 => Ok(Self::BF16),
            DType::F32 => Ok(Self::F32),
            got => Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16, DType::F32],
                got,
            }),
        }
    }

    fn type_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::BF16 => "bfloat16_t",
            Self::F16 => "half",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_varlen_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: VarlenPrefillDType,
    quantized_cache: bool,
    output: &Buffer,
    q: &Buffer,
    q_offset: usize,
    k_cache: &Buffer,
    k_cache_offset: usize,
    v_cache: &Buffer,
    v_cache_offset: usize,
    block_tables: &Buffer,
    block_tables_offset: usize,
    seq_lens: &Buffer,
    seq_lens_offset: usize,
    query_start_len: &Buffer,
    query_start_len_offset: usize,
    k_scale: Option<&Buffer>,
    v_scale: Option<&Buffer>,
    num_kv_heads: i32,
    scale: f32,
    block_table_stride: i32,
    num_seqs: i32,
    num_query_heads: i32,
    num_query_tokens: i32,
    head_size: i32,
    block_size: i32,
    softcapping: f32,
    o_stride_tokens: i32,
    sliding_window: i32,
    total_num_blocks: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
) -> Result<(), MetalKernelError> {
    const TOKEN_CHUNK_SIZE: u64 = 64;
    let suffix = if quantized_cache { "_uint8_t" } else { "" };
    let name = format!(
        "chunked_prefill_{}{}_hs{head_size}_bs{block_size}_tcs{TOKEN_CHUNK_SIZE}",
        ty.type_name(),
        suffix
    );

    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_output_buffer(0, Some(output), 0);
    encoder.set_input_buffer(1, Some(q), q_offset);
    encoder.set_input_buffer(2, Some(k_cache), k_cache_offset);
    encoder.set_input_buffer(3, Some(v_cache), v_cache_offset);
    encoder.set_bytes_raw(
        4,
        core::mem::size_of_val(&num_kv_heads),
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        5,
        core::mem::size_of_val(&scale),
        &scale as *const _ as *const c_void,
    );
    encoder.set_input_buffer(6, Some(block_tables), block_tables_offset);
    encoder.set_input_buffer(7, Some(seq_lens), seq_lens_offset);
    encoder.set_bytes_raw(
        8,
        core::mem::size_of_val(&block_table_stride),
        &block_table_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        9,
        core::mem::size_of_val(&num_seqs),
        &num_seqs as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        10,
        core::mem::size_of_val(&num_query_heads),
        &num_query_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        11,
        core::mem::size_of_val(&num_query_tokens),
        &num_query_tokens as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        12,
        core::mem::size_of_val(&softcapping),
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        13,
        core::mem::size_of_val(&o_stride_tokens),
        &o_stride_tokens as *const _ as *const c_void,
    );
    encoder.set_input_buffer(14, Some(query_start_len), query_start_len_offset);
    if let Some(k_scale) = k_scale {
        encoder.set_input_buffer(16, Some(k_scale), 0);
    }
    if let Some(v_scale) = v_scale {
        encoder.set_input_buffer(17, Some(v_scale), 0);
    }
    encoder.set_bytes_raw(
        19,
        core::mem::size_of_val(&sliding_window),
        &sliding_window as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        20,
        core::mem::size_of_val(&total_num_blocks),
        &total_num_blocks as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        21,
        core::mem::size_of_val(&kv_block_stride),
        &kv_block_stride as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        22,
        core::mem::size_of_val(&kv_head_stride),
        &kv_head_stride as *const _ as *const c_void,
    );

    let num_queries_per_kv = (num_query_heads / num_kv_heads) as u64;
    let num_token_chunks = (num_query_tokens as u64).div_ceil(TOKEN_CHUNK_SIZE).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: num_queries_per_kv as usize,
            height: num_kv_heads as usize,
            depth: num_token_chunks as usize,
        },
        MTLSize {
            width: TOKEN_CHUNK_SIZE as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_dense_varlen_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: VarlenPrefillDType,
    output: &Buffer,
    q: &Buffer,
    q_offset: usize,
    k: &Buffer,
    k_offset: usize,
    v: &Buffer,
    v_offset: usize,
    seq_lens: &Buffer,
    seq_lens_offset: usize,
    query_start_len: &Buffer,
    query_start_len_offset: usize,
    num_kv_heads: i32,
    scale: f32,
    num_seqs: i32,
    num_query_heads: i32,
    num_query_tokens: i32,
    head_size: i32,
    block_size: i32,
    softcapping: f32,
    o_stride_tokens: i32,
) -> Result<(), MetalKernelError> {
    const TOKEN_CHUNK_SIZE: u64 = 64;
    let name = format!(
        "chunked_prefill_dense_{}_hs{head_size}_bs{block_size}_tcs{TOKEN_CHUNK_SIZE}",
        ty.type_name(),
    );

    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_output_buffer(0, Some(output), 0);
    encoder.set_input_buffer(1, Some(q), q_offset);
    encoder.set_input_buffer(2, Some(k), k_offset);
    encoder.set_input_buffer(3, Some(v), v_offset);
    encoder.set_bytes_raw(
        4,
        core::mem::size_of_val(&num_kv_heads),
        &num_kv_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        5,
        core::mem::size_of_val(&scale),
        &scale as *const _ as *const c_void,
    );
    encoder.set_input_buffer(6, Some(seq_lens), seq_lens_offset);
    encoder.set_bytes_raw(
        7,
        core::mem::size_of_val(&num_seqs),
        &num_seqs as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        8,
        core::mem::size_of_val(&num_query_heads),
        &num_query_heads as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        9,
        core::mem::size_of_val(&num_query_tokens),
        &num_query_tokens as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        10,
        core::mem::size_of_val(&softcapping),
        &softcapping as *const _ as *const c_void,
    );
    encoder.set_bytes_raw(
        11,
        core::mem::size_of_val(&o_stride_tokens),
        &o_stride_tokens as *const _ as *const c_void,
    );
    encoder.set_input_buffer(12, Some(query_start_len), query_start_len_offset);

    let num_queries_per_kv = (num_query_heads / num_kv_heads) as u64;
    let num_token_chunks = (num_query_tokens as u64).div_ceil(TOKEN_CHUNK_SIZE).max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: num_queries_per_kv as usize,
            height: num_kv_heads as usize,
            depth: num_token_chunks as usize,
        },
        MTLSize {
            width: TOKEN_CHUNK_SIZE as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
