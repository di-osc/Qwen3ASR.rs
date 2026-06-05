//! Q8_0 decode matvec fast path adapted from mistral.rs `fast_mmvq`.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use candle_core::cuda_backend::CudaDType;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr};
use candle_core::cuda_backend::cudarc::driver::{CudaStream, SyncOnDrop};
use candle_core::op::BackpropOp;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36;
const MATRIX_ROW_PADDING: usize = 512;
pub const MMVQ_MAX_BATCH: usize = 8;

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

type FusedGluLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

unsafe extern "C" {
    fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_bf16_fused_glu(
        vx_gate: *const std::ffi::c_void,
        vx_up: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f16_fused_glu(
        vx_gate: *const std::ffi::c_void,
        vx_up: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f32_fused_glu(
        vx_gate: *const std::ffi::c_void,
        vx_up: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );
}

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl WorkspaceGuard<'_> {
    fn ptr_mut<'a>(&'a mut self, stream: &'a CudaStream) -> (u64, SyncOnDrop<'a>) {
        self.slot.slice.device_ptr_mut(stream)
    }
}

type WsMap = Mutex<HashMap<candle_core::cuda_backend::DeviceId, &'static Mutex<WorkspaceSlot>>>;

static WORKSPACE: OnceLock<WsMap> = OnceLock::new();

fn workspace_ensure<'a>(dev: &CudaDevice, bytes: usize) -> Result<WorkspaceGuard<'a>> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot {
                    slice,
                    cap: bytes.max(1),
                })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < bytes {
        slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.cap = bytes;
    }
    Ok(WorkspaceGuard {
        slot,
        _marker: std::marker::PhantomData,
    })
}

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

fn launcher(dtype: DType) -> Result<PlainLauncher> {
    match dtype {
        DType::BF16 => Ok(launch_mmvq_gguf_q8_0_bf16_plain),
        DType::F16 => Ok(launch_mmvq_gguf_q8_0_f16_plain),
        DType::F32 => Ok(launch_mmvq_gguf_q8_0_f32_plain),
        other => candle_core::bail!("q8_mmvq: unsupported activation dtype {other:?}"),
    }
}

fn fused_glu_launcher(dtype: DType) -> Result<FusedGluLauncher> {
    match dtype {
        DType::BF16 => Ok(launch_mmvq_gguf_q8_0_bf16_fused_glu),
        DType::F16 => Ok(launch_mmvq_gguf_q8_0_f16_fused_glu),
        DType::F32 => Ok(launch_mmvq_gguf_q8_0_f32_fused_glu),
        other => candle_core::bail!("q8_mmvq fused_glu: unsupported activation dtype {other:?}"),
    }
}

fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    stream: &'a CudaStream,
    lo: usize,
) -> (u64, SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

fn slice_ptr_mut_on_stream<'a, T: DeviceRepr>(
    v: &'a mut CudaSlice<T>,
    stream: &'a CudaStream,
    lo: usize,
) -> (u64, SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

fn tensor_from_cuda_slice<T: DeviceRepr + CudaDType>(
    slice: CudaSlice<T>,
    dev: &CudaDevice,
    shape: Shape,
) -> Tensor {
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        shape,
        BackpropOp::none(),
        false,
    )
}

pub fn can_run(w: &QTensor, xs: &Tensor) -> bool {
    if w.dtype() != GgmlDType::Q8_0 || !w.device().is_cuda() {
        return false;
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return false;
    }
    let flat_batch = xs.dims()[..xs.dims().len().saturating_sub(1)]
        .iter()
        .product::<usize>();
    (1..=MMVQ_MAX_BATCH).contains(&flat_batch)
}

pub fn can_run_fused_glu(gate_w: &QTensor, up_w: &QTensor, xs: &Tensor) -> bool {
    gate_w.dtype() == GgmlDType::Q8_0
        && up_w.dtype() == GgmlDType::Q8_0
        && gate_w.device().is_cuda()
        && up_w.device().is_cuda()
        && gate_w.device().same_device(&up_w.device())
        && gate_w.shape() == up_w.shape()
        && matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32)
        && {
            let flat_batch = xs.dims()[..xs.dims().len().saturating_sub(1)]
                .iter()
                .product::<usize>();
            (1..=MMVQ_MAX_BATCH).contains(&flat_batch)
        }
}

pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    if !can_run(w, xs) {
        candle_core::bail!("q8_mmvq: unsupported fast path input");
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("q8_mmvq: weight must live on CUDA");
    };
    let (nrows, ncols) = w.shape().dims2()?;
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("q8_mmvq: unexpected input rank {other:?}"),
    };
    if k != ncols {
        candle_core::bail!("q8_mmvq: shape mismatch weight [{nrows}, {ncols}] input tail {k}");
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("q8_mmvq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let mut workspace = workspace_ensure(&dev, scratch_bytes)?;
    let (scratch_ptr, scratch_guard) = workspace.ptr_mut(&stream);
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let weight_ptr = w.device_ptr()? as *const std::ffi::c_void;
    let launch = launcher(xs.dtype())?;

    match xs.dtype() {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            unsafe {
                launch_mmvq_gguf_quantize_q8_1_bf16(
                    xs_ptr as *const std::ffi::c_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                launch(
                    weight_ptr,
                    scratch_ptr,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    nrows as i32,
                    num_blocks_per_row as i32,
                    nrows as i32,
                    b_size as i32,
                    stream_ptr,
                );
            }
            drop(out_guard);
            drop(xs_guard);
            drop(scratch_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            unsafe {
                launch_mmvq_gguf_quantize_q8_1_f16(
                    xs_ptr as *const std::ffi::c_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                launch(
                    weight_ptr,
                    scratch_ptr,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    nrows as i32,
                    num_blocks_per_row as i32,
                    nrows as i32,
                    b_size as i32,
                    stream_ptr,
                );
            }
            drop(out_guard);
            drop(xs_guard);
            drop(scratch_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            unsafe {
                launch_mmvq_gguf_quantize_q8_1_f32(
                    xs_ptr as *const std::ffi::c_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                launch(
                    weight_ptr,
                    scratch_ptr,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    nrows as i32,
                    num_blocks_per_row as i32,
                    nrows as i32,
                    b_size as i32,
                    stream_ptr,
                );
            }
            drop(out_guard);
            drop(xs_guard);
            drop(scratch_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        _ => unreachable!(),
    }
}

pub fn fused_glu_silu(gate_w: &QTensor, up_w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    if !can_run_fused_glu(gate_w, up_w, xs) {
        candle_core::bail!("q8_mmvq fused_glu: unsupported fast path input");
    }
    let Device::Cuda(dev) = gate_w.device() else {
        candle_core::bail!("q8_mmvq fused_glu: gate weight must live on CUDA");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        candle_core::bail!("q8_mmvq fused_glu: up weight must live on CUDA");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!("q8_mmvq fused_glu: gate/up weights are on different CUDA devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        candle_core::bail!(
            "q8_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("q8_mmvq fused_glu: unexpected input rank {other:?}"),
    };
    if k != ncols {
        candle_core::bail!(
            "q8_mmvq fused_glu: shape mismatch weight [{nrows}, {ncols}] input tail {k}"
        );
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("q8_mmvq fused_glu: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let mut workspace = workspace_ensure(&dev, scratch_bytes)?;
    let (scratch_ptr, scratch_guard) = workspace.ptr_mut(&stream);
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let gate_ptr = gate_w.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up_w.device_ptr()? as *const std::ffi::c_void;
    let launch = fused_glu_launcher(xs.dtype())?;
    const GLU_SILU: i32 = 0;

    match xs.dtype() {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            unsafe {
                launch_mmvq_gguf_quantize_q8_1_bf16(
                    xs_ptr as *const std::ffi::c_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                launch(
                    gate_ptr,
                    up_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    nrows as i32,
                    num_blocks_per_row as i32,
                    nrows as i32,
                    b_size as i32,
                    GLU_SILU,
                    stream_ptr,
                );
            }
            drop(out_guard);
            drop(xs_guard);
            drop(scratch_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            unsafe {
                launch_mmvq_gguf_quantize_q8_1_f16(
                    xs_ptr as *const std::ffi::c_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                launch(
                    gate_ptr,
                    up_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    nrows as i32,
                    num_blocks_per_row as i32,
                    nrows as i32,
                    b_size as i32,
                    GLU_SILU,
                    stream_ptr,
                );
            }
            drop(out_guard);
            drop(xs_guard);
            drop(scratch_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            unsafe {
                launch_mmvq_gguf_quantize_q8_1_f32(
                    xs_ptr as *const std::ffi::c_void,
                    scratch_ptr,
                    k as i32,
                    k_padded as i32,
                    b_size as i32,
                    stream_ptr,
                );
                launch(
                    gate_ptr,
                    up_ptr,
                    scratch_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    nrows as i32,
                    num_blocks_per_row as i32,
                    nrows as i32,
                    b_size as i32,
                    GLU_SILU,
                    stream_ptr,
                );
            }
            drop(out_guard);
            drop(xs_guard);
            drop(scratch_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        _ => unreachable!(),
    }
}
