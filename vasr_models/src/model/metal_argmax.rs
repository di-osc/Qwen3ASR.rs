use std::sync::Arc;

use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
use candle_core::{backend::BackendStorage, bail};
use mistralrs_quant::metal_kernels::{
    Kernels,
    utils::{EncoderParam, EncoderProvider},
};
use objc2_metal::MTLSize;

const MAX_K: usize = 1;
const CHUNK_SIZE: usize = 2048;

/// Reusable Metal workspace for repeated decode argmax calls.
#[derive(Debug, Default)]
pub struct MetalArgmaxScratch {
    capacity_nblocks: usize,
    block_values: Option<Arc<candle_metal_kernels::metal::Buffer>>,
    block_indices: Option<Arc<candle_metal_kernels::metal::Buffer>>,
    block_maxes: Option<Arc<candle_metal_kernels::metal::Buffer>>,
    block_sums: Option<Arc<candle_metal_kernels::metal::Buffer>>,
    packed: Option<Arc<candle_metal_kernels::metal::Buffer>>,
}

impl MetalArgmaxScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn argmax_token_id(&mut self, logits: &Tensor) -> Result<u32> {
        if !logits.device().is_metal()
            || !matches!(logits.dtype(), DType::F32 | DType::F16 | DType::BF16)
        {
            bail!("metal argmax requires Metal F32/F16/BF16 logits");
        }

        let logits = if logits.is_contiguous() {
            logits.clone()
        } else {
            logits.contiguous()?
        };
        let ncols = logits.elem_count();
        if ncols == 0 {
            bail!("metal argmax got empty logits");
        }
        let nblocks = ncols.div_ceil(CHUNK_SIZE);

        let (storage, layout) = logits.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            bail!("metal argmax requires Metal logits");
        };
        let device = storage.device().clone();
        let input_offset = layout.start_offset() * logits.dtype().size_in_bytes();

        self.ensure_buffers(&device, nblocks)?;

        let block_values = self.block_values.as_ref().expect("block_values");
        let block_indices = self.block_indices.as_ref().expect("block_indices");
        let block_maxes = self.block_maxes.as_ref().expect("block_maxes");
        let block_sums = self.block_sums.as_ref().expect("block_sums");
        let packed = self.packed.as_ref().expect("packed");

        {
            let encoder = device.command_encoder()?;
            encoder.set_label("argmax-topk-packed");
            call_topk_logits_packed_with_offset(
                device.device(),
                &encoder,
                &Kernels::new(),
                logits.dtype(),
                storage.buffer(),
                input_offset,
                block_values.as_ref(),
                block_indices.as_ref(),
                block_maxes.as_ref(),
                block_sums.as_ref(),
                packed.as_ref(),
                ncols,
                MAX_K,
                CHUNK_SIZE,
                1.0,
            )
            .map_err(|err| candle_core::Error::Msg(format!("metal argmax kernel error: {err}")))?;
        }

        let packed = Tensor::from((
            Storage::Metal(MetalStorage::new(
                packed.clone(),
                device,
                2 * MAX_K + 2,
                DType::F32,
            )),
            Shape::from(vec![2 * MAX_K + 2]),
        ));
        let packed = packed.to_vec1::<f32>()?;
        Ok(packed[MAX_K] as u32)
    }

    fn ensure_buffers(&mut self, device: &candle_core::MetalDevice, nblocks: usize) -> Result<()> {
        if nblocks <= self.capacity_nblocks {
            return Ok(());
        }
        self.block_values =
            Some(device.new_buffer(nblocks * MAX_K, DType::F32, "argmax-topk-block-values")?);
        self.block_indices =
            Some(device.new_buffer(nblocks * MAX_K, DType::U32, "argmax-topk-block-indices")?);
        self.block_maxes =
            Some(device.new_buffer(nblocks, DType::F32, "argmax-topk-block-maxes")?);
        self.block_sums = Some(device.new_buffer(nblocks, DType::F32, "argmax-topk-block-sums")?);
        self.packed = Some(device.new_buffer(2 * MAX_K + 2, DType::F32, "argmax-topk-packed")?);
        self.capacity_nblocks = nblocks;
        Ok(())
    }
}

pub fn argmax_token_id(logits: &Tensor) -> Result<u32> {
    let mut scratch = MetalArgmaxScratch::new();
    scratch.argmax_token_id(logits)
}

#[allow(clippy::too_many_arguments)]
fn call_topk_logits_packed_with_offset(
    device: &candle_metal_kernels::metal::Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input_dtype: DType,
    input: &candle_metal_kernels::metal::Buffer,
    input_offset: usize,
    block_values: &candle_metal_kernels::metal::Buffer,
    block_indices: &candle_metal_kernels::metal::Buffer,
    block_maxes: &candle_metal_kernels::metal::Buffer,
    block_sums: &candle_metal_kernels::metal::Buffer,
    packed_out: &candle_metal_kernels::metal::Buffer,
    ncols: usize,
    k: usize,
    chunk_size: usize,
    inv_temperature: f32,
) -> Result<()> {
    let nblocks = ncols.div_ceil(chunk_size);
    let stage1_name = match input_dtype {
        DType::F32 => "topk_logits_stage1_f32",
        DType::BF16 => "topk_logits_stage1_bf16",
        DType::F16 => "topk_logits_stage1_f16",
        other => bail!("unsupported metal argmax dtype {other:?}"),
    };
    let stage1 = kernels
        .load_pipeline(device, stage1_name)
        .map_err(|err| candle_core::Error::Msg(format!("metal argmax kernel error: {err}")))?;
    let stage2 = kernels
        .load_pipeline(device, "topk_logits_stage2_packed_f32")
        .map_err(|err| candle_core::Error::Msg(format!("metal argmax kernel error: {err}")))?;

    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&stage1);
    encoder.set_input_buffer(0, Some(input), input_offset);
    encoder.set_output_buffer(1, Some(block_values), 0);
    encoder.set_output_buffer(2, Some(block_indices), 0);
    encoder.set_output_buffer(3, Some(block_maxes), 0);
    encoder.set_output_buffer(4, Some(block_sums), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, ncols as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, k as i32);
    <i32 as EncoderParam>::set_param(encoder, 7, chunk_size as i32);
    <f32 as EncoderParam>::set_param(encoder, 8, inv_temperature);
    encoder.set_threadgroup_memory_length(0, chunk_size);
    let group = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(
        MTLSize {
            width: nblocks,
            height: 1,
            depth: 1,
        },
        group,
    );

    encoder.set_compute_pipeline_state(&stage2);
    encoder.set_input_buffer(0, Some(block_values), 0);
    encoder.set_input_buffer(1, Some(block_indices), 0);
    encoder.set_input_buffer(2, Some(block_maxes), 0);
    encoder.set_input_buffer(3, Some(block_sums), 0);
    encoder.set_output_buffer(4, Some(packed_out), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, nblocks as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, k as i32);
    encoder.set_threadgroup_memory_length(0, (nblocks * k).max(1));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        group,
    );
    Ok(())
}
