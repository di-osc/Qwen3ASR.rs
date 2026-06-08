//! Metal fused varlen prefill attention over paged KV cache (attention.rs kernel).

use std::sync::OnceLock;

use candle_core::{DType, Device, MetalStorage, Result, Storage, Tensor};

use crate::metal::kernels::{self, Kernels, VarlenPrefillDType};

static KERNELS: OnceLock<Kernels> = OnceLock::new();

fn kernels() -> &'static Kernels {
    KERNELS.get_or_init(Kernels::new)
}

const SUPPORTED_HEAD_DIMS: [usize; 5] = [64, 96, 128, 192, 256];
const SUPPORTED_BLOCK_SIZES: [usize; 2] = [32, 64];

pub fn supports_metal_varlen_prefill(device: &Device, head_dim: usize, block_size: usize) -> bool {
    if std::env::var_os("VASR_DISABLE_METAL_VARLEN_PREFILL").is_some() {
        return false;
    }
    device.is_metal()
        && SUPPORTED_HEAD_DIMS.contains(&head_dim)
        && SUPPORTED_BLOCK_SIZES.contains(&block_size)
}

pub fn pack_query_for_varlen_prefill(q: &Tensor, query_lens: &[usize]) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    if query_lens.len() != batch {
        candle_core::bail!(
            "query_lens batch mismatch: expected={batch}, got={}",
            query_lens.len()
        );
    }
    let mut chunks = Vec::with_capacity(batch);
    for (row, &len) in query_lens.iter().enumerate() {
        if len > seq_len {
            candle_core::bail!("query_len exceeds padded seq_len: len={len} seq_len={seq_len}");
        }
        if len == 0 {
            candle_core::bail!("query_len must be non-zero for row {row}");
        }
        let flat = q
            .narrow(0, row, 1)?
            .narrow(2, 0, len)?
            .reshape((len, heads, head_dim))?;
        chunks.push(flat);
    }
    Tensor::cat(chunks.as_slice(), 0)
}

pub fn unpack_varlen_attn_to_batched(
    attn: &Tensor,
    query_lens: &[usize],
    batch: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let (total_tokens, heads, head_dim) = attn.dims3()?;
    if query_lens.len() != batch {
        candle_core::bail!(
            "query_lens batch mismatch: expected={batch}, got={}",
            query_lens.len()
        );
    }
    let expected_total: usize = query_lens.iter().sum();
    if expected_total != total_tokens {
        candle_core::bail!(
            "varlen attn token mismatch: expected={expected_total}, got={total_tokens}"
        );
    }

    let mut rows = Vec::with_capacity(batch);
    let mut offset = 0usize;
    for &len in query_lens {
        let slice = attn
            .narrow(0, offset, len)?
            .reshape((1, heads, len, head_dim))?;
        let row_out = if len < seq_len {
            let pad = Tensor::zeros(
                (1, heads, seq_len - len, head_dim),
                attn.dtype(),
                attn.device(),
            )?;
            Tensor::cat(&[slice, pad], 2)?
        } else {
            slice
        };
        rows.push(row_out);
        offset += len;
    }
    Tensor::cat(rows.as_slice(), 0)
}

fn resolve_fp8_scales(
    key_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<Option<(Tensor, Tensor)>> {
    if key_cache.dtype() != DType::F8E4M3 {
        return Ok(None);
    }
    let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) else {
        candle_core::bail!("fp8 kv cache requires k_scale and v_scale");
    };
    Ok(Some((k_scale.clone(), v_scale.clone())))
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_varlen_prefill(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    cu_seqlens_q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    if !q.device().is_metal() {
        candle_core::bail!("metal varlen prefill requires a Metal device");
    }
    let (total_tokens, num_q_heads, head_dim) = q.dims3()?;
    let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = key_cache.dims5()?;
    let (num_blocks_v, num_kv_heads_v, head_size_v, block_size_v) = value_cache.dims4()?;
    if num_blocks != num_blocks_v
        || num_kv_heads != num_kv_heads_v
        || head_size_v != head_dim
        || block_size != block_size_v
        || head_size_kc != head_dim / x
    {
        candle_core::bail!(
            "paged cache shape mismatch for varlen prefill: key={:?} value={:?} head_dim={head_dim}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    if !supports_metal_varlen_prefill(q.device(), head_dim, block_size) {
        candle_core::bail!(
            "unsupported metal varlen prefill configuration: head_dim={head_dim} block_size={block_size}"
        );
    }

    let num_seqs = cu_seqlens_q.dim(0)? - 1;
    let (num_seqs_bt, max_num_blocks_per_seq) = block_tables.dims2()?;
    if num_seqs_bt != num_seqs || context_lens.dim(0)? != num_seqs {
        candle_core::bail!(
            "varlen metadata batch mismatch: num_seqs={num_seqs} block_tables={num_seqs_bt} context_lens={}",
            context_lens.dim(0)?
        );
    }

    let ty = VarlenPrefillDType::from_candle(q.dtype())
        .map_err(|e| candle_core::Error::Msg(format!("varlen prefill dtype: {e:?}")))?;
    let fp8_scales = resolve_fp8_scales(key_cache, k_scale, v_scale)?;
    let quantized_cache = fp8_scales.is_some();

    let dev = q.device().as_metal_device()?;

    let (q_s, q_l) = q.storage_and_layout();
    let Storage::Metal(q_s) = &*q_s else {
        candle_core::bail!("query must be a metal tensor");
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let Storage::Metal(kc_s) = &*kc_s else {
        candle_core::bail!("key_cache must be a metal tensor");
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let Storage::Metal(vc_s) = &*vc_s else {
        candle_core::bail!("value_cache must be a metal tensor");
    };
    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let Storage::Metal(bt_s) = &*bt_s else {
        candle_core::bail!("block_tables must be a metal tensor");
    };
    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let Storage::Metal(cl_s) = &*cl_s else {
        candle_core::bail!("context_lens must be a metal tensor");
    };
    let (cu_s, cu_l) = cu_seqlens_q.storage_and_layout();
    let Storage::Metal(cu_s) = &*cu_s else {
        candle_core::bail!("cu_seqlens_q must be a metal tensor");
    };
    let (o_s, o_l) = q.storage_and_layout();
    let Storage::Metal(_o_s) = &*o_s else {
        candle_core::bail!("query must be a metal tensor");
    };

    let k_scale_buf = fp8_scales.as_ref().and_then(|(k, _)| metal_buffer(k).ok());
    let v_scale_buf = fp8_scales.as_ref().and_then(|(_, v)| metal_buffer(v).ok());

    let encoder = dev.command_encoder()?;
    encoder.set_label("vasr-varlen-prefill");
    let elem_count = q.elem_count();
    let out_buf = dev.new_buffer(elem_count, q.dtype(), "vasr-varlen-prefill-out")?;

    kernels::call_varlen_prefill(
        dev.device(),
        &encoder,
        kernels(),
        ty,
        quantized_cache,
        &out_buf,
        q_s.buffer(),
        q_l.start_offset() * q.dtype().size_in_bytes(),
        kc_s.buffer(),
        kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
        vc_s.buffer(),
        vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
        bt_s.buffer(),
        bt_l.start_offset() * block_tables.dtype().size_in_bytes(),
        cl_s.buffer(),
        cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
        cu_s.buffer(),
        cu_l.start_offset() * cu_seqlens_q.dtype().size_in_bytes(),
        k_scale_buf.as_ref(),
        v_scale_buf.as_ref(),
        num_kv_heads as i32,
        scale,
        max_num_blocks_per_seq as i32,
        num_seqs as i32,
        num_q_heads as i32,
        total_tokens as i32,
        head_dim as i32,
        block_size as i32,
        softcapping,
        q_l.stride()[0] as i32,
        0,
        num_blocks as i32,
        kc_l.stride()[0] as i32,
        kc_l.stride()[1] as i32,
    )
    .map_err(|e| candle_core::Error::Msg(format!("vasr varlen prefill kernel: {e}")))?;

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            dev.clone(),
            elem_count,
            q.dtype(),
        )),
        o_l.shape().clone(),
    )))
}

fn metal_buffer(t: &Tensor) -> Result<candle_metal_kernels::metal::Buffer> {
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(s) => Ok(s.buffer().clone()),
        _ => candle_core::bail!("expected metal tensor"),
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_query_for_varlen_prefill, unpack_varlen_attn_to_batched};
    use candle_core::{Device, Tensor};

    #[test]
    fn pack_unpack_roundtrip() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1f32, (2, 4, 5, 8), &device)?;
        let lens = vec![3usize, 2];
        let packed = pack_query_for_varlen_prefill(&q, &lens)?;
        assert_eq!(packed.dims3()?, (5, 4, 8));
        let restored = unpack_varlen_attn_to_batched(&packed, &lens, 2, 5)?;
        let row0 = q.narrow(0, 0, 1)?.narrow(2, 0, 3)?;
        let got0 = restored.narrow(0, 0, 1)?.narrow(2, 0, 3)?;
        let diff = (row0 - got0)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff.abs() < 1e-6);
        Ok(())
    }
}
