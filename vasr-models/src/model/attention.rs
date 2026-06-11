//! Attention primitives (causal masks, GQA helpers, etc.).

use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "flash-attn")]
use candle_flash_attn::{flash_attn, flash_attn_varlen};

/// Repeat key/value heads to match the query head count (GQA/MQA).
///
/// This is equivalent to PyTorch's `repeat_interleave` over the head dimension
/// used by HuggingFace's `repeat_kv`.
pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }

    let (batch, kv_heads, seq_len, head_dim) = hidden_states.dims4()?;
    let expanded = hidden_states.unsqueeze(2)?; // (b, kv_heads, 1, s, d)
    let expanded = expanded.broadcast_as((batch, kv_heads, n_rep, seq_len, head_dim))?;
    expanded.reshape((batch, kv_heads * n_rep, seq_len, head_dim))
}

/// Create an additive causal attention mask for eager attention.
///
/// Returns a `(batch, 1, seq_len, seq_len)` mask where values are:
/// - `0.0` for allowed attention positions
/// - a large negative value (dtype min) for masked positions
///
/// This mirrors the eager mask path in Transformers 4.57 (`create_causal_mask` -> `eager_mask`)
/// for the common no-cache prefill case.
pub fn make_causal_mask(
    attention_mask: Option<&Tensor>,
    batch_size: usize,
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if seq_len == 0 {
        return Tensor::zeros((batch_size, 1usize, 0usize, 0usize), dtype, device);
    }

    let allowed = Tensor::tril2(seq_len, DType::U8, device)?; // (seq, seq)
    let allowed = allowed
        .unsqueeze(0)?
        .broadcast_as((batch_size, seq_len, seq_len))?; // (batch, seq, seq)

    let allowed = match attention_mask {
        None => allowed,
        Some(m) => {
            let (b, s) = m.dims2()?;
            if b != batch_size || s != seq_len {
                candle_core::bail!(
                    "attention_mask dims mismatch: expected=({batch_size},{seq_len}), got=({b},{s})"
                );
            }

            let key_padding = m.ne(0u32)?; // (batch, seq)
            let key_padding = key_padding
                .unsqueeze(1)?
                .broadcast_as((batch_size, seq_len, seq_len))?; // (batch, seq, seq)
            (&allowed * &key_padding)?
        }
    };

    let allowed = allowed.unsqueeze(1)?; // (batch, 1, seq, seq)
    let shape = (batch_size, 1usize, seq_len, seq_len);
    let zeros = Tensor::zeros(shape, DType::F32, device)?;
    let neg = Tensor::full(f32::MIN, shape, device)?;
    let mask = allowed.where_cond(&zeros, &neg)?;

    if dtype == DType::F32 {
        Ok(mask)
    } else {
        mask.to_dtype(dtype)
    }
}

/// Create an additive causal attention mask for cached decoding.
///
/// Returns a `(batch, 1, new_len, total_len)` mask where:
/// - `total_len = cache_len + new_len`
/// - Each new query position `q` (0..new_len) can attend to key positions
///   up to and including its absolute position `cache_len + q`.
/// - If `attention_mask` is provided, key positions where the mask is `0` are
///   always masked out.
pub fn make_causal_mask_cached(
    attention_mask: Option<&Tensor>,
    batch_size: usize,
    cache_len: usize,
    new_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_len = cache_len.saturating_add(new_len);
    if new_len == 0 {
        return Tensor::zeros((batch_size, 1usize, 0usize, total_len), dtype, device);
    }

    let total_len_u32 = u32::try_from(total_len)
        .map_err(|_| candle_core::Error::Msg(format!("total_len overflows u32: {total_len}")))?;
    let cache_len_u32 = u32::try_from(cache_len)
        .map_err(|_| candle_core::Error::Msg(format!("cache_len overflows u32: {cache_len}")))?;
    let new_len_u32 = u32::try_from(new_len)
        .map_err(|_| candle_core::Error::Msg(format!("new_len overflows u32: {new_len}")))?;
    let q_end = cache_len_u32.checked_add(new_len_u32).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "cache_len + new_len overflows u32: cache_len={cache_len} new_len={new_len}"
        ))
    })?;

    let kv = Tensor::arange(0u32, total_len_u32, device)?;
    let q = Tensor::arange(cache_len_u32, q_end, device)?;
    let kv = kv.unsqueeze(0)?.broadcast_as((new_len, total_len))?;
    let q = q.unsqueeze(1)?.broadcast_as((new_len, total_len))?;

    let allowed = kv.le(&q)?; // (new_len, total_len) u8
    let allowed = allowed
        .unsqueeze(0)?
        .broadcast_as((batch_size, new_len, total_len))?; // (batch, new_len, total_len)

    let allowed = match attention_mask {
        None => allowed,
        Some(m) => {
            let (b, s) = m.dims2()?;
            if b != batch_size || s != total_len {
                candle_core::bail!(
                    "attention_mask dims mismatch: expected=({batch_size},{total_len}), got=({b},{s})"
                );
            }
            let key_padding = m.ne(0u32)?; // (batch, total_len)
            let key_padding = key_padding
                .unsqueeze(1)?
                .broadcast_as((batch_size, new_len, total_len))?;
            (&allowed * &key_padding)?
        }
    };

    let allowed = allowed.unsqueeze(1)?; // (batch, 1, new_len, total_len)
    let shape = (batch_size, 1usize, new_len, total_len);
    let zeros = Tensor::zeros(shape, DType::F32, device)?;
    let neg = Tensor::full(f32::MIN, shape, device)?;
    let mask = allowed.where_cond(&zeros, &neg)?;

    if dtype == DType::F32 {
        Ok(mask)
    } else {
        mask.to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
pub enum AttentionMask {
    None,
    Custom(Tensor),
}

pub use vasr_paged_attn::{FlashKMeta, FlashParams};

const SDPA_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

fn valid_sdpa_head_dim(head_dim: usize) -> bool {
    SDPA_HEAD_DIMS.contains(&head_dim)
}

fn sequence_lengths_from_cu(cu: &Tensor) -> Result<Vec<usize>> {
    let cu = cu.to_vec1::<u32>()?;
    if cu.len() < 2 {
        candle_core::bail!(
            "cumulative seqlens must contain at least two entries, got {}",
            cu.len()
        );
    }
    Ok(cu
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]) as usize)
        .collect())
}

fn cumulative_seqlens_from_lengths(lengths: &[usize], device: &Device) -> Result<Tensor> {
    let mut out = Vec::with_capacity(lengths.len().saturating_add(1));
    out.push(0u32);
    let mut total = 0u32;
    for &len in lengths {
        let len_u32 = u32::try_from(len).map_err(|_| {
            candle_core::Error::Msg(format!("sequence length overflows u32: {len}"))
        })?;
        total = total.checked_add(len_u32).ok_or_else(|| {
            candle_core::Error::Msg("cumulative sequence length overflow".to_string())
        })?;
        out.push(total);
    }
    Tensor::from_vec(out, (lengths.len() + 1,), device)
}

pub fn make_flash_params(
    query_lens: &[usize],
    kv_lens: &[usize],
    device: &Device,
    causal: bool,
) -> Result<Option<FlashParams>> {
    let cu_seqlens_q = cumulative_seqlens_from_lengths(query_lens, device)?;
    let cu_seqlens_kv = cumulative_seqlens_from_lengths(kv_lens, device)?;
    let max_query_len = query_lens.iter().copied().max();
    let max_kv_len = kv_lens.iter().copied().max();
    Ok(FlashParams::new_prefill(
        max_query_len,
        Some(cu_seqlens_q),
        max_kv_len,
        Some(cu_seqlens_kv),
        causal,
    ))
}

fn attention_is_causal(flash_params: Option<&FlashParams>, causal_default: bool) -> bool {
    flash_params.map_or(causal_default, |params| params.causal)
}

fn can_try_flash_attention(q: &Tensor, mask: &AttentionMask) -> bool {
    !matches!(mask, AttentionMask::Custom(_))
        && (q.device().is_cuda() || q.device().is_cpu() && cfg!(feature = "flash-attn"))
}

fn prepare_sdpa_mask(
    q: &Tensor,
    k: &Tensor,
    mask: &AttentionMask,
    causal: bool,
) -> Result<Option<Tensor>> {
    match mask {
        AttentionMask::None => Ok(None),
        AttentionMask::Custom(mask) => {
            // For pure causal prefill (q_len == k_len) on Metal, let the backend
            // use its internal causal kernel instead of materializing an explicit mask.
            if q.device().is_metal() && causal && q.dim(2)? == k.dim(2)? {
                Ok(None)
            } else {
                Ok(Some(mask.clone()))
            }
        }
    }
}

fn run_single_sequence_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    causal: bool,
    n_kv_groups: usize,
) -> Result<Tensor> {
    if let Some(out) = accelerated_sdpa(q, k, v, mask, softmax_scale, causal)? {
        return Ok(out);
    }

    let (k, v) = if n_kv_groups > 1 {
        (repeat_kv(k, n_kv_groups)?, repeat_kv(v, n_kv_groups)?)
    } else {
        (k.clone(), v.clone())
    };

    let q = if q.device().is_metal() || q.device().is_cuda() {
        q.contiguous()?
    } else {
        q.clone()
    };
    let v = if v.device().is_metal() || v.device().is_cuda() {
        v.contiguous()?
    } else {
        v.clone()
    };
    let k_t = if k.device().is_metal() || k.device().is_cuda() {
        k.transpose(2, 3)?.contiguous()?
    } else {
        k.transpose(2, 3)?
    };
    let mut attn_weights = q.matmul(&k_t)?.affine(softmax_scale.into(), 0.0)?;
    if let Some(mask) = mask {
        attn_weights = attn_weights.broadcast_add(mask)?;
    } else if causal && q.dim(2)? > 1 {
        let causal_mask = make_causal_mask(None, q.dim(0)?, q.dim(2)?, q.dtype(), q.device())?;
        attn_weights = attn_weights.broadcast_add(&causal_mask)?;
    }
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    attn_weights.matmul(&v)
}

#[cfg(feature = "metal-paged-attn")]
fn metal_dense_varlen_prefill_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: &FlashParams,
    softmax_scale: f32,
) -> Result<Option<Tensor>> {
    if std::env::var_os("VASR_ENABLE_METAL_DENSE_VARLEN_PREFILL").is_none() {
        return Ok(None);
    }
    let (batch, _num_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, num_kv_heads, k_seq_len, k_head_dim) = k.dims4()?;
    let (v_batch, v_kv_heads, v_seq_len, v_head_dim) = v.dims4()?;
    if !q.device().is_metal()
        || batch <= 1
        || seq_len <= 1
        || batch != k_batch
        || batch != v_batch
        || seq_len != k_seq_len
        || seq_len != v_seq_len
        || head_dim != k_head_dim
        || head_dim != v_head_dim
        || num_kv_heads != v_kv_heads
        || !flash_params.causal
    {
        return Ok(None);
    }

    const BLOCK_SIZE: usize = 32;
    if !vasr_paged_attn::supports_metal_varlen_prefill(q.device(), head_dim, BLOCK_SIZE) {
        return Ok(None);
    }

    let (Some(cu_q), Some(cu_k)) = (
        flash_params.cumulative_seqlens_q.as_ref(),
        flash_params.logical_k.cumulative_seqlens.as_ref(),
    ) else {
        return Ok(None);
    };
    let query_lens = sequence_lengths_from_cu(cu_q)?;
    let kv_lens = sequence_lengths_from_cu(cu_k)?;
    if query_lens.len() != batch || kv_lens.len() != batch {
        return Ok(None);
    }
    if query_lens
        .iter()
        .zip(kv_lens.iter())
        .any(|(&q_len, &kv_len)| q_len == 0 || q_len != kv_len || q_len > seq_len)
    {
        return Ok(None);
    }
    if query_lens.iter().all(|&len| len == seq_len) {
        return Ok(None);
    }
    let total_tokens = batch
        .checked_mul(seq_len)
        .ok_or_else(|| candle_core::Error::Msg("dense varlen token count overflow".into()))?;
    let active_tokens: usize = query_lens.iter().sum();
    let padded_tokens = total_tokens.saturating_sub(active_tokens);
    if padded_tokens < BLOCK_SIZE || padded_tokens.saturating_mul(5) < active_tokens {
        return Ok(None);
    }
    let out = vasr_paged_attn::dense_attention_varlen_prefill(
        q,
        k,
        v,
        query_lens.as_slice(),
        softmax_scale,
        1.0,
        BLOCK_SIZE,
    )?;
    Ok(Some(out))
}

#[cfg(not(feature = "metal-paged-attn"))]
fn metal_dense_varlen_prefill_attention(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _flash_params: &FlashParams,
    _softmax_scale: f32,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// Packed varlen attention for gathered paged K/V: `(1, kv_heads, total_kv, dim)`.
///
/// Right-padded query batches use valid tokens at the start of each row. On Metal this
/// avoids `unpack_gathered_kv_for_attention` pad/cat by slicing packed K/V per sequence
/// and running native GQA SDPA.
fn packed_varlen_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &AttentionMask,
    flash_params: &FlashParams,
    softmax_scale: f32,
    causal: bool,
    n_kv_groups: usize,
) -> Result<Option<Tensor>> {
    let (batch, _num_heads, seq_len, head_dim) = q.dims4()?;
    if !valid_sdpa_head_dim(head_dim) {
        return Ok(None);
    }
    if !(q.device().is_metal() || q.device().is_cpu()) {
        return Ok(None);
    }

    let (kb, _num_kv_heads, total_kv, kv_head_dim) = k.dims4()?;
    let (vb, _, _, v_head_dim) = v.dims4()?;
    if kb != 1 || vb != 1 || head_dim != kv_head_dim || head_dim != v_head_dim {
        return Ok(None);
    }

    let cu_q = flash_params
        .cumulative_seqlens_q
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("packed varlen missing cu_seqlens_q".into()))?;
    let cu_k = flash_params
        .logical_k
        .cumulative_seqlens
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("packed varlen missing cu_seqlens_k".into()))?;
    let query_lens = sequence_lengths_from_cu(cu_q)?;
    let kv_lens = sequence_lengths_from_cu(cu_k)?;
    if query_lens.len() != batch || kv_lens.len() != batch {
        candle_core::bail!(
            "packed varlen length mismatch: batch={batch} query_lens={} kv_lens={}",
            query_lens.len(),
            kv_lens.len()
        );
    }

    let mask_tensor = match mask {
        AttentionMask::None => None,
        AttentionMask::Custom(mask) => Some(mask),
    };

    if query_lens.iter().all(|&len| len == query_lens[0])
        && kv_lens.iter().all(|&len| len == kv_lens[0])
    {
        let q_len = query_lens[0];
        let kv_len = kv_lens[0];
        let total = batch
            .checked_mul(kv_len)
            .ok_or_else(|| candle_core::Error::Msg("packed kv token count overflow".into()))?;
        if total > total_kv {
            candle_core::bail!(
                "packed kv length exceeds gathered cache: total={total} gathered={total_kv}"
            );
        }

        let q_active = if q_len == seq_len {
            q.clone()
        } else {
            q.narrow(2, 0, q_len)?
        };
        let k_batched = k
            .narrow(2, 0, total)?
            .reshape((batch, kv_len, k.dim(1)?, head_dim))?
            .transpose(1, 2)?;
        let v_batched = v
            .narrow(2, 0, total)?
            .reshape((batch, kv_len, v.dim(1)?, head_dim))?
            .transpose(1, 2)?;
        let mask = match mask_tensor {
            Some(mask) => {
                let mask = if q_len < seq_len {
                    mask.narrow(2, 0, q_len)?
                } else {
                    mask.clone()
                };
                Some(mask.narrow(3, 0, kv_len)?)
            }
            None => None,
        };
        let out = run_single_sequence_attention(
            &q_active,
            &k_batched,
            &v_batched,
            mask.as_ref(),
            softmax_scale,
            causal,
            n_kv_groups,
        )?;
        let out = if q_len < seq_len {
            let pad = Tensor::zeros(
                (batch, out.dim(1)?, seq_len - q_len, head_dim),
                out.dtype(),
                out.device(),
            )?;
            Tensor::cat(&[&out, &pad], 2)?
        } else {
            out
        };
        return Ok(Some(out));
    }

    let mut kv_offset = 0usize;
    let mut rows = Vec::with_capacity(batch);
    for row in 0..batch {
        let q_len = query_lens[row];
        let kv_len = kv_lens[row];
        let kv_end = kv_offset
            .checked_add(kv_len)
            .ok_or_else(|| candle_core::Error::Msg("packed kv offset overflow".into()))?;
        if kv_end > total_kv {
            candle_core::bail!(
                "packed kv slice exceeds gathered cache: end={kv_end} gathered={total_kv}"
            );
        }

        let q_row = q.narrow(0, row, 1)?.narrow(2, 0, q_len)?;
        let k_row = k.narrow(2, kv_offset, kv_len)?;
        let v_row = v.narrow(2, kv_offset, kv_len)?;
        let mask_row = match mask_tensor {
            Some(mask) => Some(
                mask.narrow(0, row, 1)?
                    .narrow(2, 0, q_len)?
                    .narrow(3, 0, kv_len)?,
            ),
            None => None,
        };
        let out_row = run_single_sequence_attention(
            &q_row,
            &k_row,
            &v_row,
            mask_row.as_ref(),
            softmax_scale,
            causal,
            n_kv_groups,
        )?;
        let out_row = if q_len < seq_len {
            let pad = Tensor::zeros(
                (1usize, out_row.dim(1)?, seq_len - q_len, head_dim),
                out_row.dtype(),
                out_row.device(),
            )?;
            Tensor::cat(&[&out_row, &pad], 2)?
        } else {
            out_row
        };
        rows.push(out_row);
        kv_offset = kv_end;
    }

    Ok(Some(Tensor::cat(rows.as_slice(), 0)?))
}

fn should_use_packed_varlen(q: &Tensor, k: &Tensor, flash_params: Option<&FlashParams>) -> bool {
    let Some(params) = flash_params else {
        return false;
    };
    if params.cumulative_seqlens_q.is_none() || params.logical_k.cumulative_seqlens.is_none() {
        return false;
    }
    let Ok((batch, _, seq_len, _)) = q.dims4() else {
        return false;
    };
    let Ok((kb, _, total_kv, _)) = k.dims4() else {
        return false;
    };
    kb == 1 && (batch > 1 || seq_len != total_kv)
}

#[cfg(feature = "flash-attn")]
fn flash_attn_dispatch(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&FlashParams>,
    softmax_scale: f32,
    causal_default: bool,
) -> Result<Option<Tensor>> {
    if !q.device().is_cuda() || q.dtype() == DType::F32 {
        return Ok(None);
    }

    let (batch, q_len, heads, head_dim) = q.dims4()?;
    let k_len = k.dim(1)?;
    let use_varlen = flash_params
        .and_then(|params| {
            params
                .cumulative_seqlens_q
                .as_ref()
                .zip(params.logical_k.cumulative_seqlens.as_ref())
        })
        .is_some()
        || batch > 1
        || q_len != k_len;

    if use_varlen {
        if let Some(params) = flash_params {
            if let (Some(cu_q), Some(cu_k)) = (
                params.cumulative_seqlens_q.as_ref(),
                params.logical_k.cumulative_seqlens.as_ref(),
            ) {
                let query_lens = sequence_lengths_from_cu(cu_q)?;
                let total_q = query_lens.iter().try_fold(0usize, |acc, &len| {
                    if len == 0 {
                        return Err(candle_core::Error::Msg(
                            "flash varlen query length must be non-zero".to_string(),
                        ));
                    }
                    acc.checked_add(len).ok_or_else(|| {
                        candle_core::Error::Msg("flash varlen query length overflow".to_string())
                    })
                })?;
                let q_packed = if batch == 1 && q_len == total_q {
                    q.reshape((total_q, heads, head_dim))?
                } else {
                    let mut q_chunks = Vec::with_capacity(query_lens.len());
                    for (row, &len) in query_lens.iter().enumerate() {
                        if row >= batch {
                            candle_core::bail!(
                                "flash varlen query row exceeds q batch: row={row} batch={batch}"
                            );
                        }
                        if len > q_len {
                            candle_core::bail!(
                                "flash varlen query length exceeds padded q_len: row={row} len={len} q_len={q_len}"
                            );
                        }
                        q_chunks.push(
                            q.narrow(0, row, 1)?
                                .narrow(1, 0, len)?
                                .reshape((len, heads, head_dim))?,
                        );
                    }
                    Tensor::cat(q_chunks.as_slice(), 0)?
                };
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;
                let out = flash_attn_varlen(
                    &q_packed,
                    &k,
                    &v,
                    cu_q,
                    cu_k,
                    params.max_q as usize,
                    params.logical_k.max as usize,
                    softmax_scale,
                    params.causal,
                )?;

                if batch == 1 && q_len == total_q {
                    return Ok(Some(out.reshape((1usize, total_q, heads, head_dim))?));
                }

                let mut rows = Vec::with_capacity(query_lens.len());
                let mut offset = 0usize;
                for &len in &query_lens {
                    let slice = out
                        .narrow(0, offset, len)?
                        .reshape((1usize, len, heads, head_dim))?;
                    let row = if len < q_len {
                        let pad = Tensor::zeros(
                            (1usize, q_len - len, heads, head_dim),
                            out.dtype(),
                            out.device(),
                        )?;
                        Tensor::cat(&[&slice, &pad], 1)?
                    } else {
                        slice
                    };
                    rows.push(row);
                    offset = offset.saturating_add(len);
                }
                return Ok(Some(Tensor::cat(rows.as_slice(), 0)?));
            }
        }
    }

    let causal = flash_params.map_or(causal_default, |p| p.causal);
    Ok(Some(flash_attn(q, k, v, softmax_scale, causal)?))
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn_dispatch(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _flash_params: Option<&FlashParams>,
    _softmax_scale: f32,
    _causal_default: bool,
) -> Result<Option<Tensor>> {
    Ok(None)
}

pub fn supports_packed_varlen_sdpa(query: &Tensor) -> bool {
    if std::env::var_os("VASR_DISABLE_PACKED_VARLEN").is_some() {
        return false;
    }
    if query.dtype() == DType::F32 {
        return false;
    }
    if query.device().is_cpu() {
        return true;
    }
    if query.device().is_cuda() && cfg!(feature = "flash-attn") {
        return true;
    }
    query.device().is_metal() && query.dim(3).is_ok_and(valid_sdpa_head_dim)
}

pub fn run_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &AttentionMask,
    flash_params: Option<&FlashParams>,
    softmax_scale: f32,
    causal_default: bool,
    n_kv_groups: usize,
) -> Result<Tensor> {
    let causal = attention_is_causal(flash_params, causal_default);
    if can_try_flash_attention(q, mask) {
        let q_t = q.transpose(1, 2)?;
        let k_t = k.transpose(1, 2)?;
        let v_t = v.transpose(1, 2)?;
        if let Some(out) = flash_attn_dispatch(
            &q_t,
            &k_t,
            &v_t,
            flash_params,
            softmax_scale,
            causal_default,
        )? {
            return out.transpose(1, 2);
        }
    }

    if let Some(params) = flash_params {
        if let Some(out) = metal_dense_varlen_prefill_attention(q, k, v, params, softmax_scale)? {
            return Ok(out);
        }
    }

    if should_use_packed_varlen(q, k, flash_params) {
        if let Some(params) = flash_params {
            if let Some(out) =
                packed_varlen_attention(q, k, v, mask, params, softmax_scale, causal, n_kv_groups)?
            {
                return Ok(out);
            }
        }
    }

    run_attention_noflash(q, k, v, mask, softmax_scale, causal, n_kv_groups)
}

pub fn run_attention_noflash(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &AttentionMask,
    softmax_scale: f32,
    causal: bool,
    n_kv_groups: usize,
) -> Result<Tensor> {
    let mask_tensor = prepare_sdpa_mask(q, k, mask, causal)?;
    if let Some(out) = accelerated_sdpa(q, k, v, mask_tensor.as_ref(), softmax_scale, causal)? {
        return Ok(out);
    }

    run_single_sequence_attention(
        q,
        k,
        v,
        mask_tensor.as_ref(),
        softmax_scale,
        causal,
        n_kv_groups,
    )
}

/// Narrow an attention mask to the trailing `kv_seq_len` key positions.
///
/// Mirrors mistral.rs `adjust_kv_mask` for gathered/unpacked paged prefill where
/// K/V are padded to `max_kv` rather than the full prompt tensor width.
pub fn adjust_kv_mask(mask: &Tensor, kv_seq_len: usize) -> Result<Tensor> {
    let mask_dims = mask.dims();
    match mask.rank() {
        2 if mask_dims[1] > kv_seq_len => mask.narrow(1, mask_dims[1] - kv_seq_len, kv_seq_len),
        3 if mask_dims[2] > kv_seq_len => mask.narrow(2, mask_dims[2] - kv_seq_len, kv_seq_len),
        4 if mask_dims[3] > kv_seq_len => mask.narrow(3, mask_dims[3] - kv_seq_len, kv_seq_len),
        _ => Ok(mask.clone()),
    }
}

/// Try Candle's accelerator SDPA path before falling back to the manual
/// repeat-kv + matmul implementation. This mirrors the default eager path used
/// by mistral.rs on Metal for Qwen-family head dimensions.
pub fn accelerated_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    causal: bool,
) -> Result<Option<Tensor>> {
    if !q.device().is_metal() {
        return Ok(None);
    }

    let (batch, num_heads, q_len, head_dim) = q.dims4()?;
    let (kb, _num_kv_heads, k_len, k_head_dim) = k.dims4()?;
    let (vb, _num_v_heads, _v_len, v_head_dim) = v.dims4()?;
    if batch != kb || batch != vb || head_dim != k_head_dim || head_dim != v_head_dim {
        return Ok(None);
    }

    let valid_head_dims = [32usize, 64, 72, 80, 96, 128, 256, 512];
    if !valid_head_dims.contains(&head_dim) {
        return Ok(None);
    }

    let target_mask_shape = vec![batch, num_heads, q_len, k_len];
    let mask = match mask {
        Some(mask) => {
            if mask.layout().broadcast_as(target_mask_shape).is_err() {
                return Ok(None);
            }
            if q.device().is_metal() && q_len > k_len {
                return Ok(None);
            }
            Some(mask.broadcast_as((batch, num_heads, q_len, k_len))?)
        }
        None => None,
    };

    let do_causal = q_len > 1 && causal;
    let out = candle_nn::ops::sdpa(q, k, v, mask.as_ref(), do_causal, softmax_scale, 1.0)?;
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionMask, FlashKMeta, FlashParams, make_causal_mask, make_causal_mask_cached,
        packed_varlen_attention, repeat_kv,
    };
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_repeat_kv_repeats_heads() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let x =
            candle_core::Tensor::arange(0f32, 2.0 * 3.0 * 4.0, &device)?.reshape((2, 3, 2, 2))?;
        let y = repeat_kv(&x, 2)?;
        let (b, h, s, d) = y.dims4()?;
        if (b, h, s, d) != (2, 6, 2, 2) {
            anyhow::bail!("unexpected repeat_kv dims: {:?}", y.dims());
        }
        Ok(())
    }

    #[test]
    fn test_make_causal_mask_shape_and_values() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let batch = 1usize;
        let seq = 4usize;
        let mask = make_causal_mask(None, batch, seq, candle_core::DType::F32, &device)?;
        if mask.dims() != vec![batch, 1, seq, seq] {
            anyhow::bail!("unexpected mask dims: {:?}", mask.dims());
        }
        let m = mask.squeeze(1)?.to_vec3::<f32>()?;
        for (q, row) in m[0].iter().enumerate() {
            for (kv, &v) in row.iter().enumerate() {
                if kv <= q {
                    if v != 0.0 {
                        anyhow::bail!("expected 0 at ({q},{kv}), got {v}");
                    }
                } else if v != f32::MIN {
                    anyhow::bail!("expected f32::MIN at ({q},{kv}), got {v}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_make_causal_mask_cached_shape_and_values() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let batch = 1usize;
        let cache_len = 3usize;
        let new_len = 2usize;
        let total_len = cache_len + new_len;

        let mask = make_causal_mask_cached(
            None,
            batch,
            cache_len,
            new_len,
            candle_core::DType::F32,
            &device,
        )?;
        if mask.dims() != vec![batch, 1, new_len, total_len] {
            anyhow::bail!("unexpected mask dims: {:?}", mask.dims());
        }

        let m = mask.squeeze(1)?.to_vec3::<f32>()?;
        // q=0 => abs_q=3, disallow kv>3
        for (kv, &v) in m[0][0].iter().enumerate() {
            if kv <= 3 {
                if v != 0.0 {
                    anyhow::bail!("expected 0 at (q=0,kv={kv}), got {v}");
                }
            } else if v != f32::MIN {
                anyhow::bail!("expected f32::MIN at (q=0,kv={kv}), got {v}");
            }
        }
        // q=1 => abs_q=4, all kv allowed
        for (kv, &v) in m[0][1].iter().enumerate() {
            if v != 0.0 {
                anyhow::bail!("expected 0 at (q=1,kv={kv}), got {v}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_make_causal_mask_cached_decode_one_without_padding_is_noop() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let batch = 2usize;
        let cache_len = 5usize;
        let new_len = 1usize;

        let mask = make_causal_mask_cached(
            None,
            batch,
            cache_len,
            new_len,
            candle_core::DType::F32,
            &device,
        )?;
        if mask.dims() != vec![batch, 1, new_len, cache_len + new_len] {
            anyhow::bail!("unexpected mask dims: {:?}", mask.dims());
        }

        let values = mask.flatten_all()?.to_vec1::<f32>()?;
        if values.iter().any(|&v| v != 0.0) {
            anyhow::bail!("decode-one no-padding mask should be all zeros: {values:?}");
        }

        Ok(())
    }

    #[test]
    fn test_adjust_kv_mask_narrows_trailing_kv_dimension() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let mask = candle_core::Tensor::zeros((1usize, 1, 4, 6), candle_core::DType::F32, &device)?;
        let adjusted = super::adjust_kv_mask(&mask, 4)?;
        if adjusted.dims() != [1, 1, 4, 4] {
            anyhow::bail!("unexpected adjusted mask dims: {:?}", adjusted.dims());
        }
        Ok(())
    }

    #[test]
    fn test_packed_varlen_attention_matches_unpacked_reference() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let batch = 2usize;
        let seq_len = 4usize;
        let head_dim = 32usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        let q = Tensor::from_vec(
            (0..batch * num_heads * seq_len * head_dim)
                .map(|i| i as f32 * 0.01)
                .collect::<Vec<_>>(),
            (batch, num_heads, seq_len, head_dim),
            &device,
        )?;
        let query_lens = vec![3usize, 2usize];
        let kv_lens = vec![3usize, 2usize];
        let total_kv: usize = kv_lens.iter().sum();
        let packed = Tensor::from_vec(
            (0..total_kv * num_kv_heads * head_dim)
                .map(|i| (i as f32 * 0.02) + 1.0)
                .collect::<Vec<_>>(),
            (total_kv, num_kv_heads, head_dim),
            &device,
        )?;
        let k = packed.unsqueeze(0)?.transpose(1, 2)?;
        let v = k.clone();

        let cu_q = Tensor::from_vec(vec![0u32, 3, 5], (3,), &device)?;
        let cu_k = Tensor::from_vec(vec![0u32, 3, 5], (3,), &device)?;
        let flash_params = FlashParams {
            max_q: 3,
            cumulative_seqlens_q: Some(cu_q),
            logical_k: FlashKMeta {
                max: 3,
                cumulative_seqlens: Some(cu_k),
            },
            causal: true,
        };

        let packed_out = packed_varlen_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            &flash_params,
            softmax_scale,
            true,
            2,
        )?
        .ok_or_else(|| anyhow::anyhow!("packed varlen returned None"))?;

        let mut unpacked_rows = Vec::with_capacity(batch);
        let mut start = 0usize;
        for (&q_len, &kv_len) in query_lens.iter().zip(kv_lens.iter()) {
            let q_row = q.narrow(0, unpacked_rows.len(), 1)?.narrow(2, 0, q_len)?;
            let k_row = packed
                .narrow(0, start, kv_len)?
                .reshape((1, kv_len, num_kv_heads, head_dim))?
                .transpose(1, 2)?;
            let v_row = k_row.clone();
            let k_expanded = repeat_kv(&k_row, 2)?;
            let v_expanded = repeat_kv(&v_row, 2)?;
            let k_t = k_expanded.transpose(2, 3)?;
            let weights = q_row.matmul(&k_t)?.affine(softmax_scale.into(), 0.0)?;
            let causal = super::make_causal_mask(None, 1, q_len, DType::F32, &device)?;
            let weights = weights.broadcast_add(&causal)?;
            let weights = candle_nn::ops::softmax_last_dim(&weights)?;
            let out_row = weights.matmul(&v_expanded)?;
            let pad = Tensor::zeros(
                (1, num_heads, seq_len - q_len, head_dim),
                DType::F32,
                &device,
            )?;
            unpacked_rows.push(Tensor::cat(&[&out_row, &pad], 2)?);
            start += kv_len;
        }
        let unpacked_out = Tensor::cat(unpacked_rows.as_slice(), 0)?;

        let got = packed_out.flatten_all()?.to_vec1::<f32>()?;
        let expected = unpacked_out.flatten_all()?.to_vec1::<f32>()?;
        if got.len() != expected.len() {
            anyhow::bail!("length mismatch: {} vs {}", got.len(), expected.len());
        }
        for (idx, (&lhs, &rhs)) in got.iter().zip(expected.iter()).enumerate() {
            if (lhs - rhs).abs() > 1e-4 {
                anyhow::bail!("packed/unpacked mismatch at {idx}: {lhs} vs {rhs}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_supports_packed_varlen_sdpa_on_metal_head_dim() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let q = Tensor::zeros((2, 8, 4, 128), DType::BF16, &device)?;
        assert!(super::supports_packed_varlen_sdpa(&q) || !device.is_metal());
        let q_bad = Tensor::zeros((2, 8, 4, 48), DType::BF16, &device)?;
        if device.is_metal() {
            assert!(!super::supports_packed_varlen_sdpa(&q_bad));
        }
        Ok(())
    }
}
