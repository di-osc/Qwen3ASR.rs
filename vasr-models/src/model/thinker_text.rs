//! Thinker text model (decoder-only transformer with mRoPE).
//!
//! This is a faithful port of the "thinker.model" module in the official
//! Qwen3-ASR implementation:
//! `Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`.

use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder, embedding, rms_norm};

use crate::config::{RopeScaling, TextConfig};
use crate::model::isq_linear::{self, IsqLinear};
use crate::model::kv_cache::KVCache;
use crate::model::{attention, rope};

#[cfg(feature = "paged-attn")]
use crate::model::paged_kv_cache::{PagedInputMetadata, PagedKvCache};
#[cfg(all(
    feature = "paged-attn",
    any(feature = "cuda-paged-attn", feature = "metal-paged-attn")
))]
use vasr_paged_attn::mistralrs_paged_attn;

#[cfg(feature = "paged-attn")]
#[derive(Clone)]
struct PagedAttentionHandle {
    k_scale: Tensor,
    v_scale: Tensor,
}

#[cfg(feature = "paged-attn")]
impl std::fmt::Debug for PagedAttentionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedAttentionHandle").finish()
    }
}

#[cfg(feature = "flash-attn")]
use candle_core::{DType, Device};

#[cfg(feature = "flash-attn")]
use candle_flash_attn::{flash_attn, flash_attn_varlen};

fn cache_input_is_packed(tensor: &Tensor) -> Result<bool> {
    if tensor.dims().len() != 3 {
        return Ok(false);
    }
    let (_, heads, head_size) = tensor.dims3()?;
    let stride = tensor.stride();
    Ok(stride[2] == 1 && stride[1] == head_size && stride[0] == heads * head_size)
}

fn maybe_contiguous_for_accelerator(x: Tensor) -> Result<Tensor> {
    if x.device().is_metal() || x.device().is_cuda() {
        if cache_input_is_packed(&x)? {
            Ok(x)
        } else {
            x.contiguous()
        }
    } else {
        Ok(x)
    }
}

fn initial_hidden_states_for_paged_decode(inputs_embeds: &Tensor) -> Result<Tensor> {
    let seq_len = inputs_embeds.dim(1)?;
    if seq_len == 1 {
        if inputs_embeds.device().is_metal() {
            return inputs_embeds.affine(1.0, 0.0);
        }
        #[cfg(feature = "cuda-graph")]
        if inputs_embeds.device().is_cuda() {
            return inputs_embeds.affine(1.0, 0.0);
        }
    }
    Ok(inputs_embeds.clone())
}

#[cfg(feature = "paged-attn")]
fn unpack_gathered_kv_for_attention(
    packed: &Tensor,
    kv_lens: &[usize],
    num_kv_heads: usize,
    _num_kv_groups: usize,
    head_size: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if kv_lens.is_empty() {
        candle_core::bail!("unpack_gathered_kv_for_attention requires at least one sequence");
    }

    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
    let batch = kv_lens.len();
    if max_kv == 0 {
        return Tensor::zeros(
            (batch, num_kv_heads, 0usize, head_size),
            packed.dtype(),
            device,
        );
    }

    if kv_lens.iter().all(|&len| len == max_kv) {
        let total = batch
            .checked_mul(max_kv)
            .ok_or_else(|| candle_core::Error::Msg("gathered kv token count overflow".into()))?;
        return Ok(packed
            .narrow(0, 0, total)?
            .reshape((batch, max_kv, num_kv_heads, head_size))?
            .transpose(1, 2)?);
    }

    let mut start = 0usize;
    let mut unpacked: Vec<Tensor> = Vec::with_capacity(batch);

    for &kv_len in kv_lens {
        let mut seq = packed
            .narrow(0, start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        if kv_len < max_kv {
            let pad = Tensor::zeros(
                (1usize, num_kv_heads, max_kv - kv_len, head_size),
                packed.dtype(),
                device,
            )?;
            seq = Tensor::cat(&[&seq, &pad], 2)?;
        }
        unpacked.push(seq);
        start = start.saturating_add(kv_len);
    }

    Tensor::cat(&unpacked, 0)
}

#[cfg(feature = "paged-attn")]
fn paged_prefill_attention_mask(
    input_metadata: &PagedInputMetadata,
    kv_lens: &[usize],
    batch: usize,
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<attention::AttentionMask> {
    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
    if input_metadata.prefill_causal_only {
        return Ok(attention::AttentionMask::None);
    }
    let mask = if let Some(mask) = input_metadata.prefill_attention_mask.as_ref() {
        mask.clone()
    } else {
        attention::make_causal_mask(
            input_metadata.token_attention_mask.as_ref(),
            batch,
            seq_len,
            dtype,
            device,
        )?
    };
    Ok(attention::AttentionMask::Custom(attention::adjust_kv_mask(
        &mask, max_kv,
    )?))
}

#[cfg(feature = "paged-attn")]
fn metal_hybrid_paged_prefill_enabled() -> bool {
    std::env::var_os("VASR_ENABLE_METAL_PAGED_PREFILL_HYBRID").is_some()
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
fn run_paged_prefill_attention_fallback(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    input_metadata: &PagedInputMetadata,
    kv_lens: &[usize],
    paged_attn: &PagedAttentionHandle,
    batch: usize,
    seq_len: usize,
    hidden_states: &Tensor,
    scaling: f32,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let cu_kv = input_metadata
        .cu_seqlens_kv
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("paged metadata missing cu_seqlens_kv".into()))?;
    let (k_gathered, v_gathered) = mistralrs_paged_attn::gather_kv_cache(
        key_cache,
        value_cache,
        Some(&paged_attn.k_scale),
        Some(&paged_attn.v_scale),
        &input_metadata.block_tables,
        cu_kv,
        hidden_states.dtype(),
    )?;
    let mask = paged_prefill_attention_mask(
        input_metadata,
        kv_lens,
        batch,
        seq_len,
        q.dtype(),
        q.device(),
    )?;
    let flash_params = input_metadata.prefill_flash_params();
    let avoid_packed_varlen = q.device().is_metal() && metal_hybrid_paged_prefill_enabled();
    if !avoid_packed_varlen && attention::supports_packed_varlen_sdpa(q) {
        let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
        let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
        attention::run_attention(
            q,
            &k_4d,
            &v_4d,
            &mask,
            flash_params,
            scaling,
            true,
            num_key_value_groups,
        )
    } else {
        let k = unpack_gathered_kv_for_attention(
            &k_gathered,
            kv_lens,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            hidden_states.device(),
        )?;
        let v = unpack_gathered_kv_for_attention(
            &v_gathered,
            kv_lens,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            hidden_states.device(),
        )?;
        attention::run_attention(
            q,
            &k,
            &v,
            &mask,
            flash_params,
            scaling,
            true,
            num_key_value_groups,
        )
    }
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
#[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
fn run_paged_prefill_attention_fallback(
    _q: &Tensor,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _input_metadata: &PagedInputMetadata,
    _kv_lens: &[usize],
    _paged_attn: &PagedAttentionHandle,
    _batch: usize,
    _seq_len: usize,
    _hidden_states: &Tensor,
    _scaling: f32,
    _num_key_value_heads: usize,
    _num_key_value_groups: usize,
    _head_dim: usize,
) -> Result<Tensor> {
    candle_core::bail!(
        "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
    )
}

#[cfg(feature = "flash-attn")]
fn seqlens_from_attention_mask(mask: &Tensor, seq_len: usize) -> Result<Vec<usize>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != seq_len {
        candle_core::bail!("attention_mask seq_len mismatch: expected={seq_len}, got={t2}");
    }

    let mask_u8 = mask.ne(0u32)?;
    let lens_f32 = mask_u8.to_dtype(DType::F32)?.sum(1)?;
    let lens = lens_f32.to_vec1::<f32>()?;
    if lens.len() != batch {
        candle_core::bail!(
            "internal error: attention_mask lens mismatch: expected={batch}, got={}",
            lens.len()
        );
    }

    let mut out: Vec<usize> = Vec::with_capacity(batch);
    for (i, v) in lens.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            candle_core::bail!("invalid attention_mask length at {i}: {v}");
        }
        if v > seq_len as f32 + 0.01 {
            candle_core::bail!(
                "attention_mask length at {i} exceeds seq_len: len={v} seq_len={seq_len}"
            );
        }
        let len = v.round() as usize;
        if (len as f32 - v).abs() > 0.01 {
            candle_core::bail!("attention_mask length at {i} is not integral: {v}");
        }
        out.push(len);
    }
    Ok(out)
}

#[cfg(feature = "flash-attn")]
fn cu_seqlens_u32(lengths: &[usize], device: &Device) -> Result<(Tensor, usize, u32)> {
    let mut cu: Vec<u32> = Vec::with_capacity(lengths.len().saturating_add(1));
    cu.push(0);

    let mut total: u32 = 0;
    let mut max_len: usize = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let len_u32 = u32::try_from(len).map_err(|_| {
            candle_core::Error::Msg(format!(
                "sequence length overflows u32 at index {i}: len={len}"
            ))
        })?;
        total = total.checked_add(len_u32).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "cumulative sequence length overflows u32 at index {i}: total={total} len={len}"
            ))
        })?;
        cu.push(total);
        max_len = max_len.max(len);
    }

    let cu_t = Tensor::from_vec(cu, (lengths.len().saturating_add(1),), device)?;
    Ok((cu_t, max_len, total))
}

#[cfg(feature = "flash-attn")]
fn mask_nonzero_indices_u32(mask: &Tensor, seq_len: usize) -> Result<Vec<u32>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != seq_len {
        candle_core::bail!("attention_mask seq_len mismatch: expected={seq_len}, got={t2}");
    }

    let rows = mask.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    if rows.len() != batch {
        candle_core::bail!(
            "internal error: attention_mask row mismatch: expected={batch}, got={}",
            rows.len()
        );
    }

    let mut idxs = Vec::new();
    for (b, row) in rows.iter().enumerate() {
        if row.len() != seq_len {
            candle_core::bail!(
                "attention_mask row len mismatch at batch index {b}: expected={seq_len}, got={}",
                row.len()
            );
        }
        let base = b
            .checked_mul(seq_len)
            .ok_or_else(|| candle_core::Error::Msg("index overflow".to_string()))?;
        for (j, &value) in row.iter().enumerate() {
            if value == 0 {
                continue;
            }
            let pos = base
                .checked_add(j)
                .ok_or_else(|| candle_core::Error::Msg("index overflow".to_string()))?;
            idxs.push(
                u32::try_from(pos).map_err(|_| {
                    candle_core::Error::Msg(format!("index overflows u32: pos={pos}"))
                })?,
            );
        }
    }

    if idxs.is_empty() && batch > 0 {
        candle_core::bail!("attention_mask contains no valid tokens");
    }

    Ok(idxs)
}

#[derive(Debug, Clone)]
struct ThinkerTextRotaryEmbedding {
    rope: rope::mrope::MultimodalRotaryEmbedding,
    mrope_section: Vec<usize>,
    interleaved: bool,
}

impl ThinkerTextRotaryEmbedding {
    fn load(cfg: &TextConfig, device: &candle_core::Device) -> Result<Self> {
        let scaling = cfg.rope_scaling.as_ref();
        let rope = match scaling {
            None => rope::mrope::MultimodalRotaryEmbedding::new(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                device,
            )?,
            Some(s) => rope::mrope::MultimodalRotaryEmbedding::with_scaling(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                s,
                device,
            )?,
        };

        let (mrope_section, interleaved) = match scaling {
            None => (vec![24usize, 20, 20], false),
            Some(s) => (
                if s.mrope_section.is_empty() {
                    vec![24usize, 20, 20]
                } else {
                    s.mrope_section.clone()
                },
                s.mrope_interleaved || s.interleaved,
            ),
        };

        Ok(Self {
            rope,
            mrope_section,
            interleaved,
        })
    }

    fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        self.rope.forward(x, position_ids)
    }

    fn forward_first_modality(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.rope.forward_first_modality(x, position_ids)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextMlp {
    gate_proj: IsqLinear,
    up_proj: IsqLinear,
    down_proj: IsqLinear,
    hidden_act: String,
}

impl ThinkerTextMlp {
    fn load(cfg: &TextConfig, vb: VarBuilder, isq: Option<&str>) -> Result<Self> {
        let gate_proj = isq_linear::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("gate_proj"),
            isq,
        )?;
        let up_proj = isq_linear::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("up_proj"),
            isq,
        )?;
        let down_proj = isq_linear::linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj"),
            isq,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_act: cfg.hidden_act.clone(),
        })
    }
}

impl Module for ThinkerTextMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(self.hidden_act.as_str(), "silu" | "swish") {
            if let Some(hidden) =
                isq_linear::try_fused_q8_silu_gate_up(&self.gate_proj, &self.up_proj, xs)?
            {
                return self.down_proj.forward(&hidden);
            }
        }
        #[cfg(feature = "metal-paged-attn")]
        if matches!(self.hidden_act.as_str(), "silu" | "swish") {
            if let Some(hidden) =
                isq_linear::try_fused_afq_silu_gate_up(&self.gate_proj, &self.up_proj, xs)?
            {
                return self.down_proj.forward(&hidden);
            }
        }

        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let gate = match self.hidden_act.as_str() {
            "silu" | "swish" => candle_nn::ops::silu(&gate)?,
            other => candle_core::bail!("unsupported hidden_act={other:?}"),
        };
        let hidden = gate.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextAttention {
    use_flash_attn: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    scaling: f64,
    q_proj: IsqLinear,
    k_proj: IsqLinear,
    v_proj: IsqLinear,
    o_proj: IsqLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    #[cfg(feature = "paged-attn")]
    paged_attn: Option<PagedAttentionHandle>,
}

impl ThinkerTextAttention {
    fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        #[cfg(not(feature = "paged-attn"))]
        let _ = device;

        let head_dim = cfg.head_dim;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let num_key_value_groups = num_attention_heads / num_key_value_heads;

        let q_out = num_attention_heads * head_dim;
        let kv_out = num_key_value_heads * head_dim;

        let q_proj = isq_linear::linear_b(
            cfg.hidden_size,
            q_out,
            cfg.attention_bias,
            vb.pp("q_proj"),
            isq,
        )?;
        let k_proj = isq_linear::linear_b(
            cfg.hidden_size,
            kv_out,
            cfg.attention_bias,
            vb.pp("k_proj"),
            isq,
        )?;
        let v_proj = isq_linear::linear_b(
            cfg.hidden_size,
            kv_out,
            cfg.attention_bias,
            vb.pp("v_proj"),
            isq,
        )?;
        let o_proj = isq_linear::linear_b(
            q_out,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
            isq,
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        #[cfg(feature = "paged-attn")]
        let paged_attn = Some(PagedAttentionHandle {
            k_scale: Tensor::new(1f32, device)?,
            v_scale: Tensor::new(1f32, device)?,
        });

        Ok(Self {
            use_flash_attn,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            #[cfg(feature = "paged-attn")]
            paged_attn,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &ThinkerTextRotaryEmbedding,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((batch, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?, v)
        } else {
            let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
            let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
            let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
            let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
            let v = v
                .reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let (cos, sin) = position_embeddings;
        let (q, k) = rope::mrope::apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            rope_scaling.mrope_section.as_slice(),
            rope_scaling.interleaved,
        )?;

        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = token_attention_mask;
                candle_core::bail!("flash-attn support is not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;

                let q4 = q.transpose(1, 2)?.contiguous()?; // (b, s, h, d)
                let k4 = k.transpose(1, 2)?.contiguous()?; // (b, s, kv, d)
                let v4 = v.transpose(1, 2)?.contiguous()?; // (b, s, kv, d)

                let Some(tok_mask) = token_attention_mask else {
                    let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                    let attn =
                        attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&attn);
                };

                let seqlens = seqlens_from_attention_mask(tok_mask, seq_len)?;
                let (cu, max_len, total_u32) =
                    cu_seqlens_u32(seqlens.as_slice(), hidden_states.device())?;
                let total = usize::try_from(total_u32).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "total sequence length overflows usize: total={total_u32}"
                    ))
                })?;

                let flat_total = batch.checked_mul(seq_len).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "batch*seq_len overflow: batch={batch} seq_len={seq_len}"
                    ))
                })?;

                if total == flat_total {
                    let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                    let attn =
                        attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&attn);
                }

                let idxs = mask_nonzero_indices_u32(tok_mask, seq_len)?;
                if idxs.len() != total {
                    candle_core::bail!(
                        "internal error: index len mismatch: idxs={} total={total}",
                        idxs.len()
                    );
                }

                let idx = Tensor::from_vec(idxs, (total,), hidden_states.device())?;

                let q_flat = q4.reshape((flat_total, self.num_attention_heads, self.head_dim))?;
                let k_flat = k4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                let v_flat = v4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;

                let q_unpad = q_flat.index_select(&idx, 0)?.contiguous()?;
                let k_unpad = k_flat.index_select(&idx, 0)?.contiguous()?;
                let v_unpad = v_flat.index_select(&idx, 0)?.contiguous()?;

                let out_unpad = flash_attn_varlen(
                    &q_unpad,
                    &k_unpad,
                    &v_unpad,
                    &cu,
                    &cu,
                    max_len,
                    max_len,
                    softmax_scale,
                    true,
                )?;

                let out = {
                    let zeros = Tensor::zeros(
                        (flat_total, self.num_attention_heads, self.head_dim),
                        out_unpad.dtype(),
                        out_unpad.device(),
                    )?;
                    let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                    flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                };

                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        let mask = match attention_mask {
            Some(mask) => attention::AttentionMask::Custom(mask.clone()),
            None => attention::AttentionMask::None,
        };
        let flash_params = if seq_len > 1 {
            let query_lens = vec![seq_len; batch];
            attention::make_flash_params(
                query_lens.as_slice(),
                query_lens.as_slice(),
                hidden_states.device(),
                true,
            )?
        } else {
            None
        };
        let q = maybe_contiguous_for_accelerator(q)?;
        let k = maybe_contiguous_for_accelerator(k)?;
        let v = maybe_contiguous_for_accelerator(v)?;
        let attn_output = attention::run_attention(
            &q,
            &k,
            &v,
            &mask,
            flash_params.as_ref(),
            self.scaling as f32,
            true,
            self.num_key_value_groups,
        )?;
        let attn_output = attn_output.transpose(1, 2)?; // (b, s, h, d)
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &ThinkerTextRotaryEmbedding,
        layer_cache: (&mut KVCache, usize),
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let (kv_cache, layer_idx) = layer_cache;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((batch, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?, v)
        } else {
            let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
            let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
            let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
            let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
            let v = v
                .reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let (cos, sin) = position_embeddings;
        let (q, k) = rope::mrope::apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            rope_scaling.mrope_section.as_slice(),
            rope_scaling.interleaved,
        )?;

        // Cache rotated keys to match the attention math.
        let (k, v) = kv_cache.update(layer_idx, &k, &v)?;

        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = token_attention_mask;
                candle_core::bail!("flash-attn support is not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;
                let Some(token_attention_mask) = token_attention_mask else {
                    candle_core::bail!(
                        "flash-attn cached decode without a token attention mask is unsupported"
                    );
                };

                let (_b2, total_len) = token_attention_mask.dims2()?;
                if _b2 != batch {
                    candle_core::bail!(
                        "token_attention_mask batch mismatch: expected={batch}, got={_b2}"
                    );
                }
                let cache_len = total_len.checked_sub(seq_len).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "attention mask shorter than new seq_len: total_len={total_len} new_len={seq_len}"
                    ))
                })?;

                let q4 = q.transpose(1, 2)?.contiguous()?; // (b, s_q, h, d)
                let k4 = k.transpose(1, 2)?.contiguous()?; // (b, s_k, kv, d)
                let v4 = v.transpose(1, 2)?.contiguous()?; // (b, s_k, kv, d)

                if cache_len == 0 {
                    let seqlens = seqlens_from_attention_mask(token_attention_mask, seq_len)?;
                    let (cu, max_len, total_u32) =
                        cu_seqlens_u32(seqlens.as_slice(), hidden_states.device())?;
                    let total = usize::try_from(total_u32).map_err(|_| {
                        candle_core::Error::Msg(format!(
                            "total sequence length overflows usize: total={total_u32}"
                        ))
                    })?;

                    let flat_total = batch.checked_mul(seq_len).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "batch*seq_len overflow: batch={batch} seq_len={seq_len}"
                        ))
                    })?;

                    if total == flat_total {
                        let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                        let attn = attn.reshape((
                            batch,
                            seq_len,
                            self.num_attention_heads * self.head_dim,
                        ))?;
                        return self.o_proj.forward(&attn);
                    }

                    let idxs = mask_nonzero_indices_u32(token_attention_mask, seq_len)?;
                    if idxs.len() != total {
                        candle_core::bail!(
                            "internal error: index len mismatch: idxs={} total={total}",
                            idxs.len()
                        );
                    }
                    let idx = Tensor::from_vec(idxs, (total,), hidden_states.device())?;

                    let q_flat =
                        q4.reshape((flat_total, self.num_attention_heads, self.head_dim))?;
                    let k_flat =
                        k4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                    let v_flat =
                        v4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;

                    let q_unpad = q_flat.index_select(&idx, 0)?.contiguous()?;
                    let k_unpad = k_flat.index_select(&idx, 0)?.contiguous()?;
                    let v_unpad = v_flat.index_select(&idx, 0)?.contiguous()?;

                    let out_unpad = flash_attn_varlen(
                        &q_unpad,
                        &k_unpad,
                        &v_unpad,
                        &cu,
                        &cu,
                        max_len,
                        max_len,
                        softmax_scale,
                        true,
                    )?;

                    let out = {
                        let zeros = Tensor::zeros(
                            (flat_total, self.num_attention_heads, self.head_dim),
                            out_unpad.dtype(),
                            out_unpad.device(),
                        )?;
                        let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                        flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                    };

                    let out =
                        out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&out);
                }

                // Cached decode step: q has no padding, but k/v include left padding.
                let k_total = batch
                    .checked_mul(total_len)
                    .ok_or_else(|| candle_core::Error::Msg("k/v size overflow".to_string()))?;
                let q_total = batch
                    .checked_mul(seq_len)
                    .ok_or_else(|| candle_core::Error::Msg("q size overflow".to_string()))?;

                let q3 = q4.reshape((q_total, self.num_attention_heads, self.head_dim))?;

                let seqlens_k = seqlens_from_attention_mask(token_attention_mask, total_len)?;
                let (cu_k, max_k, total_k_u32) =
                    cu_seqlens_u32(seqlens_k.as_slice(), hidden_states.device())?;
                let total_k = usize::try_from(total_k_u32).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "total_k overflows usize: total_k={total_k_u32}"
                    ))
                })?;

                let seqlens_q = vec![seq_len; batch];
                let (cu_q, max_q, total_q_u32) =
                    cu_seqlens_u32(seqlens_q.as_slice(), hidden_states.device())?;
                let total_q = usize::try_from(total_q_u32).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "total_q overflows usize: total_q={total_q_u32}"
                    ))
                })?;

                if total_q != q_total {
                    candle_core::bail!(
                        "internal error: q total mismatch: expected={q_total}, got={total_q}"
                    );
                }

                let idxs_k = mask_nonzero_indices_u32(token_attention_mask, total_len)?;
                if idxs_k.len() != total_k {
                    candle_core::bail!(
                        "internal error: k index len mismatch: idxs={} total_k={total_k}",
                        idxs_k.len()
                    );
                }
                let idx_k = Tensor::from_vec(idxs_k, (total_k,), hidden_states.device())?;

                let k3 = k4
                    .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                    .index_select(&idx_k, 0)?
                    .contiguous()?;
                let v3 = v4
                    .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                    .index_select(&idx_k, 0)?
                    .contiguous()?;

                let out = flash_attn_varlen(
                    &q3,
                    &k3,
                    &v3,
                    &cu_q,
                    &cu_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    true,
                )?;
                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        let mask = match attention_mask {
            Some(mask) => attention::AttentionMask::Custom(mask.clone()),
            None => attention::AttentionMask::None,
        };
        let flash_params = if seq_len > 1 {
            let cache_len = kv_cache.seq_len();
            let query_lens = vec![seq_len; batch];
            let kv_lens = vec![cache_len.saturating_add(seq_len); batch];
            attention::make_flash_params(
                query_lens.as_slice(),
                kv_lens.as_slice(),
                hidden_states.device(),
                true,
            )?
        } else {
            None
        };
        let q = maybe_contiguous_for_accelerator(q)?;
        let k = maybe_contiguous_for_accelerator(k)?;
        let v = maybe_contiguous_for_accelerator(v)?;
        let attn_output = attention::run_attention(
            &q,
            &k,
            &v,
            &mask,
            flash_params.as_ref(),
            self.scaling as f32,
            true,
            self.num_key_value_groups,
        )?;
        let attn_output = attn_output.transpose(1, 2)?; // (b, s, h, d)
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    #[cfg(feature = "paged-attn")]
    fn forward_with_paged_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        input_metadata: &PagedInputMetadata,
        rope_scaling: &ThinkerTextRotaryEmbedding,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let paged_attn = self
            .paged_attn
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("paged attention is not initialized".into()))?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((batch, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?, v)
        } else {
            let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
            let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
            let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
            let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
            let v = v
                .reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let (cos, sin) = position_embeddings;
        let (q, k) = rope::mrope::apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            rope_scaling.mrope_section.as_slice(),
            rope_scaling.interleaved,
        )?;

        let flat_tokens = batch.checked_mul(seq_len).ok_or_else(|| {
            candle_core::Error::Msg("paged attention token count overflow".into())
        })?;
        let k_flat = if seq_len == 1 {
            maybe_contiguous_for_accelerator(k.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        } else {
            maybe_contiguous_for_accelerator(k.transpose(1, 2)?.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        };
        let v_flat = if seq_len == 1 {
            maybe_contiguous_for_accelerator(v.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        } else {
            maybe_contiguous_for_accelerator(v.transpose(1, 2)?.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        };
        #[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
        {
            mistralrs_paged_attn::reshape_and_cache(
                &k_flat,
                &v_flat,
                Some(&paged_attn.k_scale),
                Some(&paged_attn.v_scale),
                key_cache,
                value_cache,
                &input_metadata.slot_mapping,
            )?;
        }
        #[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
        {
            return candle_core::bail!(
                "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
            );
        }

        let attn = if seq_len == 1 {
            #[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
            {
                let q_flat = maybe_contiguous_for_accelerator(q.reshape((
                    batch,
                    self.num_attention_heads,
                    self.head_dim,
                ))?)?;
                mistralrs_paged_attn::paged_attention(
                    &q_flat,
                    Some(&paged_attn.k_scale),
                    Some(&paged_attn.v_scale),
                    key_cache,
                    value_cache,
                    &input_metadata.block_tables,
                    &input_metadata.context_lens,
                    None,
                    input_metadata.max_context_len,
                    self.scaling as f32,
                    1.0,
                    None,
                )?
            }
            #[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
            {
                return candle_core::bail!(
                    "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
                );
            }
        } else {
            let kv_lens = input_metadata
                .kv_lens
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("paged metadata missing kv_lens".into()))?;
            let attn_output = {
                #[cfg(feature = "metal-paged-attn")]
                {
                    let block_size = key_cache.dim(3)?;
                    if !metal_hybrid_paged_prefill_enabled()
                        && vasr_paged_attn::supports_metal_varlen_prefill(
                            q.device(),
                            self.head_dim,
                            block_size,
                        )
                    {
                        if let (Some(cu_q), Some(query_lens)) = (
                            input_metadata.cu_seqlens_q.as_ref(),
                            input_metadata.query_lens.as_ref(),
                        ) {
                            let q_contig = maybe_contiguous_for_accelerator(q.clone())?;
                            let q_packed = vasr_paged_attn::pack_query_for_varlen_prefill(
                                &q_contig, query_lens,
                            )?;
                            let attn_packed = vasr_paged_attn::paged_attention_varlen_prefill(
                                &q_packed,
                                key_cache,
                                value_cache,
                                &input_metadata.block_tables,
                                &input_metadata.context_lens,
                                cu_q,
                                Some(&paged_attn.k_scale),
                                Some(&paged_attn.v_scale),
                                self.scaling as f32,
                                1.0,
                            )?;
                            vasr_paged_attn::unpack_varlen_attn_to_batched(
                                &attn_packed,
                                query_lens,
                                batch,
                                seq_len,
                            )?
                        } else {
                            run_paged_prefill_attention_fallback(
                                &q,
                                key_cache,
                                value_cache,
                                input_metadata,
                                kv_lens.as_slice(),
                                &paged_attn,
                                batch,
                                seq_len,
                                hidden_states,
                                self.scaling as f32,
                                self.num_key_value_heads,
                                self.num_key_value_groups,
                                self.head_dim,
                            )?
                        }
                    } else {
                        run_paged_prefill_attention_fallback(
                            &q,
                            key_cache,
                            value_cache,
                            input_metadata,
                            kv_lens.as_slice(),
                            &paged_attn,
                            batch,
                            seq_len,
                            hidden_states,
                            self.scaling as f32,
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            self.head_dim,
                        )?
                    }
                }
                #[cfg(not(feature = "metal-paged-attn"))]
                {
                    run_paged_prefill_attention_fallback(
                        &q,
                        key_cache,
                        value_cache,
                        input_metadata,
                        kv_lens.as_slice(),
                        &paged_attn,
                        batch,
                        seq_len,
                        hidden_states,
                        self.scaling as f32,
                        self.num_key_value_heads,
                        self.num_key_value_groups,
                        self.head_dim,
                    )?
                }
            };
            attn_output.transpose(1, 2)?.reshape((
                batch,
                seq_len,
                self.num_attention_heads * self.head_dim,
            ))?
        };
        let attn = attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextDecoderLayer {
    self_attn: ThinkerTextAttention,
    mlp: ThinkerTextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl ThinkerTextDecoderLayer {
    fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        let self_attn =
            ThinkerTextAttention::load(cfg, vb.pp("self_attn"), device, use_flash_attn, isq)?;
        let mlp = ThinkerTextMlp::load(cfg, vb.pp("mlp"), isq)?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &ThinkerTextRotaryEmbedding,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope_scaling,
        )?;
        let x = (&residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }

    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &ThinkerTextRotaryEmbedding,
        layer_cache: (&mut KVCache, usize),
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_with_kv_cache(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope_scaling,
            layer_cache,
        )?;
        let x = (&residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }

    #[cfg(feature = "paged-attn")]
    fn forward_with_paged_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        input_metadata: &PagedInputMetadata,
        rope_scaling: &ThinkerTextRotaryEmbedding,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_with_paged_cache(
            &x,
            position_embeddings,
            input_metadata,
            rope_scaling,
            key_cache,
            value_cache,
        )?;
        let x = (&residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

/// Text part of Qwen3-ASR thinker (decoder-only transformer).
#[derive(Debug, Clone)]
pub struct ThinkerTextModel {
    embed_tokens: Embedding,
    layers: Vec<ThinkerTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: ThinkerTextRotaryEmbedding,
    hidden_size: usize,
    use_flash_attn: bool,
}

impl ThinkerTextModel {
    pub fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(ThinkerTextDecoderLayer::load(
                cfg,
                vb.pp("layers").pp(idx.to_string()),
                device,
                use_flash_attn,
                isq,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = ThinkerTextRotaryEmbedding::load(cfg, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            hidden_size: cfg.hidden_size,
            use_flash_attn,
        })
    }

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[cfg(feature = "paged-attn")]
    pub fn num_key_value_heads(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.self_attn.num_key_value_heads)
            .unwrap_or(0)
    }

    #[cfg(feature = "paged-attn")]
    pub fn head_dim(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.self_attn.head_dim)
            .unwrap_or(0)
    }

    pub fn forward(
        &self,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(attention::make_causal_mask(
                attention_mask,
                batch,
                seq_len,
                dtype,
                device,
            )?)
        };

        // Shape is used only for dtype/device in the rope kernel, matching the official behavior.
        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                position_embeddings,
                causal_mask.as_ref(),
                attention_mask,
                &self.rotary_emb,
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    pub fn forward_with_kv_cache(
        &self,
        attention_mask: &Tensor,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let (b2, total_len) = attention_mask.dims2()?;
        if b2 != batch {
            candle_core::bail!("attention_mask batch mismatch: expected={batch}, got={b2}");
        }

        let cache_len = kv_cache.seq_len();
        if total_len != cache_len.saturating_add(seq_len) {
            candle_core::bail!(
                "attention_mask total_len mismatch vs cache: total_len={total_len} cache_len={cache_len} new_len={seq_len}"
            );
        }

        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(attention::make_causal_mask_cached(
                Some(attention_mask),
                batch,
                cache_len,
                seq_len,
                dtype,
                device,
            )?)
        };

        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = initial_hidden_states_for_paged_decode(inputs_embeds)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_kv_cache(
                &hidden_states,
                position_embeddings,
                causal_mask.as_ref(),
                Some(attention_mask),
                &self.rotary_emb,
                (&mut *kv_cache, layer_idx),
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    #[cfg(feature = "paged-attn")]
    pub fn forward_with_paged_cache(
        &self,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor> {
        let (_batch, _seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = initial_hidden_states_for_paged_decode(inputs_embeds)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (key_cache, value_cache) = paged_cache.key_value_cache(layer_idx)?;
            hidden_states = layer.forward_with_paged_cache(
                &hidden_states,
                position_embeddings,
                input_metadata,
                &self.rotary_emb,
                key_cache,
                value_cache,
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    pub fn forward_decode_one_without_padding(
        &self,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }
        if seq_len != 1 {
            candle_core::bail!("decode-one fast path expects seq_len=1, got={seq_len}");
        }

        let token_attention_mask = if self.use_flash_attn {
            let total_len = kv_cache.seq_len().saturating_add(seq_len);
            Some(Tensor::ones(
                (batch, total_len),
                candle_core::DType::U32,
                inputs_embeds.device(),
            )?)
        } else {
            None
        };

        // seq_len==1: only the first (temporal) modality is needed for mRoPE
        let (cos, sin) = self
            .rotary_emb
            .forward_first_modality(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_kv_cache(
                &hidden_states,
                position_embeddings,
                None,
                token_attention_mask.as_ref(),
                &self.rotary_emb,
                (&mut *kv_cache, layer_idx),
            )?;
        }

        self.norm.forward(&hidden_states)
    }
}

fn _require_mrope_enabled(cfg: &TextConfig) -> Result<&RopeScaling> {
    cfg.rope_scaling.as_ref().ok_or_else(|| {
        candle_core::Error::Msg(
            "text_config.rope_scaling is required (Qwen3-ASR uses mRoPE)".to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::ThinkerTextRotaryEmbedding;
    #[cfg(feature = "paged-attn")]
    use super::unpack_gathered_kv_for_attention;

    #[test]
    fn test_rotary_embedding_loads_without_rope_scaling() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cfg = crate::config::TextConfig::default();
        let _ = ThinkerTextRotaryEmbedding::load(&cfg, &device)?;
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn test_unpack_gathered_kv_for_attention_repeats_groups() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let packed = candle_core::Tensor::from_vec(
            vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0],
            (3usize, 2usize, 1usize),
            &device,
        )?;
        let out = unpack_gathered_kv_for_attention(&packed, &[2, 1], 2, 2, 1, &device)?;
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = vec![1.0, 2.0, 10.0, 20.0, 3.0, 0.0, 30.0, 0.0];
        if got != expected {
            anyhow::bail!("unexpected unpacked kv: expected={expected:?} got={got:?}");
        }
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn test_unpack_gathered_kv_for_attention_equal_length_fast_path() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let packed = candle_core::Tensor::from_vec(
            vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            (4usize, 2usize, 1usize),
            &device,
        )?;
        let out = unpack_gathered_kv_for_attention(&packed, &[2, 2], 2, 2, 1, &device)?;
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        if got != expected {
            anyhow::bail!("unexpected equal-length unpacked kv: expected={expected:?} got={got:?}");
        }
        Ok(())
    }
}
