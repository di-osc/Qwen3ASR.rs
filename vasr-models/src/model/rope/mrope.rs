//! Multimodal RoPE (mRoPE) helpers.

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use crate::config::RopeScaling;
use crate::model::rope::core::RopeCore;
use crate::model::rope::scaling::RopeScalingType;

/// Whether to use the mistralrs Metal/CUDA rotary kernel for seq_len==1 decode.
fn use_accelerated_rotary(device: &Device) -> bool {
    (device.is_metal() || device.is_cuda()) && std::env::var_os("VASR_DISABLE_ACCEL_ROPE").is_none()
}

/// Multimodal rotary embedding generator for 3D positions.
///
/// `position_ids` is shaped `(3, batch, seq_len)` where the leading dimension
/// corresponds to `[temporal, height, width]`.
#[derive(Debug, Clone)]
pub struct MultimodalRotaryEmbedding {
    core: RopeCore,
}

impl MultimodalRotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let core = RopeCore::new(head_dim, max_position_embeddings, rope_theta, device)?;
        Ok(Self { core })
    }

    pub fn with_scaling(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        scaling: &RopeScaling,
        device: &Device,
    ) -> Result<Self> {
        let core = RopeCore::with_scaling(
            head_dim,
            max_position_embeddings,
            rope_theta,
            scaling,
            device,
        )?;
        Ok(Self { core })
    }

    pub fn attention_scaling(&self) -> f64 {
        self.core.attention_scaling
    }

    pub fn scaling_type(&self) -> RopeScalingType {
        self.core.scaling_type
    }

    /// Compute cos and sin for multimodal positions.
    ///
    /// Returns half-dim embeddings of shape `(3, batch, seq_len, head_dim/2)`.
    pub fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let seq_len = position_ids.dim(2)?;

        let inv_freq = self.core.get_inv_freq(seq_len)?;

        // inv_freq: (half_dim,) -> (1, 1, half_dim, 1)
        let inv_freq = inv_freq
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(3)?
            .to_dtype(DType::F32)?;

        // position_ids: (3, batch, seq_len) -> (3, batch, 1, seq_len)
        let position_ids = position_ids.unsqueeze(2)?.to_dtype(DType::F32)?;

        // freqs: (3, batch, half_dim, seq_len) -> (3, batch, seq_len, half_dim)
        let freqs = inv_freq.broadcast_mul(&position_ids)?;
        let freqs = freqs.transpose(2, 3)?.contiguous()?;

        let cos = (freqs.cos()? * self.core.attention_scaling)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * self.core.attention_scaling)?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    /// Compute cos/sin for only the first modality (temporal), used in seq_len==1 decode.
    ///
    /// Returns `(cos, sin)` each of shape `(batch, seq_len, head_dim/2)`.
    pub fn forward_first_modality(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let seq_len = position_ids.dim(2)?;

        let inv_freq = self.core.get_inv_freq(seq_len)?;

        // Use only the first (temporal) position row: (1, batch, seq_len)
        let pos_first = position_ids.i(0)?;

        // inv_freq: (half_dim,) -> (1, 1, half_dim, 1)
        let inv_freq = inv_freq
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(3)?
            .to_dtype(DType::F32)?;

        // pos_first: (batch, seq_len) -> (batch, 1, 1, seq_len)
        let pos_first = pos_first
            .unsqueeze(1)?
            .unsqueeze(1)?
            .to_dtype(DType::F32)?;

        // freqs: (batch, 1, half_dim, seq_len) -> (batch, 1, seq_len, half_dim)
        let freqs = inv_freq.broadcast_mul(&pos_first)?;
        let freqs = freqs.transpose(2, 3)?.contiguous()?;

        let cos = (freqs.cos()? * self.core.attention_scaling)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * self.core.attention_scaling)?.to_dtype(dtype)?;
        // Squeeze the modality dimension: (batch, 1, seq_len, half_dim) -> (batch, seq_len, half_dim)
        let cos = cos.squeeze(1)?;
        let sin = sin.squeeze(1)?;
        Ok((cos, sin))
    }
}

/// Apply multimodal rotary position embedding for 3D positions.
///
/// For seq_len==1 on Metal/CUDA devices, uses the accelerated `mistralrs_quant::rotary`
/// kernel instead of the Candle fallback.
pub fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    // seq_len==1 fast path: use Metal/CUDA accelerated rotary kernel
    if q.dim(2).unwrap_or(0) == 1 && use_accelerated_rotary(q.device()) {
        return apply_multimodal_rotary_pos_emb_seq_one(q, k, cos, sin, interleaved);
    }

    if interleaved {
        apply_multimodal_rotary_pos_emb_interleaved(q, k, cos, sin, mrope_section)
    } else {
        apply_multimodal_rotary_pos_emb_standard(q, k, cos, sin, mrope_section)
    }
}

/// Seq-len-1 fast path: use the accelerated rotary kernel for single-token decode.
///
/// For mRoPE with seq_len==1, only the first (temporal) modality matters.
/// We extract that slice and feed it to `mistralrs_quant::rotary::apply_rotary_qk`.
fn apply_multimodal_rotary_pos_emb_seq_one(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    // cos/sin can be either:
    //   (3, batch, 1, half_dim)  — full mRoPE from `forward()`
    //   (batch, 1, half_dim)     — single modality from `forward_first_modality()`
    // Normalize to (batch, half_dim).
    let (cos_first, sin_first) = if cos.rank() == 4 && cos.dim(0)? == 3 {
        // Full 3-modality: extract first (temporal) modality
        (
            cos.i(0)?.squeeze(1)?, // (batch, half_dim)
            sin.i(0)?.squeeze(1)?, // (batch, half_dim)
        )
    } else {
        // Already single modality: just squeeze the seq dim
        (
            cos.squeeze(1)?, // (batch, half_dim)
            sin.squeeze(1)?, // (batch, half_dim)
        )
    };

    // Qwen3 uses NeoX-style RoPE (half-dimension rotation, not GPT-NeoX's interleaved)
    // For interleaved mRoPE the order differs but the accelerator kernel handles both
    // via the is_neox parameter: interleaved=true means GPT-NeoX style
    let is_neox = !interleaved;

    #[cfg(any(
        feature = "metal-paged-attn",
        feature = "cuda-paged-attn",
        feature = "cuda-quant"
    ))]
    {
        vasr_quant::apply_rotary_qk(q, k, &cos_first, &sin_first, is_neox)
    }
    #[cfg(not(any(
        feature = "metal-paged-attn",
        feature = "cuda-paged-attn",
        feature = "cuda-quant"
    )))]
    {
        // Fallback to Candle-based RoPE for CPU or non-accelerated builds
        let _ = (cos_first, sin_first, is_neox);
        apply_rope_batched_generic(q, k, cos, sin)
    }
}

#[cfg(not(any(
    feature = "metal-paged-attn",
    feature = "cuda-paged-attn",
    feature = "cuda-quant"
)))]
fn apply_rope_batched_generic(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Simplified non-accelerated path: take first modality and use apply_rope_batched
    let cos_first = cos.i(0)?.squeeze(1)?;
    let sin_first = sin.i(0)?.squeeze(1)?;
    let q_embed = apply_rope_batched(q, &cos_first, &sin_first)?;
    let k_embed = apply_rope_batched(k, &cos_first, &sin_first)?;
    Ok((q_embed, k_embed))
}

fn apply_multimodal_rotary_pos_emb_standard(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let mut cos_parts: Vec<Tensor> = Vec::new();
    let mut sin_parts: Vec<Tensor> = Vec::new();
    let mut offset = 0usize;

    for (i, &section_size) in mrope_section.iter().enumerate() {
        let cos_modality = cos.i(i)?;
        let sin_modality = sin.i(i)?;

        let cos_section = cos_modality.narrow(candle_core::D::Minus1, offset, section_size)?;
        let sin_section = sin_modality.narrow(candle_core::D::Minus1, offset, section_size)?;

        cos_parts.push(cos_section);
        sin_parts.push(sin_section);
        offset = offset.saturating_add(section_size);
    }

    let cos_half = Tensor::cat(
        &cos_parts.iter().collect::<Vec<_>>(),
        candle_core::D::Minus1,
    )?
    .contiguous()?;
    let sin_half = Tensor::cat(
        &sin_parts.iter().collect::<Vec<_>>(),
        candle_core::D::Minus1,
    )?
    .contiguous()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = apply_rope_batched(&q, &cos_half, &sin_half)?;
    let k_embed = apply_rope_batched(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}

fn apply_multimodal_rotary_pos_emb_interleaved(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let (_modalities, _batch, _seq_len, half_dim) = cos.dims4()?;
    let modality_num = mrope_section.len();

    let original_dtype = cos.dtype();
    let cos_half = cos.contiguous()?.to_dtype(DType::F32)?;
    let sin_half = sin.contiguous()?.to_dtype(DType::F32)?;

    let m1_end = if mrope_section.len() > 1 {
        (mrope_section[1] * modality_num).min(half_dim)
    } else {
        0
    };
    let m2_end = if mrope_section.len() > 2 {
        (mrope_section[2] * modality_num).min(half_dim)
    } else {
        0
    };

    let cos_m0 = cos_half.i(0)?.contiguous()?;
    let sin_m0 = sin_half.i(0)?.contiguous()?;
    let cos_m1 = cos_half.i(1)?.contiguous()?;
    let sin_m1 = sin_half.i(1)?.contiguous()?;
    let cos_m2 = cos_half.i(2)?.contiguous()?;
    let sin_m2 = sin_half.i(2)?.contiguous()?;

    let mut cos_parts: Vec<Tensor> = Vec::with_capacity(half_dim);
    let mut sin_parts: Vec<Tensor> = Vec::with_capacity(half_dim);

    for pos in 0..half_dim {
        let modality = if modality_num >= 3 && mrope_section.len() >= 3 {
            if pos >= 1 && pos < m1_end && (pos - 1) % modality_num == 0 {
                1
            } else if pos >= 2 && pos < m2_end && (pos - 2) % modality_num == 0 {
                2
            } else {
                0
            }
        } else {
            0
        };

        let (cos_src, sin_src) = if modality == 0 {
            (&cos_m0, &sin_m0)
        } else if modality == 1 {
            (&cos_m1, &sin_m1)
        } else if modality == 2 {
            (&cos_m2, &sin_m2)
        } else {
            candle_core::bail!("invalid modality={modality} for interleaved mRoPE")
        };

        let cos_col = cos_src.narrow(2, pos, 1)?;
        let sin_col = sin_src.narrow(2, pos, 1)?;
        cos_parts.push(cos_col);
        sin_parts.push(sin_col);
    }

    let cos_half = Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), 2)?;
    let sin_half = Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), 2)?;

    let cos_half = cos_half.to_dtype(original_dtype)?.contiguous()?;
    let sin_half = sin_half.to_dtype(original_dtype)?.contiguous()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = apply_rope_batched(&q, &cos_half, &sin_half)?;
    let k_embed = apply_rope_batched(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}

fn apply_rope_batched(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let (batch, _heads, seq_len, head_dim) = x.dims4()?;
    let (cos_batch, cos_seq, cos_half_dim) = cos_half.dims3()?;
    let (sin_batch, sin_seq, sin_half_dim) = sin_half.dims3()?;
    if cos_batch != batch || sin_batch != batch {
        candle_core::bail!("mRoPE batch mismatch: x={batch}, cos={cos_batch}, sin={sin_batch}");
    }
    if cos_seq < seq_len || sin_seq < seq_len {
        candle_core::bail!("mRoPE seq mismatch: x={seq_len}, cos={cos_seq}, sin={sin_seq}");
    }
    if cos_half_dim * 2 != head_dim || sin_half_dim * 2 != head_dim {
        candle_core::bail!(
            "mRoPE dim mismatch: x={head_dim}, cos_half={cos_half_dim}, sin_half={sin_half_dim}"
        );
    }

    let cos = Tensor::cat(&[cos_half, cos_half], candle_core::D::Minus1)?
        .narrow(1, 0, seq_len)?
        .unsqueeze(1)?;
    let sin = Tensor::cat(&[sin_half, sin_half], candle_core::D::Minus1)?
        .narrow(1, 0, seq_len)?
        .unsqueeze(1)?;
    (x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?)?.contiguous()
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(
        candle_core::D::Minus1,
        last_dim / 2,
        last_dim - last_dim / 2,
    )?;
    Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::{MultimodalRotaryEmbedding, apply_multimodal_rotary_pos_emb};

    #[test]
    fn test_multimodal_rope_shapes() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;

        // head_dim=128 => half_dim=64, mrope_section sums to 64.
        let head_dim = 128usize;
        let rope = MultimodalRotaryEmbedding::new(head_dim, 1024, 10000.0, &device)?;

        let batch = 2usize;
        let seq_len = 7usize;

        // Dummy x: only dtype/device are used by forward.
        let x = candle_core::Tensor::zeros(
            (batch, 1, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;

        // position_ids: (3, batch, seq_len)
        let pos1 = candle_core::Tensor::arange(0i64, seq_len as i64, &device)?.unsqueeze(0)?;
        let pos1 = pos1.broadcast_as((batch, seq_len))?;
        let position_ids = candle_core::Tensor::stack(&[&pos1, &pos1, &pos1], 0)?;

        let (cos, sin) = rope.forward(&x, &position_ids)?;
        let (m, b, s, half) = cos.dims4()?;
        if (m, b, s, half) != (3, batch, seq_len, head_dim / 2) {
            anyhow::bail!("unexpected cos dims: {:?}", cos.dims());
        }
        let (m, b, s, half) = sin.dims4()?;
        if (m, b, s, half) != (3, batch, seq_len, head_dim / 2) {
            anyhow::bail!("unexpected sin dims: {:?}", sin.dims());
        }

        // Apply mRoPE to q/k (shape checks only).
        let q = candle_core::Tensor::zeros(
            (batch, 4, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;
        let k = candle_core::Tensor::zeros(
            (batch, 4, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;
        let mrope_section = &[24usize, 20, 20];

        let (q1, k1) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, mrope_section, false)?;
        if q1.dims() != q.dims() || k1.dims() != k.dims() {
            anyhow::bail!("unexpected output dims for standard mRoPE");
        }

        let (q2, k2) = apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin, mrope_section, true)?;
        if q2.dims() != q.dims() || k2.dims() != k.dims() {
            anyhow::bail!("unexpected output dims for interleaved mRoPE");
        }

        Ok(())
    }
}
