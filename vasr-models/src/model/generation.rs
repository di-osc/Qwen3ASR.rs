//! Autoregressive generation loop (greedy first, then sampling).
//!
//! This module is intentionally small and conservative: it prioritizes correctness
//! and parity bring-up over performance. KV caching is a separate concern.

use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::model::kv_cache::KVCache;
#[cfg(feature = "paged-attn")]
use crate::model::paged_cache_runtime::SharedPagedCacheRuntime;
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::model::thinker::get_rope_index;
#[cfg(feature = "paged-attn")]
use vasr_paged_attn::PagedKvCache;
use vasr_quant::isq_linear::set_linear_is_prefill;

#[cfg(feature = "paged-attn")]
const PAGED_CACHE_BLOCK_SIZE: usize = 32;

#[cfg(feature = "timing")]
fn duration_to_us(d: std::time::Duration) -> u64 {
    let us = d.as_micros();
    if us > u128::from(u64::MAX) {
        u64::MAX
    } else {
        us as u64
    }
}

#[cfg(feature = "paged-attn")]
pub(crate) fn use_paged_attn_on_device(device: &Device) -> bool {
    if std::env::var_os("VASR_DISABLE_PAGED_ATTN").is_some() {
        return false;
    }
    if device.is_metal() {
        #[cfg(feature = "metal-paged-attn")]
        {
            // Metal defaults to eager KV-cache decode (SDPA) for best single-stream
            // throughput; paged-attn is opt-in via VASR_FORCE_PAGED_ATTN=1.
            // See docs/performance/qwen3-asr-metal-isq8-optimization-notes.md Pass 9.
            if std::env::var_os("VASR_FORCE_PAGED_ATTN").is_some() {
                return true;
            }
            return false;
        }
        #[cfg(not(feature = "metal-paged-attn"))]
        {
            return false;
        }
    }
    device.is_cuda()
}

#[cfg(feature = "paged-attn")]
fn should_use_paged_attention(device: &Device, _batch: usize) -> bool {
    if std::env::var_os("VASR_DISABLE_PAGED_ATTN").is_some() {
        return false;
    }
    use_paged_attn_on_device(device)
}

fn argmax_token_id(logits: &Tensor) -> Result<u32> {
    #[cfg(feature = "metal-paged-attn")]
    if logits.device().is_metal()
        && std::env::var_os("VASR_DISABLE_METAL_ARGMAX").is_none()
        && matches!(logits.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        return Ok(vasr_quant::argmax_token_id(logits)?);
    }

    Ok(logits.argmax(0usize)?.to_scalar::<u32>()?)
}

#[cfg(feature = "metal-paged-attn")]
fn metal_argmax_scratch_for_device(device: &Device) -> Option<vasr_quant::MetalArgmaxScratch> {
    if device.is_metal() && std::env::var_os("VASR_DISABLE_METAL_ARGMAX").is_none() {
        Some(vasr_quant::MetalArgmaxScratch::new())
    } else {
        None
    }
}

#[cfg(not(feature = "metal-paged-attn"))]
fn metal_argmax_scratch_for_device(_device: &Device) -> Option<()> {
    None
}

#[cfg(feature = "metal-paged-attn")]
fn argmax_token_id_with_scratch(
    logits: &Tensor,
    scratch: Option<&mut vasr_quant::MetalArgmaxScratch>,
) -> Result<u32> {
    if let Some(scratch) = scratch {
        if logits.device().is_metal()
            && std::env::var_os("VASR_DISABLE_METAL_ARGMAX").is_none()
            && matches!(logits.dtype(), DType::F32 | DType::F16 | DType::BF16)
        {
            return Ok(scratch.argmax_token_id(logits)?);
        }
    }
    argmax_token_id(logits)
}

#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GenerationTimings {
    /// Time spent building the initial prompt tensors and transferring them to device.
    pub prompt_tensors_us: u64,
    /// Time spent running the initial prompt prefill forward pass (including mRoPE indices).
    pub prefill_us: u64,
    /// Time spent building prefill token/audio embeddings.
    pub prefill_inputs_us: u64,
    /// Time spent computing prefill mRoPE position ids.
    pub prefill_rope_us: u64,
    /// Time spent allocating paged-cache slots and building prefill metadata.
    pub prefill_metadata_us: u64,
    /// Time spent building explicit prefill attention masks.
    pub prefill_mask_us: u64,
    /// Time spent launching/running the prefill model forward.
    pub prefill_forward_us: u64,
    /// Time spent gathering last logits from prefill outputs.
    pub prefill_gather_us: u64,
    /// Time spent preparing decode metadata/positions after prefill.
    pub prefill_decode_setup_us: u64,
    /// Time spent selecting the first next token after prefill.
    pub prefill_argmax_us: u64,
    /// Time spent in the token-by-token decode loop (including mRoPE indices).
    pub decode_us: u64,
    /// Time spent creating the one-token decode input tensor.
    pub decode_token_tensor_us: u64,
    /// Time spent embedding the one-token decode input.
    pub decode_embed_us: u64,
    /// Time spent creating decode position ids.
    pub decode_position_us: u64,
    /// Time spent creating paged-attention decode metadata.
    pub decode_metadata_us: u64,
    /// Time spent replaying the CUDA graph.
    pub decode_graph_replay_us: u64,
    /// Time spent in paged decode forward passes (excluding argmax sync).
    pub decode_forward_us: u64,
    /// Optional probe: time spent synchronizing the device immediately before argmax.
    pub decode_pre_argmax_sync_us: u64,
    /// Time spent selecting and copying the next token back to host.
    pub decode_argmax_us: u64,
    /// Prompt/context length at the start of token-by-token decoding.
    pub prompt_len: usize,
    /// Number of decode loop iterations executed (<= `max_new_tokens`).
    pub steps: usize,
    /// Total non-EOS tokens produced across the batch.
    pub tokens_generated: usize,
}

#[cfg(not(feature = "timing"))]
#[derive(Debug, Clone, Default)]
pub struct GenerationTimings;

#[cfg(feature = "timing")]
fn timing_sync_before_argmax_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("VASR_TIMING_SYNC_BEFORE_ARGMAX")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

/// Greedy decode for a single sequence (batch=1).
///
/// Returns generated token ids (excluding the input prompt ids).
pub fn greedy_generate(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    if input_ids.is_empty() {
        bail!("input_ids is empty");
    }
    if attention_mask.len() != input_ids.len() {
        bail!(
            "attention_mask length mismatch: expected={}, got={}",
            input_ids.len(),
            attention_mask.len()
        );
    }
    if eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }
    if max_new_tokens == 0 {
        return Ok(vec![]);
    }

    let mut ids: Vec<u32> = input_ids.to_vec();
    let mut attn: Vec<u32> = attention_mask.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders(input_ids, thinker.audio_token_id())
    } else {
        0
    };

    for _step in 0..max_new_tokens {
        let seq_len = ids.len();
        let input_ids_t = Tensor::from_vec(ids.clone(), (1usize, seq_len), device)?;
        let attention_mask_t = Tensor::from_vec(attn.clone(), (1usize, seq_len), device)?;

        let logits = thinker.forward_with_audio_features(
            &input_ids_t,
            &attention_mask_t,
            audio_features,
            audio_placeholder_count,
        )?;

        let last = logits.i((0usize, seq_len.saturating_sub(1)))?;
        let next_id = argmax_token_id(&last)?;

        if eos_token_ids.contains(&next_id) {
            break;
        }

        generated.push(next_id);
        ids.push(next_id);
        attn.push(1);
    }

    Ok(generated)
}

/// Greedy decode with a KV cache (single sequence, batch=1).
///
/// This is the practical default for transcription: without caching, decoding
/// is prohibitively slow because each step would re-run the full prompt.
fn kv_cache_for_generation(
    thinker: &ThinkerForConditionalGeneration,
    seq_len: usize,
    max_new_tokens: usize,
) -> Result<KVCache> {
    let max_seq_len = seq_len
        .checked_add(max_new_tokens)
        .ok_or_else(|| anyhow::anyhow!("kv cache capacity overflow"))?;
    Ok(KVCache::with_max_seq_len(
        thinker.num_text_layers(),
        max_seq_len,
    ))
}

pub(crate) fn argmax_token_ids_from_logits(logits: &Tensor, batch: usize) -> Result<Vec<u32>> {
    if batch == 1 {
        let row = if logits.rank() == 1 {
            logits.clone()
        } else {
            logits.i((0,))?
        };
        return Ok(vec![argmax_token_id(&row)?]);
    }
    let next = logits.argmax(1usize)?.to_vec1::<u32>()?;
    if next.len() != batch {
        bail!(
            "internal error: batch argmax mismatch: expected={batch}, got={}",
            next.len()
        );
    }
    Ok(next)
}

pub fn greedy_generate_cached(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    if input_ids.is_empty() {
        bail!("input_ids is empty");
    }
    if attention_mask.len() != input_ids.len() {
        bail!(
            "attention_mask length mismatch: expected={}, got={}",
            input_ids.len(),
            attention_mask.len()
        );
    }
    if eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }
    if max_new_tokens == 0 {
        return Ok(vec![]);
    }

    let mut ids: Vec<u32> = input_ids.to_vec();
    let mut generated: Vec<u32> = Vec::new();

    let seq_len = ids.len();
    let mut kv_cache = kv_cache_for_generation(thinker, seq_len, max_new_tokens)?;

    // Prefill the cache with the full prompt (including audio features).
    let input_ids_t = Tensor::from_vec(ids.clone(), (1usize, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attention_mask.to_vec(), (1usize, seq_len), device)?;
    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders(input_ids, thinker.audio_token_id())
    } else {
        0
    };
    let inputs_embeds = thinker.inputs_embeds_with_audio_features(
        &input_ids_t,
        audio_features,
        audio_placeholder_count,
    )?;
    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let logits = {
        let _linear_prefill_guard = set_linear_is_prefill(true);
        thinker.forward_embeds_with_kv_cache(
            &attention_mask_t,
            &position_ids,
            &inputs_embeds,
            &mut kv_cache,
        )?
    };

    let mut next_id = argmax_token_id(&logits.i((0usize, seq_len.saturating_sub(1)))?)?;

    let dense_attention = attention_mask_is_dense(attention_mask);
    let prompt_len = prompt_len_from_left_padded_mask(attention_mask, seq_len)?;
    let decode_position_ids = position_ids_for_decode_steps(prompt_len, max_new_tokens, device)?;
    let ones_col = if dense_attention {
        None
    } else {
        Some(Tensor::ones((1usize, 1usize), DType::U32, device)?)
    };
    let mut attention_mask_total = attention_mask_t;
    #[cfg(feature = "metal-paged-attn")]
    let mut metal_argmax_scratch = metal_argmax_scratch_for_device(device);
    let _linear_decode_guard = set_linear_is_prefill(false);

    for step in 0..max_new_tokens {
        if eos_token_ids.contains(&next_id) {
            break;
        }

        generated.push(next_id);
        ids.push(next_id);

        let input_ids_new = Tensor::from_vec(vec![next_id], (1usize, 1usize), device)?;
        let inputs_embeds_new = thinker.embed_tokens(&input_ids_new)?;

        let position_ids_new = decode_position_ids
            .get(step)
            .ok_or_else(|| {
                anyhow::anyhow!("missing single-seq decode position ids for step {step}")
            })?
            .clone();

        let logits_new = {
            if dense_attention {
                thinker.forward_decode_one_without_padding(
                    &position_ids_new,
                    &inputs_embeds_new,
                    &mut kv_cache,
                )?
            } else {
                let ones_col = ones_col
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing decode attention mask column"))?;
                attention_mask_total = Tensor::cat(&[&attention_mask_total, ones_col], 1)?;
                thinker.forward_embeds_with_kv_cache(
                    &attention_mask_total,
                    &position_ids_new,
                    &inputs_embeds_new,
                    &mut kv_cache,
                )?
            }
        };
        next_id = {
            let step_logits = logits_new.i((0usize, 0usize))?;
            #[cfg(feature = "metal-paged-attn")]
            let next = argmax_token_id_with_scratch(&step_logits, metal_argmax_scratch.as_mut())?;
            #[cfg(not(feature = "metal-paged-attn"))]
            let next = argmax_token_id(&step_logits)?;
            next
        };
    }

    Ok(generated)
}

/// Greedy decode with a KV cache for a batch of sequences.
///
/// This mirrors the shape semantics used by Transformers `generate()`:
/// - `input_ids` and `attention_mask` are left-padded to a common `seq_len`.
/// - Each step generates one token per sequence until EOS or `max_new_tokens`.
/// - Once a sequence hits EOS, we keep feeding an EOS token for it so we can
///   continue decoding the remaining batch with a shared KV cache.
///
/// Returns generated token ids (excluding the input prompt ids), one vec per sample.
pub fn greedy_generate_cached_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<Vec<u32>>> {
    let opts = GreedyGenerateBatchOpts {
        audio_features,
        max_new_tokens,
        eos_token_ids,
        #[cfg(feature = "paged-attn")]
        paged_runtime: None,
    };
    greedy_generate_cached_batch_impl(thinker, device, input_ids, attention_mask, opts, None)
}

#[cfg(feature = "paged-attn")]
pub fn greedy_generate_cached_batch_with_paged_runtime(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    paged_runtime: Option<&SharedPagedCacheRuntime>,
) -> Result<Vec<Vec<u32>>> {
    let opts = GreedyGenerateBatchOpts {
        audio_features,
        max_new_tokens,
        eos_token_ids,
        #[cfg(feature = "paged-attn")]
        paged_runtime,
    };
    greedy_generate_cached_batch_impl(thinker, device, input_ids, attention_mask, opts, None)
}

#[cfg(feature = "timing")]
pub fn greedy_generate_cached_batch_timed(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<(Vec<Vec<u32>>, GenerationTimings)> {
    let mut timings = GenerationTimings::default();
    let opts = GreedyGenerateBatchOpts {
        audio_features,
        max_new_tokens,
        eos_token_ids,
        #[cfg(feature = "paged-attn")]
        paged_runtime: None,
    };
    let out = greedy_generate_cached_batch_impl(
        thinker,
        device,
        input_ids,
        attention_mask,
        opts,
        Some(&mut timings),
    )?;
    timings.tokens_generated = out.iter().map(Vec::len).sum();
    Ok((out, timings))
}

#[cfg(all(feature = "timing", feature = "paged-attn"))]
pub fn greedy_generate_cached_batch_timed_with_paged_runtime(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    paged_runtime: Option<&SharedPagedCacheRuntime>,
) -> Result<(Vec<Vec<u32>>, GenerationTimings)> {
    let mut timings = GenerationTimings::default();
    let opts = GreedyGenerateBatchOpts {
        audio_features,
        max_new_tokens,
        eos_token_ids,
        paged_runtime,
    };
    let out = greedy_generate_cached_batch_impl(
        thinker,
        device,
        input_ids,
        attention_mask,
        opts,
        Some(&mut timings),
    )?;
    timings.tokens_generated = out.iter().map(Vec::len).sum();
    Ok((out, timings))
}

struct GreedyGenerateBatchOpts<'a> {
    audio_features: Option<&'a Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &'a [u32],
    #[cfg(feature = "paged-attn")]
    paged_runtime: Option<&'a SharedPagedCacheRuntime>,
}

fn greedy_generate_cached_batch_impl(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    opts: GreedyGenerateBatchOpts<'_>,
    #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    #[cfg(not(feature = "timing"))] _timings: Option<&mut GenerationTimings>,
) -> Result<Vec<Vec<u32>>> {
    let batch = input_ids.len();
    if batch == 0 {
        return Ok(vec![]);
    }
    if attention_mask.len() != batch {
        bail!(
            "attention_mask batch mismatch: expected={}, got={}",
            batch,
            attention_mask.len()
        );
    }

    let seq_len = input_ids
        .first()
        .map(|r| r.len())
        .ok_or_else(|| anyhow::anyhow!("input_ids missing first row"))?;
    if seq_len == 0 {
        bail!("input_ids rows are empty");
    }
    if opts.eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }

    for (i, row) in input_ids.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "input_ids[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }
    for (i, row) in attention_mask.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "attention_mask[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }

    if opts.max_new_tokens == 0 {
        return Ok(vec![vec![]; batch]);
    }

    #[cfg(feature = "paged-attn")]
    let use_paged_attn = should_use_paged_attention(device, batch);

    #[cfg(all(feature = "paged-attn", feature = "timing"))]
    if batch == 1 && use_paged_attn && attention_masks_are_dense(attention_mask) {
        let out = greedy_generate_paged(
            thinker,
            device,
            input_ids[0],
            attention_mask[0],
            opts.audio_features,
            opts.max_new_tokens,
            opts.eos_token_ids,
            timings.as_deref_mut(),
        )?;
        return Ok(vec![out]);
    }
    #[cfg(all(feature = "paged-attn", feature = "timing"))]
    if batch > 1 && use_paged_attn && opts.paged_runtime.is_some() {
        return greedy_generate_paged_batch(
            thinker,
            device,
            input_ids,
            attention_mask,
            opts.audio_features,
            opts.max_new_tokens,
            opts.eos_token_ids,
            opts.paged_runtime,
            timings.as_deref_mut(),
        );
    }
    #[cfg(all(feature = "paged-attn", not(feature = "timing")))]
    if batch == 1 && use_paged_attn && attention_masks_are_dense(attention_mask) {
        let out = greedy_generate_paged(
            thinker,
            device,
            input_ids[0],
            attention_mask[0],
            opts.audio_features,
            opts.max_new_tokens,
            opts.eos_token_ids,
            None,
        )?;
        return Ok(vec![out]);
    }
    #[cfg(all(feature = "paged-attn", not(feature = "timing")))]
    if batch > 1 && use_paged_attn && opts.paged_runtime.is_some() {
        return greedy_generate_paged_batch(
            thinker,
            device,
            input_ids,
            attention_mask,
            opts.audio_features,
            opts.max_new_tokens,
            opts.eos_token_ids,
            opts.paged_runtime,
            None,
        );
    }

    let eos_fill_id = *opts
        .eos_token_ids
        .first()
        .ok_or_else(|| anyhow::anyhow!("eos_token_ids is empty"))?;

    let mut ids_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    let mut attn_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    for i in 0..batch {
        let ids = input_ids
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing input_ids row {i}"))?;
        let attn = attention_mask
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing attention_mask row {i}"))?;
        ids_flat.extend_from_slice(ids);
        attn_flat.extend_from_slice(attn);
    }

    #[cfg(feature = "timing")]
    let start_tensors = std::time::Instant::now();
    let input_ids_t = Tensor::from_vec(ids_flat, (batch, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attn_flat, (batch, seq_len), device)?;
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prompt_tensors_us = t
            .prompt_tensors_us
            .saturating_add(duration_to_us(start_tensors.elapsed()));
    }

    let mut kv_cache = kv_cache_for_generation(thinker, seq_len, opts.max_new_tokens)?;

    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prompt_len = seq_len;
    }

    // Prefill the cache with the full prompt (including audio features).
    #[cfg(feature = "timing")]
    let start_prefill = std::time::Instant::now();
    let audio_placeholder_count = if opts.audio_features.is_some() {
        count_audio_placeholders_batch(input_ids, thinker.audio_token_id())?
    } else {
        0
    };
    #[cfg(feature = "timing")]
    let start_inputs = std::time::Instant::now();
    let inputs_embeds = thinker.inputs_embeds_with_audio_features(
        &input_ids_t,
        opts.audio_features,
        audio_placeholder_count,
    )?;
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prefill_inputs_us = t
            .prefill_inputs_us
            .saturating_add(duration_to_us(start_inputs.elapsed()));
    }
    #[cfg(feature = "timing")]
    let start_rope = std::time::Instant::now();
    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prefill_rope_us = t
            .prefill_rope_us
            .saturating_add(duration_to_us(start_rope.elapsed()));
    }
    #[cfg(feature = "timing")]
    let start_forward = std::time::Instant::now();
    let logits = {
        let _linear_prefill_guard = set_linear_is_prefill(true);
        thinker.forward_embeds_with_kv_cache(
            &attention_mask_t,
            &position_ids,
            &inputs_embeds,
            &mut kv_cache,
        )?
    };
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prefill_forward_us = t
            .prefill_forward_us
            .saturating_add(duration_to_us(start_forward.elapsed()));
    }

    let prompt_lens = prompt_lens_from_attention_masks(attention_mask, seq_len)?;
    let last_logits = gather_last_logits_for_prompt_lens(&logits, prompt_lens.as_slice())?;
    let mut next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prefill_us = t
            .prefill_us
            .saturating_add(duration_to_us(start_prefill.elapsed()));
    }

    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch];
    let mut finished: Vec<bool> = next_ids
        .iter()
        .map(|id| opts.eos_token_ids.contains(id))
        .collect();
    let dense_attention = attention_masks_are_dense(attention_mask);
    let decode_position_ids =
        position_ids_for_decode_steps_batch(prompt_lens.as_slice(), opts.max_new_tokens, device)?;
    let ones_col = if dense_attention {
        None
    } else {
        Some(Tensor::ones((batch, 1usize), DType::U32, device)?)
    };
    let mut attention_mask_total = attention_mask_t;

    #[cfg(feature = "timing")]
    let start_decode = std::time::Instant::now();
    #[cfg(feature = "timing")]
    let mut decode_token_tensor_us = 0u64;
    #[cfg(feature = "timing")]
    let mut decode_embed_us = 0u64;
    #[cfg(feature = "timing")]
    let mut decode_forward_us = 0u64;
    #[cfg(feature = "timing")]
    let mut decode_argmax_us = 0u64;
    #[cfg(feature = "metal-paged-attn")]
    let mut metal_argmax_scratch = metal_argmax_scratch_for_device(device);
    let _linear_decode_guard = set_linear_is_prefill(false);
    for step in 0..opts.max_new_tokens {
        if finished.iter().all(|&x| x) {
            break;
        }
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.steps = t.steps.saturating_add(1);
        }

        #[cfg(feature = "timing")]
        let start_token_tensor = std::time::Instant::now();
        let mut tokens_in: Vec<u32> = Vec::with_capacity(batch);
        for i in 0..batch {
            if finished.get(i).copied().unwrap_or(true) {
                tokens_in.push(eos_fill_id);
                continue;
            }

            let tok = next_ids
                .get(i)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing next_id for batch index {i}"))?;
            if opts.eos_token_ids.contains(&tok) {
                finished[i] = true;
                tokens_in.push(eos_fill_id);
            } else {
                generated[i].push(tok);
                tokens_in.push(tok);
            }
        }

        #[cfg(feature = "timing")]
        {
            decode_token_tensor_us =
                decode_token_tensor_us.saturating_add(duration_to_us(start_token_tensor.elapsed()));
        }
        let input_ids_new = Tensor::from_vec(tokens_in, (batch, 1usize), device)?;
        #[cfg(feature = "timing")]
        let start_embed = std::time::Instant::now();
        let inputs_embeds_new = thinker.embed_tokens(&input_ids_new)?;
        #[cfg(feature = "timing")]
        {
            decode_embed_us = decode_embed_us.saturating_add(duration_to_us(start_embed.elapsed()));
        }

        let position_ids_new = decode_position_ids
            .get(step)
            .ok_or_else(|| {
                anyhow::anyhow!("missing eager batch decode position ids for step {step}")
            })?
            .clone();

        #[cfg(feature = "timing")]
        let start_forward = std::time::Instant::now();
        let logits_new = {
            if dense_attention {
                thinker.forward_decode_one_without_padding(
                    &position_ids_new,
                    &inputs_embeds_new,
                    &mut kv_cache,
                )?
            } else {
                let ones_col = ones_col
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing decode attention mask column"))?;
                attention_mask_total = Tensor::cat(&[&attention_mask_total, ones_col], 1)?;
                thinker.forward_embeds_with_kv_cache(
                    &attention_mask_total,
                    &position_ids_new,
                    &inputs_embeds_new,
                    &mut kv_cache,
                )?
            }
        };
        #[cfg(feature = "timing")]
        {
            decode_forward_us =
                decode_forward_us.saturating_add(duration_to_us(start_forward.elapsed()));
        }

        #[cfg(feature = "timing")]
        let start_argmax = std::time::Instant::now();
        let next = if batch == 1 {
            let step_logits = logits_new.i((0usize, 0usize))?;
            let next = {
                #[cfg(feature = "metal-paged-attn")]
                {
                    argmax_token_id_with_scratch(&step_logits, metal_argmax_scratch.as_mut())?
                }
                #[cfg(not(feature = "metal-paged-attn"))]
                {
                    argmax_token_id(&step_logits)?
                }
            };
            vec![next]
        } else {
            argmax_token_ids_from_logits(&logits_new.squeeze(1)?, batch)?
        };
        #[cfg(feature = "timing")]
        {
            decode_argmax_us =
                decode_argmax_us.saturating_add(duration_to_us(start_argmax.elapsed()));
        }
        next_ids = next;
    }
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.decode_us = t
            .decode_us
            .saturating_add(duration_to_us(start_decode.elapsed()));
        t.decode_token_tensor_us = t
            .decode_token_tensor_us
            .saturating_add(decode_token_tensor_us);
        t.decode_embed_us = t.decode_embed_us.saturating_add(decode_embed_us);
        t.decode_forward_us = t.decode_forward_us.saturating_add(decode_forward_us);
        t.decode_argmax_us = t.decode_argmax_us.saturating_add(decode_argmax_us);
    }

    Ok(generated)
}

#[cfg(feature = "paged-attn")]
fn greedy_generate_paged(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    #[cfg(not(feature = "timing"))] _timings: Option<&mut GenerationTimings>,
) -> Result<Vec<u32>> {
    if input_ids.is_empty() {
        bail!("input_ids is empty");
    }
    if attention_mask.len() != input_ids.len() {
        bail!(
            "attention_mask length mismatch: expected={}, got={}",
            input_ids.len(),
            attention_mask.len()
        );
    }
    if !attention_mask_is_dense(attention_mask) {
        bail!("paged attention fast path requires a dense single-sequence attention mask");
    }

    #[cfg(feature = "timing")]
    let start_tensors = std::time::Instant::now();

    let seq_len = input_ids.len();
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prompt_len = seq_len;
    }
    let input_ids_t = Tensor::from_vec(input_ids.to_vec(), (1usize, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attention_mask.to_vec(), (1usize, seq_len), device)?;

    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prompt_tensors_us = t
            .prompt_tensors_us
            .saturating_add(duration_to_us(start_tensors.elapsed()));
    }

    #[cfg(feature = "timing")]
    let start_prefill = std::time::Instant::now();

    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders(input_ids, thinker.audio_token_id())
    } else {
        0
    };
    let inputs_embeds = thinker.inputs_embeds_with_audio_features(
        &input_ids_t,
        audio_features,
        audio_placeholder_count,
    )?;
    let (num_layers, num_key_value_heads, head_dim) = thinker.paged_cache_config();
    let max_tokens = seq_len
        .checked_add(max_new_tokens)
        .ok_or_else(|| anyhow::anyhow!("paged cache max token capacity overflow"))?;
    let paged_cache = PagedKvCache::new(
        num_layers,
        num_key_value_heads,
        head_dim,
        PAGED_CACHE_BLOCK_SIZE,
        max_tokens,
        inputs_embeds.dtype(),
        device,
    )?;
    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let input_metadata = paged_prefill_metadata(&paged_cache, seq_len, device)?;
    let logits = {
        let _linear_prefill_guard = set_linear_is_prefill(true);
        thinker.forward_embeds_with_paged_cache(
            &position_ids,
            &inputs_embeds,
            &paged_cache,
            &input_metadata,
        )?
    };
    #[cfg(feature = "metal-paged-attn")]
    let mut metal_argmax_scratch = metal_argmax_scratch_for_device(device);
    let mut next_id = {
        let last_logits = logits.i((0usize, seq_len.saturating_sub(1)))?;
        #[cfg(feature = "metal-paged-attn")]
        {
            argmax_token_id_with_scratch(&last_logits, metal_argmax_scratch.as_mut())?
        }
        #[cfg(not(feature = "metal-paged-attn"))]
        {
            argmax_token_id(&last_logits)?
        }
    };

    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.prefill_us = t
            .prefill_us
            .saturating_add(duration_to_us(start_prefill.elapsed()));
    }

    let prompt_len = i64::try_from(seq_len)
        .map_err(|_| anyhow::anyhow!("prompt length overflows i64: {seq_len}"))?;
    let decode_metadata = paged_cache.decode_metadata_for_steps(seq_len, max_new_tokens, device)?;
    let decode_position_ids = position_ids_for_decode_steps(prompt_len, max_new_tokens, device)?;
    let mut generated = Vec::new();

    #[cfg(feature = "timing")]
    let start_decode = std::time::Instant::now();
    for step in 0..max_new_tokens {
        if eos_token_ids.contains(&next_id) {
            break;
        }

        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.steps = t.steps.saturating_add(1);
        }

        generated.push(next_id);

        #[cfg(feature = "timing")]
        let start_token_tensor = std::time::Instant::now();
        let input_ids_new = Tensor::from_vec(vec![next_id], (1usize, 1usize), device)?;
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.decode_token_tensor_us = t
                .decode_token_tensor_us
                .saturating_add(duration_to_us(start_token_tensor.elapsed()));
        }

        #[cfg(feature = "timing")]
        let start_embed = std::time::Instant::now();
        let inputs_embeds_new = thinker.embed_tokens(&input_ids_new)?;
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.decode_embed_us = t
                .decode_embed_us
                .saturating_add(duration_to_us(start_embed.elapsed()));
        }

        #[cfg(feature = "timing")]
        let start_position = std::time::Instant::now();
        let position_ids_new = decode_position_ids
            .get(step)
            .ok_or_else(|| anyhow::anyhow!("missing decode position ids for step {step}"))?
            .clone();
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.decode_position_us = t
                .decode_position_us
                .saturating_add(duration_to_us(start_position.elapsed()));
        }

        #[cfg(feature = "timing")]
        let start_metadata = std::time::Instant::now();
        let input_metadata = decode_metadata
            .get(step)
            .ok_or_else(|| anyhow::anyhow!("missing decode metadata for step {step}"))?;
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.decode_metadata_us = t
                .decode_metadata_us
                .saturating_add(duration_to_us(start_metadata.elapsed()));
        }

        let logits_new = {
            #[cfg(feature = "timing")]
            let start_forward = std::time::Instant::now();
            let _linear_decode_guard = set_linear_is_prefill(false);
            let logits_new = thinker.forward_embeds_with_paged_cache(
                &position_ids_new,
                &inputs_embeds_new,
                &paged_cache,
                input_metadata,
            )?;
            #[cfg(feature = "timing")]
            if let Some(t) = timings.as_mut() {
                t.decode_forward_us = t
                    .decode_forward_us
                    .saturating_add(duration_to_us(start_forward.elapsed()));
            }
            logits_new
        };

        #[cfg(feature = "timing")]
        if timing_sync_before_argmax_enabled() {
            let start_sync = std::time::Instant::now();
            device.synchronize()?;
            if let Some(t) = timings.as_mut() {
                t.decode_pre_argmax_sync_us = t
                    .decode_pre_argmax_sync_us
                    .saturating_add(duration_to_us(start_sync.elapsed()));
            }
        }
        #[cfg(feature = "timing")]
        let start_argmax = std::time::Instant::now();
        let step_logits = logits_new.i((0usize, 0usize))?;
        next_id = {
            #[cfg(feature = "metal-paged-attn")]
            {
                argmax_token_id_with_scratch(&step_logits, metal_argmax_scratch.as_mut())?
            }
            #[cfg(not(feature = "metal-paged-attn"))]
            {
                argmax_token_id(&step_logits)?
            }
        };
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_mut() {
            t.decode_argmax_us = t
                .decode_argmax_us
                .saturating_add(duration_to_us(start_argmax.elapsed()));
        }
    }
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_mut() {
        t.decode_us = t
            .decode_us
            .saturating_add(duration_to_us(start_decode.elapsed()));
    }

    Ok(generated)
}

#[cfg(feature = "paged-attn")]
fn greedy_generate_paged_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    paged_runtime: Option<&SharedPagedCacheRuntime>,
    #[cfg(feature = "timing")] _timings: Option<&mut GenerationTimings>,
    #[cfg(not(feature = "timing"))] _timings: Option<&mut GenerationTimings>,
) -> Result<Vec<Vec<u32>>> {
    use crate::model::paged_batch_engine::{PagedBatchConfig, paged_batch_run};

    let mut runtime_guard = paged_runtime
        .ok_or_else(|| anyhow::anyhow!("paged batch generation requires a shared paged runtime"))?
        .lock()
        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
    let config = PagedBatchConfig::for_static_batch(max_new_tokens, eos_token_ids);
    paged_batch_run(
        thinker,
        device,
        &mut runtime_guard,
        input_ids,
        attention_mask,
        audio_features,
        &config,
    )
}

#[cfg(feature = "paged-attn")]
fn paged_prefill_metadata(
    cache: &PagedKvCache,
    seq_len: usize,
    device: &Device,
) -> Result<crate::model::paged_kv_cache::PagedInputMetadata> {
    Ok(cache.input_metadata_for_range(0, seq_len, seq_len, device)?)
}

#[allow(dead_code)]
fn decode_slot_and_context_len(prompt_len: usize, step: usize) -> Result<(usize, usize)> {
    let slot_pos = prompt_len
        .checked_add(step)
        .ok_or_else(|| anyhow::anyhow!("decode slot position overflow"))?;
    let context_len = slot_pos
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("decode context length overflow"))?;
    Ok((slot_pos, context_len))
}

fn count_audio_placeholders(ids: &[u32], audio_token_id: u32) -> usize {
    ids.iter().filter(|&&id| id == audio_token_id).count()
}

pub(crate) fn count_audio_placeholders_batch(
    input_ids: &[&[u32]],
    audio_token_id: u32,
) -> Result<usize> {
    let mut total = 0usize;
    for (i, row) in input_ids.iter().enumerate() {
        let n = row.iter().filter(|&&id| id == audio_token_id).count();
        total = total.checked_add(n).ok_or_else(|| {
            anyhow::anyhow!("audio placeholder count overflow at batch index {i}: adding {n}")
        })?;
    }
    Ok(total)
}

#[cfg(feature = "paged-attn")]
pub(crate) fn normalize_batch_for_paged_prefill(
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<usize>)> {
    let batch = input_ids.len();
    if batch != attention_mask.len() {
        bail!(
            "input_ids/attention_mask batch mismatch: input_ids={} attention_mask={}",
            batch,
            attention_mask.len()
        );
    }
    if batch == 0 {
        return Ok((vec![], vec![], vec![]));
    }
    let seq_len = input_ids[0].len();
    let mut ids_out = Vec::with_capacity(batch);
    let mut masks_out = Vec::with_capacity(batch);
    let mut lens_out = Vec::with_capacity(batch);

    for (i, (ids, mask)) in input_ids.iter().zip(attention_mask.iter()).enumerate() {
        if ids.len() != seq_len || mask.len() != seq_len {
            bail!(
                "batch row len mismatch at {i}: ids={} mask={} expected={seq_len}",
                ids.len(),
                mask.len()
            );
        }
        let len = mask.iter().filter(|&&v| v != 0).count();
        if len == 0 || len > seq_len {
            bail!("invalid prompt length at batch index {i}: len={len} seq_len={seq_len}");
        }
        let pad_id = ids
            .iter()
            .zip(mask.iter())
            .find_map(|(&id, &m)| if m == 0 { Some(id) } else { None })
            .unwrap_or(ids[0]);
        let mut row_ids = vec![pad_id; seq_len];
        let mut row_mask = vec![0u32; seq_len];
        let mut write_idx = 0usize;
        for (&id, &m) in ids.iter().zip(mask.iter()) {
            if m != 0 {
                row_ids[write_idx] = id;
                row_mask[write_idx] = 1;
                write_idx += 1;
            }
        }
        if write_idx != len {
            bail!(
                "failed to normalize prompt row {i}: expected {len} valid tokens, copied {write_idx}"
            );
        }
        ids_out.push(row_ids);
        masks_out.push(row_mask);
        lens_out.push(len);
    }

    Ok((ids_out, masks_out, lens_out))
}

fn attention_mask_is_dense(attention_mask: &[u32]) -> bool {
    attention_mask.iter().all(|&v| v != 0)
}

fn attention_masks_are_dense(attention_mask: &[&[u32]]) -> bool {
    attention_mask
        .iter()
        .all(|row| attention_mask_is_dense(row))
}

#[cfg(feature = "paged-attn")]
fn attention_mask_is_right_padded(attention_mask: &[u32]) -> bool {
    let mut seen_pad = false;
    for &v in attention_mask {
        if v == 0 {
            seen_pad = true;
        } else if seen_pad {
            return false;
        }
    }
    true
}

#[cfg(feature = "paged-attn")]
pub(crate) fn attention_masks_are_right_padded(attention_mask: &[&[u32]]) -> bool {
    attention_mask
        .iter()
        .all(|row| attention_mask_is_right_padded(row))
}

fn prompt_len_from_left_padded_mask(attention_mask: &[u32], seq_len: usize) -> Result<i64> {
    if attention_mask.len() != seq_len {
        bail!(
            "attention_mask length mismatch: expected={seq_len}, got={}",
            attention_mask.len()
        );
    }

    let mut sum: usize = 0;
    for &v in attention_mask {
        if v != 0 {
            sum = sum
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("attention_mask sum overflow"))?;
        }
    }

    i64::try_from(sum)
        .map_err(|_| anyhow::anyhow!("attention_mask length overflows i64: sum={sum}"))
}

fn prompt_lens_from_attention_masks(attention_mask: &[&[u32]], seq_len: usize) -> Result<Vec<i64>> {
    let batch = attention_mask.len();
    let mut out: Vec<i64> = Vec::with_capacity(batch);
    for (i, row) in attention_mask.iter().enumerate() {
        let len = prompt_len_from_left_padded_mask(row, seq_len).map_err(|e| {
            anyhow::anyhow!("failed to compute prompt length for attention_mask[{i}]: {e:#}")
        })?;
        out.push(len);
    }
    Ok(out)
}

pub(crate) fn gather_last_logits_for_prompt_lens(
    logits: &Tensor,
    prompt_lens: &[i64],
) -> Result<Tensor> {
    let (batch, seq_len, vocab) = logits.dims3()?;
    if prompt_lens.len() != batch {
        bail!(
            "prompt_lens batch mismatch: expected={batch}, got={}",
            prompt_lens.len()
        );
    }
    let mut gather_idx = Vec::with_capacity(batch);
    for (i, &len_i64) in prompt_lens.iter().enumerate() {
        if len_i64 <= 0 {
            bail!("prompt length must be > 0 at batch index {i}: {len_i64}");
        }
        let len = usize::try_from(len_i64)
            .map_err(|_| anyhow::anyhow!("prompt length overflows usize at {i}: {len_i64}"))?;
        if len > seq_len {
            bail!("prompt length exceeds seq_len at batch index {i}: len={len} seq_len={seq_len}");
        }
        let flat = i
            .checked_mul(seq_len)
            .and_then(|base| base.checked_add(len - 1))
            .ok_or_else(|| anyhow::anyhow!("gather index overflow for batch index {i}"))?;
        gather_idx.push(u32::try_from(flat).map_err(|_| {
            anyhow::anyhow!("gather index overflows u32 at batch index {i}: {flat}")
        })?);
    }
    let flat_logits = logits.reshape((batch * seq_len, vocab))?;
    let idx = Tensor::from_vec(gather_idx, (batch,), logits.device())?;
    Ok(flat_logits.index_select(&idx, 0)?)
}

pub(crate) fn position_ids_for_step(
    prompt_lens: &[i64],
    step: usize,
    device: &Device,
) -> Result<Tensor> {
    if prompt_lens.is_empty() {
        bail!("prompt_lens is empty");
    }
    let batch = prompt_lens.len();
    let step_i64 =
        i64::try_from(step).map_err(|_| anyhow::anyhow!("step overflows i64: {step}"))?;

    let mut pos: Vec<i64> = Vec::with_capacity(batch);
    for (i, &base) in prompt_lens.iter().enumerate() {
        let p = base.checked_add(step_i64).ok_or_else(|| {
            anyhow::anyhow!("position id overflow at batch index {i}: base={base} step={step_i64}")
        })?;
        pos.push(p);
    }

    let pos_t = Tensor::from_vec(pos, (batch, 1usize), device)?;
    Ok(pos_t.unsqueeze(0)?.broadcast_as((3usize, batch, 1usize))?)
}

fn position_ids_for_decode_steps(
    prompt_len: i64,
    max_steps: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    (0..max_steps)
        .map(|step| position_ids_for_step(&[prompt_len], step, device))
        .collect()
}

pub(crate) fn position_ids_for_decode_steps_batch(
    prompt_lens: &[i64],
    max_steps: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    (0..max_steps)
        .map(|step| position_ids_for_step(prompt_lens, step, device))
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "paged-attn")]
    use super::normalize_batch_for_paged_prefill;
    use super::{
        attention_mask_is_dense, attention_masks_are_dense, decode_slot_and_context_len,
        position_ids_for_step, prompt_lens_from_attention_masks,
    };
    #[cfg(feature = "paged-attn")]
    use super::{attention_mask_is_right_padded, attention_masks_are_right_padded};

    #[test]
    fn test_position_ids_for_step_matches_get_rope_index_last_col() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let masks: Vec<Vec<u32>> = vec![
            vec![0, 0, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1],
        ];

        let batch = masks.len();
        let seq_len = masks
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing masks"))?
            .len();
        let rows: Vec<&[u32]> = masks.iter().map(|r| r.as_slice()).collect();
        let prompt_lens = prompt_lens_from_attention_masks(rows.as_slice(), seq_len)?;

        for step in 0..4usize {
            let mut flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
            for row in &masks {
                flat.extend_from_slice(row.as_slice());
            }
            let base = candle_core::Tensor::from_vec(flat, (batch, seq_len), &device)?;

            let extra = step.saturating_add(1);
            let ones = candle_core::Tensor::ones((batch, extra), candle_core::DType::U32, &device)?;
            let mask_total = candle_core::Tensor::cat(&[&base, &ones], 1)?;
            let total_len = seq_len.saturating_add(extra);

            let (expected, _deltas) = crate::model::thinker::get_rope_index(&mask_total)?;
            let expected_last = expected.narrow(2, total_len.saturating_sub(1), 1)?;

            let got = position_ids_for_step(prompt_lens.as_slice(), step, &device)?;

            let exp = expected_last.to_vec3::<i64>()?;
            let got = got.to_vec3::<i64>()?;
            if exp != got {
                anyhow::bail!("position_ids mismatch at step {step}: expected={exp:?} got={got:?}");
            }
        }

        Ok(())
    }

    #[test]
    fn test_attention_mask_dense_detection() {
        assert!(attention_mask_is_dense(&[1, 1, 2]));
        assert!(!attention_mask_is_dense(&[0, 1, 1]));

        let dense_rows: Vec<&[u32]> = vec![&[1, 1], &[2, 1]];
        assert!(attention_masks_are_dense(dense_rows.as_slice()));

        let padded_rows: Vec<&[u32]> = vec![&[1, 1], &[0, 1]];
        assert!(!attention_masks_are_dense(padded_rows.as_slice()));
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn test_attention_mask_right_padding_detection() {
        assert!(attention_mask_is_right_padded(&[1, 1, 1, 0, 0]));
        assert!(attention_mask_is_right_padded(&[1, 1, 1]));
        assert!(attention_mask_is_right_padded(&[0, 0, 0]));
        assert!(!attention_mask_is_right_padded(&[0, 1, 1]));
        assert!(!attention_mask_is_right_padded(&[1, 0, 1]));

        let right_rows: Vec<&[u32]> = vec![&[1, 1, 0], &[1, 1, 1]];
        assert!(attention_masks_are_right_padded(right_rows.as_slice()));

        let mixed_rows: Vec<&[u32]> = vec![&[1, 1, 0], &[0, 1, 1]];
        assert!(!attention_masks_are_right_padded(mixed_rows.as_slice()));
    }

    #[test]
    fn test_decode_slot_and_context_len_matches_paged_decode_steps() -> anyhow::Result<()> {
        assert_eq!(decode_slot_and_context_len(42, 0)?, (42, 43));
        assert_eq!(decode_slot_and_context_len(42, 7)?, (49, 50));
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn test_normalize_batch_for_paged_prefill_handles_left_and_right_padding() -> anyhow::Result<()>
    {
        let input_ids: Vec<&[u32]> = vec![
            &[9, 9, 11, 12, 13],
            &[21, 22, 23, 0, 0],
            &[31, 32, 33, 34, 35],
        ];
        let attention_mask: Vec<&[u32]> =
            vec![&[0, 0, 1, 1, 1], &[1, 1, 1, 0, 0], &[1, 1, 1, 1, 1]];

        let (ids, masks, lens) =
            normalize_batch_for_paged_prefill(input_ids.as_slice(), attention_mask.as_slice())?;
        assert_eq!(lens, vec![3, 3, 5]);
        assert_eq!(ids[0], vec![11, 12, 13, 9, 9]);
        assert_eq!(ids[1], vec![21, 22, 23, 0, 0]);
        assert_eq!(ids[2], vec![31, 32, 33, 34, 35]);
        assert_eq!(masks[0], vec![1, 1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 0, 0]);
        assert_eq!(masks[2], vec![1, 1, 1, 1, 1]);
        Ok(())
    }
}
