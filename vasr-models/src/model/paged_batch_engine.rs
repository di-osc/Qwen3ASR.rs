//! Paged-attention batch engine: prefill + single decode steps for continuous batching.
//!
//! `greedy_generate_paged_batch` delegates here. Schedulers call `paged_batch_prefill` and
//! `paged_batch_decode_step` directly to drive batch membership across steps.

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

use crate::model::attention;
use crate::model::generation::{
    argmax_token_ids_from_logits, attention_masks_are_right_padded, count_audio_placeholders_batch,
    gather_last_logits_for_prompt_lens, normalize_batch_for_paged_prefill, position_ids_for_step,
};
use crate::model::paged_cache_runtime::PagedCacheRuntime;
use crate::model::paged_kv_cache::PagedInputMetadata;
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::model::thinker::get_rope_index;
use vasr_quant::isq_linear::set_linear_is_prefill;

#[derive(Debug, Clone)]
pub struct PagedBatchConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    /// First `request_id` assigned to row 0; row `i` uses `request_id_base + i`.
    pub request_id_base: usize,
}

#[derive(Debug)]
pub struct PagedBatchState {
    pub batch: usize,
    pub seq_len: usize,
    pub prompt_lens_usize: Vec<usize>,
    pub prompt_lens_i64: Vec<i64>,
    pub request_ids: Vec<usize>,
    pub block_tables: Vec<Vec<usize>>,
    pub next_ids: Vec<u32>,
    pub generated: Vec<Vec<u32>>,
    pub finished: Vec<bool>,
    pub decode_step: usize,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
}

impl PagedBatchState {
    pub fn all_finished(&self) -> bool {
        self.finished.iter().all(|&done| done)
    }

    pub fn active_indices(&self) -> Vec<usize> {
        (0..self.batch).filter(|&i| !self.finished[i]).collect()
    }
}

/// Per-sequence decode state for continuous batching (each slot tracks its own step).
#[derive(Debug, Clone)]
pub struct PagedDecodeSlot {
    pub request_id: usize,
    pub prompt_len_usize: usize,
    pub prompt_len_i64: i64,
    pub block_table: Vec<usize>,
    pub next_id: u32,
    pub generated: Vec<u32>,
    pub finished: bool,
    pub decode_step: usize,
}

impl PagedDecodeSlot {
    pub fn from_prefill_state(state: &PagedBatchState, row: usize) -> Result<Self> {
        Ok(Self {
            request_id: *state
                .request_ids
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing request_id for row {row}"))?,
            prompt_len_usize: *state
                .prompt_lens_usize
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing prompt_len for row {row}"))?,
            prompt_len_i64: *state
                .prompt_lens_i64
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing prompt_len_i64 for row {row}"))?,
            block_table: state
                .block_tables
                .get(row)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing block table for row {row}"))?,
            next_id: *state
                .next_ids
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing next_id for row {row}"))?,
            generated: Vec::new(),
            finished: *state.finished.get(row).unwrap_or(&false),
            decode_step: 0,
        })
    }
}

pub fn paged_batch_free_slots(runtime: &mut PagedCacheRuntime, request_ids: &[usize]) {
    runtime.manager_mut().free_many(request_ids);
}

pub fn paged_batch_prefill_row(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    request_id: usize,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<PagedDecodeSlot> {
    let config = PagedBatchConfig {
        max_new_tokens,
        eos_token_ids: eos_token_ids.to_vec(),
        request_id_base: request_id,
    };
    let state = paged_batch_prefill(
        thinker,
        device,
        runtime,
        &[input_ids],
        &[attention_mask],
        audio_features,
        &config,
    )?;
    PagedDecodeSlot::from_prefill_state(&state, 0)
}

/// Greedy decode step for `slots` at a shared `step` (all slots must share the same `decode_step`).
pub fn paged_batch_decode_slots_at_step(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    slots: &mut [PagedDecodeSlot],
    step: usize,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<()> {
    if slots.is_empty() {
        return Ok(());
    }
    if step >= max_new_tokens {
        for slot in slots.iter_mut() {
            slot.finished = true;
        }
        return Ok(());
    }

    let eos_fill_id = *eos_token_ids
        .first()
        .ok_or_else(|| anyhow::anyhow!("eos_token_ids is empty"))?;

    let mut forward_indices: Vec<usize> = (0..slots.len())
        .filter(|&i| !slots[i].finished && slots[i].decode_step == step)
        .collect();
    tracing::info!(
        "paged_batch_decode_slots_at_step: step={step} slots={} forward={}",
        slots.len(),
        forward_indices.len(),
    );
    if forward_indices.is_empty() {
        return Ok(());
    }

    let mut decodable_indices = Vec::with_capacity(forward_indices.len());
    for &i in &forward_indices {
        let slot = &slots[i];
        let num_tokens = slot
            .prompt_len_usize
            .checked_add(slot.generated.len())
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("paged slot token count overflow"))?;
        let alloc_ok = runtime
            .manager_mut()
            .try_allocate_slots(slot.request_id, num_tokens)?;
        tracing::info!(
            "  slot[{i}] req={} prompt_len={} generated={} num_tokens={num_tokens} alloc={alloc_ok}",
            slot.request_id,
            slot.prompt_len_usize,
            slot.generated.len(),
        );
        if alloc_ok {
            decodable_indices.push(i);
        }
    }

    if decodable_indices.is_empty() {
        return Ok(());
    }

    let mut active_indices = Vec::with_capacity(decodable_indices.len());
    for &i in &decodable_indices {
        let tok = slots[i].next_id;
        if eos_token_ids.contains(&tok) {
            slots[i].finished = true;
            runtime.manager_mut().trim_request_to_num_tokens(
                slots[i].request_id,
                slots[i].prompt_len_usize + slots[i].generated.len(),
            );
            continue;
        }
        slots[i].generated.push(tok);
        active_indices.push(i);
    }

    if active_indices.is_empty() {
        for i in 0..slots.len() {
            if !slots[i].finished && slots[i].decode_step == step {
                slots[i].decode_step = slots[i].decode_step.saturating_add(1);
            }
        }
        return Ok(());
    }

    let tokens_in: Vec<u32> = active_indices
        .iter()
        .map(|&i| {
            slots[i]
                .generated
                .last()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing generated token for slot {i}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let active_batch = active_indices.len();
    let input_ids_new = Tensor::from_vec(tokens_in, (active_batch, 1usize), device)?;

    let prompt_lens_active: Vec<i64> = active_indices
        .iter()
        .map(|&i| slots[i].prompt_len_i64)
        .collect();
    let position_ids_new = position_ids_for_step(prompt_lens_active.as_slice(), step, device)?;

    let request_ids: Vec<usize> = active_indices
        .iter()
        .map(|&i| slots[i].request_id)
        .collect();
    let block_tables_active = runtime.block_tables_for(&request_ids)?;
    let prompt_lens_active_usize: Vec<usize> = active_indices
        .iter()
        .map(|&i| slots[i].prompt_len_usize)
        .collect();
    let input_metadata = runtime.cache().decode_metadata_for_batch_step(
        &block_tables_active,
        prompt_lens_active_usize.as_slice(),
        step,
        device,
    )?;

    let logits_new = paged_batch_decode_forward(
        thinker,
        runtime,
        device,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = argmax_token_ids_from_logits(&logits_new.squeeze(1)?, active_batch)?;
    for (slot_idx, &i) in active_indices.iter().enumerate() {
        slots[i].next_id = *next_active
            .get(slot_idx)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for slot index {slot_idx}"))?;
        slots[i].decode_step = slots[i].decode_step.saturating_add(1);
    }

    // Silence unused when eos_fill would be used in non-compact paths.
    let _ = eos_fill_id;
    Ok(())
}

pub fn paged_batch_run(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    config: &PagedBatchConfig,
) -> Result<Vec<Vec<u32>>> {
    let batch = input_ids.len();
    if batch == 0 {
        return Ok(vec![]);
    }
    if config.max_new_tokens == 0 {
        return Ok(vec![vec![]; batch]);
    }

    // Defensively free any stale request IDs left over from a previous failed run
    // that shares the same `request_id_base` (commonly 0 for static batches).
    {
        let request_ids: Vec<usize> = (0..batch)
            .map(|i| config.request_id_base.saturating_add(i))
            .collect();
        runtime.manager_mut().free_many(&request_ids);
    }

    let mut state = paged_batch_prefill(
        thinker,
        device,
        runtime,
        input_ids,
        attention_mask,
        audio_features,
        config,
    )?;

    while state.decode_step < state.max_new_tokens && !state.all_finished() {
        let continued = paged_batch_decode_step(thinker, device, runtime, &mut state)?;
        if !continued {
            break;
        }
    }

    runtime.manager_mut().free_many(&state.request_ids);
    Ok(state.generated)
}

pub fn paged_batch_prefill(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    config: &PagedBatchConfig,
) -> Result<PagedBatchState> {
    let batch = input_ids.len();
    if batch == 0 {
        bail!("paged batch prefill requires at least one row");
    }
    if config.eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }

    let seq_len = input_ids
        .first()
        .map(|row| row.len())
        .ok_or_else(|| anyhow::anyhow!("missing first input row"))?;
    if seq_len == 0 {
        bail!("input_ids rows are empty");
    }
    for (i, row) in input_ids.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "paged batch input_ids[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }
    for (i, row) in attention_mask.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "paged batch attention_mask[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }

    let normalized_batch = if attention_masks_are_right_padded(attention_mask) {
        None
    } else {
        Some(normalize_batch_for_paged_prefill(
            input_ids,
            attention_mask,
        )?)
    };
    let (paged_id_rows, paged_mask_rows, prompt_lens_usize): (
        Vec<&[u32]>,
        Vec<&[u32]>,
        Vec<usize>,
    ) = if let Some((ids, masks, lens)) = normalized_batch.as_ref() {
        (
            ids.iter().map(Vec::as_slice).collect(),
            masks.iter().map(Vec::as_slice).collect(),
            lens.clone(),
        )
    } else {
        (
            input_ids.to_vec(),
            attention_mask.to_vec(),
            attention_mask
                .iter()
                .map(|row| row.iter().filter(|&&v| v != 0).count())
                .collect(),
        )
    };

    let mut ids_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    let mut attn_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    for i in 0..batch {
        ids_flat.extend_from_slice(paged_id_rows[i]);
        attn_flat.extend_from_slice(paged_mask_rows[i]);
    }
    let input_ids_t = Tensor::from_vec(ids_flat, (batch, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attn_flat, (batch, seq_len), device)?;

    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders_batch(paged_id_rows.as_slice(), thinker.audio_token_id())?
    } else {
        0
    };
    let inputs_embeds = thinker.inputs_embeds_with_audio_features(
        &input_ids_t,
        audio_features,
        audio_placeholder_count,
    )?;

    let request_ids: Vec<usize> = (0..batch)
        .map(|i| config.request_id_base.saturating_add(i))
        .collect();
    for (i, &request_id) in request_ids.iter().enumerate() {
        runtime
            .manager_mut()
            .allocate_slots(request_id, prompt_lens_usize[i])?;
    }
    let block_tables = runtime.block_tables_for(&request_ids)?;

    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let mut input_metadata = runtime.cache().input_metadata_from_attention_masks(
        &block_tables,
        paged_mask_rows.as_slice(),
        device,
    )?;
    if !input_metadata.prefill_causal_only {
        input_metadata.prefill_attention_mask = Some(attention::make_causal_mask(
            input_metadata.token_attention_mask.as_ref(),
            batch,
            seq_len,
            inputs_embeds.dtype(),
            device,
        )?);
    }

    let logits = {
        let _linear_prefill_guard = set_linear_is_prefill(true);
        thinker.forward_embeds_with_paged_cache(
            &position_ids,
            &inputs_embeds,
            runtime.cache(),
            &input_metadata,
        )?
    };

    let prompt_lens_i64 = input_metadata
        .query_lens
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("paged prefill metadata missing query_lens"))?
        .iter()
        .map(|&len| {
            i64::try_from(len).map_err(|_| anyhow::anyhow!("query_len overflows i64: {len}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let last_logits = gather_last_logits_for_prompt_lens(&logits, prompt_lens_i64.as_slice())?;
    let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
    let finished: Vec<bool> = next_ids
        .iter()
        .map(|id| config.eos_token_ids.contains(id))
        .collect();
    for i in 0..batch {
        if finished[i] {
            runtime
                .manager_mut()
                .trim_request_to_num_tokens(request_ids[i], prompt_lens_usize[i]);
        }
    }

    Ok(PagedBatchState {
        batch,
        seq_len,
        prompt_lens_usize,
        prompt_lens_i64,
        request_ids,
        block_tables,
        next_ids,
        generated: vec![Vec::new(); batch],
        finished,
        decode_step: 0,
        max_new_tokens: config.max_new_tokens,
        eos_token_ids: config.eos_token_ids.clone(),
    })
}

/// Run one decode step. Returns `Ok(false)` when all sequences are finished.
///
/// Finished sequences are compacted out of the forward batch so only active
/// sequences consume compute.
pub fn paged_batch_decode_step(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    state: &mut PagedBatchState,
) -> Result<bool> {
    if state.all_finished() {
        return Ok(false);
    }
    if state.decode_step >= state.max_new_tokens {
        return Ok(false);
    }

    let step = state.decode_step;
    let eos_token_ids = state.eos_token_ids.as_slice();

    let mut forward_indices: Vec<usize> = state.active_indices();
    if forward_indices.is_empty() {
        state.decode_step = state.decode_step.saturating_add(1);
        return Ok(!state.all_finished());
    }

    for &i in &forward_indices {
        // `step + 1` covers decode position (prompt_len + step) + 1,
        // ensuring the block table always has enough entries.
        let num_tokens = state.prompt_lens_usize[i]
            .checked_add(step)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("paged batch token count overflow"))?;
        runtime
            .manager_mut()
            .allocate_slots(state.request_ids[i], num_tokens)?;
    }
    state.block_tables = runtime.block_tables_for(&state.request_ids)?;

    let mut tokens_in = Vec::with_capacity(forward_indices.len());
    for &i in &forward_indices {
        let tok = state
            .next_ids
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("next token id missing for batch row {i}"))?;
        if eos_token_ids.contains(&tok) {
            state.finished[i] = true;
            runtime.manager_mut().trim_request_to_num_tokens(
                state.request_ids[i],
                state.prompt_lens_usize[i] + state.generated[i].len(),
            );
            continue;
        }
        state.generated[i].push(tok);
        tokens_in.push(tok);
    }

    forward_indices.retain(|&i| !state.finished[i]);
    if forward_indices.is_empty() {
        state.decode_step = state.decode_step.saturating_add(1);
        return Ok(!state.all_finished());
    }
    tokens_in = forward_indices
        .iter()
        .map(|&i| {
            state
                .generated
                .get(i)
                .and_then(|g| g.last())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing generated token for row {i}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let active_batch = forward_indices.len();
    let input_ids_new = Tensor::from_vec(tokens_in, (active_batch, 1usize), device)?;

    let prompt_lens_active: Vec<i64> = forward_indices
        .iter()
        .map(|&i| state.prompt_lens_i64[i])
        .collect();
    let position_ids_new = position_ids_for_step(prompt_lens_active.as_slice(), step, device)?;

    let block_tables_active: Vec<Vec<usize>> = forward_indices
        .iter()
        .map(|&i| {
            state
                .block_tables
                .get(i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing block table for row {i}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let prompt_lens_active_usize: Vec<usize> = forward_indices
        .iter()
        .map(|&i| state.prompt_lens_usize[i])
        .collect();
    let input_metadata = runtime.cache().decode_metadata_for_batch_step(
        &block_tables_active,
        prompt_lens_active_usize.as_slice(),
        step,
        device,
    )?;

    let logits_new = paged_batch_decode_forward(
        thinker,
        runtime,
        device,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = argmax_token_ids_from_logits(&logits_new.squeeze(1)?, active_batch)?;
    for (slot, &row) in forward_indices.iter().enumerate() {
        state.next_ids[row] = *next_active
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for active slot {slot}"))?;
    }

    state.decode_step = state.decode_step.saturating_add(1);
    Ok(!state.all_finished())
}

fn paged_batch_decode_forward(
    thinker: &ThinkerForConditionalGeneration,
    runtime: &mut PagedCacheRuntime,
    _device: &Device,
    input_ids_new: &Tensor,
    position_ids_new: &Tensor,
    input_metadata: &PagedInputMetadata,
) -> Result<Tensor> {
    #[cfg(feature = "cuda-graph")]
    {
        if runtime.cuda_decode_graph_enabled_for(input_ids_new, input_metadata)? {
            match runtime.cuda_decode_graph(
                thinker,
                input_ids_new,
                position_ids_new,
                input_metadata,
            ) {
                Ok(logits) => return Ok(logits),
                Err(err) => {
                    tracing::warn!("CUDA decode graph key disabled after decode error: {err}");
                    if let Err(disable_err) =
                        runtime.disable_cuda_decode_graph_for(input_ids_new, input_metadata)
                    {
                        tracing::warn!(
                            "failed to disable CUDA decode graph key after decode error: {disable_err}"
                        );
                    }
                }
            }
        }
    }

    let inputs_embeds_new = thinker.embed_tokens(input_ids_new)?;
    let _linear_decode_guard = set_linear_is_prefill(false);
    thinker.forward_embeds_with_paged_cache(
        position_ids_new,
        &inputs_embeds_new,
        runtime.cache(),
        input_metadata,
    )
}

#[cfg(test)]
mod tests {
    use super::PagedBatchState;

    #[test]
    fn active_indices_skips_finished_rows() {
        let state = PagedBatchState {
            batch: 4,
            seq_len: 8,
            prompt_lens_usize: vec![8; 4],
            prompt_lens_i64: vec![8; 4],
            request_ids: vec![0, 1, 2, 3],
            block_tables: vec![vec![1]; 4],
            next_ids: vec![1, 2, 3, 4],
            generated: vec![vec![]; 4],
            finished: vec![false, true, false, true],
            decode_step: 0,
            max_new_tokens: 16,
            eos_token_ids: vec![0],
        };
        assert_eq!(state.active_indices(), vec![0, 2]);
    }
}
