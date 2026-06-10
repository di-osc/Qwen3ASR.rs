//! ASR batch scheduler: waiting/running queues and continuous paged-attention batching.
//!
//! GPU task serialization remains in `vasr_server::InferenceScheduler`.
//!
//! `waiting` holds segment indices (typically one VAD slice each). Upstream code prepares
//! segments with [`AsrProcessor::prepare_batch_varlen`] so admit waves can pad locally and
//! prefill with per-row `query_lens` / flash varlen metadata.

use std::collections::{HashMap, VecDeque};

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

use crate::model::paged_batch_engine::{
    PagedBatchConfig, PagedDecodeSlot, paged_batch_decode_slots_at_step, paged_batch_free_slots,
    paged_batch_prefill, paged_batch_run,
};
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::processor::asr_processor::PreparedInputs;
use vasr_paged_attn::{PagedBlockManager, PagedCacheRuntime, SharedPagedCacheRuntime};

#[derive(Debug, Clone)]
pub struct AsrBatchSchedulerConfig {
    pub max_num_seqs: usize,
    pub continuous_batch: bool,
}

impl Default for AsrBatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 8,
            continuous_batch: continuous_batch_enabled(),
        }
    }
}

impl AsrBatchSchedulerConfig {
    pub fn from_max_batch_size(max_batch_size: usize) -> Self {
        let mut config = Self::default();
        if max_batch_size > 0 {
            config.max_num_seqs = max_batch_size;
        }
        config
    }
}

fn continuous_batch_enabled() -> bool {
    std::env::var_os("VASR_DISABLE_CONTINUOUS_BATCH").is_none()
}

pub(crate) fn continuous_paged_batch_enabled() -> bool {
    crate::inference::utils::continuous_paged_batch_enabled()
}

#[derive(Debug)]
struct RunningSlot {
    client_index: usize,
    slot: PagedDecodeSlot,
}

#[derive(Debug)]
pub struct AsrBatchScheduler {
    config: AsrBatchSchedulerConfig,
    next_request_id: usize,
}

fn free_all_running(runtime: &mut PagedCacheRuntime, running: &mut [Option<RunningSlot>]) {
    for slot in running.iter_mut() {
        if let Some(entry) = slot.take() {
            paged_batch_free_slots(runtime, &[entry.slot.request_id]);
        }
    }
}

fn prepared_prompt_len(row: &PreparedInputs) -> usize {
    row.attention_mask.iter().filter(|&&v| v != 0).count()
}

fn running_kv_blocks(running: &[Option<RunningSlot>], manager: &PagedBlockManager) -> usize {
    running
        .iter()
        .filter_map(|entry| entry.as_ref())
        .map(|entry| manager.blocks_for_request(entry.slot.request_id))
        .sum()
}

/// Whether admitting one more sequence keeps headroom for running decode growth.
fn can_admit_with_projected_kv(
    manager: &PagedBlockManager,
    kv_block_capacity: usize,
    prompt_len: usize,
    projected_used_blocks: usize,
    projected_running: usize,
) -> bool {
    let prompt_blocks = prompt_len.div_ceil(manager.block_size());
    let decode_headroom = projected_running.max(1);
    projected_used_blocks
        .saturating_add(prompt_blocks)
        .saturating_add(decode_headroom)
        <= kv_block_capacity
}

fn free_finished_running(
    running: &mut [Option<RunningSlot>],
    outputs: &mut [Vec<u32>],
    runtime: &mut PagedCacheRuntime,
) -> bool {
    let mut freed = false;
    for slot in running.iter_mut() {
        let Some(entry) = slot.as_mut() else {
            continue;
        };
        if !entry.slot.finished {
            continue;
        }
        outputs[entry.client_index] = entry.slot.generated.clone();
        paged_batch_free_slots(runtime, &[entry.slot.request_id]);
        *slot = None;
        freed = true;
    }
    freed
}

/// Right-pad selected rows to the max effective prompt length within the admit wave.
fn pad_prepared_wave_for_prefill(
    prepared: &[PreparedInputs],
    wave_indices: &[usize],
    pad_id: u32,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    if wave_indices.is_empty() {
        bail!("pad_prepared_wave_for_prefill requires at least one row");
    }
    let max_prompt_len = wave_indices
        .iter()
        .map(|&idx| prepared_prompt_len(&prepared[idx]))
        .max()
        .unwrap_or(0);
    if max_prompt_len == 0 {
        bail!("admit wave contains zero-length prompts");
    }

    let mut ids_rows = Vec::with_capacity(wave_indices.len());
    let mut attn_rows = Vec::with_capacity(wave_indices.len());
    for &idx in wave_indices {
        let row = prepared
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing prepared row {idx}"))?;
        let prompt_len = prepared_prompt_len(row);
        if prompt_len > row.input_ids.len() || prompt_len > row.attention_mask.len() {
            bail!(
                "prepared[{idx}] effective prompt len {prompt_len} exceeds row capacity: ids={} mask={}",
                row.input_ids.len(),
                row.attention_mask.len()
            );
        }
        let mut ids = row.input_ids[..prompt_len].to_vec();
        let mut attn = row.attention_mask[..prompt_len].to_vec();
        if prompt_len < max_prompt_len {
            ids.extend(std::iter::repeat_n(pad_id, max_prompt_len - prompt_len));
            attn.extend(std::iter::repeat_n(0u32, max_prompt_len - prompt_len));
        }
        ids_rows.push(ids);
        attn_rows.push(attn);
    }
    Ok((ids_rows, attn_rows))
}

fn concat_prepared_audio(
    audio_features: &Tensor,
    audio_offsets: &[usize],
    row_audio_lens: &[usize],
    client_indices: &[usize],
) -> Result<Tensor> {
    if client_indices.is_empty() {
        bail!("concat_prepared_audio requires at least one row");
    }
    if client_indices.len() == 1 {
        let client_index = client_indices[0];
        return Ok(audio_features.narrow(
            0,
            audio_offsets[client_index],
            row_audio_lens[client_index],
        )?);
    }
    let parts = client_indices
        .iter()
        .map(|&client_index| {
            audio_features.narrow(0, audio_offsets[client_index], row_audio_lens[client_index])
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::cat(parts.as_slice(), 0)?)
}

impl AsrBatchScheduler {
    pub fn new(config: AsrBatchSchedulerConfig) -> Self {
        Self {
            config,
            next_request_id: 0,
        }
    }

    pub fn config(&self) -> &AsrBatchSchedulerConfig {
        &self.config
    }

    fn allocate_request_id(&mut self) -> usize {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);
        id
    }

    /// Run one homogeneous micro-batch (same padded `seq_len` across rows).
    pub fn run_micro_batch(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        input_ids: &[&[u32]],
        attention_mask: &[&[u32]],
        audio_features: Option<&Tensor>,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let batch = input_ids.len();
        if batch > self.config.max_num_seqs {
            bail!(
                "micro-batch size {batch} exceeds scheduler max_num_seqs {}",
                self.config.max_num_seqs
            );
        }

        let request_id_base = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(batch.max(1));

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let engine_config = PagedBatchConfig {
            max_new_tokens,
            eos_token_ids: eos_token_ids.to_vec(),
            request_id_base,
        };
        paged_batch_run(
            thinker,
            device,
            &mut runtime_guard,
            input_ids,
            attention_mask,
            audio_features,
            &engine_config,
        )
    }

    /// Continuous batching for variable-length prepared segments (VAD slices).
    ///
    /// `waiting` starts as all segment indices; each admit wave is right-padded locally and
    /// prefilled with per-row `query_lens` for flash varlen metadata.
    pub fn run_continuous_prepared(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        prepared: &[PreparedInputs],
        audio_features: &Tensor,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
        pad_id: u32,
    ) -> Result<Vec<Vec<u32>>> {
        let batch = prepared.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        if max_new_tokens == 0 {
            return Ok(vec![vec![]; batch]);
        }

        let max_slots = self.config.max_num_seqs.max(1).min(batch);
        let mut waiting: VecDeque<usize> = (0..batch).collect();
        let mut running: Vec<Option<RunningSlot>> = (0..max_slots).map(|_| None).collect();
        let mut outputs: Vec<Vec<u32>> = vec![vec![]; batch];

        let audio_token_id = thinker.audio_token_id();
        let row_audio_lens: Vec<usize> = prepared
            .iter()
            .map(|row| {
                row.input_ids
                    .iter()
                    .filter(|&&id| id == audio_token_id)
                    .count()
            })
            .collect();
        let mut audio_offsets = Vec::with_capacity(batch);
        let mut audio_cursor = 0usize;
        for &len in &row_audio_lens {
            audio_offsets.push(audio_cursor);
            audio_cursor = audio_cursor.saturating_add(len);
        }
        let total_audio_tokens = audio_features.dims2().map(|(n, _)| n).unwrap_or(0);
        if audio_cursor != total_audio_tokens {
            bail!(
                "continuous batch audio token mismatch: prepared placeholders={audio_cursor} audio_features={total_audio_tokens}"
            );
        }

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let kv_block_capacity =
            PagedBlockManager::allocatable_block_capacity(runtime_guard.stats().num_blocks);

        let result = (|| -> Result<Vec<Vec<u32>>> {
            loop {
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry {
                        if !entry.slot.finished {
                            active_steps.push((slot_idx, entry.slot.decode_step));
                        }
                    }
                }

                if !active_steps.is_empty() {
                    let step_groups = group_running_by_decode_step(active_steps.as_slice());
                    for (step, slot_indices) in step_groups {
                        let mut batch_slots: Vec<PagedDecodeSlot> = slot_indices
                            .iter()
                            .map(|&slot_idx| {
                                running[slot_idx]
                                    .as_ref()
                                    .map(|entry| entry.slot.clone())
                                    .ok_or_else(|| {
                                        anyhow::anyhow!("missing running slot {slot_idx}")
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?;

                        paged_batch_decode_slots_at_step(
                            thinker,
                            device,
                            &mut runtime_guard,
                            batch_slots.as_mut_slice(),
                            step,
                            max_new_tokens,
                            eos_token_ids,
                        )?;

                        for (&slot_idx, slot) in slot_indices.iter().zip(batch_slots.iter()) {
                            if let Some(entry) = running[slot_idx].as_mut() {
                                entry.slot = slot.clone();
                            }
                        }
                    }
                }

                for entry in running.iter_mut() {
                    let Some(entry) = entry.as_mut() else {
                        continue;
                    };
                    if !entry.slot.finished && entry.slot.decode_step >= max_new_tokens {
                        entry.slot.finished = true;
                    }
                }

                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );

                let mut admitted = false;
                let manager = runtime_guard.manager();
                let mut wave_client_indices: Vec<usize> = Vec::new();
                let mut wave_slot_indices: Vec<usize> = Vec::new();
                let mut projected_used = running_kv_blocks(running.as_slice(), manager);
                let mut projected_running = running.iter().filter(|entry| entry.is_some()).count();
                for slot_idx in 0..max_slots {
                    if running[slot_idx].is_some() {
                        continue;
                    }
                    if projected_running >= max_slots {
                        break;
                    }
                    let Some(client_index) = waiting.pop_front() else {
                        break;
                    };
                    let prompt_len = prepared_prompt_len(&prepared[client_index]);
                    let next_running = projected_running.saturating_add(1);
                    if !can_admit_with_projected_kv(
                        manager,
                        kv_block_capacity,
                        prompt_len,
                        projected_used,
                        next_running,
                    ) {
                        waiting.push_front(client_index);
                        break;
                    }
                    projected_used =
                        projected_used.saturating_add(prompt_len.div_ceil(manager.block_size()));
                    projected_running = next_running;
                    wave_client_indices.push(client_index);
                    wave_slot_indices.push(slot_idx);
                }

                if !wave_client_indices.is_empty() {
                    let wave = wave_client_indices.len();
                    let request_id_base = self.next_request_id;
                    self.next_request_id = self.next_request_id.saturating_add(wave);
                    let (wave_ids, wave_masks) = pad_prepared_wave_for_prefill(
                        prepared,
                        wave_client_indices.as_slice(),
                        pad_id,
                    )?;
                    let ids_rows: Vec<&[u32]> = wave_ids.iter().map(Vec::as_slice).collect();
                    let attn_rows: Vec<&[u32]> = wave_masks.iter().map(Vec::as_slice).collect();
                    let wave_audio = concat_prepared_audio(
                        audio_features,
                        audio_offsets.as_slice(),
                        row_audio_lens.as_slice(),
                        wave_client_indices.as_slice(),
                    )?;
                    let engine_config = PagedBatchConfig {
                        max_new_tokens,
                        eos_token_ids: eos_token_ids.to_vec(),
                        request_id_base,
                    };
                    let state = paged_batch_prefill(
                        thinker,
                        device,
                        &mut runtime_guard,
                        ids_rows.as_slice(),
                        attn_rows.as_slice(),
                        Some(&wave_audio),
                        &engine_config,
                    )?;
                    for (wave_row, (&client_index, &slot_idx)) in wave_client_indices
                        .iter()
                        .zip(wave_slot_indices.iter())
                        .enumerate()
                    {
                        running[slot_idx] = Some(RunningSlot {
                            client_index,
                            slot: PagedDecodeSlot::from_prefill_state(&state, wave_row)?,
                        });
                    }
                    let running_count = running.iter().filter(|s| s.is_some()).count();
                    let waiting_count = waiting.len();
                    tracing::info!(
                        target: "vasr_transcribe::pipeline",
                        "continuous | admit wave={wave} running={running_count}/{max_slots} waiting={waiting_count}",
                    );
                    admitted = true;
                }

                let has_running = running.iter().any(|entry| entry.is_some());
                if waiting.is_empty() && !has_running {
                    break;
                }
                if !admitted && active_steps.is_empty() && !has_running && !waiting.is_empty() {
                    bail!(
                        "paged KV cache exhausted: {} sequence(s) waiting, free_blocks={}",
                        waiting.len(),
                        runtime_guard.manager().free_blocks()
                    );
                }
            }

            Ok(outputs)
        })();

        free_all_running(&mut runtime_guard, running.as_mut_slice());
        result
    }
}

pub(crate) fn group_running_by_decode_step(active: &[(usize, usize)]) -> Vec<(usize, Vec<usize>)> {
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(slot_idx, step) in active {
        groups.entry(step).or_default().push(slot_idx);
    }
    let mut out: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    out.sort_by_key(|(step, _)| *step);
    out
}

/// Convenience entry when a scheduler instance is not needed.
pub fn run_paged_micro_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    max_batch_size: usize,
) -> Result<Vec<Vec<u32>>> {
    let config = AsrBatchSchedulerConfig::from_max_batch_size(max_batch_size);
    let mut scheduler = AsrBatchScheduler::new(config);
    scheduler.run_micro_batch(
        thinker,
        device,
        runtime,
        input_ids,
        attention_mask,
        audio_features,
        max_new_tokens,
        eos_token_ids,
    )
}

pub fn run_paged_prepared_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    prepared: &[PreparedInputs],
    audio_features: &Tensor,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    max_batch_size: usize,
    pad_id: u32,
) -> Result<Vec<Vec<u32>>> {
    let config = AsrBatchSchedulerConfig::from_max_batch_size(max_batch_size);
    let mut scheduler = AsrBatchScheduler::new(config);
    if scheduler.config.continuous_batch {
        scheduler.run_continuous_prepared(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            max_new_tokens,
            eos_token_ids,
            pad_id,
        )
    } else {
        let ids_rows: Vec<&[u32]> = prepared.iter().map(|p| p.input_ids.as_slice()).collect();
        let attn_rows: Vec<&[u32]> = prepared
            .iter()
            .map(|p| p.attention_mask.as_slice())
            .collect();
        scheduler.run_micro_batch(
            thinker,
            device,
            runtime,
            ids_rows.as_slice(),
            attn_rows.as_slice(),
            Some(audio_features),
            max_new_tokens,
            eos_token_ids,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AsrBatchSchedulerConfig, PagedBlockManager, can_admit_with_projected_kv,
        continuous_batch_enabled, group_running_by_decode_step,
        pad_prepared_wave_for_prefill,
    };
    use crate::processor::asr_processor::PreparedInputs;

    #[test]
    fn continuous_batch_defaults_on() {
        if std::env::var_os("VASR_DISABLE_CONTINUOUS_BATCH").is_none() {
            assert!(continuous_batch_enabled());
            assert!(AsrBatchSchedulerConfig::default().continuous_batch);
        }
    }

    #[test]
    fn group_running_by_decode_step_batches_same_step() {
        let groups = group_running_by_decode_step(&[(0, 2), (1, 2), (2, 0), (3, 1)]);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].0, 0);
        assert_eq!(groups[0].1, vec![2]);
        assert_eq!(groups[1].0, 1);
        assert_eq!(groups[1].1, vec![3]);
        assert_eq!(groups[2].0, 2);
        assert_eq!(groups[2].1, vec![0, 1]);
    }

    #[test]
    fn projected_kv_admission_respects_block_capacity() {
        let manager = PagedBlockManager::new(5, 32);
        let capacity = PagedBlockManager::allocatable_block_capacity(5);
        assert_eq!(capacity, 4);
        assert!(can_admit_with_projected_kv(&manager, capacity, 32, 0, 1));
        assert!(!can_admit_with_projected_kv(&manager, capacity, 128, 3, 1));
    }

    #[test]
    fn pad_prepared_wave_for_prefill_builds_varlen_masks() {
        let prepared = vec![
            PreparedInputs {
                input_ids: vec![1, 2, 3],
                attention_mask: vec![1, 1, 1],
                input_features: vec![],
                feature_attention_mask: vec![],
            },
            PreparedInputs {
                input_ids: vec![4, 5, 6, 7, 8],
                attention_mask: vec![1, 1, 1, 1, 1],
                input_features: vec![],
                feature_attention_mask: vec![],
            },
        ];
        let (ids, masks) = pad_prepared_wave_for_prefill(prepared.as_slice(), &[0, 1], 0).unwrap();
        assert_eq!(ids[0].len(), 5);
        assert_eq!(ids[1].len(), 5);
        assert_eq!(masks[0], vec![1, 1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 1, 1]);
    }
}
