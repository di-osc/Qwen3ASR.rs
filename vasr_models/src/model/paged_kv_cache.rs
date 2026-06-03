//! Paged KV cache metadata and storage for mistral.rs paged-attention backed decoding.

use candle_core::{DType, Device, Result, Tensor};

use crate::model::attention::{FlashKMeta, FlashParams};

pub const PAD_SLOT_ID: i64 = -1;

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    block_table_host: Vec<u32>,
    block_size: usize,
    num_blocks: usize,
    num_blocks_per_sequence: usize,
    batch_size: usize,
    max_tokens_per_sequence: usize,
    max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct PagedInputMetadata {
    pub slot_mapping: Tensor,
    pub block_tables: Tensor,
    pub context_lens: Tensor,
    pub max_context_len: usize,
    pub token_attention_mask: Option<Tensor>,
    pub prefill_attention_mask: Option<Tensor>,
    pub prefill_causal_only: bool,
    pub query_lens: Option<Vec<usize>>,
    pub kv_lens: Option<Vec<usize>>,
    pub cu_seqlens_q: Option<Tensor>,
    pub cu_seqlens_kv: Option<Tensor>,
    pub max_query_len: Option<usize>,
    pub max_kv_len: Option<usize>,
}

impl PagedInputMetadata {
    pub fn flash_params(&self, causal: bool) -> Option<FlashParams> {
        Some(FlashParams {
            max_q: self.max_query_len?.try_into().ok()?,
            cumulative_seqlens_q: self.cu_seqlens_q.clone(),
            logical_k: FlashKMeta {
                max: self.max_kv_len?.try_into().ok()?,
                cumulative_seqlens: self.cu_seqlens_kv.clone(),
            },
            causal,
        })
    }
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

fn slot_for_block_table_position(
    block_tables: &[Vec<usize>],
    block_size: usize,
    num_blocks: usize,
    seq: usize,
    position: usize,
) -> Result<i64> {
    let table = block_tables.get(seq).ok_or_else(|| {
        candle_core::Error::Msg(format!("missing block table for sequence {seq}"))
    })?;
    let block_idx = position / block_size;
    let offset = position % block_size;
    let block_id = *table.get(block_idx).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "missing block table entry: sequence={seq} block_idx={block_idx}"
        ))
    })?;
    if block_id >= num_blocks {
        candle_core::bail!(
            "block id exceeds paged cache capacity: block_id={block_id} num_blocks={num_blocks}"
        );
    }
    let slot = block_id
        .checked_mul(block_size)
        .and_then(|base| base.checked_add(offset))
        .ok_or_else(|| candle_core::Error::Msg("paged slot mapping overflow".to_string()))?;
    i64::try_from(slot)
        .map_err(|_| candle_core::Error::Msg(format!("paged slot overflows i64: {slot}")))
}

fn build_varlen_metadata(
    query_lens: Option<Vec<usize>>,
    kv_lens: Option<Vec<usize>>,
    device: &Device,
) -> Result<(Option<Tensor>, Option<Tensor>, Option<usize>, Option<usize>)> {
    let cu_seqlens_q = query_lens
        .as_ref()
        .map(|lens| cumulative_seqlens_from_lengths(lens.as_slice(), device))
        .transpose()?;
    let cu_seqlens_kv = kv_lens
        .as_ref()
        .map(|lens| cumulative_seqlens_from_lengths(lens.as_slice(), device))
        .transpose()?;
    let max_query_len = query_lens
        .as_ref()
        .map(|lens| lens.iter().copied().max().unwrap_or(0));
    let max_kv_len = kv_lens
        .as_ref()
        .map(|lens| lens.iter().copied().max().unwrap_or(0));
    Ok((cu_seqlens_q, cu_seqlens_kv, max_query_len, max_kv_len))
}

impl PagedKvCache {
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_tokens: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_batched(
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            1,
            max_tokens,
            dtype,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_batched(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        batch_size: usize,
        max_tokens_per_sequence: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let num_blocks_per_sequence = max_tokens_per_sequence.div_ceil(block_size);
        let num_blocks = batch_size
            .checked_mul(num_blocks_per_sequence)
            .ok_or_else(|| {
                candle_core::Error::Msg("paged kv cache block count overflow".to_string())
            })?;
        Self::new_inner(
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            num_blocks_per_sequence,
            batch_size,
            max_tokens_per_sequence,
            dtype,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_pool(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let max_tokens = num_blocks.checked_mul(block_size).ok_or_else(|| {
            candle_core::Error::Msg("paged kv cache token capacity overflow".to_string())
        })?;
        Self::new_inner(
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            num_blocks,
            1,
            max_tokens,
            dtype,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        num_blocks_per_sequence: usize,
        batch_size: usize,
        max_tokens_per_sequence: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if num_layers == 0 {
            candle_core::bail!("paged kv cache requires at least one layer");
        }
        if num_kv_heads == 0 {
            candle_core::bail!("paged kv cache requires at least one kv head");
        }
        if head_dim == 0 {
            candle_core::bail!("paged kv cache requires non-zero head_dim");
        }
        if block_size == 0 {
            candle_core::bail!("paged kv cache requires non-zero block_size");
        }
        if batch_size == 0 {
            candle_core::bail!("paged kv cache requires non-zero batch_size");
        }
        if max_tokens_per_sequence == 0 {
            candle_core::bail!("paged kv cache requires non-zero max_tokens");
        }
        if num_blocks == 0 {
            candle_core::bail!("paged kv cache requires non-zero num_blocks");
        }

        let dtype_bytes = dtype.size_in_bytes();
        if dtype_bytes == 0 || 16 % dtype_bytes != 0 {
            candle_core::bail!("unsupported paged kv cache dtype: {dtype:?}");
        }
        let key_pack = 16 / dtype_bytes;
        if head_dim % key_pack != 0 {
            candle_core::bail!(
                "head_dim must be divisible by key cache pack size: head_dim={head_dim} pack={key_pack}"
            );
        }

        let max_tokens = num_blocks.checked_mul(block_size).ok_or_else(|| {
            candle_core::Error::Msg("paged kv cache token capacity overflow".to_string())
        })?;
        let block_table_host: Vec<u32> = (0..num_blocks)
            .map(|idx| {
                u32::try_from(idx).map_err(|_| {
                    candle_core::Error::Msg(format!("block index overflows u32: {idx}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let (key_shape, value_shape) = (
            (
                num_blocks,
                num_kv_heads,
                head_dim / key_pack,
                block_size,
                key_pack,
            ),
            (num_blocks, num_kv_heads, head_dim, block_size),
        );

        let mut key_cache = Vec::with_capacity(num_layers);
        let mut value_cache = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            key_cache.push(Tensor::zeros(key_shape, dtype, device)?);
            value_cache.push(Tensor::zeros(value_shape, dtype, device)?);
        }

        Ok(Self {
            key_cache,
            value_cache,
            block_table_host,
            block_size,
            num_blocks,
            num_blocks_per_sequence,
            batch_size,
            max_tokens_per_sequence,
            max_tokens,
        })
    }

    pub fn estimated_bytes(&self) -> usize {
        self.key_cache
            .iter()
            .chain(self.value_cache.iter())
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn num_blocks_per_sequence(&self) -> usize {
        self.num_blocks_per_sequence
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn max_tokens_per_sequence(&self) -> usize {
        self.max_tokens_per_sequence
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn block_table_host(&self) -> &[u32] {
        self.block_table_host.as_slice()
    }

    pub fn slot_for_position(&self, position: usize) -> Result<i64> {
        self.slot_for_sequence_position(0, position)
    }

    pub fn slot_for_sequence_position(&self, sequence: usize, position: usize) -> Result<i64> {
        if sequence >= self.batch_size {
            candle_core::bail!(
                "sequence exceeds paged kv cache batch: sequence={sequence} batch={}",
                self.batch_size
            );
        }
        if position >= self.max_tokens_per_sequence {
            candle_core::bail!(
                "position exceeds paged kv cache capacity: position={position} max_tokens_per_sequence={}",
                self.max_tokens_per_sequence
            );
        }
        let base = sequence
            .checked_mul(self.num_blocks_per_sequence)
            .and_then(|blocks| blocks.checked_mul(self.block_size))
            .ok_or_else(|| candle_core::Error::Msg("slot sequence base overflow".to_string()))?;
        let slot = base
            .checked_add(position)
            .ok_or_else(|| candle_core::Error::Msg("slot position overflow".to_string()))?;
        i64::try_from(slot)
            .map_err(|_| candle_core::Error::Msg(format!("slot overflows i64: {slot}")))
    }

    pub fn slot_mapping_for_range(
        &self,
        start: usize,
        len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| candle_core::Error::Msg("slot range overflow".to_string()))?;
        if end > self.max_tokens {
            candle_core::bail!(
                "slot range exceeds paged kv cache capacity: start={start} len={len} max_tokens={}",
                self.max_tokens
            );
        }
        let slots: Vec<i64> = (start..end)
            .map(|position| self.slot_for_position(position))
            .collect::<Result<Vec<_>>>()?;
        Tensor::from_vec(slots, (len,), device)
    }

    pub fn block_tables_tensor(&self, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            self.block_table_host.clone(),
            (1usize, self.num_blocks),
            device,
        )
    }

    pub fn block_tables_tensor_for_batch(&self, batch: usize, device: &Device) -> Result<Tensor> {
        if batch > self.batch_size {
            candle_core::bail!(
                "requested block table batch exceeds cache batch: requested={batch} cache_batch={}",
                self.batch_size
            );
        }
        let len = batch
            .checked_mul(self.num_blocks_per_sequence)
            .ok_or_else(|| candle_core::Error::Msg("block table slice overflow".to_string()))?;
        Tensor::from_vec(
            self.block_table_host[..len].to_vec(),
            (batch, self.num_blocks_per_sequence),
            device,
        )
    }

    pub fn context_lens_tensor(&self, context_len: usize, device: &Device) -> Result<Tensor> {
        if context_len > self.max_tokens {
            candle_core::bail!(
                "context length exceeds paged kv cache capacity: context_len={context_len} max_tokens={}",
                self.max_tokens
            );
        }
        let len = u32::try_from(context_len).map_err(|_| {
            candle_core::Error::Msg(format!("context length overflows u32: {context_len}"))
        })?;
        Tensor::from_vec(vec![len], (1usize,), device)
    }

    pub fn input_metadata_for_range(
        &self,
        start: usize,
        len: usize,
        context_len: usize,
        device: &Device,
    ) -> Result<PagedInputMetadata> {
        let query_lens = Some(vec![len]);
        let kv_lens = Some(vec![context_len]);
        let (cu_seqlens_q, cu_seqlens_kv, max_query_len, max_kv_len) =
            build_varlen_metadata(query_lens.clone(), kv_lens.clone(), device)?;
        Ok(PagedInputMetadata {
            slot_mapping: self.slot_mapping_for_range(start, len, device)?,
            block_tables: self.block_tables_tensor(device)?,
            context_lens: self.context_lens_tensor(context_len, device)?,
            max_context_len: context_len,
            token_attention_mask: None,
            prefill_attention_mask: None,
            prefill_causal_only: false,
            query_lens,
            kv_lens,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_query_len,
            max_kv_len,
        })
    }

    pub fn input_metadata_for_batch_ranges(
        &self,
        starts: &[usize],
        len: usize,
        context_lens: &[usize],
        device: &Device,
    ) -> Result<PagedInputMetadata> {
        let batch = starts.len();
        if batch == 0 {
            candle_core::bail!("paged metadata batch must be non-empty");
        }
        if context_lens.len() != batch {
            candle_core::bail!(
                "context_lens batch mismatch: starts={} context_lens={}",
                starts.len(),
                context_lens.len()
            );
        }
        if batch > self.batch_size {
            candle_core::bail!(
                "metadata batch exceeds cache batch: batch={batch} cache_batch={}",
                self.batch_size
            );
        }

        let mut slots: Vec<i64> = Vec::with_capacity(batch.saturating_mul(len));
        for (seq, &start) in starts.iter().enumerate() {
            let end = start
                .checked_add(len)
                .ok_or_else(|| candle_core::Error::Msg("slot range overflow".to_string()))?;
            if end > self.max_tokens_per_sequence {
                candle_core::bail!(
                    "slot range exceeds paged kv cache sequence capacity: sequence={seq} start={start} len={len} max_tokens_per_sequence={}",
                    self.max_tokens_per_sequence
                );
            }
            for position in start..end {
                slots.push(self.slot_for_sequence_position(seq, position)?);
            }
        }

        let mut context_host = Vec::with_capacity(batch);
        let mut max_context_len = 0usize;
        for &context_len in context_lens {
            if context_len > self.max_tokens_per_sequence {
                candle_core::bail!(
                    "context length exceeds paged kv cache sequence capacity: context_len={context_len} max_tokens_per_sequence={}",
                    self.max_tokens_per_sequence
                );
            }
            max_context_len = max_context_len.max(context_len);
            context_host.push(u32::try_from(context_len).map_err(|_| {
                candle_core::Error::Msg(format!("context length overflows u32: {context_len}"))
            })?);
        }

        let query_lens = Some(vec![len; batch]);
        let kv_lens = Some(context_lens.to_vec());
        let (cu_seqlens_q, cu_seqlens_kv, max_query_len, max_kv_len) =
            build_varlen_metadata(query_lens.clone(), kv_lens.clone(), device)?;

        Ok(PagedInputMetadata {
            slot_mapping: Tensor::from_vec(slots, (batch * len,), device)?,
            block_tables: self.block_tables_tensor_for_batch(batch, device)?,
            context_lens: Tensor::from_vec(context_host, (batch,), device)?,
            max_context_len,
            token_attention_mask: None,
            prefill_attention_mask: None,
            prefill_causal_only: false,
            query_lens,
            kv_lens,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_query_len,
            max_kv_len,
        })
    }

    pub fn input_metadata_from_block_tables(
        &self,
        block_tables: &[Vec<usize>],
        starts: &[usize],
        len: usize,
        context_lens: &[usize],
        device: &Device,
    ) -> Result<PagedInputMetadata> {
        let batch = block_tables.len();
        if batch == 0 {
            candle_core::bail!("paged metadata batch must be non-empty");
        }
        if starts.len() != batch || context_lens.len() != batch {
            candle_core::bail!(
                "paged metadata batch mismatch: block_tables={} starts={} context_lens={}",
                batch,
                starts.len(),
                context_lens.len()
            );
        }
        let max_blocks = block_tables.iter().map(Vec::len).max().unwrap_or(0);
        if max_blocks == 0 {
            candle_core::bail!("paged metadata requires at least one block per sequence");
        }

        let mut slots: Vec<i64> = Vec::with_capacity(batch.saturating_mul(len));
        for seq in 0..batch {
            let table = &block_tables[seq];
            let start = starts[seq];
            let end = start
                .checked_add(len)
                .ok_or_else(|| candle_core::Error::Msg("slot range overflow".to_string()))?;
            for position in start..end {
                let block_idx = position / self.block_size;
                let offset = position % self.block_size;
                let block_id = *table.get(block_idx).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "missing block table entry: sequence={seq} block_idx={block_idx}"
                    ))
                })?;
                if block_id >= self.num_blocks {
                    candle_core::bail!(
                        "block id exceeds paged cache capacity: block_id={block_id} num_blocks={}",
                        self.num_blocks
                    );
                }
                let slot = block_id
                    .checked_mul(self.block_size)
                    .and_then(|base| base.checked_add(offset))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("paged slot mapping overflow".to_string())
                    })?;
                slots.push(i64::try_from(slot).map_err(|_| {
                    candle_core::Error::Msg(format!("paged slot overflows i64: {slot}"))
                })?);
            }
        }

        let mut block_table_host: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for table in block_tables {
            for &block_id in table {
                block_table_host.push(u32::try_from(block_id).map_err(|_| {
                    candle_core::Error::Msg(format!("block id overflows u32: {block_id}"))
                })?);
            }
            for _ in table.len()..max_blocks {
                block_table_host.push(0);
            }
        }

        let mut context_host = Vec::with_capacity(batch);
        let mut max_context_len = 0usize;
        for &context_len in context_lens {
            max_context_len = max_context_len.max(context_len);
            context_host.push(u32::try_from(context_len).map_err(|_| {
                candle_core::Error::Msg(format!("context length overflows u32: {context_len}"))
            })?);
        }

        let query_lens = Some(vec![len; batch]);
        let kv_lens = Some(context_lens.to_vec());
        let (cu_seqlens_q, cu_seqlens_kv, max_query_len, max_kv_len) =
            build_varlen_metadata(query_lens.clone(), kv_lens.clone(), device)?;

        Ok(PagedInputMetadata {
            slot_mapping: Tensor::from_vec(slots, (batch * len,), device)?,
            block_tables: Tensor::from_vec(block_table_host, (batch, max_blocks), device)?,
            context_lens: Tensor::from_vec(context_host, (batch,), device)?,
            max_context_len,
            token_attention_mask: None,
            prefill_attention_mask: None,
            prefill_causal_only: false,
            query_lens,
            kv_lens,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_query_len,
            max_kv_len,
        })
    }

    pub fn input_metadata_from_attention_masks(
        &self,
        block_tables: &[Vec<usize>],
        attention_masks: &[&[u32]],
        device: &Device,
    ) -> Result<PagedInputMetadata> {
        let batch = attention_masks.len();
        if batch == 0 {
            candle_core::bail!("paged metadata batch must be non-empty");
        }
        if block_tables.len() != batch {
            candle_core::bail!(
                "paged metadata batch mismatch: block_tables={} attention_masks={}",
                block_tables.len(),
                batch
            );
        }
        let seq_len = attention_masks
            .first()
            .map(|row| row.len())
            .ok_or_else(|| candle_core::Error::Msg("missing first attention mask".into()))?;
        if seq_len == 0 {
            candle_core::bail!("attention mask rows must be non-empty");
        }

        let max_blocks = block_tables.iter().map(Vec::len).max().unwrap_or(0);
        if max_blocks == 0 {
            candle_core::bail!("paged metadata requires at least one block per sequence");
        }

        let mut slots: Vec<i64> = Vec::with_capacity(batch.saturating_mul(seq_len));
        let mut attn_host: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
        let mut context_host: Vec<u32> = Vec::with_capacity(batch);
        let mut query_lens = Vec::with_capacity(batch);
        let mut max_context_len = 0usize;

        for (seq, row) in attention_masks.iter().enumerate() {
            if row.len() != seq_len {
                candle_core::bail!(
                    "attention_mask row len mismatch: expected={seq_len}, got={}",
                    row.len()
                );
            }
            let table = &block_tables[seq];
            let mut context_len = 0usize;
            for (position, &mask) in row.iter().enumerate() {
                attn_host.push(mask);
                if mask == 0 {
                    slots.push(PAD_SLOT_ID);
                    continue;
                }
                let block_idx = position / self.block_size;
                let offset = position % self.block_size;
                let block_id = *table.get(block_idx).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "missing block table entry: sequence={seq} block_idx={block_idx}"
                    ))
                })?;
                if block_id >= self.num_blocks {
                    candle_core::bail!(
                        "block id exceeds paged cache capacity: block_id={block_id} num_blocks={}",
                        self.num_blocks
                    );
                }
                let slot = block_id
                    .checked_mul(self.block_size)
                    .and_then(|base| base.checked_add(offset))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("paged slot mapping overflow".to_string())
                    })?;
                slots.push(i64::try_from(slot).map_err(|_| {
                    candle_core::Error::Msg(format!("paged slot overflows i64: {slot}"))
                })?);
                context_len = context_len.checked_add(1).ok_or_else(|| {
                    candle_core::Error::Msg("context length overflow".to_string())
                })?;
            }
            max_context_len = max_context_len.max(context_len);
            query_lens.push(context_len);
            context_host.push(u32::try_from(context_len).map_err(|_| {
                candle_core::Error::Msg(format!("context length overflows u32: {context_len}"))
            })?);
        }

        let mut block_table_host: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for table in block_tables {
            for &block_id in table {
                block_table_host.push(u32::try_from(block_id).map_err(|_| {
                    candle_core::Error::Msg(format!("block id overflows u32: {block_id}"))
                })?);
            }
            for _ in table.len()..max_blocks {
                block_table_host.push(0);
            }
        }

        let kv_lens = Some(query_lens.clone());
        let query_lens = Some(query_lens);
        let (cu_seqlens_q, cu_seqlens_kv, max_query_len, max_kv_len) =
            build_varlen_metadata(query_lens.clone(), kv_lens.clone(), device)?;
        let prefill_causal_only = query_lens
            .as_ref()
            .is_some_and(|lens| lens.iter().all(|&len| len == seq_len));

        Ok(PagedInputMetadata {
            slot_mapping: Tensor::from_vec(slots, (batch * seq_len,), device)?,
            block_tables: Tensor::from_vec(block_table_host, (batch, max_blocks), device)?,
            context_lens: Tensor::from_vec(context_host, (batch,), device)?,
            max_context_len,
            token_attention_mask: Some(Tensor::from_vec(attn_host, (batch, seq_len), device)?),
            prefill_attention_mask: None,
            prefill_causal_only,
            query_lens,
            kv_lens,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_query_len,
            max_kv_len,
        })
    }

    pub fn decode_metadata_for_steps(
        &self,
        prompt_len: usize,
        steps: usize,
        device: &Device,
    ) -> Result<Vec<PagedInputMetadata>> {
        let end = prompt_len
            .checked_add(steps)
            .ok_or_else(|| candle_core::Error::Msg("decode metadata range overflow".to_string()))?;
        if end > self.max_tokens {
            candle_core::bail!(
                "decode metadata range exceeds paged kv cache capacity: prompt_len={prompt_len} steps={steps} max_tokens={}",
                self.max_tokens
            );
        }

        let slots: Vec<i64> = (prompt_len..end)
            .map(|position| self.slot_for_position(position))
            .collect::<Result<Vec<_>>>()?;
        let contexts: Vec<u32> = ((prompt_len + 1)..=(prompt_len + steps))
            .map(|context_len| {
                u32::try_from(context_len).map_err(|_| {
                    candle_core::Error::Msg(format!("context length overflows u32: {context_len}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let all_slots = Tensor::from_vec(slots, (steps,), device)?;
        let all_contexts = Tensor::from_vec(contexts, (steps,), device)?;
        let block_tables = self.block_tables_tensor(device)?;

        (0..steps)
            .map(|step| {
                Ok(PagedInputMetadata {
                    slot_mapping: all_slots.narrow(0, step, 1)?,
                    block_tables: block_tables.clone(),
                    context_lens: all_contexts.narrow(0, step, 1)?,
                    max_context_len: prompt_len + step + 1,
                    token_attention_mask: None,
                    prefill_attention_mask: None,
                    prefill_causal_only: false,
                    query_lens: None,
                    kv_lens: None,
                    cu_seqlens_q: None,
                    cu_seqlens_kv: None,
                    max_query_len: None,
                    max_kv_len: None,
                })
            })
            .collect()
    }

    pub fn decode_metadata_for_batch_steps(
        &self,
        block_tables: &[Vec<usize>],
        prompt_lens: &[usize],
        steps: usize,
        device: &Device,
    ) -> Result<Vec<PagedInputMetadata>> {
        let batch = block_tables.len();
        if batch == 0 {
            candle_core::bail!("paged batch decode metadata requires at least one sequence");
        }
        if prompt_lens.len() != batch {
            candle_core::bail!(
                "paged batch decode metadata mismatch: block_tables={batch} prompt_lens={}",
                prompt_lens.len()
            );
        }

        let max_blocks = block_tables.iter().map(Vec::len).max().unwrap_or(0);
        if max_blocks == 0 {
            candle_core::bail!(
                "paged batch decode metadata requires at least one block per sequence"
            );
        }

        let mut block_table_host: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for table in block_tables {
            for &block_id in table {
                block_table_host.push(u32::try_from(block_id).map_err(|_| {
                    candle_core::Error::Msg(format!("block id overflows u32: {block_id}"))
                })?);
            }
            for _ in table.len()..max_blocks {
                block_table_host.push(0);
            }
        }
        let block_tables_tensor = Tensor::from_vec(block_table_host, (batch, max_blocks), device)?;

        let mut all_slots: Vec<i64> = Vec::with_capacity(steps.saturating_mul(batch));
        let mut all_contexts: Vec<u32> = Vec::with_capacity(steps.saturating_mul(batch));
        let mut max_context_lens = Vec::with_capacity(steps);

        for step in 0..steps {
            let mut step_max_context = 0usize;
            for seq in 0..batch {
                let position = prompt_lens[seq].checked_add(step).ok_or_else(|| {
                    candle_core::Error::Msg("batch decode slot position overflow".to_string())
                })?;
                let context_len = position.checked_add(1).ok_or_else(|| {
                    candle_core::Error::Msg("batch decode context length overflow".to_string())
                })?;
                if context_len > self.max_tokens_per_sequence {
                    candle_core::bail!(
                        "batch decode context length exceeds paged kv cache sequence capacity: context_len={context_len} max_tokens_per_sequence={}",
                        self.max_tokens_per_sequence
                    );
                }
                let slot = slot_for_block_table_position(
                    block_tables,
                    self.block_size,
                    self.num_blocks,
                    seq,
                    position,
                )?;
                all_slots.push(slot);
                all_contexts.push(u32::try_from(context_len).map_err(|_| {
                    candle_core::Error::Msg(format!("context length overflows u32: {context_len}"))
                })?);
                step_max_context = step_max_context.max(context_len);
            }
            max_context_lens.push(step_max_context);
        }

        let all_slots = Tensor::from_vec(all_slots, (steps, batch), device)?;
        let all_contexts = Tensor::from_vec(all_contexts, (steps, batch), device)?;
        let query_lens = vec![1usize; batch];

        (0..steps)
            .map(|step| {
                let kv_lens = prompt_lens
                    .iter()
                    .map(|&prompt_len| prompt_len + step + 1)
                    .collect::<Vec<_>>();
                let (cu_seqlens_q, cu_seqlens_kv, max_query_len, max_kv_len) =
                    build_varlen_metadata(Some(query_lens.clone()), Some(kv_lens.clone()), device)?;
                Ok(PagedInputMetadata {
                    slot_mapping: all_slots.narrow(0, step, 1)?.flatten_all()?,
                    block_tables: block_tables_tensor.clone(),
                    context_lens: all_contexts.narrow(0, step, 1)?.flatten_all()?,
                    max_context_len: max_context_lens[step],
                    token_attention_mask: None,
                    prefill_attention_mask: None,
                    prefill_causal_only: false,
                    query_lens: Some(query_lens.clone()),
                    kv_lens: Some(kv_lens),
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_query_len,
                    max_kv_len,
                })
            })
            .collect()
    }

    pub fn key_value_cache(&self, layer_idx: usize) -> Result<(&Tensor, &Tensor)> {
        let key = self.key_cache.get(layer_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("paged kv cache layer out of range: {layer_idx}"))
        })?;
        let value = self.value_cache.get(layer_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("paged kv cache layer out of range: {layer_idx}"))
        })?;
        Ok((key, value))
    }
}

#[cfg(test)]
mod tests {
    use super::{PAD_SLOT_ID, PagedKvCache};

    #[test]
    fn test_paged_cache_single_sequence_layout() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new(2, 4, 8, 16, 40, candle_core::DType::F32, &device)?;
        assert_eq!(cache.block_size(), 16);
        assert_eq!(cache.num_blocks(), 3);
        assert_eq!(cache.block_table_host(), &[0, 1, 2]);
        assert_eq!(cache.slot_for_position(0)?, 0);
        assert_eq!(cache.slot_for_position(16)?, 16);
        assert_eq!(cache.slot_for_position(39)?, 39);
        Ok(())
    }

    #[test]
    fn test_paged_cache_metadata_tensors() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new(1, 2, 8, 4, 10, candle_core::DType::F32, &device)?;

        let slots = cache
            .slot_mapping_for_range(3, 4, &device)?
            .to_vec1::<i64>()?;
        assert_eq!(slots, vec![3, 4, 5, 6]);

        let blocks = cache.block_tables_tensor(&device)?.to_vec2::<u32>()?;
        assert_eq!(blocks, vec![vec![0, 1, 2]]);

        let context = cache.context_lens_tensor(7, &device)?.to_vec1::<u32>()?;
        assert_eq!(context, vec![7]);

        let (key, value) = cache.key_value_cache(0)?;
        assert_eq!(key.dims(), &[3, 2, 2, 4, 4]);
        assert_eq!(value.dims(), &[3, 2, 8, 4]);
        Ok(())
    }

    #[test]
    fn test_paged_cache_multi_sequence_layout() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new_batched(1, 2, 8, 4, 2, 10, candle_core::DType::F32, &device)?;

        assert_eq!(cache.batch_size(), 2);
        assert_eq!(cache.max_tokens_per_sequence(), 10);
        assert_eq!(cache.num_blocks_per_sequence(), 3);
        assert_eq!(cache.num_blocks(), 6);
        assert_eq!(cache.slot_for_sequence_position(0, 0)?, 0);
        assert_eq!(cache.slot_for_sequence_position(0, 9)?, 9);
        assert_eq!(cache.slot_for_sequence_position(1, 0)?, 12);
        assert_eq!(cache.slot_for_sequence_position(1, 9)?, 21);

        let meta = cache.input_metadata_for_batch_ranges(&[0, 0], 3, &[3, 3], &device)?;
        assert_eq!(
            meta.slot_mapping.to_vec1::<i64>()?,
            vec![0, 1, 2, 12, 13, 14]
        );
        assert_eq!(
            meta.block_tables.to_vec2::<u32>()?,
            vec![vec![0, 1, 2], vec![3, 4, 5]]
        );
        assert_eq!(meta.context_lens.to_vec1::<u32>()?, vec![3, 3]);
        assert_eq!(meta.max_context_len, 3);

        Ok(())
    }

    #[test]
    fn test_paged_cache_metadata_from_physical_block_tables() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new_pool(1, 2, 8, 4, 8, candle_core::DType::F32, &device)?;

        let meta = cache.input_metadata_from_block_tables(
            &[vec![3, 4], vec![6, 7]],
            &[2, 4],
            2,
            &[4, 6],
            &device,
        )?;
        assert_eq!(meta.slot_mapping.to_vec1::<i64>()?, vec![14, 15, 28, 29]);
        assert_eq!(
            meta.block_tables.to_vec2::<u32>()?,
            vec![vec![3, 4], vec![6, 7]]
        );
        assert_eq!(meta.context_lens.to_vec1::<u32>()?, vec![4, 6]);
        assert_eq!(meta.max_context_len, 6);

        Ok(())
    }

    #[test]
    fn test_paged_cache_metadata_from_attention_masks_tracks_query_lens() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new_pool(1, 2, 8, 4, 8, candle_core::DType::F32, &device)?;

        let masks = [vec![1u32, 1, 1, 1, 0, 0], vec![1u32, 1, 1, 1, 1, 0]];
        let mask_rows: Vec<&[u32]> = masks.iter().map(Vec::as_slice).collect();
        let meta = cache.input_metadata_from_attention_masks(
            &[vec![3, 4], vec![6, 7]],
            mask_rows.as_slice(),
            &device,
        )?;

        assert_eq!(
            meta.slot_mapping.to_vec1::<i64>()?,
            vec![
                12,
                13,
                14,
                15,
                PAD_SLOT_ID,
                PAD_SLOT_ID,
                24,
                25,
                26,
                27,
                28,
                PAD_SLOT_ID
            ]
        );
        assert_eq!(
            meta.token_attention_mask
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing token_attention_mask"))?
                .to_vec2::<u32>()?,
            vec![vec![1, 1, 1, 1, 0, 0], vec![1, 1, 1, 1, 1, 0]]
        );
        assert_eq!(meta.context_lens.to_vec1::<u32>()?, vec![4, 5]);
        assert_eq!(
            meta.query_lens
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing query_lens"))?,
            &vec![4, 5]
        );
        assert_eq!(
            meta.kv_lens
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing kv_lens"))?,
            &vec![4, 5]
        );
        assert_eq!(
            meta.cu_seqlens_q
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing cu_seqlens_q"))?
                .to_vec1::<u32>()?,
            vec![0, 4, 9]
        );
        assert_eq!(
            meta.cu_seqlens_kv
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing cu_seqlens_kv"))?
                .to_vec1::<u32>()?,
            vec![0, 4, 9]
        );
        assert_eq!(meta.max_query_len, Some(5));
        assert_eq!(meta.max_kv_len, Some(5));
        assert_eq!(meta.max_context_len, 5);
        if meta.prefill_causal_only {
            anyhow::bail!("expected prefill_causal_only=false for padded attention masks");
        }
        let flash = meta
            .flash_params(true)
            .ok_or_else(|| anyhow::anyhow!("missing flash params"))?;
        assert_eq!(flash.max_q, 5);
        assert_eq!(flash.logical_k.max, 5);
        assert!(flash.causal);
        assert_eq!(
            flash
                .cumulative_seqlens_q
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing cumulative_seqlens_q"))?
                .to_vec1::<u32>()?,
            vec![0, 4, 9]
        );
        assert_eq!(
            flash
                .logical_k
                .cumulative_seqlens
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing cumulative_seqlens_k"))?
                .to_vec1::<u32>()?,
            vec![0, 4, 9]
        );
        Ok(())
    }

    #[test]
    fn test_decode_metadata_for_steps_matches_per_step_builder() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new_pool(1, 2, 8, 4, 64, candle_core::DType::F32, &device)?;
        let prompt_len = 10usize;
        let steps = 4usize;

        let precomputed = cache.decode_metadata_for_steps(prompt_len, steps, &device)?;
        for step in 0..steps {
            let start = prompt_len + step;
            let context_len = prompt_len + step + 1;
            let expected = cache.input_metadata_for_range(start, 1, context_len, &device)?;
            let actual = precomputed
                .get(step)
                .ok_or_else(|| anyhow::anyhow!("missing precomputed metadata for step {step}"))?;
            assert_eq!(
                actual.slot_mapping.to_vec1::<i64>()?,
                expected.slot_mapping.to_vec1::<i64>()?
            );
            assert_eq!(
                actual.context_lens.to_vec1::<u32>()?,
                expected.context_lens.to_vec1::<u32>()?
            );
            assert_eq!(actual.max_context_len, expected.max_context_len);
            assert!(actual.cu_seqlens_q.is_none());
            assert!(actual.cu_seqlens_kv.is_none());
        }

        Ok(())
    }

    #[test]
    fn test_decode_metadata_for_batch_steps_matches_per_step_builder() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cache = PagedKvCache::new_pool(1, 2, 8, 4, 16, candle_core::DType::F32, &device)?;
        let block_tables = [vec![3usize, 4], vec![6usize, 7]];
        let prompt_lens = [4usize, 5usize];
        let steps = 3usize;

        let precomputed =
            cache.decode_metadata_for_batch_steps(&block_tables, &prompt_lens, steps, &device)?;

        for step in 0..steps {
            let starts: Vec<usize> = prompt_lens.iter().map(|&len| len + step).collect();
            let context_lens: Vec<usize> = prompt_lens.iter().map(|&len| len + step + 1).collect();
            let expected = cache.input_metadata_from_block_tables(
                &block_tables,
                &starts,
                1,
                &context_lens,
                &device,
            )?;
            let actual = precomputed
                .get(step)
                .ok_or_else(|| anyhow::anyhow!("missing precomputed metadata for step {step}"))?;
            assert_eq!(
                actual.slot_mapping.to_vec1::<i64>()?,
                expected.slot_mapping.to_vec1::<i64>()?
            );
            assert_eq!(
                actual.block_tables.to_vec2::<u32>()?,
                expected.block_tables.to_vec2::<u32>()?
            );
            assert_eq!(
                actual.context_lens.to_vec1::<u32>()?,
                expected.context_lens.to_vec1::<u32>()?
            );
            assert_eq!(actual.max_context_len, expected.max_context_len);
            assert_eq!(actual.query_lens, expected.query_lens);
            assert_eq!(actual.kv_lens, expected.kv_lens);
        }

        Ok(())
    }
}
