//! Paged KV cache metadata and storage for mistral.rs paged-attention backed decoding.

use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    block_table_host: Vec<u32>,
    block_size: usize,
    num_blocks: usize,
    max_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct PagedInputMetadata {
    pub slot_mapping: Tensor,
    pub block_tables: Tensor,
    pub context_lens: Tensor,
    pub max_context_len: usize,
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
        if max_tokens == 0 {
            candle_core::bail!("paged kv cache requires non-zero max_tokens");
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

        let num_blocks = max_tokens.div_ceil(block_size);
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
            max_tokens,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn block_table_host(&self) -> &[u32] {
        self.block_table_host.as_slice()
    }

    pub fn slot_for_position(&self, position: usize) -> Result<i64> {
        if position >= self.max_tokens {
            candle_core::bail!(
                "position exceeds paged kv cache capacity: position={position} max_tokens={}",
                self.max_tokens
            );
        }
        i64::try_from(position)
            .map_err(|_| candle_core::Error::Msg(format!("position overflows i64: {position}")))
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
        Ok(PagedInputMetadata {
            slot_mapping: self.slot_mapping_for_range(start, len, device)?,
            block_tables: self.block_tables_tensor(device)?,
            context_lens: self.context_lens_tensor(context_len, device)?,
            max_context_len: context_len,
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
    use super::PagedKvCache;

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
}
