//! KV cache implementation for autoregressive generation.
//!
//! This is a minimal, Candle-friendly cache for transformer self-attention key/value
//! tensors. It mirrors the conceptual API of HuggingFace "dynamic cache".

use candle_core::{Result, Tensor};

const CACHE_GROW_SIZE: usize = 256;

/// Per-layer key/value cache entry.
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Backing key tensor: (batch, num_kv_heads, capacity_seq_len, head_dim)
    pub key: Tensor,
    /// Backing value tensor: (batch, num_kv_heads, capacity_seq_len, head_dim)
    pub value: Tensor,
    seq_len: usize,
    capacity_seq_len: usize,
}

impl KVCacheEntry {
    pub fn new(key: Tensor, value: Tensor) -> Result<Self> {
        let seq_len = key.dim(2)?;
        let capacity_seq_len = seq_len.saturating_add(CACHE_GROW_SIZE);
        let key = backing_tensor(&key, capacity_seq_len)?;
        let value = backing_tensor(&value, capacity_seq_len)?;
        Ok(Self {
            key,
            value,
            seq_len,
            capacity_seq_len,
        })
    }

    pub fn seq_len(&self) -> Result<usize> {
        Ok(self.seq_len)
    }

    fn current_key_value(&self) -> Result<(Tensor, Tensor)> {
        Ok((
            self.key.narrow(2, 0, self.seq_len)?,
            self.value.narrow(2, 0, self.seq_len)?,
        ))
    }

    fn append_to_backing(&mut self, new_key: &Tensor, new_value: &Tensor) -> Result<()> {
        let new_key = new_key.contiguous()?;
        let new_value = new_value.contiguous()?;
        let new_len = new_key.dim(2)?;
        let needed = self.seq_len.saturating_add(new_len);
        if needed > self.capacity_seq_len {
            let diff = needed - self.capacity_seq_len;
            let blocks = diff.div_ceil(CACHE_GROW_SIZE);
            self.capacity_seq_len += blocks * CACHE_GROW_SIZE;
            let old_key = self.key.narrow(2, 0, self.seq_len)?;
            let old_value = self.value.narrow(2, 0, self.seq_len)?;
            self.key = backing_tensor(&old_key, self.capacity_seq_len)?;
            self.value = backing_tensor(&old_value, self.capacity_seq_len)?;
        }

        self.key.slice_set(&new_key, 2, self.seq_len)?;
        self.value.slice_set(&new_value, 2, self.seq_len)?;
        self.seq_len = needed;
        Ok(())
    }

    pub fn update(&mut self, new_key: &Tensor, new_value: &Tensor) -> Result<(Tensor, Tensor)> {
        self.append_to_backing(new_key, new_value)?;
        self.current_key_value()
    }
}

fn backing_tensor(src: &Tensor, capacity_seq_len: usize) -> Result<Tensor> {
    let src = src.contiguous()?;
    let mut shape = src.dims().to_vec();
    shape[2] = capacity_seq_len;
    let backing = Tensor::zeros(shape, src.dtype(), src.device())?;
    backing.slice_set(&src, 2, 0)?;
    Ok(backing)
}

/// Dynamic KV-cache that grows as generation progresses.
#[derive(Debug, Clone, Default)]
pub struct KVCache {
    entries: Vec<Option<KVCacheEntry>>,
    seq_len: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            seq_len: 0,
        }
    }

    pub fn with_num_layers(num_layers: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
            seq_len: 0,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn get(&self, layer_idx: usize) -> Option<&KVCacheEntry> {
        self.entries.get(layer_idx).and_then(|e| e.as_ref())
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        while self.entries.len() <= layer_idx {
            self.entries.push(None);
        }

        let new_len = key.dim(2)?;

        match &mut self.entries[layer_idx] {
            Some(entry) => {
                let result = entry.update(key, value)?;
                if layer_idx == 0 {
                    self.seq_len = self.seq_len.saturating_add(new_len);
                }
                Ok(result)
            }
            None => {
                self.entries[layer_idx] = Some(KVCacheEntry::new(key.clone(), value.clone())?);
                if layer_idx == 0 {
                    self.seq_len = new_len;
                }
                let entry = self
                    .entries
                    .get(layer_idx)
                    .and_then(|e| e.as_ref())
                    .ok_or_else(|| candle_core::Error::Msg("missing kv cache entry".into()))?;
                entry.current_key_value()
            }
        }
    }

    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.seq_len = 0;
    }

    /// The next position index (0-based) to fill when appending.
    pub fn cache_position(&self) -> usize {
        self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::KVCache;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_kv_cache_basic() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut cache = KVCache::with_num_layers(2);

        if !cache.is_empty() {
            anyhow::bail!("expected empty cache");
        }
        if cache.seq_len() != 0 {
            anyhow::bail!("expected seq_len=0");
        }

        let key1 = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;
        let value1 = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;

        let (k, v) = cache.update(0, &key1, &value1)?;
        if k.dims() != [1, 4, 5, 64] {
            anyhow::bail!("unexpected key dims: {:?}", k.dims());
        }
        if v.dims() != [1, 4, 5, 64] {
            anyhow::bail!("unexpected value dims: {:?}", v.dims());
        }
        if cache.seq_len() != 5 {
            anyhow::bail!("expected seq_len=5, got {}", cache.seq_len());
        }

        let key2 = Tensor::zeros((1, 4, 1, 64), DType::F32, &device)?;
        let value2 = Tensor::zeros((1, 4, 1, 64), DType::F32, &device)?;

        let (k, v) = cache.update(0, &key2, &value2)?;
        if k.dims() != [1, 4, 6, 64] {
            anyhow::bail!("unexpected key dims after append: {:?}", k.dims());
        }
        if v.dims() != [1, 4, 6, 64] {
            anyhow::bail!("unexpected value dims after append: {:?}", v.dims());
        }
        if cache.seq_len() != 6 {
            anyhow::bail!("expected seq_len=6, got {}", cache.seq_len());
        }

        Ok(())
    }

    #[test]
    fn test_kv_cache_clear() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let mut cache = KVCache::with_num_layers(1);

        let key = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;
        let value = Tensor::zeros((1, 4, 5, 64), DType::F32, &device)?;

        cache.update(0, &key, &value)?;
        if cache.is_empty() {
            anyhow::bail!("expected non-empty cache after update");
        }

        cache.clear();
        if !cache.is_empty() {
            anyhow::bail!("expected empty cache after clear");
        }
        if cache.seq_len() != 0 {
            anyhow::bail!("expected seq_len=0 after clear");
        }

        Ok(())
    }
}
