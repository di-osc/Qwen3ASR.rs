use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result};

use crate::model::paged_kv_cache::PagedKvCache;

#[derive(Debug, Clone, Copy)]
pub enum PagedCacheMemory {
    ContextSize(usize),
    Blocks(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct PagedCacheConfig {
    pub block_size: usize,
    pub memory: PagedCacheMemory,
}

impl Default for PagedCacheConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            memory: PagedCacheMemory::ContextSize(4096),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PagedCacheStats {
    pub block_size: usize,
    pub num_blocks: usize,
    pub free_blocks: usize,
    pub max_context_tokens: usize,
    pub bytes: usize,
}

#[derive(Debug)]
struct RequestBlocks {
    block_ids: Vec<usize>,
}

#[derive(Debug)]
pub struct PagedBlockManager {
    block_size: usize,
    free: VecDeque<usize>,
    request_blocks: HashMap<usize, RequestBlocks>,
}

impl PagedBlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut free = VecDeque::with_capacity(num_blocks.saturating_sub(1));
        for block_id in 1..num_blocks {
            free.push_back(block_id);
        }
        Self {
            block_size,
            free,
            request_blocks: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn allocate_slots(&mut self, request_id: usize, num_tokens: usize) -> Result<()> {
        let required = num_tokens.div_ceil(self.block_size);
        let existing = self
            .request_blocks
            .get(&request_id)
            .map(|blocks| blocks.block_ids.len())
            .unwrap_or(0);
        let needed = required.saturating_sub(existing);
        if needed == 0 {
            return Ok(());
        }
        if needed > self.free.len() {
            candle_core::bail!(
                "paged KV cache exhausted: request_id={request_id} needed_blocks={needed} free_blocks={}",
                self.free.len()
            );
        }
        let entry = self
            .request_blocks
            .entry(request_id)
            .or_insert_with(|| RequestBlocks {
                block_ids: Vec::with_capacity(required),
            });
        for _ in 0..needed {
            let block_id = self
                .free
                .pop_front()
                .ok_or_else(|| candle_core::Error::Msg("paged KV free list underflow".into()))?;
            entry.block_ids.push(block_id);
        }
        Ok(())
    }

    pub fn block_ids(&self, request_id: usize) -> Result<&[usize]> {
        self.request_blocks
            .get(&request_id)
            .map(|blocks| blocks.block_ids.as_slice())
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged KV request has no allocated blocks: request_id={request_id}"
                ))
            })
    }

    pub fn free_request(&mut self, request_id: usize) {
        if let Some(mut blocks) = self.request_blocks.remove(&request_id) {
            blocks.block_ids.reverse();
            for block_id in blocks.block_ids {
                if block_id != 0 {
                    self.free.push_back(block_id);
                }
            }
        }
    }

    pub fn free_many(&mut self, request_ids: &[usize]) {
        for &request_id in request_ids {
            self.free_request(request_id);
        }
    }
}

#[derive(Debug)]
pub struct PagedCacheRuntime {
    cache: PagedKvCache,
    manager: PagedBlockManager,
    stats: PagedCacheStats,
}

impl PagedCacheRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
        config: PagedCacheConfig,
    ) -> Result<Self> {
        if !matches!(config.block_size, 8 | 16 | 32) {
            candle_core::bail!(
                "paged cache block size must be 8, 16, or 32; got {}",
                config.block_size
            );
        }
        let num_blocks = match config.memory {
            PagedCacheMemory::ContextSize(tokens) => tokens.div_ceil(config.block_size) + 1,
            PagedCacheMemory::Blocks(blocks) => blocks,
        };
        if num_blocks <= 1 {
            candle_core::bail!("paged cache requires at least 2 blocks including null block");
        }
        let cache = PagedKvCache::new_pool(
            num_layers,
            num_kv_heads,
            head_dim,
            config.block_size,
            num_blocks,
            dtype,
            device,
        )?;
        let bytes = cache.estimated_bytes();
        let stats = PagedCacheStats {
            block_size: config.block_size,
            num_blocks,
            free_blocks: num_blocks - 1,
            max_context_tokens: (num_blocks - 1) * config.block_size,
            bytes,
        };
        Ok(Self {
            cache,
            manager: PagedBlockManager::new(num_blocks, config.block_size),
            stats,
        })
    }

    pub fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    pub fn manager(&self) -> &PagedBlockManager {
        &self.manager
    }

    pub fn manager_mut(&mut self) -> &mut PagedBlockManager {
        &mut self.manager
    }

    pub fn stats(&self) -> PagedCacheStats {
        let mut stats = self.stats.clone();
        stats.free_blocks = self.manager.free_blocks();
        stats
    }
}

pub type SharedPagedCacheRuntime = Arc<Mutex<PagedCacheRuntime>>;

#[cfg(test)]
mod tests {
    use super::PagedBlockManager;

    #[test]
    fn block_manager_allocates_extends_and_frees_requests() -> anyhow::Result<()> {
        let mut manager = PagedBlockManager::new(8, 4);
        assert_eq!(manager.free_blocks(), 7);

        manager.allocate_slots(10, 5)?;
        assert_eq!(manager.block_ids(10)?.len(), 2);
        assert_eq!(manager.free_blocks(), 5);

        manager.allocate_slots(10, 9)?;
        assert_eq!(manager.block_ids(10)?.len(), 3);
        assert_eq!(manager.free_blocks(), 4);

        manager.free_request(10);
        assert_eq!(manager.free_blocks(), 7);
        Ok(())
    }
}
