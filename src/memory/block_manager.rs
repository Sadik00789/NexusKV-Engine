// src/memory/block_manager.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum MemoryError {
    #[error("Physical VRAM Out Of Memory: No free blocks remaining")]
    OutOfMemory,
    #[error("Sequence ID `{0}` not found in page table")]
    SequenceNotFound(usize),
    #[error("Attempted to free an already unreferenced block: {0}")]
    DoubleFree(usize),
    #[error("Capacity exceeded: Max blocks per sequence reached")]
    CapacityExceeded,
}

/// Metadata for a physical 16-token page in GPU memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhysicalBlock {
    pub block_id: usize,
    pub ref_count: usize,
    pub is_free: bool,
    pub mapped_seq_id: Option<usize>,
    pub is_prefix_cached: bool,
    pub is_speculative: bool,
}

/// Sequence page table mapping logical sequence tokens to physical blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SequencePageTable {
    pub seq_id: usize,
    pub logical_tokens: usize,
    pub physical_block_ids: Vec<usize>,
}

/// Block telemetry snapshot used for the interactive Canvas Inspector and SSE telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockInspectInfo {
    pub block_id: usize,
    pub ref_count: usize,
    pub is_free: bool,
    pub mapped_seq_id: Option<usize>,
    pub is_prefix_cached: bool,
    pub is_speculative: bool,
}

/// Trie Node representation for the Radix Prefix Cache.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct RadixTrieNode {
    pub physical_block_id: Option<usize>,
    pub token_chunk: Vec<u32>,
    pub children: HashMap<Vec<u32>, usize>, // maps token_chunk to node_index in nodes vector
}

/// Radix Tree Prefix Cache index to identify shared prefix tokens across incoming prompts.
#[derive(Debug, Clone)]
pub struct RadixPrefixCache {
    nodes: Vec<RadixTrieNode>,
}

impl RadixPrefixCache {
    pub fn new() -> Self {
        Self {
            nodes: vec![RadixTrieNode::default()], // root node at index 0
        }
    }

    /// Clears the radix tree
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(RadixTrieNode::default());
    }
}

impl Default for RadixPrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe Physical Block Allocator and Virtual Page Table with Radix Prefix Caching.
pub struct BlockSpaceManager {
    pub total_blocks: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,
    physical_blocks: Vec<PhysicalBlock>,
    free_stack: Vec<usize>,
    page_tables: HashMap<usize, SequencePageTable>,
    prefix_cache: RadixPrefixCache,
    pub prefix_cache_lookups: usize,
    pub prefix_cache_hits: usize,
}

impl BlockSpaceManager {
    pub fn new(total_blocks: usize, block_size: usize, max_blocks_per_seq: usize) -> Self {
        let mut physical_blocks = Vec::with_capacity(total_blocks);
        let mut free_stack = Vec::with_capacity(total_blocks);

        // Populate free stack in reverse so block 0 is popped first
        for id in (0..total_blocks).rev() {
            free_stack.push(id);
        }

        for id in 0..total_blocks {
            physical_blocks.push(PhysicalBlock {
                block_id: id,
                ref_count: 0,
                is_free: true,
                mapped_seq_id: None,
                is_prefix_cached: false,
                is_speculative: false,
            });
        }

        Self {
            total_blocks,
            block_size,
            max_blocks_per_seq,
            physical_blocks,
            free_stack,
            page_tables: HashMap::new(),
            prefix_cache: RadixPrefixCache::new(),
            prefix_cache_lookups: 0,
            prefix_cache_hits: 0,
        }
    }

    /// O(1) Allocation from the physical free stack.
    pub fn allocate_block(&mut self) -> Result<usize, MemoryError> {
        let block_id = self.free_stack.pop().ok_or(MemoryError::OutOfMemory)?;
        let block = &mut self.physical_blocks[block_id];
        block.is_free = false;
        block.ref_count = 1;
        block.is_prefix_cached = false;
        block.is_speculative = false;
        Ok(block_id)
    }

    /// O(1) Decrement & Free back to pool when reference count reaches zero.
    pub fn free_block(&mut self, block_id: usize) -> Result<(), MemoryError> {
        if block_id >= self.total_blocks {
            return Err(MemoryError::DoubleFree(block_id));
        }

        let block = &mut self.physical_blocks[block_id];
        if block.ref_count == 0 {
            return Err(MemoryError::DoubleFree(block_id));
        }

        block.ref_count -= 1;

        if block.ref_count == 0 {
            block.is_free = true;
            block.mapped_seq_id = None;
            block.is_prefix_cached = false;
            block.is_speculative = false;
            self.free_stack.push(block_id);
        }

        Ok(())
    }

    /// Registers a new logical sequence entry in the page table.
    pub fn register_sequence(&mut self, seq_id: usize) -> Result<(), MemoryError> {
        self.page_tables.insert(
            seq_id,
            SequencePageTable {
                seq_id,
                logical_tokens: 0,
                physical_block_ids: Vec::new(),
            },
        );
        Ok(())
    }

    /// Matches prompt tokens against the Radix Prefix Cache and allocates missing blocks.
    /// Returns the number of cached prefix blocks successfully reused.
    pub fn match_or_allocate_prefix(
        &mut self,
        seq_id: usize,
        prompt_tokens: &[u32],
    ) -> Result<usize, MemoryError> {
        if !self.page_tables.contains_key(&seq_id) {
            self.register_sequence(seq_id)?;
        }

        let block_size = self.block_size;
        let num_full_blocks = prompt_tokens.len() / block_size;
        let mut curr_node_idx = 0;
        let mut matched_cached_blocks = 0;
        let mut allocated_block_ids = Vec::new();

        self.prefix_cache_lookups += 1;

        // 1. Traverse / Insert prefix chunks into Radix Trie
        for chunk_idx in 0..num_full_blocks {
            let start = chunk_idx * block_size;
            let end = start + block_size;
            let chunk = prompt_tokens[start..end].to_vec();

            let next_node_idx = self.prefix_cache.nodes[curr_node_idx]
                .children
                .get(&chunk)
                .copied();

            if let Some(child_idx) = next_node_idx {
                // Prefix hit: reuse physical block
                if let Some(phys_block_id) = self.prefix_cache.nodes[child_idx].physical_block_id {
                    self.physical_blocks[phys_block_id].ref_count += 1;
                    self.physical_blocks[phys_block_id].is_prefix_cached = true;
                    self.physical_blocks[phys_block_id].mapped_seq_id = Some(seq_id);
                    allocated_block_ids.push(phys_block_id);
                    matched_cached_blocks += 1;
                    self.prefix_cache_hits += 1;
                    curr_node_idx = child_idx;
                    continue;
                }
            }

            // Prefix miss: allocate a new physical block and register in Trie
            let new_block_id = self.allocate_block()?;
            self.physical_blocks[new_block_id].mapped_seq_id = Some(seq_id);
            self.physical_blocks[new_block_id].is_prefix_cached = true;
            allocated_block_ids.push(new_block_id);

            let new_node_idx = self.prefix_cache.nodes.len();
            self.prefix_cache.nodes.push(RadixTrieNode {
                physical_block_id: Some(new_block_id),
                token_chunk: chunk.clone(),
                children: HashMap::new(),
            });

            self.prefix_cache.nodes[curr_node_idx]
                .children
                .insert(chunk, new_node_idx);
            curr_node_idx = new_node_idx;
        }

        // 2. Handle remainder tokens (partial block) if prompt tokens is not an exact multiple
        let remainder = prompt_tokens.len() % block_size;
        if remainder > 0 {
            let remainder_block_id = self.allocate_block()?;
            self.physical_blocks[remainder_block_id].mapped_seq_id = Some(seq_id);
            allocated_block_ids.push(remainder_block_id);
        }

        // 3. Update Sequence Page Table
        let table = self
            .page_tables
            .get_mut(&seq_id)
            .ok_or(MemoryError::SequenceNotFound(seq_id))?;
        table.physical_block_ids = allocated_block_ids;
        table.logical_tokens = prompt_tokens.len();

        Ok(matched_cached_blocks)
    }

    /// Appends logical tokens to a sequence, allocating additional physical pages on demand.
    pub fn append_tokens(&mut self, seq_id: usize, num_new_tokens: usize) -> Result<(), MemoryError> {
        let block_size = self.block_size;
        let max_blocks = self.max_blocks_per_seq;

        // 1. Inspect existing token and block count
        let (old_tokens, current_block_count) = {
            let table = self
                .page_tables
                .get(&seq_id)
                .ok_or(MemoryError::SequenceNotFound(seq_id))?;
            (table.logical_tokens, table.physical_block_ids.len())
        };

        let new_tokens = old_tokens + num_new_tokens;
        let needed_blocks = if new_tokens == 0 {
            0
        } else {
            (new_tokens + block_size - 1) / block_size
        };

        if needed_blocks > max_blocks {
            return Err(MemoryError::CapacityExceeded);
        }

        // 2. Allocate newly needed physical blocks
        let num_to_allocate = needed_blocks.saturating_sub(current_block_count);
        let mut new_blocks = Vec::with_capacity(num_to_allocate);
        for _ in 0..num_to_allocate {
            let block_id = self.allocate_block()?;
            self.physical_blocks[block_id].mapped_seq_id = Some(seq_id);
            new_blocks.push(block_id);
        }

        // 3. Mutate the page table
        let table = self.page_tables.get_mut(&seq_id).unwrap();
        table.physical_block_ids.extend(new_blocks);
        table.logical_tokens = new_tokens;

        Ok(())
    }

    /// Fork Sequence with Copy-on-Write (CoW) semantics for Speculative Tree Branches.
    #[allow(dead_code)]
    pub fn fork_sequence_cow(&mut self, parent_seq_id: usize, child_seq_id: usize) -> Result<(), MemoryError> {
        let parent_table = self
            .page_tables
            .get(&parent_seq_id)
            .ok_or(MemoryError::SequenceNotFound(parent_seq_id))?
            .clone();

        for &block_id in &parent_table.physical_block_ids {
            self.physical_blocks[block_id].ref_count += 1;
        }

        self.page_tables.insert(
            child_seq_id,
            SequencePageTable {
                seq_id: child_seq_id,
                logical_tokens: parent_table.logical_tokens,
                physical_block_ids: parent_table.physical_block_ids,
            },
        );

        Ok(())
    }

    /// Instantly rolls back rejected speculative draft tokens in O(1) time.
    #[allow(dead_code)]
    pub fn rollback_sequence(&mut self, seq_id: usize, tokens_to_retain: usize) -> Result<(), MemoryError> {
        let block_size = self.block_size;
        let needed_blocks = if tokens_to_retain == 0 {
            0
        } else {
            (tokens_to_retain + block_size - 1) / block_size
        };

        // 1. Pop excess blocks from table
        let blocks_to_free = {
            let table = self
                .page_tables
                .get_mut(&seq_id)
                .ok_or(MemoryError::SequenceNotFound(seq_id))?;

            let mut freed = Vec::new();
            while table.physical_block_ids.len() > needed_blocks {
                freed.push(table.physical_block_ids.pop().unwrap());
            }
            table.logical_tokens = tokens_to_retain;
            freed
        };

        // 2. Free the blocks back to the pool
        for block_id in blocks_to_free {
            self.free_block(block_id)?;
        }

        Ok(())
    }

    /// Frees an entire sequence and recycles all mapped physical blocks.
    pub fn free_sequence(&mut self, seq_id: usize) -> Result<(), MemoryError> {
        if let Some(table) = self.page_tables.remove(&seq_id) {
            for block_id in table.physical_block_ids {
                self.free_block(block_id)?;
            }
        }
        Ok(())
    }

    /// Sets speculative draft flag on a physical block.
    #[allow(dead_code)]
    pub fn set_block_speculative(&mut self, block_id: usize, is_speculative: bool) {
        if block_id < self.total_blocks {
            self.physical_blocks[block_id].is_speculative = is_speculative;
        }
    }

    /// Flattens page tables for CUDA kernel consumption.
    #[allow(dead_code)]
    pub fn export_cuda_block_tables(&self, active_seq_ids: &[usize]) -> (Vec<i32>, Vec<i32>) {
        let mut flat_tables = vec![-1i32; active_seq_ids.len() * self.max_blocks_per_seq];
        let mut seq_lens = Vec::with_capacity(active_seq_ids.len());

        for (i, &seq_id) in active_seq_ids.iter().enumerate() {
            if let Some(table) = self.page_tables.get(&seq_id) {
                seq_lens.push(table.logical_tokens as i32);
                for (j, &block_id) in table.physical_block_ids.iter().enumerate() {
                    if j < self.max_blocks_per_seq {
                        flat_tables[i * self.max_blocks_per_seq + j] = block_id as i32;
                    }
                }
            } else {
                seq_lens.push(0);
            }
        }

        (flat_tables, seq_lens)
    }

    /// Returns list of physical block IDs assigned to a specific sequence.
    pub fn get_sequence_blocks(&self, seq_id: usize) -> Option<Vec<usize>> {
        self.page_tables.get(&seq_id).map(|t| t.physical_block_ids.clone())
    }

    /// Returns telemetry inspection array for all physical blocks.
    pub fn get_block_inspect_info(&self) -> Vec<BlockInspectInfo> {
        self.physical_blocks
            .iter()
            .map(|b| BlockInspectInfo {
                block_id: b.block_id,
                ref_count: b.ref_count,
                is_free: b.is_free,
                mapped_seq_id: b.mapped_seq_id,
                is_prefix_cached: b.is_prefix_cached,
                is_speculative: b.is_speculative,
            })
            .collect()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.free_stack.len()
    }

    pub fn allocated_blocks_count(&self) -> usize {
        self.total_blocks - self.free_stack.len()
    }

    pub fn memory_usage_ratio(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.allocated_blocks_count() as f64 / self.total_blocks as f64
        }
    }

    pub fn prefix_cache_hit_ratio(&self) -> f64 {
        if self.prefix_cache_lookups == 0 {
            0.0
        } else {
            self.prefix_cache_hits as f64 / self.prefix_cache_lookups as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_and_free() {
        let mut manager = BlockSpaceManager::new(4, 16, 4);
        assert_eq!(manager.free_blocks_count(), 4);

        let b1 = manager.allocate_block().unwrap();
        let b2 = manager.allocate_block().unwrap();
        assert_eq!(manager.free_blocks_count(), 2);

        manager.free_block(b1).unwrap();
        assert_eq!(manager.free_blocks_count(), 3);

        manager.free_block(b2).unwrap();
        assert_eq!(manager.free_blocks_count(), 4);
    }

    #[test]
    fn test_sequence_lifecycle_and_rollback() {
        let mut manager = BlockSpaceManager::new(10, 16, 4);
        manager.register_sequence(101).unwrap();

        manager.append_tokens(101, 20).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 2);

        manager.rollback_sequence(101, 12).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 1);

        manager.free_sequence(101).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 0);
    }

    #[test]
    fn test_copy_on_write_fork() {
        let mut manager = BlockSpaceManager::new(10, 16, 4);
        manager.register_sequence(1).unwrap();
        manager.append_tokens(1, 16).unwrap();

        manager.fork_sequence_cow(1, 2).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 1);

        manager.free_sequence(1).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 1);

        manager.free_sequence(2).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 0);
    }

    #[test]
    fn test_radix_prefix_cache_hit_and_sharing() {
        let mut manager = BlockSpaceManager::new(16, 16, 8);

        // Create a prompt with 32 tokens (exactly 2 blocks of 16 tokens)
        let prompt_tokens: Vec<u32> = (0..32).collect();

        // First sequence: allocates 2 new blocks in the prefix cache
        let matched1 = manager.match_or_allocate_prefix(1, &prompt_tokens).unwrap();
        assert_eq!(matched1, 0);
        assert_eq!(manager.allocated_blocks_count(), 2);

        // Second sequence with identical prompt: should hit prefix cache and reuse both blocks
        let matched2 = manager.match_or_allocate_prefix(2, &prompt_tokens).unwrap();
        assert_eq!(matched2, 2);
        assert_eq!(manager.allocated_blocks_count(), 2); // No new blocks allocated!

        // Freeing sequence 1 keeps blocks alive because ref_count == 1 (held by sequence 2)
        manager.free_sequence(1).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 2);

        // Freeing sequence 2 reclaims the blocks
        manager.free_sequence(2).unwrap();
        assert_eq!(manager.allocated_blocks_count(), 0);
    }

    #[test]
    fn test_export_cuda_block_tables() {
        let mut manager = BlockSpaceManager::new(16, 16, 4);
        manager.register_sequence(1).unwrap();
        manager.append_tokens(1, 32).unwrap();

        let (flat_tables, seq_lens) = manager.export_cuda_block_tables(&[1]);
        assert_eq!(seq_lens, vec![32]);
        assert_eq!(flat_tables.len(), 4);
        assert!(flat_tables[0] >= 0);
        assert!(flat_tables[1] >= 0);
        assert_eq!(flat_tables[2], -1);
    }
}