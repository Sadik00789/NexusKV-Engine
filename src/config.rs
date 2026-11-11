// src/config.rs
use serde::{Deserialize, Serialize};

/// Engine runtime configuration for the memory pool and continuous scheduler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineConfig {
    /// Total number of physical blocks in the KV pool.
    pub total_physical_blocks: usize,
    /// Number of tokens stored per physical block.
    pub block_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Maximum number of concurrent sequences.
    pub max_num_seqs: usize,
    /// Maximum physical blocks assignable to one sequence.
    pub max_blocks_per_seq: usize,
    /// Model layer count.
    pub num_layers: usize,
    /// Token budget for one scheduler iteration.
    pub max_num_batched_tokens: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            total_physical_blocks: 512,
            block_size: 16,
            num_heads: 8,
            head_dim: 64,
            max_num_seqs: 16,
            max_blocks_per_seq: 32,
            num_layers: 1,
            max_num_batched_tokens: 256,
        }
    }
}

impl EngineConfig {
    pub fn max_token_capacity(&self) -> usize {
        self.total_physical_blocks * self.block_size
    }

    pub fn max_seq_token_length(&self) -> usize {
        self.max_blocks_per_seq * self.block_size
    }
}
