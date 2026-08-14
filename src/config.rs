// src/config.rs
use serde::{Deserialize, Serialize};

/// Engine runtime configuration parameters for NexusKV memory and scheduling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(dead_code)]
pub struct EngineConfig {
    /// Total number of physical blocks allocated in the GPU VRAM pool.
    pub total_physical_blocks: usize,
    /// Number of tokens stored per physical block (typically 16).
    pub block_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per attention head (e.g., 64 or 128).
    pub head_dim: usize,
    /// Maximum number of concurrent sequences in the active batch.
    pub max_num_seqs: usize,
    /// Maximum physical blocks assignable to a single sequence.
    pub max_blocks_per_seq: usize,
    /// Default simulated or actual GPU layer count.
    pub num_layers: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            total_physical_blocks: 512, // 512 blocks * 16 tokens = 8,192 tokens max capacity
            block_size: 16,             // Standard 16 tokens per block
            num_heads: 8,
            head_dim: 64,
            max_num_seqs: 16,
            max_blocks_per_seq: 32,     // 32 blocks * 16 = 512 tokens max per sequence
            num_layers: 1,
        }
    }
}

#[allow(dead_code)]
impl EngineConfig {
    /// Total token capacity of the physical memory pool.
    pub fn max_token_capacity(&self) -> usize {
        self.total_physical_blocks * self.block_size
    }

    /// Maximum token length permissible for an individual sequence.
    pub fn max_seq_token_length(&self) -> usize {
        self.max_blocks_per_seq * self.block_size
    }
}