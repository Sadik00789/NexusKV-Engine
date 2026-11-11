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
        // RTX 3050 4GB / 8GB RAM target: 192 blocks x 16 tokens is a ~1.5GB
        // f32 pool (344MB at f16) for Qwen3-0.6B's 28 layers / 8 KV heads.
        Self {
            total_physical_blocks: 192,
            block_size: 16,
            num_heads: 16,
            head_dim: 128,
            max_num_seqs: 8,
            // A 512-token shared prompt needs 32 blocks plus output blocks. This
            // is a per-sequence cap, not a reservation: the pool remains 192.
            max_blocks_per_seq: 40,
            num_layers: 28,
            max_num_batched_tokens: 256,
        }
    }
}

impl EngineConfig {
    /// f32 weights plus a 192-block f32 KV pool do not fit in 4 GB. The CUDA
    /// f32 fallback uses the documented 96-block / 4-sequence configuration.
    pub fn clamp_for_f32_cuda(&mut self) {
        self.total_physical_blocks = 96;
        self.max_num_seqs = 4;
    }

    pub fn max_token_capacity(&self) -> usize {
        self.total_physical_blocks * self.block_size
    }

    pub fn max_seq_token_length(&self) -> usize {
        self.max_blocks_per_seq * self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_cuda_defaults_and_f32_fallback_are_the_documented_budgets() {
        let default = EngineConfig::default();
        assert_eq!(default.total_physical_blocks, 192);
        assert_eq!(default.max_num_seqs, 8);

        let mut fallback = EngineConfig::default();
        fallback.clamp_for_f32_cuda();
        assert_eq!(fallback.total_physical_blocks, 96);
        assert_eq!(fallback.max_num_seqs, 4);
        assert_eq!(fallback.block_size, 16);
    }
}
