// src/engine/speculative.rs
use crate::memory::{BlockSpaceManager, MemoryError};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Clone)]
#[allow(dead_code)]
pub enum SpeculativeError {
    #[error("Memory error during speculative verification: {0}")]
    Memory(#[from] MemoryError),
    #[error("Mismatched candidate and verification tensor lengths")]
    LengthMismatch,
    #[error("Invalid speculative tree node ID: {0}")]
    InvalidNodeId(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DraftNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,
    pub token_id: u32,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SpeculativeTree {
    pub seq_id: usize,
    pub nodes: Vec<DraftNode>,
}

#[allow(dead_code)]
impl SpeculativeTree {
    pub fn new(seq_id: usize) -> Self {
        Self {
            seq_id,
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, parent_id: Option<usize>, token_id: u32, probability: f32) -> usize {
        let node_id = self.nodes.len();
        self.nodes.push(DraftNode {
            node_id,
            parent_id,
            token_id,
            probability,
        });
        node_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct VerificationResult {
    pub accepted_tokens: Vec<u32>,
    pub num_accepted: usize,
    pub num_rejected: usize,
    pub acceptance_rate: f64,
}

#[allow(dead_code)]
pub struct SpeculativeVerifier {
    pub total_proposed: usize,
    pub total_accepted: usize,
    block_manager: Arc<RwLock<BlockSpaceManager>>,
}

impl SpeculativeVerifier {
    pub fn new(block_manager: Arc<RwLock<BlockSpaceManager>>) -> Self {
        Self {
            total_proposed: 0,
            total_accepted: 0,
            block_manager,
        }
    }

    /// Verifies draft tokens against target model logits in parallel.
    /// Rolls back physical VRAM pages in O(1) for any rejected tokens.
    #[allow(dead_code)]
    pub fn verify_and_rollback(
        &mut self,
        seq_id: usize,
        draft_tokens: &[u32],
        target_predicted_tokens: &[u32],
        baseline_token_length: usize,
    ) -> Result<VerificationResult, SpeculativeError> {
        if draft_tokens.len() != target_predicted_tokens.len() {
            return Err(SpeculativeError::LengthMismatch);
        }

        let mut accepted = Vec::new();
        let mut reject_idx = draft_tokens.len();

        // Exact match verification check
        for (i, (&draft, &target)) in draft_tokens.iter().zip(target_predicted_tokens.iter()).enumerate() {
            if draft == target {
                accepted.push(draft);
            } else {
                // First divergence encountered
                reject_idx = i;
                // Accept the target model's correction token
                accepted.push(target);
                break;
            }
        }

        let num_accepted = accepted.len();
        let num_rejected = draft_tokens.len().saturating_sub(reject_idx);

        self.total_proposed += draft_tokens.len();
        self.total_accepted += num_accepted;

        // Perform instant O(1) memory rollback on physical cache pages
        let final_token_count = baseline_token_length + num_accepted;
        let mut block_mgr = self.block_manager.write();
        block_mgr.rollback_sequence(seq_id, final_token_count)?;

        let acceptance_rate = if self.total_proposed > 0 {
            self.total_accepted as f64 / self.total_proposed as f64
        } else {
            0.0
        };

        Ok(VerificationResult {
            accepted_tokens: accepted,
            num_accepted,
            num_rejected,
            acceptance_rate,
        })
    }

    pub fn overall_acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            0.85 // Default baseline estimate before samples
        } else {
            self.total_accepted as f64 / self.total_proposed as f64
        }
    }

    #[allow(dead_code)]
    pub fn reset_metrics(&mut self) {
        self.total_proposed = 0;
        self.total_accepted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_verification_and_rollback() {
        let block_mgr = Arc::new(RwLock::new(BlockSpaceManager::new(32, 16, 4)));
        {
            let mut mgr = block_mgr.write();
            mgr.register_sequence(1).unwrap();
            // Baseline 16 tokens (1 block) + 4 speculative draft tokens = 20 tokens (2 blocks)
            mgr.append_tokens(1, 20).unwrap();
            assert_eq!(mgr.allocated_blocks_count(), 2);
        }

        let mut verifier = SpeculativeVerifier::new(Arc::clone(&block_mgr));

        let draft_tokens = vec![100, 101, 102, 103];
        let target_tokens = vec![100, 101, 999, 103]; // Divergence at index 2

        let result = verifier
            .verify_and_rollback(1, &draft_tokens, &target_tokens, 16)
            .unwrap();

        // Should accept [100, 101] + correction [999] = 3 tokens
        assert_eq!(result.accepted_tokens, vec![100, 101, 999]);
        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.num_rejected, 2);

        // Final retained tokens: 16 (baseline) + 3 (accepted) = 19 tokens (still requires 2 blocks)
        assert_eq!(block_mgr.read().allocated_blocks_count(), 2);
    }
}