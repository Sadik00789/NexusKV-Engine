#![allow(dead_code)]
// tests/common/mod.rs
use nexuskv::config::EngineConfig;
use nexuskv::memory::{BlockSpaceManager, SeqId};
use nexuskv::model::{Executor, MockExecutor, ModelError};
use std::time::Duration;

pub fn test_config() -> EngineConfig {
    EngineConfig {
        total_physical_blocks: 64,
        block_size: 4,
        num_heads: 1,
        head_dim: 8,
        max_num_seqs: 4,
        max_blocks_per_seq: 16,
        num_layers: 1,
        max_num_batched_tokens: 256,
    }
}

/// Wraps the mock executor with a deterministic delay so integration tests can
/// observe concurrent sequences and client disconnects.
pub struct SlowExecutor {
    inner: MockExecutor,
    delay: Duration,
}

impl SlowExecutor {
    pub fn new(delay_ms: u64) -> Self {
        Self {
            inner: MockExecutor::new(),
            delay: Duration::from_millis(delay_ms),
        }
    }
}

impl Executor for SlowExecutor {
    fn forward(
        &mut self,
        block_manager: &mut BlockSpaceManager,
        seq: SeqId,
        tokens: &[u32],
        start: usize,
        all_logits: bool,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        std::thread::sleep(self.delay);
        self.inner
            .forward(block_manager, seq, tokens, start, all_logits)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn is_mock(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "mock"
    }
}
