// src/engine/scheduler.rs
use crate::memory::{BlockSpaceManager, MemoryError};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum SchedulerError {
    #[error("Memory allocation failed: {0}")]
    Memory(#[from] MemoryError),
    #[error("Sequence `{0}` not found in scheduler pool")]
    SequenceNotFound(usize),
    #[error("Sequence `{0}` is already finished")]
    SequenceAlreadyFinished(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished(FinishReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokensReached,
    StopTokenEncountered,
    Aborted,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Sequence {
    pub id: usize,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub status: SequenceStatus,
    pub created_at: Instant,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
}

#[allow(dead_code)]
impl Sequence {
    pub fn new(id: usize, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: Vec::new(),
            max_tokens,
            status: SequenceStatus::Waiting,
            created_at: Instant::now(),
            first_token_at: None,
            last_token_at: None,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.status, SequenceStatus::Finished(_))
    }

    pub fn ttft(&self) -> Option<Duration> {
        self.first_token_at.map(|t| t.duration_since(self.created_at))
    }

    pub fn itl(&self) -> Option<Duration> {
        if let (Some(first), Some(last)) = (self.first_token_at, self.last_token_at) {
            let num_generated = self.output_tokens.len();
            if num_generated > 1 {
                return Some(last.duration_since(first) / (num_generated - 1) as u32);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchType {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledBatch {
    pub batch_type: BatchType,
    pub sequence_ids: Vec<usize>,
    pub token_counts: Vec<usize>,
}

pub struct ContinuousScheduler {
    pub max_num_seqs: usize,
    waiting_queue: VecDeque<Sequence>,
    running_pool: HashMap<usize, Sequence>,
    finished_pool: HashMap<usize, Sequence>,
    block_manager: Arc<RwLock<BlockSpaceManager>>,
}

impl ContinuousScheduler {
    pub fn new(max_num_seqs: usize, block_manager: Arc<RwLock<BlockSpaceManager>>) -> Self {
        Self {
            max_num_seqs,
            waiting_queue: VecDeque::new(),
            running_pool: HashMap::new(),
            finished_pool: HashMap::new(),
            block_manager,
        }
    }

    /// Submits a new sequence request into the waiting queue.
    pub fn submit_sequence(&mut self, seq: Sequence) {
        self.waiting_queue.push_back(seq);
    }

    /// Schedules an iteration step, prioritizing running decodes and admitting prefill sequences.
    pub fn schedule_iteration(&mut self) -> Result<Option<ScheduledBatch>, SchedulerError> {
        // 1. Process active running sequences (Decode step)
        if !self.running_pool.is_empty() {
            let mut decode_seq_ids = Vec::with_capacity(self.running_pool.len());
            let mut decode_token_counts = Vec::with_capacity(self.running_pool.len());

            for (seq_id, seq) in self.running_pool.iter() {
                decode_seq_ids.push(*seq_id);
                decode_token_counts.push(seq.total_tokens());
            }

            return Ok(Some(ScheduledBatch {
                batch_type: BatchType::Decode,
                sequence_ids: decode_seq_ids,
                token_counts: decode_token_counts,
            }));
        }

        // 2. Admit sequences from waiting queue (Prefill step)
        if self.running_pool.len() < self.max_num_seqs {
            if let Some(mut next_seq) = self.waiting_queue.pop_front() {
                let seq_id = next_seq.id;

                let mut block_mgr = self.block_manager.write();
                // Attempt prefix-matching allocation
                if let Err(err) = block_mgr.match_or_allocate_prefix(seq_id, &next_seq.prompt_tokens) {
                    let _ = block_mgr.free_sequence(seq_id);
                    self.waiting_queue.push_front(next_seq);
                    return Err(SchedulerError::Memory(err));
                }

                next_seq.status = SequenceStatus::Running;
                let prompt_len = next_seq.prompt_tokens.len();
                self.running_pool.insert(seq_id, next_seq);

                return Ok(Some(ScheduledBatch {
                    batch_type: BatchType::Prefill,
                    sequence_ids: vec![seq_id],
                    token_counts: vec![prompt_len],
                }));
            }
        }

        Ok(None)
    }

    /// Advances a sequence by appending newly generated token(s).
    pub fn step_sequence(&mut self, seq_id: usize, token_id: u32, is_stop_token: bool) -> Result<(), SchedulerError> {
        let now = Instant::now();
        let mut block_mgr = self.block_manager.write();

        let seq = self
            .running_pool
            .get_mut(&seq_id)
            .ok_or(SchedulerError::SequenceNotFound(seq_id))?;

        if seq.first_token_at.is_none() {
            seq.first_token_at = Some(now);
        }
        seq.last_token_at = Some(now);

        seq.output_tokens.push(token_id);
        block_mgr.append_tokens(seq_id, 1)?;

        if is_stop_token {
            seq.status = SequenceStatus::Finished(FinishReason::StopTokenEncountered);
        } else if seq.output_tokens.len() >= seq.max_tokens {
            seq.status = SequenceStatus::Finished(FinishReason::MaxTokensReached);
        }

        if seq.is_finished() {
            let finished_seq = self.running_pool.remove(&seq_id).unwrap();
            block_mgr.free_sequence(seq_id)?;
            self.finished_pool.insert(seq_id, finished_seq);
        }

        Ok(())
    }

    /// Aborts an active or waiting sequence and immediately reclaims all allocated VRAM blocks.
    pub fn abort_sequence(&mut self, seq_id: usize) -> Result<(), SchedulerError> {
        let mut block_mgr = self.block_manager.write();

        if let Some(mut seq) = self.running_pool.remove(&seq_id) {
            seq.status = SequenceStatus::Finished(FinishReason::Aborted);
            let _ = block_mgr.free_sequence(seq_id);
            self.finished_pool.insert(seq_id, seq);
            return Ok(());
        }

        // Check waiting queue
        if let Some(pos) = self.waiting_queue.iter().position(|s| s.id == seq_id) {
            if let Some(mut seq) = self.waiting_queue.remove(pos) {
                seq.status = SequenceStatus::Finished(FinishReason::Aborted);
                let _ = block_mgr.free_sequence(seq_id);
                self.finished_pool.insert(seq_id, seq);
                return Ok(());
            }
        }

        // Check if already finished
        if self.finished_pool.contains_key(&seq_id) {
            return Err(SchedulerError::SequenceAlreadyFinished(seq_id));
        }

        Err(SchedulerError::SequenceNotFound(seq_id))
    }

    #[allow(dead_code)]
    pub fn get_sequence(&self, seq_id: usize) -> Option<&Sequence> {
        self.running_pool
            .get(&seq_id)
            .or_else(|| self.finished_pool.get(&seq_id))
    }

    pub fn active_sequence_ids(&self) -> Vec<usize> {
        self.running_pool.keys().copied().collect()
    }

    pub fn running_count(&self) -> usize {
        self.running_pool.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    #[allow(dead_code)]
    pub fn finished_count(&self) -> usize {
        self.finished_pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_scheduling_lifecycle() {
        let block_mgr = Arc::new(RwLock::new(BlockSpaceManager::new(64, 16, 8)));
        let mut scheduler = ContinuousScheduler::new(4, Arc::clone(&block_mgr));

        let seq1 = Sequence::new(1, vec![10, 20, 30], 2);
        scheduler.submit_sequence(seq1);

        // First step: Prefill Batch
        let batch = scheduler.schedule_iteration().unwrap().unwrap();
        assert_eq!(batch.batch_type, BatchType::Prefill);
        assert_eq!(batch.sequence_ids, vec![1]);
        assert_eq!(scheduler.running_count(), 1);

        // Advance 1 token (Decode)
        scheduler.step_sequence(1, 101, false).unwrap();
        let batch2 = scheduler.schedule_iteration().unwrap().unwrap();
        assert_eq!(batch2.batch_type, BatchType::Decode);

        // Advance final token (Max tokens reached)
        scheduler.step_sequence(1, 102, false).unwrap();
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.finished_count(), 1);
        assert_eq!(block_mgr.read().allocated_blocks_count(), 0);
    }

    #[test]
    fn test_sequence_abort_reclaims_blocks() {
        let block_mgr = Arc::new(RwLock::new(BlockSpaceManager::new(64, 16, 8)));
        let mut scheduler = ContinuousScheduler::new(4, Arc::clone(&block_mgr));

        let seq = Sequence::new(42, vec![1, 2, 3, 4, 5], 10);
        scheduler.submit_sequence(seq);

        let _ = scheduler.schedule_iteration().unwrap().unwrap();
        assert_eq!(scheduler.running_count(), 1);
        assert!(block_mgr.read().allocated_blocks_count() > 0);

        // Abort sequence mid-flight
        scheduler.abort_sequence(42).unwrap();
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.finished_count(), 1);
        assert_eq!(block_mgr.read().allocated_blocks_count(), 0);
    }
}