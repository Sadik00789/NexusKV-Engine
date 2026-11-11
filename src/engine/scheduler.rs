// src/engine/scheduler.rs
use super::actor::TokenMsg;
use crate::memory::{BlockSpaceManager, MemoryError, SeqId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokensReached,
    StopTokenEncountered,
    Aborted,
    Error,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokensReached => "length",
            Self::StopTokenEncountered => "stop",
            Self::Aborted => "aborted",
            Self::Error => "error",
        }
    }
}

/// One in-flight sequence. `computed` is the number of tokens whose KV has
/// actually been written into the paged pool.
pub struct Seq {
    pub id: SeqId,
    pub prompt: Vec<u32>,
    pub out: Vec<u32>,
    pub computed: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub arrival: Instant,
    pub first_token_at: Option<Instant>,
    pub last_token_at: Option<Instant>,
    pub tx: UnboundedSender<TokenMsg>,
    pub spec: bool,
    /// A checkout sequence already owns a page table attached by the session
    /// store. Admission must not re-run prefix matching for it.
    pub attached: bool,
    pub rng_state: u64,
}

impl Seq {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: SeqId,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        tx: UnboundedSender<TokenMsg>,
        spec: bool,
    ) -> Self {
        let rng_state = (id as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x2545_f491_4f6c_dd1d);
        Self {
            id,
            prompt,
            out: Vec::new(),
            computed: 0,
            max_tokens: max_tokens.max(1),
            temperature,
            top_p,
            arrival: Instant::now(),
            first_token_at: None,
            last_token_at: None,
            tx,
            spec,
            attached: false,
            rng_state,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt.len() + self.out.len()
    }

    pub fn needs_prefill(&self) -> bool {
        self.computed < self.prompt.len()
    }

    pub fn ready_for_decode(&self) -> bool {
        self.total_tokens().checked_sub(1) == Some(self.computed) && !self.out.is_empty()
    }

    fn out_is_empty(&self) -> bool {
        self.out.is_empty()
    }

    pub fn ttft(&self) -> Option<f64> {
        self.first_token_at
            .map(|first| first.duration_since(self.arrival).as_secs_f64() * 1000.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillChunk {
    pub seq: SeqId,
    pub start: usize,
    pub len: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Batch {
    pub prefill: Vec<PrefillChunk>,
    pub decode: Vec<SeqId>,
    /// Sequences preempted while building this batch. The actor resets their
    /// drafters and the batch entries for them were already removed.
    pub preempted: Vec<SeqId>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }

    pub fn token_count(&self) -> usize {
        self.prefill.iter().map(|chunk| chunk.len).sum::<usize>() + self.decode.len()
    }
}

/// Continuous batching scheduler.
///
/// Iteration order: running decodes get one token each first, then running
/// prefills are chunked, then waiting sequences are admitted while budget and
/// sequence slots remain. Admission OOM preempts the newest running sequence
/// and requeues it at the front of the waiting queue.
pub struct Scheduler {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    waiting: VecDeque<Seq>,
    running: HashMap<SeqId, Seq>,
    order: VecDeque<SeqId>,
    pub preemptions: u64,
}

impl Scheduler {
    pub fn new(max_num_seqs: usize, max_num_batched_tokens: usize) -> Self {
        assert!(max_num_seqs > 0, "max_num_seqs must be positive");
        assert!(
            max_num_batched_tokens > 0,
            "max_num_batched_tokens must be positive"
        );
        Self {
            max_num_seqs,
            max_num_batched_tokens,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            order: VecDeque::new(),
            preemptions: 0,
        }
    }

    pub fn submit(&mut self, seq: Seq) {
        self.waiting.push_back(seq);
    }

    /// Checkout sequences already own an attached page table, so they skip
    /// prefix matching on admission. They are queued at the front to preserve
    /// the cheap-resume path.
    pub fn submit_checkout(&mut self, seq: Seq) {
        debug_assert!(seq.attached, "checkout sequences must be pre-attached");
        self.waiting.push_front(seq);
    }

    pub fn schedule(&mut self, block_manager: &mut BlockSpaceManager) -> Batch {
        let mut budget = self.max_num_batched_tokens;
        let mut batch = Batch::default();
        let running_ids: Vec<SeqId> = self.order.iter().copied().collect();

        // Decodes first: one budget token each.
        for id in &running_ids {
            if budget == 0 {
                break;
            }
            if let Some(seq) = self.running.get(id)
                && seq.ready_for_decode()
            {
                batch.decode.push(*id);
                budget -= 1;
            }
        }

        // Running prefills are chunked into the remaining budget.
        // TODO(kv-valid): skip prefill compute on resident trie hits. v1
        // recomputes prompt tokens even when match_or_allocate_prefix hit.
        for id in &running_ids {
            if budget == 0 {
                break;
            }
            if let Some(seq) = self.running.get(id) {
                let remaining = seq.prompt.len().saturating_sub(seq.computed);
                if remaining > 0 {
                    let len = remaining.min(budget);
                    batch.prefill.push(PrefillChunk {
                        seq: *id,
                        start: seq.computed,
                        len,
                    });
                    budget -= len;
                }
            }
        }

        // Admit waiting sequences even while other sequences are decoding.
        let mut admission_attempts = 0usize;
        while budget > 0 && self.running.len() < self.max_num_seqs && !self.waiting.is_empty() {
            admission_attempts += 1;
            if admission_attempts > self.max_num_seqs + self.waiting.len() + 8 {
                break;
            }
            let mut seq = self.waiting.pop_front().expect("checked non-empty");
            let seq_id = seq.id;
            let prompt_len = seq.prompt.len();
            let admission = if seq.attached {
                Ok(())
            } else {
                block_manager
                    .match_or_allocate_prefix(seq_id, &seq.prompt)
                    .map(|_| ())
            };
            match admission {
                Ok(()) => {
                    if !seq.attached {
                        seq.computed = 0;
                    }
                    let computed = seq.computed;
                    self.order.push_back(seq_id);
                    self.running.insert(seq_id, seq);
                    let remaining = prompt_len.saturating_sub(computed);
                    if remaining > 0 {
                        let len = remaining.min(budget);
                        batch.prefill.push(PrefillChunk {
                            seq: seq_id,
                            start: computed,
                            len,
                        });
                        budget -= len;
                    }
                }
                Err(MemoryError::OutOfMemory) => {
                    if let Some(victim) = self.preempt_newest(block_manager) {
                        batch.decode.retain(|id| *id != victim);
                        batch.prefill.retain(|chunk| chunk.seq != victim);
                        batch.preempted.push(victim);
                        self.preemptions += 1;
                        self.waiting.push_front(seq);
                        continue;
                    }
                    self.waiting.push_front(seq);
                    break;
                }
                Err(err) => {
                    let _ = seq.tx.send(TokenMsg::Error {
                        message: err.to_string(),
                    });
                    let _ = seq.tx.send(TokenMsg::Done {
                        reason: FinishReason::Error,
                    });
                }
            }
        }

        debug_assert!(batch.token_count() <= self.max_num_batched_tokens);
        batch
    }

    /// Preempt the newest running sequence that has not yet produced output in
    /// its current residency. Sequences that have made progress are allowed to
    /// finish; otherwise two equal-sized sequences that both need the whole
    /// pool would preempt each other forever.
    fn preempt_newest(&mut self, block_manager: &mut BlockSpaceManager) -> Option<SeqId> {
        let victim = self
            .order
            .iter()
            .rev()
            .copied()
            .find(|id| self.running.get(id).map(Seq::out_is_empty).unwrap_or(false))?;
        self.order.retain(|running| *running != victim);
        let mut seq = self.running.remove(&victim)?;
        let _ = block_manager.free_sequence(victim);
        seq.computed = 0;
        seq.out.clear();
        seq.attached = false;
        seq.first_token_at = None;
        seq.last_token_at = None;
        self.waiting.push_front(seq);
        Some(victim)
    }

    pub fn finish(&mut self, id: SeqId, block_manager: &mut BlockSpaceManager) -> Option<Seq> {
        self.order.retain(|running| *running != id);
        let seq = self.running.remove(&id)?;
        let _ = block_manager.free_sequence(id);
        Some(seq)
    }

    /// Remove a sequence from either queue. Returns `None` if it already finished.
    pub fn abort(&mut self, id: SeqId, block_manager: &mut BlockSpaceManager) -> Option<Seq> {
        if let Some(seq) = self.running.remove(&id) {
            self.order.retain(|running| *running != id);
            let _ = block_manager.free_sequence(id);
            return Some(seq);
        }
        if let Some(position) = self.waiting.iter().position(|seq| seq.id == id) {
            let seq = self.waiting.remove(position).expect("position exists");
            let _ = block_manager.free_sequence(id);
            return Some(seq);
        }
        None
    }

    pub fn contains(&self, id: SeqId) -> bool {
        self.running.contains_key(&id) || self.waiting.iter().any(|seq| seq.id == id)
    }

    pub fn seq(&self, id: SeqId) -> Option<&Seq> {
        self.running.get(&id)
    }

    pub fn seq_mut(&mut self, id: SeqId) -> Option<&mut Seq> {
        self.running.get_mut(&id)
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    pub fn active_sequence_ids(&self) -> Vec<SeqId> {
        self.order
            .iter()
            .copied()
            .filter(|id| self.running.contains_key(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::BlockSpaceManager;
    use std::time::Instant;
    use tokio::sync::mpsc::unbounded_channel;

    fn new_seq(id: SeqId, prompt: Vec<u32>, max_tokens: usize) -> Seq {
        let (tx, _rx) = unbounded_channel();
        Seq::new(id, prompt, max_tokens, 0.0, 1.0, tx, false)
    }

    fn apply_prefill(
        scheduler: &mut Scheduler,
        block_manager: &mut BlockSpaceManager,
        chunk: &PrefillChunk,
    ) {
        let seq = scheduler.seq_mut(chunk.seq).unwrap();
        seq.computed += chunk.len;
        let prompt_done = seq.computed == seq.prompt.len();
        let needs_first = seq.out.is_empty();
        if prompt_done && needs_first {
            seq.out.push(7);
            seq.first_token_at = Some(Instant::now());
            block_manager.append_tokens(chunk.seq, 1).unwrap();
        }
    }

    #[test]
    fn mixed_batch_and_starvation_regression() {
        let mut block_manager = BlockSpaceManager::new(64, 4, 32);
        let mut scheduler = Scheduler::new(4, 16);
        scheduler.submit(new_seq(1, vec![1, 2, 3, 4, 5, 6], 8));
        scheduler.submit(new_seq(2, (10..20).collect(), 8));

        let batch = scheduler.schedule(&mut block_manager);
        assert_eq!(batch.token_count(), 16);
        assert_eq!(batch.prefill.len(), 2);
        assert!(batch.decode.is_empty());
        for chunk in &batch.prefill {
            apply_prefill(&mut scheduler, &mut block_manager, chunk);
        }

        // Third sequence arrives while the first two are decoding.
        scheduler.submit(new_seq(3, (30..38).collect(), 8));
        let batch = scheduler.schedule(&mut block_manager);
        assert_eq!(batch.decode.len(), 2, "running sequences must decode");
        assert_eq!(batch.prefill.len(), 1, "waiting sequence must not starve");
        assert_eq!(batch.prefill[0].seq, 3);
        assert_eq!(batch.prefill[0].len, 8);
        assert!(batch.token_count() <= 16);
        assert_eq!(scheduler.running_count(), 3);
    }

    #[test]
    fn three_chunk_prefill_spans_three_iterations() {
        let mut block_manager = BlockSpaceManager::new(64, 4, 32);
        let mut scheduler = Scheduler::new(1, 4);
        scheduler.submit(new_seq(1, (0..10).collect(), 4));

        let mut chunks = Vec::new();
        for _ in 0..3 {
            let batch = scheduler.schedule(&mut block_manager);
            assert_eq!(batch.decode.len(), 0);
            assert_eq!(batch.prefill.len(), 1);
            let chunk = batch.prefill[0].clone();
            assert!(chunk.len <= 4);
            chunks.push((chunk.start, chunk.len));
            apply_prefill(&mut scheduler, &mut block_manager, &chunk);
        }
        assert_eq!(chunks, vec![(0, 4), (4, 4), (8, 2)]);
        assert_eq!(scheduler.seq(1).unwrap().computed, 10);
        assert_eq!(scheduler.seq(1).unwrap().out, vec![7]);
    }

    #[test]
    fn preemption_requeues_and_both_sequences_complete() {
        let mut block_manager = BlockSpaceManager::new(2, 4, 4);
        let mut scheduler = Scheduler::new(2, 16);
        scheduler.submit(new_seq(1, (0..6).collect(), 2));
        scheduler.submit(new_seq(2, (20..26).collect(), 2));

        let mut finished = 0usize;
        for _ in 0..200 {
            let batch = scheduler.schedule(&mut block_manager);
            if batch.is_empty() && scheduler.is_idle() {
                break;
            }
            for chunk in batch.prefill.clone() {
                assert_eq!(chunk.seq, scheduler.seq(chunk.seq).unwrap().id);
                apply_prefill(&mut scheduler, &mut block_manager, &chunk);
            }
            for id in batch.decode {
                let seq = scheduler.seq_mut(id).unwrap();
                seq.computed += 1;
                seq.out.push(7);
                block_manager.append_tokens(id, 1).unwrap();
            }
            for id in scheduler.active_sequence_ids() {
                let done = scheduler
                    .seq(id)
                    .map(|seq| seq.out.len() >= seq.max_tokens)
                    .unwrap_or(false);
                if done {
                    scheduler.finish(id, &mut block_manager).unwrap();
                    finished += 1;
                }
            }
        }

        assert_eq!(finished, 2, "preempted sequence must eventually complete");
        assert!(scheduler.preemptions > 0, "memory pressure must preempt");
        assert_eq!(block_manager.active_blocks(), 0);
        block_manager.assert_invariants();
    }

    #[test]
    fn abort_waiting_and_running() {
        let mut block_manager = BlockSpaceManager::new(8, 4, 8);
        let mut scheduler = Scheduler::new(1, 8);
        scheduler.submit(new_seq(1, vec![1, 2, 3, 4], 4));
        scheduler.submit(new_seq(2, vec![5, 6, 7, 8], 4));

        let batch = scheduler.schedule(&mut block_manager);
        assert_eq!(batch.prefill[0].seq, 1);
        assert_eq!(scheduler.waiting_count(), 1);

        assert!(scheduler.abort(2, &mut block_manager).is_some());
        assert_eq!(scheduler.waiting_count(), 0);
        assert!(scheduler.abort(1, &mut block_manager).is_some());
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(block_manager.active_blocks(), 0);
        assert!(scheduler.abort(1, &mut block_manager).is_none());
    }
}
