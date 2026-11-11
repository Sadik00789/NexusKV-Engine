// src/engine/actor.rs
use super::scheduler::{Batch, FinishReason, PrefillChunk, Scheduler, Seq};
use super::session::{SessionStore, SessionTree};
use super::speculative::{
    AdaptiveK, DRAFT_NGRAM, LookupDrafter, SpecRng, SpecStats, probabilities, sample_from_logits,
    verify_greedy, verify_sampled,
};
use crate::memory::{
    BlockId, BlockInspectInfo, BlockSpaceManager, FlightRecorder, MemoryError, SeqId,
};
use crate::model::{Executor, Tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

#[derive(Debug, Clone)]
pub enum TokenMsg {
    Token {
        text: String,
        id: u32,
        step: usize,
        block_ids: Vec<BlockId>,
    },
    Done {
        reason: FinishReason,
    },
    Error {
        message: String,
    },
}

#[derive(Debug)]
pub struct GenerateJob {
    pub seq: SeqId,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub spec: bool,
    pub tx: UnboundedSender<TokenMsg>,
}

pub enum Request {
    Generate(GenerateJob),
    Abort {
        seq: SeqId,
    },
    Commit {
        session: String,
        parent: Option<u64>,
        label: String,
        seq: SeqId,
        reply: oneshot::Sender<Result<u64, String>>,
    },
    Checkout {
        node: u64,
        job: GenerateJob,
    },
    Forget {
        node: u64,
        reply: oneshot::Sender<Result<(), String>>,
    },
    SessionTree {
        session: String,
        reply: oneshot::Sender<SessionTree>,
    },
}

/// Telemetry published by the actor after every iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub executor: String,
    pub attention_backend: String,
    pub model: String,
    pub total_blocks: usize,
    pub blocks: usize,
    pub block_size: usize,
    pub allocated_blocks: usize,
    pub active_blocks: usize,
    pub cached_blocks: usize,
    pub free_blocks: usize,
    pub memory_usage_ratio: f64,
    pub memory_usage_percent: f64,
    pub running_sequences: usize,
    pub waiting_sequences: usize,
    pub prefix_cache_hit_ratio: f64,
    pub prefix_cache_lookups: u64,
    pub prefix_cache_hits: u64,
    /// Prompt tokens whose KV was reused rather than recomputed.
    pub prefix_skip_tokens: u64,
    /// Prompt tokens actually pushed through a forward pass (measured).
    pub prefill_tokens: u64,
    pub speculative_acceptance_rate: f64,
    pub acceptance: f64,
    pub speculative_proposed: u64,
    pub speculative_accepted: u64,
    pub adaptive_k: usize,
    pub tokens_per_sec: f64,
    pub ttft_p50_ms: f64,
    pub evictions: u64,
    pub page_faults: u64,
    pub cow_copies: u64,
    pub swapped_out: u64,
    pub swapped_in: u64,
    pub swap_bytes_out: u64,
    pub swap_bytes_in: u64,
    pub disk_spills: u64,
    pub disk_loads: u64,
    pub preemptions: u64,
    pub total_tokens_generated: u64,
    pub event_tick: u64,
    pub active_sequence_ids: Vec<SeqId>,
    pub seq_blocks: HashMap<SeqId, Vec<BlockId>>,
    pub block_inspect_info: Vec<BlockInspectInfo>,
}

impl Snapshot {
    pub fn initial(
        model: &str,
        executor: &str,
        attention_backend: &str,
        total_blocks: usize,
        block_size: usize,
    ) -> Self {
        Self {
            executor: executor.to_string(),
            attention_backend: attention_backend.to_string(),
            model: model.to_string(),
            total_blocks,
            blocks: total_blocks,
            block_size,
            allocated_blocks: 0,
            active_blocks: 0,
            cached_blocks: 0,
            free_blocks: total_blocks,
            memory_usage_ratio: 0.0,
            memory_usage_percent: 0.0,
            running_sequences: 0,
            waiting_sequences: 0,
            prefix_cache_hit_ratio: 0.0,
            prefix_cache_lookups: 0,
            prefix_cache_hits: 0,
            prefix_skip_tokens: 0,
            prefill_tokens: 0,
            speculative_acceptance_rate: 0.0,
            acceptance: 0.0,
            speculative_proposed: 0,
            speculative_accepted: 0,
            adaptive_k: 0,
            tokens_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            evictions: 0,
            page_faults: 0,
            cow_copies: 0,
            swapped_out: 0,
            swapped_in: 0,
            swap_bytes_out: 0,
            swap_bytes_in: 0,
            disk_spills: 0,
            disk_loads: 0,
            preemptions: 0,
            total_tokens_generated: 0,
            event_tick: 0,
            active_sequence_ids: Vec::new(),
            seq_blocks: HashMap::new(),
            block_inspect_info: Vec::new(),
        }
    }
}

struct ThroughputWindow {
    events: VecDeque<(Instant, usize)>,
}

impl ThroughputWindow {
    fn new() -> Self {
        Self {
            events: VecDeque::new(),
        }
    }

    fn push(&mut self, tokens: usize) {
        self.events.push_back((Instant::now(), tokens));
    }

    fn prune(&mut self, now: Instant) {
        while let Some((at, _)) = self.events.front() {
            if now.duration_since(*at).as_secs_f64() > 2.0 {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    fn tokens_per_sec(&self, now: Instant) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        let total: usize = self.events.iter().map(|(_, count)| count).sum();
        let elapsed = now
            .duration_since(self.events.front().expect("non-empty").0)
            .as_secs_f64()
            .max(0.001);
        total as f64 / elapsed
    }
}

/// Handles the actor uses to publish telemetry and flight-recorder keyframes.
pub struct TelemetrySink {
    pub snapshot: Arc<Mutex<Snapshot>>,
    pub recorder: Arc<Mutex<FlightRecorder>>,
}

pub struct EngineActor {
    rx: UnboundedReceiver<Request>,
    scheduler: Scheduler,
    block_manager: BlockSpaceManager,
    executor: Box<dyn Executor>,
    tokenizer: Arc<dyn Tokenizer>,
    snapshot: Arc<Mutex<Snapshot>>,
    spec_stats: SpecStats,
    bandit: AdaptiveK,
    drafters: HashMap<SeqId, LookupDrafter>,
    sessions: SessionStore,
    recorder: Arc<Mutex<FlightRecorder>>,
    throughput: ThroughputWindow,
    ttfts: VecDeque<f64>,
    total_tokens: u64,
    model_id: String,
    /// Measured final-prefill logits keyed by the full prompt. A repeated
    /// full-prefix hit can emit its first token with zero prefill forwards.
    prefill_logits: HashMap<u64, Vec<f32>>,
    prefill_logits_order: VecDeque<u64>,
    prefill_tokens: u64,
}

const PREFILL_LOGIT_CACHE_ENTRIES: usize = 8;

fn prompt_key(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ (tokens.len() as u64);
    for &token in tokens {
        hash = (hash ^ u64::from(token)).wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

impl EngineActor {
    pub fn new(
        rx: UnboundedReceiver<Request>,
        scheduler: Scheduler,
        block_manager: BlockSpaceManager,
        executor: Box<dyn Executor>,
        tokenizer: Arc<dyn Tokenizer>,
        telemetry: TelemetrySink,
        model_id: String,
    ) -> Self {
        Self {
            rx,
            scheduler,
            block_manager,
            executor,
            tokenizer,
            snapshot: telemetry.snapshot,
            spec_stats: SpecStats::default(),
            bandit: AdaptiveK::new(),
            drafters: HashMap::new(),
            sessions: SessionStore::new(),
            recorder: telemetry.recorder,
            throughput: ThroughputWindow::new(),
            ttfts: VecDeque::new(),
            total_tokens: 0,
            model_id,
            prefill_logits: HashMap::new(),
            prefill_logits_order: VecDeque::new(),
            prefill_tokens: 0,
        }
    }

    pub fn run(mut self) {
        // Keyframe the empty allocator so time-travel can replay from tick 0.
        self.publish();
        loop {
            let mut disconnected = false;
            loop {
                match self.rx.try_recv() {
                    Ok(request) => self.handle_request(request),
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }

            if disconnected && self.scheduler.is_idle() {
                self.publish();
                return;
            }

            let batch = self.scheduler.schedule(&mut self.block_manager);
            if !batch.is_empty() {
                for seq_id in &batch.preempted {
                    if let Some(seq) = self.scheduler.seq(*seq_id) {
                        let mut drafter = LookupDrafter::new(DRAFT_NGRAM);
                        drafter.observe(&seq.prompt);
                        self.drafters.insert(*seq_id, drafter);
                    }
                }
                let produced = self.execute_batch(&batch);
                if produced > 0 {
                    self.throughput.push(produced);
                }
                #[cfg(debug_assertions)]
                self.block_manager.assert_invariants();
                self.publish();
                continue;
            }

            if self.scheduler.is_idle() {
                self.publish();
                if disconnected {
                    return;
                }
                match self.rx.blocking_recv() {
                    Some(request) => self.handle_request(request),
                    None => {
                        self.publish();
                        return;
                    }
                }
                continue;
            }

            // Waiting sequences exist but cannot be admitted, and running
            // sequences have no scheduled work. Yield rather than spin hot.
            std::thread::yield_now();
        }
    }

    fn handle_request(&mut self, request: Request) {
        match request {
            Request::Generate(job) => {
                if self.scheduler.contains(job.seq) {
                    let _ = job.tx.send(TokenMsg::Error {
                        message: format!("duplicate sequence id {}", job.seq),
                    });
                    return;
                }
                if job.prompt.is_empty() {
                    let _ = job.tx.send(TokenMsg::Error {
                        message: "prompt must not be empty".to_string(),
                    });
                    let _ = job.tx.send(TokenMsg::Done {
                        reason: FinishReason::Error,
                    });
                    return;
                }
                let mut drafter = LookupDrafter::new(DRAFT_NGRAM);
                drafter.observe(&job.prompt);
                self.drafters.insert(job.seq, drafter);
                self.executor.prepare_sequence(job.seq, &job.prompt);
                let seq = Seq::new(
                    job.seq,
                    job.prompt,
                    job.max_tokens,
                    job.temperature,
                    job.top_p,
                    job.tx,
                    job.spec,
                );
                self.scheduler.submit(seq);
            }
            Request::Abort { seq } => {
                self.drafters.remove(&seq);
                if let Some(seq) = self.scheduler.abort(seq, &mut self.block_manager) {
                    let _ = seq.tx.send(TokenMsg::Done {
                        reason: FinishReason::Aborted,
                    });
                }
            }
            Request::Commit {
                session,
                parent,
                label,
                seq,
                reply,
            } => {
                let tokens = match self.scheduler.seq(seq) {
                    Some(seq) => {
                        let mut tokens = seq.prompt.clone();
                        tokens.extend_from_slice(&seq.out);
                        tokens
                    }
                    None => {
                        let _ = reply.send(Err(format!("sequence {seq} is not resident")));
                        return;
                    }
                };
                let result = self.sessions.commit(
                    &session,
                    parent,
                    &label,
                    seq,
                    tokens,
                    &mut self.block_manager,
                );
                let _ = reply.send(result);
            }
            Request::Checkout { node, job } => {
                if self.scheduler.contains(job.seq) {
                    let _ = job.tx.send(TokenMsg::Error {
                        message: format!("duplicate sequence id {}", job.seq),
                    });
                    return;
                }
                match self
                    .sessions
                    .checkout(node, job.seq, &mut self.block_manager)
                {
                    Ok(snapshot) => {
                        let mut drafter = LookupDrafter::new(DRAFT_NGRAM);
                        drafter.observe(&snapshot.tokens);
                        self.drafters.insert(job.seq, drafter);
                        self.executor.prepare_sequence(job.seq, &snapshot.tokens);
                        let mut seq = Seq::new(
                            job.seq,
                            snapshot.tokens,
                            job.max_tokens,
                            job.temperature,
                            job.top_p,
                            job.tx,
                            job.spec,
                        );
                        seq.computed = snapshot.computed;
                        seq.attached = true;
                        self.scheduler.submit_checkout(seq);
                    }
                    Err(error) => {
                        let _ = job.tx.send(TokenMsg::Error { message: error });
                        let _ = job.tx.send(TokenMsg::Done {
                            reason: FinishReason::Error,
                        });
                    }
                }
            }
            Request::Forget { node, reply } => {
                let result = self.sessions.forget(node, &mut self.block_manager);
                let _ = reply.send(result);
            }
            Request::SessionTree { session, reply } => {
                let _ = reply.send(self.sessions.tree(&session));
            }
        }
    }

    fn execute_batch(&mut self, batch: &Batch) -> usize {
        let mut produced = 0usize;
        for chunk in &batch.prefill {
            produced += self.run_prefill(chunk);
        }
        for seq in &batch.first_token {
            produced += self.emit_first_token(*seq);
        }
        for seq in &batch.decode {
            produced += self.run_decode(*seq);
        }
        self.apply_pending_copies();
        produced
    }

    fn run_prefill(&mut self, chunk: &PrefillChunk) -> usize {
        let seq_id = chunk.seq;
        let (tokens, temperature, top_p) = {
            let Some(seq) = self.scheduler.seq(seq_id) else {
                return 0;
            };
            let end = chunk.start + chunk.len;
            if end > seq.prompt.len() {
                return 0;
            }
            (
                seq.prompt[chunk.start..end].to_vec(),
                seq.temperature,
                seq.top_p,
            )
        };

        let mut logits = match self.executor.forward(
            &mut self.block_manager,
            seq_id,
            &tokens,
            chunk.start,
            false,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                self.fail_seq(seq_id, format!("prefill forward failed: {err}"));
                return 0;
            }
        };
        self.apply_pending_copies();
        self.prefill_tokens += chunk.len as u64;
        // The forward pass wrote these slots; only now may a future trie hit
        // skip their compute.
        self.block_manager
            .mark_kv_valid(seq_id, chunk.start, chunk.len);

        let (prompt_done, key) = {
            let Some(seq) = self.scheduler.seq_mut(seq_id) else {
                return 0;
            };
            seq.computed += chunk.len;
            (seq.computed == seq.prompt.len(), prompt_key(&seq.prompt))
        };
        if !prompt_done {
            return 0;
        }

        let last_logits = logits.pop().unwrap_or_default();
        self.cache_prefill_logits(key, last_logits.clone());
        self.emit_sampled_token(seq_id, last_logits, temperature, top_p)
    }

    /// Emit the first token for a sequence whose prompt is already fully valid.
    /// Uses measured logits from the earlier prefill; if they are absent (for
    /// example a partially pre-filled commit), it falls back to a one-token
    /// forward and records that cost honestly.
    fn emit_first_token(&mut self, seq_id: SeqId) -> usize {
        let (prompt, temperature, top_p, key) = {
            let Some(seq) = self.scheduler.seq(seq_id) else {
                return 0;
            };
            if seq.prompt.is_empty() {
                return 0;
            }
            (
                seq.prompt.clone(),
                seq.temperature,
                seq.top_p,
                prompt_key(&seq.prompt),
            )
        };
        let logits = match self.prefill_logits.get(&key).cloned() {
            Some(logits) => logits,
            None => {
                let start = prompt.len() - 1;
                let mut rows = match self.executor.forward(
                    &mut self.block_manager,
                    seq_id,
                    &prompt[start..],
                    start,
                    false,
                ) {
                    Ok(rows) => rows,
                    Err(err) => {
                        self.fail_seq(seq_id, format!("first-token forward failed: {err}"));
                        return 0;
                    }
                };
                self.apply_pending_copies();
                self.prefill_tokens += 1;
                let row = rows.pop().unwrap_or_default();
                self.cache_prefill_logits(key, row.clone());
                row
            }
        };
        self.emit_sampled_token(seq_id, logits, temperature, top_p)
    }

    fn cache_prefill_logits(&mut self, key: u64, logits: Vec<f32>) {
        if self.prefill_logits.contains_key(&key) {
            return;
        }
        if self.prefill_logits_order.len() >= PREFILL_LOGIT_CACHE_ENTRIES
            && let Some(evicted) = self.prefill_logits_order.pop_front()
        {
            self.prefill_logits.remove(&evicted);
        }
        self.prefill_logits.insert(key, logits);
        self.prefill_logits_order.push_back(key);
    }

    /// Shared sampling/emission tail for prefill and cached-first-token paths.
    fn emit_sampled_token(
        &mut self,
        seq_id: SeqId,
        last_logits: Vec<f32>,
        temperature: f32,
        top_p: f32,
    ) -> usize {
        let mut rng = SpecRng::new(
            self.scheduler
                .seq(seq_id)
                .map(|seq| seq.rng_state)
                .unwrap_or(1),
        );
        let token = sample_from_logits(&last_logits, temperature, top_p, &mut rng);
        if let Some(seq) = self.scheduler.seq_mut(seq_id) {
            seq.rng_state = rng.state();
        }
        if let Err(err) = self.block_manager.append_tokens(seq_id, 1) {
            self.fail_seq(seq_id, format!("KV allocation failed: {err}"));
            return 0;
        }
        self.apply_pending_copies();
        let eos = self.emit_token(seq_id, token);
        let maxed = self
            .scheduler
            .seq(seq_id)
            .map(|seq| seq.out.len() >= seq.max_tokens)
            .unwrap_or(false);
        let reason = if eos {
            Some(FinishReason::StopTokenEncountered)
        } else if maxed {
            Some(FinishReason::MaxTokensReached)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.finish_sequence(seq_id, reason);
        }
        1
    }

    fn run_decode(&mut self, seq_id: SeqId) -> usize {
        let (last_token, start, temperature, top_p, spec, prompt_len, out_len) = {
            let Some(seq) = self.scheduler.seq(seq_id) else {
                return 0;
            };
            if !seq.ready_for_decode() {
                return 0;
            }
            let Some(&last) = seq.out.last() else {
                return 0;
            };
            (
                last,
                seq.computed,
                seq.temperature,
                seq.top_p,
                seq.spec,
                seq.prompt.len(),
                seq.out.len(),
            )
        };

        let selected_k = if spec { self.bandit.select() } else { 0 };
        let mut drafts = if selected_k > 0 {
            self.drafters
                .get(&seq_id)
                .map(|drafter| drafter.propose(selected_k))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Drafts must fit the per-sequence page-table capacity. The pool itself
        // may still reject the append; that falls back to a plain decode below.
        let capacity = self.block_manager.max_blocks_per_seq * self.block_manager.block_size;
        let available = capacity.saturating_sub(prompt_len + out_len);
        if available <= 1 {
            drafts.clear();
        } else {
            drafts.truncate(available - 1);
        }

        let mut input = Vec::with_capacity(drafts.len() + 1);
        input.push(last_token);
        input.extend_from_slice(&drafts);
        loop {
            match self.block_manager.append_tokens(seq_id, input.len()) {
                Ok(()) => break,
                Err(MemoryError::OutOfMemory) | Err(MemoryError::CapacityExceeded)
                    if input.len() > 1 =>
                {
                    input.truncate(1);
                }
                Err(err) => {
                    self.fail_seq(seq_id, format!("KV allocation failed: {err}"));
                    return 0;
                }
            }
        }
        let drafts = &input[1..];

        let started = Instant::now();
        let logits =
            match self
                .executor
                .forward(&mut self.block_manager, seq_id, &input, start, true)
            {
                Ok(logits) => logits,
                Err(err) => {
                    self.fail_seq(seq_id, format!("decode forward failed: {err}"));
                    return 0;
                }
            };
        self.apply_pending_copies();

        let mut rng = SpecRng::new(
            self.scheduler
                .seq(seq_id)
                .map(|seq| seq.rng_state)
                .unwrap_or(1),
        );
        let verification = if temperature <= 1e-3 {
            verify_greedy(&logits, drafts)
        } else {
            let target: Vec<Vec<f64>> = logits
                .iter()
                .map(|row| probabilities(row, temperature, top_p))
                .collect();
            verify_sampled(&target, drafts, &mut rng)
        };
        let verification = match verification {
            Ok(verification) => verification,
            Err(err) => {
                self.fail_seq(seq_id, format!("speculative verification failed: {err}"));
                return 0;
            }
        };
        let elapsed_ms = (started.elapsed().as_secs_f64() * 1000.0).max(1e-6);

        if spec {
            self.spec_stats
                .record_step(drafts.len(), verification.accepted);
            let reward = verification.accepted as f64 / elapsed_ms;
            self.bandit.update(selected_k, reward);
            self.spec_stats.adaptive_k = self.bandit.current_k();
        }

        let remaining = self
            .scheduler
            .seq(seq_id)
            .map(|seq| seq.max_tokens.saturating_sub(seq.out.len()))
            .unwrap_or(0);
        let mut emitted = 0usize;
        let mut eos = false;
        for &token in &verification.tokens {
            if emitted >= remaining {
                break;
            }
            eos = self.emit_token(seq_id, token);
            emitted += 1;
            if eos {
                break;
            }
        }
        if emitted > 0
            && let Some(drafter) = self.drafters.get_mut(&seq_id)
        {
            drafter.observe(&verification.tokens[..emitted]);
        }

        if let Err(err) = self
            .block_manager
            .rollback(seq_id, prompt_len + out_len + emitted)
        {
            self.fail_seq(seq_id, format!("speculative rollback failed: {err}"));
            return 0;
        }
        let kv_valid = start + 1 + verification.accepted;
        let logical_after = prompt_len + out_len + emitted;
        if let Some(seq) = self.scheduler.seq_mut(seq_id) {
            seq.computed = kv_valid.min(logical_after.saturating_sub(1));
            seq.rng_state = rng.state();
        }

        let maxed = self
            .scheduler
            .seq(seq_id)
            .map(|seq| seq.out.len() >= seq.max_tokens)
            .unwrap_or(false);
        let reason = if eos {
            Some(FinishReason::StopTokenEncountered)
        } else if maxed {
            Some(FinishReason::MaxTokensReached)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.finish_sequence(seq_id, reason);
        }
        emitted
    }

    /// Push one committed token into the sequence and notify the client.
    /// Returns whether the token is an end-of-sequence marker. KV allocation is
    /// the caller's responsibility because speculative steps batch it.
    fn emit_token(&mut self, seq_id: SeqId, token: u32) -> bool {
        let now = Instant::now();
        // Capture the real page-table blocks at emission time. The HTTP relay
        // may drain this message after the sequence finishes and its snapshot
        // entry disappears, so block ids must travel with the token.
        let block_ids = self.block_manager.blocks_of(seq_id);
        let Some(seq) = self.scheduler.seq_mut(seq_id) else {
            return false;
        };
        if seq.first_token_at.is_none() {
            seq.first_token_at = Some(now);
        }
        seq.last_token_at = Some(now);
        seq.out.push(token);
        let step = seq.out.len();
        self.total_tokens += 1;

        let text = self.tokenizer.decode(&[token]);
        let is_eos = self.tokenizer.eos_token_ids().contains(&token);
        let _ = seq.tx.send(TokenMsg::Token {
            text,
            id: token,
            step,
            block_ids,
        });
        is_eos
    }

    fn finish_sequence(&mut self, seq_id: SeqId, reason: FinishReason) {
        if let Some(seq) = self.scheduler.finish(seq_id, &mut self.block_manager) {
            if let Some(ttft) = seq.ttft() {
                self.ttfts.push_back(ttft);
                while self.ttfts.len() > 512 {
                    self.ttfts.pop_front();
                }
            }
            let _ = seq.tx.send(TokenMsg::Done { reason });
        }
        self.drafters.remove(&seq_id);
    }

    fn fail_seq(&mut self, seq_id: SeqId, message: String) {
        self.drafters.remove(&seq_id);
        if let Some(seq) = self.scheduler.abort(seq_id, &mut self.block_manager) {
            let _ = seq.tx.send(TokenMsg::Error { message });
            let _ = seq.tx.send(TokenMsg::Done {
                reason: FinishReason::Error,
            });
        }
    }

    fn apply_pending_copies(&mut self) {
        for (src, dst) in self.block_manager.take_pending_copies() {
            if let Err(err) = self.executor.copy_block(src, dst) {
                tracing::error!("KV copy-on-write from block {src} to {dst} failed: {err}");
            }
        }
    }

    fn publish(&mut self) {
        let now = Instant::now();
        self.throughput.prune(now);
        let tps = self.throughput.tokens_per_sec(now);
        let stats = self.block_manager.stats();
        let io_stats = self.block_manager.io_stats();
        let mut ttfts: Vec<f64> = self.ttfts.iter().copied().collect();
        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let ttft_p50 = if ttfts.is_empty() {
            0.0
        } else {
            ttfts[ttfts.len() / 2]
        };

        let active_sequence_ids = self.scheduler.active_sequence_ids();
        let mut seq_blocks = HashMap::new();
        for id in &active_sequence_ids {
            seq_blocks.insert(*id, self.block_manager.blocks_of(*id));
        }
        let allocated = self.block_manager.allocated();
        let total = self.block_manager.total_blocks;
        let ratio = self.block_manager.memory_usage_ratio();
        let acceptance = self.spec_stats.acceptance();
        let event_tick = self.block_manager.event_tick();
        let inspect = self.block_manager.inspect();
        if let Ok(mut recorder) = self.recorder.lock() {
            recorder.maybe_keyframe(event_tick, inspect.clone());
        }

        let snapshot = Snapshot {
            executor: self.executor.name().to_string(),
            attention_backend: self.executor.attention_backend().to_string(),
            model: self.model_id.clone(),
            total_blocks: total,
            blocks: total,
            block_size: self.block_manager.block_size,
            allocated_blocks: allocated,
            active_blocks: self.block_manager.active_blocks(),
            cached_blocks: self.block_manager.cached_blocks(),
            free_blocks: self.block_manager.free_blocks_count(),
            memory_usage_ratio: ratio,
            memory_usage_percent: ratio * 100.0,
            running_sequences: self.scheduler.running_count(),
            waiting_sequences: self.scheduler.waiting_count(),
            prefix_cache_hit_ratio: self.block_manager.prefix_cache_hit_ratio(),
            prefix_cache_lookups: stats.lookups,
            prefix_cache_hits: stats.hits,
            prefix_skip_tokens: self.scheduler.skipped_prefill_tokens,
            prefill_tokens: self.prefill_tokens,
            speculative_acceptance_rate: acceptance,
            acceptance,
            speculative_proposed: self.spec_stats.proposed,
            speculative_accepted: self.spec_stats.accepted,
            adaptive_k: self.spec_stats.adaptive_k,
            tokens_per_sec: tps,
            ttft_p50_ms: ttft_p50,
            evictions: stats.evictions,
            page_faults: stats.page_faults,
            cow_copies: stats.cow_copies,
            swapped_out: io_stats.swapped_out,
            swapped_in: io_stats.swapped_in,
            swap_bytes_out: io_stats.swap_bytes_out,
            swap_bytes_in: io_stats.swap_bytes_in,
            disk_spills: io_stats.disk_spills,
            disk_loads: io_stats.disk_loads,
            preemptions: self.scheduler.preemptions,
            total_tokens_generated: self.total_tokens,
            event_tick,
            active_sequence_ids,
            seq_blocks,
            block_inspect_info: inspect,
        };
        let mut guard = self
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = snapshot;
    }
}
