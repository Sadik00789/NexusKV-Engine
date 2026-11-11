// src/engine/actor.rs
use super::scheduler::{Batch, FinishReason, PrefillChunk, Scheduler, Seq};
use super::speculative::SpecStats;
use crate::memory::{BlockId, BlockInspectInfo, BlockSpaceManager, SeqId};
use crate::model::{Executor, Tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

#[derive(Debug, Clone)]
pub enum TokenMsg {
    Token { text: String, id: u32, step: usize },
    Done { reason: FinishReason },
    Error { message: String },
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

#[derive(Debug)]
pub enum Request {
    Generate(GenerateJob),
    Abort { seq: SeqId },
}

/// Telemetry published by the actor after every iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub executor: String,
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
    pub speculative_acceptance_rate: f64,
    pub acceptance: f64,
    pub adaptive_k: usize,
    pub tokens_per_sec: f64,
    pub ttft_p50_ms: f64,
    pub evictions: u64,
    pub page_faults: u64,
    pub cow_copies: u64,
    pub preemptions: u64,
    pub total_tokens_generated: u64,
    pub active_sequence_ids: Vec<SeqId>,
    pub seq_blocks: HashMap<SeqId, Vec<BlockId>>,
    pub block_inspect_info: Vec<BlockInspectInfo>,
}

impl Snapshot {
    pub fn initial(model: &str, executor: &str, total_blocks: usize, block_size: usize) -> Self {
        Self {
            executor: executor.to_string(),
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
            speculative_acceptance_rate: 0.0,
            acceptance: 0.0,
            adaptive_k: 0,
            tokens_per_sec: 0.0,
            ttft_p50_ms: 0.0,
            evictions: 0,
            page_faults: 0,
            cow_copies: 0,
            preemptions: 0,
            total_tokens_generated: 0,
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

pub struct EngineActor {
    rx: UnboundedReceiver<Request>,
    scheduler: Scheduler,
    block_manager: BlockSpaceManager,
    executor: Box<dyn Executor>,
    tokenizer: Arc<dyn Tokenizer>,
    snapshot: Arc<Mutex<Snapshot>>,
    spec_stats: SpecStats,
    throughput: ThroughputWindow,
    ttfts: VecDeque<f64>,
    total_tokens: u64,
    model_id: String,
}

impl EngineActor {
    pub fn new(
        rx: UnboundedReceiver<Request>,
        scheduler: Scheduler,
        block_manager: BlockSpaceManager,
        executor: Box<dyn Executor>,
        tokenizer: Arc<dyn Tokenizer>,
        snapshot: Arc<Mutex<Snapshot>>,
        model_id: String,
    ) -> Self {
        Self {
            rx,
            scheduler,
            block_manager,
            executor,
            tokenizer,
            snapshot,
            spec_stats: SpecStats::default(),
            throughput: ThroughputWindow::new(),
            ttfts: VecDeque::new(),
            total_tokens: 0,
            model_id,
        }
    }

    pub fn run(mut self) {
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
                if let Some(seq) = self.scheduler.abort(seq, &mut self.block_manager) {
                    let _ = seq.tx.send(TokenMsg::Done {
                        reason: FinishReason::Aborted,
                    });
                }
            }
        }
    }

    fn execute_batch(&mut self, batch: &Batch) -> usize {
        let mut produced = 0usize;
        for chunk in &batch.prefill {
            produced += self.run_prefill(chunk);
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

        let prompt_done = {
            let Some(seq) = self.scheduler.seq_mut(seq_id) else {
                return 0;
            };
            seq.computed += chunk.len;
            seq.computed == seq.prompt.len()
        };
        if !prompt_done {
            return 0;
        }

        let last_logits = logits.pop().unwrap_or_default();
        let mut rng = self
            .scheduler
            .seq(seq_id)
            .map(|seq| seq.rng_state)
            .unwrap_or(1);
        let token = sample_token(&last_logits, temperature, top_p, &mut rng);
        if let Some(seq) = self.scheduler.seq_mut(seq_id) {
            seq.rng_state = rng;
        }
        self.emit_token(seq_id, token);
        1
    }

    fn run_decode(&mut self, seq_id: SeqId) -> usize {
        let (last_token, start, temperature, top_p) = {
            let Some(seq) = self.scheduler.seq(seq_id) else {
                return 0;
            };
            if !seq.ready_for_decode() {
                return 0;
            }
            let Some(&last) = seq.out.last() else {
                return 0;
            };
            (last, seq.computed, seq.temperature, seq.top_p)
        };

        let mut logits = match self.executor.forward(
            &mut self.block_manager,
            seq_id,
            &[last_token],
            start,
            false,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                self.fail_seq(seq_id, format!("decode forward failed: {err}"));
                return 0;
            }
        };
        self.apply_pending_copies();

        let mut rng = {
            let Some(seq) = self.scheduler.seq_mut(seq_id) else {
                return 0;
            };
            seq.computed += 1;
            seq.rng_state
        };
        let last_logits = logits.pop().unwrap_or_default();
        let token = sample_token(&last_logits, temperature, top_p, &mut rng);
        if let Some(seq) = self.scheduler.seq_mut(seq_id) {
            seq.rng_state = rng;
        }
        self.emit_token(seq_id, token);
        1
    }

    fn emit_token(&mut self, seq_id: SeqId, token: u32) {
        let now = Instant::now();
        let (step, max_tokens) = {
            let Some(seq) = self.scheduler.seq_mut(seq_id) else {
                return;
            };
            if seq.first_token_at.is_none() {
                seq.first_token_at = Some(now);
            }
            seq.last_token_at = Some(now);
            seq.out.push(token);
            (seq.out.len(), seq.max_tokens)
        };

        if let Err(err) = self.block_manager.append_tokens(seq_id, 1) {
            self.fail_seq(seq_id, format!("KV allocation failed: {err}"));
            return;
        }
        self.apply_pending_copies();
        self.total_tokens += 1;

        let text = self.tokenizer.decode(&[token]);
        let is_eos = self.tokenizer.eos_token_ids().contains(&token);
        let finished_reason = if is_eos {
            Some(FinishReason::StopTokenEncountered)
        } else if step >= max_tokens {
            Some(FinishReason::MaxTokensReached)
        } else {
            None
        };

        if let Some(seq) = self.scheduler.seq(seq_id) {
            let _ = seq.tx.send(TokenMsg::Token {
                text,
                id: token,
                step,
            });
        }

        if let Some(reason) = finished_reason
            && let Some(seq) = self.scheduler.finish(seq_id, &mut self.block_manager)
        {
            if let Some(ttft) = seq.ttft() {
                self.ttfts.push_back(ttft);
                while self.ttfts.len() > 512 {
                    self.ttfts.pop_front();
                }
            }
            let _ = seq.tx.send(TokenMsg::Done { reason });
        }
    }

    fn fail_seq(&mut self, seq_id: SeqId, message: String) {
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

        let snapshot = Snapshot {
            executor: self.executor.name().to_string(),
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
            speculative_acceptance_rate: acceptance,
            acceptance,
            adaptive_k: self.spec_stats.adaptive_k,
            tokens_per_sec: tps,
            ttft_p50_ms: ttft_p50,
            evictions: stats.evictions,
            page_faults: stats.page_faults,
            cow_copies: stats.cow_copies,
            preemptions: self.scheduler.preemptions,
            total_tokens_generated: self.total_tokens,
            active_sequence_ids,
            seq_blocks,
            block_inspect_info: self.block_manager.inspect(),
        };
        let mut guard = self
            .snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = snapshot;
    }
}

fn sample_token(logits: &[f32], temperature: f32, top_p: f32, rng: &mut u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 1e-3 {
        let mut best = 0usize;
        for index in 1..logits.len() {
            if logits[index] > logits[best] {
                best = index;
            }
        }
        return best as u32;
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f64> = logits
        .iter()
        .map(|logit| ((*logit - max) as f64 / temperature as f64).exp())
        .collect();
    let sum: f64 = probabilities.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return 0;
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }

    let mut ranked: Vec<(usize, f64)> = probabilities.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_p = top_p.clamp(1e-6, 1.0) as f64;
    let mut cumulative = 0.0;
    let mut cutoff = ranked.len();
    for (index, (_, probability)) in ranked.iter().enumerate() {
        cumulative += probability;
        if cumulative >= top_p {
            cutoff = index + 1;
            break;
        }
    }
    let total: f64 = ranked[..cutoff]
        .iter()
        .map(|(_, probability)| probability)
        .sum();
    let sample = next_unit(rng) * total;
    let mut accumulated = 0.0;
    for (index, probability) in &ranked[..cutoff] {
        accumulated += probability;
        if sample < accumulated {
            return *index as u32;
        }
    }
    ranked[cutoff.saturating_sub(1)].0 as u32
}

fn next_unit(rng: &mut u64) -> f64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*rng >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_is_argmax_and_deterministic() {
        let logits = vec![0.1, 0.9, 0.3, 0.9];
        let mut rng = 7u64;
        assert_eq!(sample_token(&logits, 0.0, 1.0, &mut rng), 1);
        assert_eq!(sample_token(&logits, 0.0, 1.0, &mut rng), 1);
    }

    #[test]
    fn sampled_tokens_respect_top_p_support() {
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let mut rng = 99u64;
        for _ in 0..100 {
            assert_eq!(sample_token(&logits, 1.0, 0.5, &mut rng), 0);
        }
    }
}
