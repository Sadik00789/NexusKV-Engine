// src/model/mod.rs
use crate::memory::{BlockSpaceManager, SeqId};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    #[error("model backend error: {0}")]
    Backend(String),
    #[error("invalid forward request: {0}")]
    Invalid(String),
}

/// One sequence inside a batched decode step.
#[derive(Debug, Clone, Copy)]
pub struct DecodeInput<'a> {
    pub seq: SeqId,
    /// Input tokens for this sequence, at absolute positions
    /// `start .. start + tokens.len()`. Decode passes one token plus any
    /// speculative drafts.
    pub tokens: &'a [u32],
    /// Absolute position of `tokens[0]`.
    pub start: usize,
}

/// A model executor writes KV for `tokens` at absolute positions
/// `[start, start + tokens.len())` into the paged pool and returns logits.
///
/// `all_logits == false` returns exactly one vector (the last position).
/// `all_logits == true` returns one vector per input position, which is what
/// speculative verification needs.
pub trait Executor: Send {
    fn forward(
        &mut self,
        block_manager: &mut BlockSpaceManager,
        seq: SeqId,
        tokens: &[u32],
        start: usize,
        all_logits: bool,
    ) -> Result<Vec<Vec<f32>>, ModelError>;

    /// Batched decode: one forward pass over the model for several sequences.
    ///
    /// Returns one logits row list per input, in input order, aligned with each
    /// input's `tokens` (one vocabulary-sized vector per token), so speculative
    /// verification can consume it directly.
    ///
    /// The default implementation keeps one forward per sequence, so executors
    /// that cannot batch (the mock) stay correct. Real executors override it
    /// with a fused pass and report `is_batched_decode() == true`.
    fn forward_decode_batch(
        &mut self,
        block_manager: &mut BlockSpaceManager,
        inputs: &[DecodeInput<'_>],
    ) -> Result<Vec<Vec<Vec<f32>>>, ModelError> {
        let mut rows = Vec::with_capacity(inputs.len());
        for input in inputs {
            rows.push(self.forward(block_manager, input.seq, input.tokens, input.start, true)?);
        }
        Ok(rows)
    }

    /// Whether `forward_decode_batch` performs one fused forward pass. Telemetry
    /// reports this so a per-sequence loop is never presented as batching.
    fn is_batched_decode(&self) -> bool {
        false
    }

    /// Called once when a sequence is admitted. Executors that derive state
    /// from the prompt (the mock's deterministic offset) use the full prompt
    /// here, so KV-valid prefix skipping cannot change the continuation rule.
    fn prepare_sequence(&mut self, _seq: SeqId, _prompt: &[u32]) {}

    /// Copy one physical KV block. Called when the allocator reports CoW.
    fn copy_block(&mut self, _src: usize, _dst: usize) -> Result<(), ModelError> {
        Ok(())
    }

    fn vocab_size(&self) -> usize;

    fn is_mock(&self) -> bool;

    fn name(&self) -> &'static str;

    /// Honest attention implementation label for telemetry.
    fn attention_backend(&self) -> &'static str {
        "gather"
    }
}

/// Tokenization and detokenization shared by the HTTP layer and the actor.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn encode_chat(&self, messages: &[(String, String)]) -> Vec<u32>;
    fn eos_token_ids(&self) -> &[u32];
}

/// Byte-level tokenizer used by the mock executor. It is explicitly labeled
/// as a mock tokenizer in `/v1/models`; it is not a model vocabulary.
pub struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| match id {
                32..=126 => char::from_u32(id).unwrap_or('?'),
                _ => '·',
            })
            .collect()
    }

    fn encode_chat(&self, messages: &[(String, String)]) -> Vec<u32> {
        let mut text = String::new();
        for (role, content) in messages {
            text.push_str(role);
            text.push_str(": ");
            text.push_str(content);
            text.push('\n');
        }
        text.push_str("assistant: ");
        self.encode(&text)
    }

    fn eos_token_ids(&self) -> &[u32] {
        &[]
    }
}

/// Deterministic echo model used without weights.
///
/// It remembers the token history per sequence. The continuation rule is
/// `next = (3*last + 7*previous + 1 + offset(seq, prompt)) % vocab`, where the
/// offset is fixed per sequence from the first prompt chunk it sees. The rule
/// depends only on the last two tokens and the prompt-derived offset, so an n-gram
/// lookup drafter (phase 4) sees consistent continuations, and speculative
/// verification can be proven token-identical to plain greedy decoding.
pub struct MockExecutor {
    vocab: usize,
    history: HashMap<SeqId, Vec<u32>>,
    offsets: HashMap<SeqId, u32>,
}

impl MockExecutor {
    pub fn new() -> Self {
        Self::with_vocab(256)
    }

    pub fn with_vocab(vocab: usize) -> Self {
        assert!(vocab > 1);
        Self {
            vocab,
            history: HashMap::new(),
            offsets: HashMap::new(),
        }
    }

    fn prompt_offset(prompt: &[u32]) -> u32 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &token in prompt {
            hash = (hash ^ u64::from(token)).wrapping_mul(0x1000_0000_01b3);
        }
        (hash as u32) % 256
    }

    fn predict(&self, seq: SeqId, previous: u32, last: u32) -> u32 {
        let offset = self.offsets.get(&seq).copied().unwrap_or(0);
        last.wrapping_mul(3)
            .wrapping_add(previous.wrapping_mul(7))
            .wrapping_add(1)
            .wrapping_add(offset)
            % self.vocab as u32
    }

    fn logits_for(&self, seq: SeqId, previous: u32, last: u32) -> Vec<f32> {
        let predicted = self.predict(seq, previous, last) as usize;
        let rest = 0.05f32 / (self.vocab - 1) as f32;
        let mut logits = vec![rest; self.vocab];
        logits[predicted] = 0.95;
        logits
    }

    fn record(&mut self, seq: SeqId, position: usize, token: u32) {
        let history = self.history.entry(seq).or_default();
        if history.len() <= position {
            history.resize(position + 1, 0);
        }
        history[position] = token;
    }

    pub fn history_len(&self, seq: SeqId) -> usize {
        self.history.get(&seq).map(Vec::len).unwrap_or(0)
    }
}

impl Default for MockExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for MockExecutor {
    fn prepare_sequence(&mut self, seq: SeqId, prompt: &[u32]) {
        self.offsets
            .entry(seq)
            .or_insert_with(|| Self::prompt_offset(prompt));
        // Seed the history for prompt positions that a KV-valid hit may skip.
        // Decode reads position-1, so an unseeded history would change the
        // deterministic continuation even though the KV bytes are correct.
        self.history.insert(seq, prompt.to_vec());
    }

    fn forward(
        &mut self,
        _block_manager: &mut BlockSpaceManager,
        seq: SeqId,
        tokens: &[u32],
        start: usize,
        all_logits: bool,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Invalid(
                "forward requires at least one token".to_string(),
            ));
        }
        if !self.offsets.contains_key(&seq) {
            // Direct executor users (unit tests) may call forward without
            // prepare_sequence; fall back to the first chunk as before.
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for &token in tokens {
                hash = (hash ^ u64::from(token)).wrapping_mul(0x1000_0000_01b3);
            }
            self.offsets.insert(seq, (hash as u32) % self.vocab as u32);
        }

        let mut logits = Vec::with_capacity(tokens.len());
        for (offset, &token) in tokens.iter().enumerate() {
            if token as usize >= self.vocab {
                return Err(ModelError::Invalid(format!(
                    "token id {token} outside mock vocab {}",
                    self.vocab
                )));
            }
            let position = start + offset;
            self.record(seq, position, token);
            let previous = if position > 0 {
                self.history
                    .get(&seq)
                    .and_then(|history| history.get(position - 1))
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            logits.push(self.logits_for(seq, previous, token));
        }
        if all_logits {
            Ok(logits)
        } else {
            Ok(vec![logits.pop().expect("non-empty logits")])
        }
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn is_mock(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn attention_backend(&self) -> &'static str {
        "mock"
    }
}

#[cfg(feature = "candle-model")]
pub mod qwen;
