// src/engine/speculative.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Candidate draft lengths explored by the adaptive bandit.
pub const DRAFT_ARMS: [usize; 5] = [0, 2, 4, 8, 16];
/// N-gram length used by the prompt-lookup drafter.
pub const DRAFT_NGRAM: usize = 3;

/// Honest speculative-decoding counters.
///
/// With no drafter proposals the acceptance rate is exactly 0.0. There is
/// deliberately no default estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecStats {
    pub proposed: u64,
    pub accepted: u64,
    pub speculative_steps: u64,
    pub adaptive_k: usize,
}

impl SpecStats {
    pub fn acceptance(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    pub fn record_step(&mut self, proposed: usize, accepted: usize) {
        self.proposed += proposed as u64;
        self.accepted += accepted as u64;
        self.speculative_steps += 1;
    }
}

/// Deterministic LCG shared by all engines in this project. Tests use the same
/// constants, so results are reproducible across runs and platforms.
#[derive(Debug, Clone)]
pub struct SpecRng(u64);

impl SpecRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn state(&self) -> u64 {
        self.0
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    best as u32
}

/// Convert logits into a target distribution, applying temperature and
/// nucleus (top-p) truncation. Greedy decoding is represented as a one-hot
/// distribution, which keeps the verification code uniform.
pub fn probabilities(logits: &[f32], temperature: f32, top_p: f32) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    if temperature <= 1e-3 {
        let mut one_hot = vec![0.0; logits.len()];
        one_hot[argmax(logits) as usize] = 1.0;
        return one_hot;
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    let mut values: Vec<f64> = logits
        .iter()
        .map(|logit| ((*logit - max) as f64 / temperature as f64).exp())
        .collect();
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    for value in &mut values {
        *value /= sum;
    }

    let mut ranked: Vec<usize> = (0..values.len()).collect();
    ranked.sort_by(|a, b| {
        values[*b]
            .partial_cmp(&values[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_p = top_p.clamp(1e-6, 1.0) as f64;
    let mut cumulative = 0.0;
    let mut keep = ranked.len();
    for (position, index) in ranked.iter().enumerate() {
        cumulative += values[*index];
        if cumulative >= top_p {
            keep = position + 1;
            break;
        }
    }
    for (position, index) in ranked.iter().enumerate() {
        if position >= keep {
            values[*index] = 0.0;
        }
    }
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / values.len() as f64; values.len()];
    }
    for value in &mut values {
        *value /= total;
    }
    values
}

/// Sample one token from a normalized probability vector.
pub fn sample_probabilities(probabilities: &[f64], rng: &mut SpecRng) -> u32 {
    if probabilities.is_empty() {
        return 0;
    }
    let mut sample = rng.unit();
    for (index, probability) in probabilities.iter().enumerate() {
        sample -= *probability;
        if sample < 0.0 {
            return index as u32;
        }
    }
    probabilities
        .iter()
        .rposition(|probability| *probability > 0.0)
        .unwrap_or(0) as u32
}

/// Sample from logits with temperature/top-p handling. Temperature <= 1e-3 is
/// the deterministic greedy path.
pub fn sample_from_logits(logits: &[f32], temperature: f32, top_p: f32, rng: &mut SpecRng) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 1e-3 {
        return argmax(logits);
    }
    sample_probabilities(&probabilities(logits, temperature, top_p), rng)
}

/// Prompt-lookup drafter. It indexes every observed n-gram and proposes the
/// continuation that followed the most recent occurrence of the current suffix.
#[derive(Debug, Clone)]
pub struct LookupDrafter {
    n: usize,
    tokens: Vec<u32>,
    index: HashMap<Vec<u32>, Vec<usize>>,
}

impl LookupDrafter {
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "n-gram length must be positive");
        Self {
            n,
            tokens: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn observe(&mut self, tokens: &[u32]) {
        for &token in tokens {
            let position = self.tokens.len();
            if position >= self.n {
                let gram = self.tokens[position - self.n..position].to_vec();
                self.index.entry(gram).or_default().push(position);
            }
            self.tokens.push(token);
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Propose up to `k` continuation tokens. Returns an empty vector when no
    /// earlier occurrence of the current n-gram exists.
    pub fn propose(&self, k: usize) -> Vec<u32> {
        if k == 0 || self.tokens.len() < self.n {
            return Vec::new();
        }
        let gram = &self.tokens[self.tokens.len() - self.n..];
        let Some(positions) = self.index.get(gram) else {
            return Vec::new();
        };
        let Some(&position) = positions.last() else {
            return Vec::new();
        };
        let end = (position + k).min(self.tokens.len());
        self.tokens[position..end].to_vec()
    }

    pub fn reset(&mut self, tokens: &[u32]) {
        self.tokens.clear();
        self.index.clear();
        self.observe(tokens);
    }
}

/// UCB1 bandit over draft lengths. The reward is accepted tokens per
/// millisecond of verification work.
#[derive(Debug, Clone)]
pub struct AdaptiveK {
    counts: [u64; DRAFT_ARMS.len()],
    means: [f64; DRAFT_ARMS.len()],
    total: u64,
    last_arm: usize,
}

impl AdaptiveK {
    pub fn new() -> Self {
        Self {
            counts: [0; DRAFT_ARMS.len()],
            means: [0.0; DRAFT_ARMS.len()],
            total: 0,
            last_arm: 0,
        }
    }

    /// Select the next draft length. Untried arms are explored first; after
    /// that, UCB1 maximizes `mean + sqrt(2 ln t / n)`.
    pub fn select(&mut self) -> usize {
        let arm = if let Some(index) = (0..DRAFT_ARMS.len()).find(|index| self.counts[*index] == 0)
        {
            index
        } else {
            let logarithm = (self.total.max(1) as f64).ln();
            let mut best = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for index in 0..DRAFT_ARMS.len() {
                let count = self.counts[index] as f64;
                let score = self.means[index] + (2.0 * logarithm / count).sqrt();
                if score > best_score {
                    best_score = score;
                    best = index;
                }
            }
            best
        };
        self.last_arm = arm;
        DRAFT_ARMS[arm]
    }

    pub fn update(&mut self, k: usize, reward: f64) {
        let Some(index) = DRAFT_ARMS.iter().position(|arm| *arm == k) else {
            return;
        };
        if !reward.is_finite() {
            return;
        }
        self.counts[index] += 1;
        self.total += 1;
        let count = self.counts[index] as f64;
        self.means[index] += (reward - self.means[index]) / count;
    }

    pub fn current_k(&self) -> usize {
        DRAFT_ARMS[self.last_arm]
    }

    pub fn mean_for(&self, k: usize) -> Option<f64> {
        DRAFT_ARMS
            .iter()
            .position(|arm| *arm == k)
            .map(|index| self.means[index])
    }

    pub fn arm_count(&self, k: usize) -> u64 {
        DRAFT_ARMS
            .iter()
            .position(|arm| *arm == k)
            .map(|index| self.counts[index])
            .unwrap_or(0)
    }
}

impl Default for AdaptiveK {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// Accepted drafts plus exactly one correction/bonus token.
    pub tokens: Vec<u32>,
    pub accepted: usize,
    pub all_accepted: bool,
}

/// Greedy verification: accept a draft iff it equals the target argmax.
/// Rejection produces the argmax correction and stops the chain.
pub fn verify_greedy(logits: &[Vec<f32>], drafts: &[u32]) -> Result<Verification, String> {
    if logits.len() != drafts.len() + 1 {
        return Err(format!(
            "verification needs {} logit rows, got {}",
            drafts.len() + 1,
            logits.len()
        ));
    }
    if logits.iter().any(Vec::is_empty) {
        return Err("verification received an empty logit row".to_string());
    }
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    for (index, &draft) in drafts.iter().enumerate() {
        let target = argmax(&logits[index]);
        if draft == target {
            tokens.push(draft);
        } else {
            tokens.push(target);
            return Ok(Verification {
                tokens,
                accepted: index,
                all_accepted: false,
            });
        }
    }
    tokens.push(argmax(&logits[drafts.len()]));
    Ok(Verification {
        tokens,
        accepted: drafts.len(),
        all_accepted: true,
    })
}

/// Distribution-exact rejection sampling for one-hot draft distributions.
///
/// Draft `x` is accepted with probability `p_target(x)`. On rejection the
/// correction token is sampled from the target distribution with `x` removed,
/// and the draft chain stops. When every draft is accepted, a bonus token is
/// sampled from the final target distribution.
pub fn verify_sampled(
    probabilities: &[Vec<f64>],
    drafts: &[u32],
    rng: &mut SpecRng,
) -> Result<Verification, String> {
    if probabilities.len() != drafts.len() + 1 {
        return Err(format!(
            "verification needs {} probability rows, got {}",
            drafts.len() + 1,
            probabilities.len()
        ));
    }
    for (index, draft) in drafts.iter().enumerate() {
        let row = &probabilities[index];
        if *draft as usize >= row.len() {
            return Err(format!(
                "draft {draft} outside target distribution of size {}",
                row.len()
            ));
        }
        let accept = row[*draft as usize].clamp(0.0, 1.0);
        if rng.unit() < accept {
            continue;
        }
        let correction = sample_excluding(row, *draft, rng)
            .ok_or_else(|| "target distribution has no mass after rejection".to_string())?;
        let mut tokens = drafts[..index].to_vec();
        tokens.push(correction);
        return Ok(Verification {
            tokens,
            accepted: index,
            all_accepted: false,
        });
    }

    let mut tokens = drafts.to_vec();
    tokens.push(sample_probabilities(&probabilities[drafts.len()], rng));
    Ok(Verification {
        tokens,
        accepted: drafts.len(),
        all_accepted: true,
    })
}

fn sample_excluding(probabilities: &[f64], excluded: u32, rng: &mut SpecRng) -> Option<u32> {
    let total: f64 = probabilities
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != excluded as usize)
        .map(|(_, probability)| *probability)
        .sum();
    if total <= 0.0 {
        return None;
    }
    let mut sample = rng.unit() * total;
    let mut fallback = None;
    for (index, probability) in probabilities.iter().enumerate() {
        if index == excluded as usize || *probability <= 0.0 {
            continue;
        }
        fallback = Some(index as u32);
        sample -= *probability;
        if sample <= 0.0 {
            return Some(index as u32);
        }
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(seed: u64, count: usize, p: &[f64], draft: u32) -> Vec<usize> {
        let mut rng = SpecRng::new(seed);
        let mut histogram = vec![0usize; p.len()];
        for _ in 0..count {
            let verification = verify_sampled(&[p.to_vec(), p.to_vec()], &[draft], &mut rng)
                .expect("verification");
            assert!(!verification.tokens.is_empty());
            // With a one-hot draft, the first emitted token is the acceptance
            // decision: `draft` on acceptance, otherwise the residual sample.
            histogram[verification.tokens[0] as usize] += 1;
        }
        histogram
    }

    #[test]
    fn acceptance_is_zero_without_proposals() {
        let stats = SpecStats::default();
        assert_eq!(stats.acceptance(), 0.0);
    }

    #[test]
    fn acceptance_counts_are_real() {
        let mut stats = SpecStats::default();
        stats.record_step(4, 3);
        stats.record_step(2, 0);
        assert_eq!(stats.proposed, 6);
        assert_eq!(stats.accepted, 3);
        assert!((stats.acceptance() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejection_sampling_matches_target_distribution() {
        // Seeded random target over a 6-token vocabulary, with a one-hot draft.
        let mut source = SpecRng::new(0x5eed);
        let mut p: Vec<f64> = (0..6).map(|_| source.unit() + 0.01).collect();
        let total: f64 = p.iter().sum();
        for probability in &mut p {
            *probability /= total;
        }
        let draft = 3u32;
        let histogram = draws(2026, 400_000, &p, draft);
        let total: usize = histogram.iter().sum();
        for (index, count) in histogram.iter().enumerate() {
            let empirical = *count as f64 / total as f64;
            assert!(
                (empirical - p[index]).abs() < 0.01,
                "token {index}: empirical {empirical} vs target {}",
                p[index]
            );
        }
        let corrections: usize = total - histogram[draft as usize];
        assert!(corrections > 0, "residual branch must be exercised");
    }

    #[test]
    fn correction_stops_the_chain() {
        let p = vec![0.0, 0.0, 1.0, 0.0];
        let mut rng = SpecRng::new(7);
        let verification = verify_sampled(
            &[p.clone(), p.clone(), p.clone(), p.clone()],
            &[0, 1, 2],
            &mut rng,
        )
        .unwrap();
        assert_eq!(verification.accepted, 0);
        assert_eq!(verification.tokens, vec![2]);
        assert!(!verification.all_accepted);
    }

    #[test]
    fn greedy_verification_accepts_or_corrects() {
        let logits = vec![vec![0.0, 2.0, 1.0], vec![3.0, 0.0, 1.0]];
        let accepted = verify_greedy(&logits, &[1]).unwrap();
        assert_eq!(accepted.tokens, vec![1, 0]);
        assert_eq!(accepted.accepted, 1);
        assert!(accepted.all_accepted);

        let rejected = verify_greedy(&logits, &[2]).unwrap();
        assert_eq!(rejected.tokens, vec![1]);
        assert_eq!(rejected.accepted, 0);
        assert!(!rejected.all_accepted);
    }

    #[test]
    fn full_acceptance_emits_a_bonus_token() {
        let p = vec![0.0, 0.0, 1.0, 0.0];
        let mut rng = SpecRng::new(9);
        let verification =
            verify_sampled(&[p.clone(), p.clone(), p.clone()], &[2, 2], &mut rng).unwrap();
        assert_eq!(verification.tokens, vec![2, 2, 2]);
        assert_eq!(verification.accepted, 2);
        assert!(verification.all_accepted);
    }

    #[test]
    fn empty_drafts_fall_back_to_a_plain_sample() {
        let probabilities = vec![vec![0.0, 0.0, 1.0]];
        let mut rng = SpecRng::new(11);
        let verification = verify_sampled(&probabilities, &[], &mut rng).unwrap();
        assert_eq!(verification.tokens, vec![2]);
        assert_eq!(verification.accepted, 0);
        assert!(verification.all_accepted);
    }

    #[test]
    fn drafter_proposes_the_most_recent_continuation() {
        let mut drafter = LookupDrafter::new(DRAFT_NGRAM);
        drafter.observe(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 3, 4, 5]);
        // The suffix [3, 4, 5] previously continued with [6, 9].
        assert_eq!(drafter.propose(2), vec![6, 9]);

        let mut no_match = LookupDrafter::new(DRAFT_NGRAM);
        no_match.observe(&[1, 2, 3]);
        assert!(no_match.propose(4).is_empty());
        assert!(no_match.propose(0).is_empty());
    }

    #[test]
    fn bandit_converges_to_the_best_cost_adjusted_arm() {
        const ACCEPT: f64 = 0.72;
        const COST_BASE: f64 = 0.9;
        const COST_PER_TOKEN: f64 = 0.12;
        let mut bandit = AdaptiveK::new();
        let mut rng = SpecRng::new(4242);
        let mut last_choices = vec![0usize; DRAFT_ARMS.len()];

        for step in 0..5000 {
            let k = bandit.select();
            let accepted = (0..k).filter(|_| rng.unit() < ACCEPT).count();
            let cost = COST_BASE + COST_PER_TOKEN * k as f64;
            bandit.update(k, accepted as f64 / cost);
            if step >= 4500 {
                let index = DRAFT_ARMS.iter().position(|arm| *arm == k).unwrap();
                last_choices[index] += 1;
            }
        }

        let best = last_choices[4];
        assert!(
            best > 300,
            "best arm must dominate the last 500 iterations: {last_choices:?}"
        );
        let mean_16 = bandit.mean_for(16).unwrap();
        let mean_8 = bandit.mean_for(8).unwrap();
        let mean_4 = bandit.mean_for(4).unwrap();
        let mean_2 = bandit.mean_for(2).unwrap();
        let mean_0 = bandit.mean_for(0).unwrap();
        assert!(mean_16 > mean_8 && mean_8 > mean_4 && mean_4 > mean_2 && mean_2 > mean_0);
    }

    #[test]
    fn greedy_sampling_is_argmax() {
        let logits = vec![0.1, 0.9, 0.3, 0.9];
        let mut rng = SpecRng::new(7);
        assert_eq!(sample_from_logits(&logits, 0.0, 1.0, &mut rng), 1);
    }

    #[test]
    fn probabilities_respect_top_p_support() {
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let probabilities = probabilities(&logits, 1.0, 0.5);
        assert!((probabilities[0] - 1.0).abs() < 1e-12);
        assert!(
            probabilities[1..]
                .iter()
                .all(|probability| *probability == 0.0)
        );
    }

    /// Mirrors the actor's speculative KV accounting: append the draft input,
    /// verify, roll back to the committed token count, and check the page table
    /// after every step. OOM/capacity errors are allowed and must be
    /// transactional.
    #[test]
    fn spec_step_memory_accounting_fuzz() {
        use crate::memory::{BlockSpaceManager, MemoryError};

        let block_size = 4;
        let mut manager = BlockSpaceManager::new(16, block_size, 4);
        let mut rng = SpecRng::new(2048);
        let prompt: Vec<u32> = (1..=4).collect();
        manager.match_or_allocate_prefix(1, &prompt).unwrap();
        let mut out: Vec<u32> = Vec::new();

        for _ in 0..2000 {
            if out.len() >= 8 {
                manager.rollback(1, prompt.len()).unwrap();
                out.clear();
            }
            let k = rng.next_u64() % 5;
            let drafts: Vec<u32> = (0..k).map(|_| (rng.next_u64() % 256) as u32).collect();
            let mut input = vec![*out.last().unwrap_or(&7)];
            input.extend_from_slice(&drafts);

            match manager.append_tokens(1, input.len()) {
                Ok(()) => {}
                Err(MemoryError::OutOfMemory) | Err(MemoryError::CapacityExceeded) => {
                    manager.assert_invariants();
                    continue;
                }
                Err(error) => panic!("unexpected append error: {error}"),
            }
            manager.assert_invariants();

            let logits: Vec<Vec<f32>> = (0..input.len())
                .map(|_| (0..8).map(|_| rng.unit() as f32).collect())
                .collect();
            let verification = verify_greedy(&logits, &drafts).unwrap();
            let mut emitted = verification.tokens;
            let remaining = 12usize.saturating_sub(out.len());
            emitted.truncate(remaining);
            out.extend_from_slice(&emitted);
            manager
                .rollback(1, prompt.len() + out.len())
                .expect("rollback above prompt");
            assert_eq!(manager.logical_tokens(1), Some(prompt.len() + out.len()));
            manager.assert_invariants();
        }
    }
}
