// src/memory/fuzz.rs
use super::*;

/// Deterministic LCG specified by the implementation directive.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, upper: u64) -> u64 {
        if upper == 0 { 0 } else { self.next() % upper }
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

struct LiveSequence {
    id: SeqId,
    prompt_len: usize,
    tokens: usize,
}

fn shared_bases() -> Vec<Vec<u32>> {
    vec![
        (1..=16).collect(),
        vec![1, 2, 3, 4, 9, 8, 7, 6, 5, 4],
        vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        vec![],
    ]
}

fn fuzz_run(seed: u64, tier: bool, blocks: usize, operations: usize) {
    let io = TestIo::new();
    let mut manager = if tier {
        BlockSpaceManager::with_io(blocks, 4, 10, Box::new(io.clone()))
    } else {
        BlockSpaceManager::new(blocks, 4, 10)
    };
    let bases = shared_bases();
    let mut rng = Lcg::new(seed);
    let mut next_seq: SeqId = 1;
    let mut live: Vec<LiveSequence> = Vec::new();
    let mut pins: Vec<BlockId> = Vec::new();

    for _ in 0..operations {
        if live.is_empty() {
            // Seed the run with a match.
            let base = &bases[rng.below(bases.len() as u64) as usize];
            let take = rng.below(base.len() as u64 + 1) as usize;
            let mut prompt = base[..take].to_vec();
            for _ in 0..rng.below(6) {
                prompt.push(rng.below(64) as u32);
            }
            let seq = next_seq;
            next_seq += 1;
            if manager.match_or_allocate_prefix(seq, &prompt).is_ok()
                && let Some(table) = manager.page_table(seq)
            {
                live.push(LiveSequence {
                    id: seq,
                    prompt_len: table.prompt_len,
                    tokens: table.logical_tokens,
                });
            }
            manager.assert_invariants();
            continue;
        }

        let choice = rng.below(100);
        if choice < 20 {
            let base = &bases[rng.below(bases.len() as u64) as usize];
            let take = rng.below(base.len() as u64 + 1) as usize;
            let mut prompt = base[..take].to_vec();
            for _ in 0..rng.below(7) {
                prompt.push(rng.below(64) as u32);
            }
            let seq = next_seq;
            next_seq += 1;
            if manager.match_or_allocate_prefix(seq, &prompt).is_ok()
                && let Some(table) = manager.page_table(seq)
            {
                live.push(LiveSequence {
                    id: seq,
                    prompt_len: table.prompt_len,
                    tokens: table.logical_tokens,
                });
            }
        } else if choice < 45 {
            let idx = rng.below(live.len() as u64) as usize;
            let seq = live[idx].id;
            let tokens = 1 + rng.below(8) as usize;
            if manager.append_tokens(seq, tokens).is_ok() {
                live[idx].tokens = manager.logical_tokens(seq).unwrap_or(live[idx].tokens);
            }
        } else if choice < 60 {
            let idx = rng.below(live.len() as u64) as usize;
            let parent = live[idx].id;
            let child = next_seq;
            next_seq += 1;
            if manager.fork(parent, child).is_ok() {
                live.push(LiveSequence {
                    id: child,
                    prompt_len: live[idx].prompt_len,
                    tokens: live[idx].tokens,
                });
            }
        } else if choice < 75 {
            let idx = rng.below(live.len() as u64) as usize;
            let seq = live[idx].id;
            let prompt_len = manager.prompt_len(seq).unwrap_or(0);
            let current = manager.logical_tokens(seq).unwrap_or(prompt_len);
            let keep = if rng.chance(10) {
                prompt_len.saturating_sub(1)
            } else {
                prompt_len + rng.below((current - prompt_len + 1) as u64) as usize
            };
            if manager.rollback(seq, keep).is_ok() {
                live[idx].tokens = manager.logical_tokens(seq).unwrap_or(current);
            }
        } else if choice < 90 {
            if !pins.is_empty() && (pins.len() > 8 || rng.chance(60)) {
                let idx = rng.below(pins.len() as u64) as usize;
                let block = pins.swap_remove(idx);
                let _ = manager.unpin(block);
            } else {
                let idx = rng.below(live.len() as u64) as usize;
                let seq = live[idx].id;
                let blocks = manager.blocks_of(seq);
                if !blocks.is_empty() {
                    let block = blocks[rng.below(blocks.len() as u64) as usize];
                    if manager.pin(block).is_ok() {
                        pins.push(block);
                    }
                }
            }
        } else {
            let idx = rng.below(live.len() as u64) as usize;
            let removed = live.swap_remove(idx);
            manager.free_sequence(removed.id).unwrap();
        }

        manager.assert_invariants();

        if live.len() > 10 {
            let idx = rng.below(live.len() as u64) as usize;
            let removed = live.swap_remove(idx);
            let _ = manager.free_sequence(removed.id);
            manager.assert_invariants();
        }
    }
}

#[test]
fn fuzz_all_invariants() {
    for seed in [1337u64, 42, 7, 99] {
        fuzz_run(seed, false, 24, 8000);
        fuzz_run(seed, true, 24, 8000);
    }
    // High-pressure run with a tiny pool.
    fuzz_run(2024, true, 10, 12_000);
    fuzz_run(2025, false, 10, 12_000);
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        max_shrink_iters: 0,
        ..ProptestConfig::default()
    })]

    #[test]
    fn proptest_refcount_conservation(
        seed in any::<u64>(),
        tier in any::<bool>(),
        operations in 20usize..120,
    ) {
        fuzz_run(seed, tier, 16, operations);
    }
}
