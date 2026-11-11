// benches/bookkeeping.rs
//! BOOKKEEPING THROUGHPUT — NOT INFERENCE.
//!
//! These benchmarks measure allocator, trie, and rollback bookkeeping only.
//! They do not run a model, do not produce tokens, and must never be quoted as
//! inference throughput.

use criterion::{Criterion, criterion_group, criterion_main};
use nexuskv::memory::BlockSpaceManager;
use std::hint::black_box;

fn allocator(c: &mut Criterion) {
    eprintln!("BOOKKEEPING THROUGHPUT — NOT INFERENCE");
    let mut group = c.benchmark_group("bookkeeping/allocator");
    group.bench_function("match_and_free_unique_16_token_prompts", |b| {
        let mut block_manager = BlockSpaceManager::new(4096, 16, 64);
        let mut seq = 1usize;
        b.iter(|| {
            let prompt: Vec<u32> = (0..16)
                .map(|index| (seq as u32).wrapping_mul(31).wrapping_add(index))
                .collect();
            block_manager
                .match_or_allocate_prefix(seq, &prompt)
                .expect("allocate");
            block_manager.free_sequence(seq).expect("free");
            seq = seq.wrapping_add(1);
        });
    });
    group.finish();
}

fn trie(c: &mut Criterion) {
    eprintln!("BOOKKEEPING TRIE HITS — ALLOCATOR BOOKKEEPING, NOT INFERENCE");
    let mut group = c.benchmark_group("bookkeeping/trie");
    let prompt: Vec<u32> = (0..256).collect();
    let mut block_manager = BlockSpaceManager::new(4096, 16, 64);
    block_manager
        .match_or_allocate_prefix(1, &prompt)
        .expect("seed prefix");
    block_manager.free_sequence(1).expect("seed release");
    let mut seq = 2usize;
    group.bench_function("resident_prefix_hit", |b| {
        b.iter(|| {
            let hits = block_manager
                .match_or_allocate_prefix(seq, black_box(&prompt))
                .expect("prefix hit");
            black_box(hits);
            block_manager.free_sequence(seq).expect("release");
            seq = seq.wrapping_add(1);
        });
    });
    group.finish();
}

fn rollback(c: &mut Criterion) {
    eprintln!("BOOKKEEPING THROUGHPUT — NOT INFERENCE");
    let mut group = c.benchmark_group("bookkeeping/rollback");
    let prompt: Vec<u32> = (0..16).collect();
    group.bench_function("rollback_64_decode_tokens", |b| {
        let mut block_manager = BlockSpaceManager::new(4096, 16, 64);
        block_manager
            .match_or_allocate_prefix(1, &prompt)
            .expect("allocate prompt");
        b.iter(|| {
            block_manager.append_tokens(1, 64).expect("append");
            block_manager
                .rollback(1, black_box(prompt.len()))
                .expect("rollback");
        });
        block_manager.free_sequence(1).expect("release");
    });
    group.finish();
}

criterion_group!(benches, allocator, trie, rollback);
criterion_main!(benches);
