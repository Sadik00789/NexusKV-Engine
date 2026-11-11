// tests/tiering.rs
//! Tiered KV memory: content roundtrips, host/disk budget behavior, IO failure
//! injection, and demand paging through the real block manager.

mod common;

use nexuskv::config::EngineConfig;
use nexuskv::engine::{FinishReason, GenerateJob, Request, TokenMsg};
use nexuskv::memory::{BlockIo, BlockSpaceManager, MemoryBridge, MemoryError, TierStore};
use nexuskv::server::AppState;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

fn temp_directory(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("nexuskv-tier-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

#[test]
fn tier_roundtrip_preserves_kv_bytes_through_eviction_and_page_in() {
    let directory = temp_directory("roundtrip");
    let block_bytes = 16;
    let bridge = MemoryBridge::new(block_bytes);
    let store = TierStore::new(bridge.clone(), &directory, 1 << 20).unwrap();
    let mut block_manager = BlockSpaceManager::with_io(8, 4, 8, Box::new(store));

    let prompt: Vec<u32> = (1..=8).collect();
    assert_eq!(
        block_manager.match_or_allocate_prefix(1, &prompt).unwrap(),
        0
    );
    let blocks = block_manager.blocks_of(1);
    let mut expected = HashMap::new();
    for (index, &block) in blocks.iter().enumerate() {
        let content: Vec<u8> = (0..block_bytes)
            .map(|offset| {
                (block as u8)
                    .wrapping_add(index as u8)
                    .wrapping_add(offset as u8)
            })
            .collect();
        bridge.set_block(block, content.clone());
        expected.insert(block, content);
    }
    block_manager.free_sequence(1).unwrap();
    block_manager.assert_invariants();

    // Fill the pool with attached sequences so the cached prefix must swap out.
    for seq in 2..6 {
        let prompt: Vec<u32> = (0..8).map(|index| (seq * 100 + index) as u32).collect();
        block_manager
            .match_or_allocate_prefix(seq, &prompt)
            .expect("filler sequence");
    }
    block_manager.free_sequence(5).unwrap();
    let swapped_out = block_manager.io_stats().swapped_out;
    assert!(swapped_out >= 2, "cached prefix must be saved to the tier");

    let hits = block_manager
        .match_or_allocate_prefix(10, &prompt)
        .expect("demand-page the cached prefix");
    assert_eq!(hits, 2, "both prefix blocks must be trie hits");
    let loaded = block_manager.blocks_of(10);
    for (index, &block) in loaded.iter().enumerate() {
        assert_eq!(
            bridge.block(block).as_ref(),
            expected.get(&blocks[index]),
            "page-in content for logical block {index} must match"
        );
    }
    assert!(block_manager.stats().page_faults >= 2);
    assert!(block_manager.io_stats().swapped_in >= 2);
    block_manager.assert_invariants();

    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn host_budget_spills_oldest_blocks_and_loads_from_disk() {
    let directory = temp_directory("spill");
    let block_bytes = 16;
    let bridge = MemoryBridge::new(block_bytes);
    let mut store = TierStore::new(bridge.clone(), &directory, 32).unwrap();

    for block in 0..4 {
        bridge.set_block(block, vec![block as u8; block_bytes]);
        store.save(100 + block as u64, block).unwrap();
    }
    let stats = store.stats();
    assert_eq!(stats.swapped_out, 4);
    assert_eq!(stats.disk_spills, 2, "two blocks exceed the 32-byte budget");
    assert!(store.host_bytes() <= 32);
    assert_eq!(store.host_entries(), 2);

    store.load(100, 10).unwrap();
    assert_eq!(bridge.block(10).unwrap(), vec![0; block_bytes]);
    let stats = store.stats();
    assert_eq!(stats.disk_loads, 1);
    assert_eq!(stats.swapped_in, 1);

    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn missing_and_corrupt_tier_files_fail_loudly() {
    let directory = temp_directory("corrupt");
    let block_bytes = 8;
    let bridge = MemoryBridge::new(block_bytes);
    let mut store = TierStore::new(bridge.clone(), &directory, 0).unwrap();
    bridge.set_block(0, vec![1u8; block_bytes]);
    store.save(7, 0).unwrap();
    let missing = directory.join(format!("{:016x}.kv", 7));
    std::fs::remove_file(&missing).unwrap();
    let error = store.load(7, 0).unwrap_err();
    assert!(error.contains("missing"), "unexpected error: {error}");

    bridge.set_block(1, vec![2u8; block_bytes]);
    store.save(8, 1).unwrap();
    let corrupt = directory.join(format!("{:016x}.kv", 8));
    std::fs::write(&corrupt, b"not a tier file").unwrap();
    let error = store.load(8, 1).unwrap_err();
    assert!(
        error.contains("corrupt"),
        "corruption must be loud, got: {error}"
    );

    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn io_failure_injection_keeps_allocator_invariants() {
    struct Lcg(u64);
    impl Lcg {
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
    }

    let directory = temp_directory("faults");
    let bridge = MemoryBridge::new(16);
    let store = TierStore::new(bridge, &directory, 256).unwrap();
    let faults = store.fault_handle();
    let mut block_manager = BlockSpaceManager::with_io(24, 4, 10, Box::new(store));
    let mut rng = Lcg(0x5eed_2026);
    let mut next_seq = 1usize;
    let mut live: Vec<usize> = Vec::new();
    let bases: [Vec<u32>; 3] = [
        (1..=16).collect(),
        vec![9, 8, 7, 6, 5, 4, 3, 2, 1],
        vec![100, 101, 102, 103, 104, 105],
    ];

    for _ in 0..4000 {
        faults.set_fail_save(rng.below(100) < 10);
        faults.set_fail_load(rng.below(100) < 10);
        match rng.below(100) {
            0..=24 => {
                let base = &bases[rng.below(bases.len() as u64) as usize];
                let take = rng.below(base.len() as u64 + 1) as usize;
                let mut prompt = base[..take].to_vec();
                for _ in 0..rng.below(8) {
                    prompt.push(rng.below(64) as u32);
                }
                if block_manager
                    .match_or_allocate_prefix(next_seq, &prompt)
                    .is_ok()
                {
                    live.push(next_seq);
                }
                next_seq += 1;
            }
            25..=50 => {
                if !live.is_empty() {
                    let seq = live[rng.below(live.len() as u64) as usize];
                    let _ = block_manager.append_tokens(seq, 1 + rng.below(6) as usize);
                }
            }
            51..=65 => {
                if !live.is_empty() {
                    let parent = live[rng.below(live.len() as u64) as usize];
                    let child = next_seq;
                    next_seq += 1;
                    if block_manager.fork(parent, child).is_ok() {
                        live.push(child);
                    }
                }
            }
            66..=85 => {
                if !live.is_empty() {
                    let index = rng.below(live.len() as u64) as usize;
                    let seq = live.swap_remove(index);
                    let _ = block_manager.free_sequence(seq);
                }
            }
            _ => {
                if !live.is_empty() {
                    let seq = live[rng.below(live.len() as u64) as usize];
                    if let Some(table) = block_manager.page_table(seq)
                        && table.logical_tokens > table.prompt_len
                    {
                        let keep = table.prompt_len
                            + rng.below((table.logical_tokens - table.prompt_len) as u64 + 1)
                                as usize;
                        let _ = block_manager.rollback(seq, keep);
                    }
                }
            }
        }
        block_manager.assert_invariants();
    }

    // A failed demand paging attempt must surface as an IO error, never as
    // silent wrong KV. Build a swapped prefix first, then fail the load.
    let load_directory = temp_directory("fault-load");
    let load_bridge = MemoryBridge::new(16);
    let load_store = TierStore::new(load_bridge.clone(), &load_directory, 0).unwrap();
    let load_faults = load_store.fault_handle();
    let mut manager = BlockSpaceManager::with_io(8, 4, 8, Box::new(load_store));
    let prefix: Vec<u32> = (1..=8).collect();
    manager.match_or_allocate_prefix(1, &prefix).unwrap();
    manager.free_sequence(1).unwrap();
    for seq in 2..5 {
        let prompt: Vec<u32> = (0..8).map(|index| (seq * 100 + index) as u32).collect();
        manager.match_or_allocate_prefix(seq, &prompt).unwrap();
    }
    let extra: Vec<u32> = (0..8).map(|index| (900 + index) as u32).collect();
    manager.match_or_allocate_prefix(5, &extra).unwrap();
    manager.free_sequence(5).unwrap();
    load_faults.set_fail_load(true);
    let error = manager
        .match_or_allocate_prefix(10, &prefix)
        .expect_err("page-in failure must be reported");
    assert!(
        matches!(error, MemoryError::Io(_)),
        "unexpected error: {error}"
    );
    assert!(manager.page_table(10).is_none());
    manager.assert_invariants();

    let _ = std::fs::remove_dir_all(&directory);
    let _ = std::fs::remove_dir_all(&load_directory);
}

fn tiered_mock_config() -> EngineConfig {
    EngineConfig {
        total_physical_blocks: 8,
        block_size: 16,
        num_heads: 1,
        head_dim: 8,
        max_num_seqs: 1,
        max_blocks_per_seq: 8,
        num_layers: 1,
        max_num_batched_tokens: 256,
    }
}

async fn generate(state: &AppState, prompt: &[u8], max_tokens: usize) -> Vec<u32> {
    let seq = state.next_sequence_id();
    let (tx, mut rx) = unbounded_channel();
    state
        .request_tx
        .send(Request::Generate(GenerateJob {
            seq,
            prompt: prompt.iter().map(|&byte| u32::from(byte)).collect(),
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            spec: false,
            tx,
        }))
        .unwrap();
    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("timed out waiting for tiered generation")
        {
            Some(TokenMsg::Token { id, .. }) => tokens.push(id),
            Some(TokenMsg::Done { reason }) => {
                assert!(matches!(
                    reason,
                    FinishReason::MaxTokensReached | FinishReason::StopTokenEncountered
                ));
                break;
            }
            Some(TokenMsg::Error { message }) => panic!("tiered generation failed: {message}"),
            None => panic!("channel closed before Done"),
        }
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eight_block_pool_generates_200_tokens_with_demand_paging() {
    let directory = temp_directory("e2e");
    let config = tiered_mock_config();
    let block_bytes = config.block_size * config.num_heads * config.head_dim * 4 * 2;
    let bridge = MemoryBridge::new(block_bytes);
    let store = TierStore::new(bridge, &directory, 64 * 1024).unwrap();
    let state = AppState::mock_with_io(config, Some(Box::new(store)));

    let shared: Vec<u8> = (0..60).map(|index| b'A' + (index % 26) as u8).collect();
    let mut produced = 0usize;

    for round in 0..5 {
        produced += generate(&state, &shared, 40).await.len();

        if round < 4 {
            // A unique 100-byte prompt (7 blocks) forces the shared prefix out
            // of the 8-block pool so later rounds page it back in.
            let filler: Vec<u8> = (0..100)
                .map(|index| b'a' + ((index + round * 7) % 26) as u8)
                .collect();
            assert_eq!(generate(&state, &filler, 1).await.len(), 1);
        }
    }

    assert_eq!(produced, 200, "the 8-block pool must complete 200 tokens");
    for _ in 0..500 {
        let snapshot = state.snapshot();
        if snapshot.running_sequences == 0 && snapshot.total_tokens_generated >= 200 {
            assert!(snapshot.page_faults > 0, "demand paging must occur");
            assert!(snapshot.swapped_out > 0);
            assert!(snapshot.swapped_in > 0);
            assert_eq!(
                snapshot.free_blocks + snapshot.active_blocks + snapshot.cached_blocks,
                snapshot.total_blocks
            );
            let _ = std::fs::remove_dir_all(&directory);
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("tiered generation did not publish completion telemetry");
}
