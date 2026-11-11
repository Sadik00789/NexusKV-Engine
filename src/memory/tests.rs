// src/memory/tests.rs
use super::*;

fn bm(total: usize, block_size: usize, max_blocks: usize) -> BlockSpaceManager {
    BlockSpaceManager::new(total, block_size, max_blocks)
}

fn refs(bm: &BlockSpaceManager) -> Vec<usize> {
    (0..bm.total()).map(|block| bm.ref_count(block)).collect()
}

#[test]
fn kv_valid_prefix_is_block_aligned_and_only_counts_written_blocks() {
    let mut manager = bm(16, 4, 8);
    let prompt: Vec<u32> = (1..=12).collect();
    let first = manager
        .match_or_allocate_prefix_with_validity(1, &prompt)
        .unwrap();
    assert_eq!(first.hit_blocks, 0);
    assert_eq!(first.valid_tokens, 0);

    // Only the first two full blocks were written.
    assert_eq!(manager.mark_kv_valid(1, 0, 8), 2);
    assert_eq!(manager.valid_prefix_tokens(1), 8);
    manager.assert_invariants();
    manager.free_sequence(1).unwrap();

    let second = manager
        .match_or_allocate_prefix_with_validity(2, &prompt)
        .unwrap();
    assert_eq!(second.hit_blocks, 3);
    assert_eq!(second.valid_tokens, 8);

    // The remaining block is written now; the next match can skip the whole
    // block-aligned prompt.
    assert_eq!(manager.mark_kv_valid(2, 8, 4), 1);
    manager.free_sequence(2).unwrap();
    let third = manager
        .match_or_allocate_prefix_with_validity(3, &prompt)
        .unwrap();
    assert_eq!(third.valid_tokens, 12);
    manager.assert_invariants();
}

#[test]
fn kv_valid_unaligned_prompt_skips_only_full_blocks() {
    let mut manager = bm(8, 4, 8);
    let prompt: Vec<u32> = (1..=10).collect();
    manager.match_or_allocate_prefix(1, &prompt).unwrap();
    assert_eq!(manager.mark_kv_valid(1, 0, prompt.len()), 2);
    assert_eq!(manager.valid_prefix_tokens(1), 8);
    manager.free_sequence(1).unwrap();
    let matched = manager
        .match_or_allocate_prefix_with_validity(2, &prompt)
        .unwrap();
    assert_eq!(matched.valid_tokens, 8, "partial tail cannot be skipped");
}

#[test]
fn drop_mode_eviction_invalidates_detached_nodes() {
    let mut manager = bm(4, 4, 8);
    let prompt: Vec<u32> = (1..=8).collect();
    manager.match_or_allocate_prefix(1, &prompt).unwrap();
    manager.mark_kv_valid(1, 0, prompt.len());
    manager.free_sequence(1).unwrap();
    assert!(manager.evict_one().unwrap());
    assert!(manager.evict_one().unwrap());
    manager.assert_invariants();
    let matched = manager
        .match_or_allocate_prefix_with_validity(2, &prompt)
        .unwrap();
    assert_eq!(
        matched.valid_tokens, 0,
        "dropped KV must never be reported valid"
    );
}

#[test]
fn swapped_blocks_keep_their_validity_after_page_in() {
    let io = TestIo::new();
    let mut manager = BlockSpaceManager::with_io(4, 4, 8, Box::new(io));
    let prompt: Vec<u32> = (1..=8).collect();
    manager.match_or_allocate_prefix(1, &prompt).unwrap();
    manager.mark_kv_valid(1, 0, prompt.len());
    manager.free_sequence(1).unwrap();
    assert!(manager.evict_one().unwrap(), "one leaf must be spilled");
    assert_eq!(manager.stats().page_faults, 0);

    let matched = manager
        .match_or_allocate_prefix_with_validity(2, &prompt)
        .unwrap();
    assert_eq!(matched.valid_tokens, 8);
    assert_eq!(manager.stats().page_faults, 1);
    manager.assert_invariants();
}

#[test]
fn regression_b1_stale_trie() {
    // B1: freeing a sequence must not leave the trie pointing at a block that
    // can be handed out by the free stack (dual ownership / UAF).
    let mut manager = bm(4, 4, 10);
    let prompt: Vec<u32> = (1..=8).collect(); // two full blocks

    assert_eq!(manager.match_or_allocate_prefix(1, &prompt).unwrap(), 0);
    let blocks = manager.blocks_of(1);
    assert_eq!(blocks.len(), 2);

    manager.free_sequence(1).unwrap();
    manager.assert_invariants();

    for &block in &blocks {
        assert_eq!(manager.ref_count(block), 0);
        assert!(!manager.is_free(block), "cached block must not be free");
        assert_eq!(manager.state_of(block), BlockState::Cached);
    }

    // Re-matching must resurrect the cached trie node, never a free-stack block.
    assert_eq!(manager.match_or_allocate_prefix(2, &prompt).unwrap(), 2);
    assert_eq!(manager.blocks_of(2), blocks);
    manager.assert_invariants();

    for &block in &blocks {
        assert_eq!(manager.ref_count(block), 1);
        assert!(!manager.is_free(block));
    }

    // The remaining free block is not any block referenced by the page table.
    // A bare allocate() is a transient caller-owned reference, so invariants
    // are deliberately not asserted until it is attached or released.
    let extra = manager.allocate().unwrap();
    assert!(!blocks.contains(&extra));
    assert_eq!(manager.ref_count(extra), 1);
    assert!(!manager.is_free(extra));
}

#[test]
fn regression_b2_oom_leak() {
    // Mid-match: the first miss allocates a block, the second fails. The whole
    // operation must be undone: no table, no leaked refs, no reachable new node.
    let mut manager = bm(4, 4, 10);
    manager
        .match_or_allocate_prefix(1, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        .unwrap();
    let before_refs = refs(&manager);
    let before_allocated = manager.allocated();
    let before_nodes = manager.trie_reachable_nodes();

    let err = manager
        .match_or_allocate_prefix(2, &[21, 22, 23, 24, 25, 26, 27, 28])
        .unwrap_err();
    assert_eq!(err, MemoryError::OutOfMemory);
    assert!(manager.page_table(2).is_none());
    assert_eq!(manager.allocated(), before_allocated);
    assert_eq!(refs(&manager), before_refs);
    assert_eq!(manager.trie_reachable_nodes(), before_nodes);
    manager.assert_invariants();

    // Mid-fork: the partial tail copy needs one block. With a full pool the
    // allocation fails before any shared refcount is bumped.
    let mut manager = bm(4, 4, 10);
    manager
        .match_or_allocate_prefix(1, &(100..114).collect::<Vec<u32>>())
        .unwrap();
    manager.assert_invariants();
    let before_refs = refs(&manager);
    let before_allocated = manager.allocated();

    let err = manager.fork(1, 2).unwrap_err();
    assert_eq!(err, MemoryError::OutOfMemory);
    assert!(manager.page_table(2).is_none());
    assert_eq!(manager.allocated(), before_allocated);
    assert_eq!(refs(&manager), before_refs);
    assert!(manager.take_pending_copies().is_empty());
    manager.assert_invariants();
}

#[test]
fn regression_b5_cow_tail() {
    // B5: a forked child must get a private copy of the partial tail, so appending
    // to the child cannot corrupt the parent's KV content.
    let io = TestIo::new();
    let mut manager = BlockSpaceManager::with_io(8, 4, 10, Box::new(io.clone()));
    manager
        .match_or_allocate_prefix(1, &[10, 11, 12, 13, 20, 21])
        .unwrap();
    let parent_blocks = manager.blocks_of(1);
    let parent_tail = *parent_blocks.last().unwrap();
    io.set_content(parent_tail, b"parent-kv");

    manager.fork(1, 2).unwrap();
    manager.assert_invariants();
    let child_blocks = manager.blocks_of(2);
    let child_tail = *child_blocks.last().unwrap();
    assert_ne!(parent_tail, child_tail, "child must own a private tail");
    assert_eq!(manager.blocks_of(1), parent_blocks, "parent tail changed");
    assert_eq!(manager.ref_count(parent_tail), 1);
    assert_eq!(manager.ref_count(child_tail), 1);
    assert_eq!(
        child_blocks[0], parent_blocks[0],
        "full block should be shared"
    );

    let copies = manager.take_pending_copies();
    assert_eq!(copies, vec![(parent_tail, child_tail)]);
    io.copy_block(parent_tail, child_tail);
    assert_eq!(io.content(child_tail).unwrap(), b"parent-kv");

    // Simulate the child writing into its own tail.
    manager.append_tokens(2, 1).unwrap();
    io.set_content(child_tail, b"child-kv");
    assert_eq!(io.content(parent_tail).unwrap(), b"parent-kv");
    assert_eq!(io.content(child_tail).unwrap(), b"child-kv");
    assert_eq!(manager.blocks_of(1), parent_blocks);

    // Appending to a *shared* partial tail must also CoW. A commit-style
    // snapshot is the way two live sequences can legally share a partial tail.
    let io = TestIo::new();
    let mut manager = BlockSpaceManager::with_io(8, 4, 10, Box::new(io));
    manager
        .match_or_allocate_prefix(1, &[1, 2, 3, 4, 5])
        .unwrap();
    let blocks = manager.blocks_of(1);
    let tail = *blocks.last().unwrap();
    manager.attach_snapshot(2, &blocks, 5, 5);
    assert_eq!(manager.ref_count(tail), 2);

    manager.append_tokens(1, 1).unwrap();
    let after_tail = *manager.blocks_of(1).last().unwrap();
    assert_ne!(
        tail, after_tail,
        "shared tail must be copied before writing"
    );
    assert_eq!(manager.blocks_of(2), blocks);
    assert_eq!(manager.ref_count(tail), 1);
    assert!(manager.stats().cow_copies >= 1);
    manager.assert_invariants();
}

#[test]
fn rollback_boundary() {
    let mut manager = bm(8, 4, 10);
    manager
        .match_or_allocate_prefix(1, &[1, 2, 3, 4, 5])
        .unwrap();
    manager.append_tokens(1, 5).unwrap();
    assert_eq!(manager.logical_tokens(1), Some(10));
    assert_eq!(manager.blocks_of(1).len(), 3);

    // No-op rollback.
    manager.rollback(1, 10).unwrap();
    assert_eq!(manager.blocks_of(1).len(), 3);

    // keep < prompt_len is rejected without changing state.
    let err = manager.rollback(1, 4).unwrap_err();
    assert_eq!(
        err,
        MemoryError::RollbackBelowPrompt {
            keep: 4,
            prompt_len: 5
        }
    );
    assert_eq!(manager.logical_tokens(1), Some(10));
    assert_eq!(manager.blocks_of(1).len(), 3);

    // keep == prompt_len frees every decode block.
    manager.rollback(1, 5).unwrap();
    assert_eq!(manager.logical_tokens(1), Some(5));
    assert_eq!(manager.blocks_of(1).len(), 2);
    assert_eq!(
        manager.trie_reachable_nodes(),
        2,
        "decode blocks must never be inserted into the trie"
    );
    manager.assert_invariants();

    let err = manager.rollback(1, 11).unwrap_err();
    assert_eq!(
        err,
        MemoryError::RollbackAboveCurrent {
            keep: 11,
            current: 5
        }
    );
    manager.assert_invariants();
}

#[test]
fn lru_order_and_leaf_rules() {
    // Drop mode: leaves evict oldest-first; a parent enters the LRU only after
    // its last child is gone.
    let mut manager = bm(8, 4, 10);
    manager
        .match_or_allocate_prefix(1, &(0..12).collect::<Vec<u32>>())
        .unwrap();
    let blocks = manager.blocks_of(1);
    manager.free_sequence(1).unwrap();
    manager.assert_invariants();
    assert_eq!(manager.lru_len(), 1);
    assert_eq!(manager.lru_oldest(), Some(blocks[2]));

    let mut evicted = Vec::new();
    while let Some(block) = manager.lru_oldest() {
        assert!(manager.evict_one().unwrap());
        evicted.push(block);
        manager.assert_invariants();
    }
    assert_eq!(evicted, vec![blocks[2], blocks[1], blocks[0]]);
    assert_eq!(manager.cached_blocks(), 0);
    assert_eq!(manager.free_blocks_count(), 8);

    // Tier mode: any cached block (leaf or not) is evictable and demand-paged.
    let io = TestIo::new();
    let mut manager = BlockSpaceManager::with_io(8, 4, 10, Box::new(io));
    manager
        .match_or_allocate_prefix(1, &(0..12).collect::<Vec<u32>>())
        .unwrap();
    let blocks = manager.blocks_of(1);
    manager.free_sequence(1).unwrap();
    assert_eq!(manager.lru_len(), 3);

    let oldest = manager.lru_oldest().unwrap();
    assert_eq!(
        oldest, blocks[0],
        "non-leaf is still evictable in tier mode"
    );
    assert!(manager.evict_one().unwrap());
    assert_eq!(manager.state_of(blocks[0]), BlockState::Free);
    assert_eq!(manager.stats().evictions, 1);
    manager.assert_invariants();

    // A Swapped hit demand-pages the block and counts a page fault.
    let hits = manager
        .match_or_allocate_prefix(2, &(0..12).collect::<Vec<u32>>())
        .unwrap();
    assert_eq!(hits, 3);
    assert_eq!(manager.stats().page_faults, 1);
    assert_eq!(manager.stats().hits, 3);
    manager.assert_invariants();
}

#[test]
fn tier_roundtrip_mock() {
    let io = TestIo::new();
    let mut manager = BlockSpaceManager::with_io(8, 4, 10, Box::new(io.clone()));
    let prompt: Vec<u32> = vec![10, 11, 12, 13, 20, 21, 22, 23];
    manager.match_or_allocate_prefix(1, &prompt).unwrap();
    let blocks = manager.blocks_of(1);
    io.set_content(blocks[0], b"block-zero");
    io.set_content(blocks[1], b"block-one");

    manager.free_sequence(1).unwrap();
    assert_eq!(manager.cached_blocks(), 2);
    assert!(manager.evict_one().unwrap());
    assert!(manager.evict_one().unwrap());
    assert_eq!(manager.cached_blocks(), 0);
    assert_eq!(manager.stats().evictions, 2);

    assert_eq!(manager.match_or_allocate_prefix(2, &prompt).unwrap(), 2);
    assert_eq!(manager.stats().page_faults, 2);
    let restored = manager.blocks_of(2);
    assert_eq!(io.content(restored[0]).unwrap(), b"block-zero");
    assert_eq!(io.content(restored[1]).unwrap(), b"block-one");
    manager.assert_invariants();
}

#[test]
fn pinned_blocks_survive_release_and_eviction() {
    let mut manager = bm(4, 4, 10);
    manager.match_or_allocate_prefix(1, &[1, 2, 3, 4]).unwrap();
    let block = manager.blocks_of(1)[0];
    manager.pin(block).unwrap();
    manager.free_sequence(1).unwrap();
    assert_eq!(manager.ref_count(block), 1);
    assert_eq!(manager.state_of(block), BlockState::Active);
    assert!(!manager.is_free(block));

    manager.unpin(block).unwrap();
    assert_eq!(manager.state_of(block), BlockState::Cached);
    manager.assert_invariants();
}

#[test]
fn attach_snapshot_is_o1_and_shares_blocks() {
    let mut manager = bm(8, 4, 10);
    manager
        .match_or_allocate_prefix(1, &(0..8).collect::<Vec<u32>>())
        .unwrap();
    let blocks = manager.blocks_of(1);
    manager.free_sequence(1).unwrap();
    let allocated_before = manager.allocated();

    manager.attach_snapshot(2, &blocks, 8, 8);
    manager.attach_snapshot(3, &blocks, 8, 8);
    assert_eq!(
        manager.allocated(),
        allocated_before,
        "checkout must not allocate"
    );
    for &block in &blocks {
        assert_eq!(manager.ref_count(block), 2);
    }
    manager.assert_invariants();

    manager.free_sequence(2).unwrap();
    manager.free_sequence(3).unwrap();
    assert_eq!(manager.cached_blocks(), 2);
    assert_eq!(manager.free_blocks_count(), 6);
    manager.assert_invariants();
}
