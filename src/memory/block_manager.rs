// src/memory/block_manager.rs
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use thiserror::Error;

pub type BlockId = usize;
pub type SeqId = usize;
pub type NodeId = usize;

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum MemoryError {
    #[error("Physical VRAM Out Of Memory: No free blocks remaining")]
    OutOfMemory,
    #[error("Sequence ID `{0}` not found in page table")]
    SequenceNotFound(SeqId),
    #[error("Attempted to free an already unreferenced block: {0}")]
    DoubleFree(BlockId),
    #[error("Capacity exceeded: Max blocks per sequence reached")]
    CapacityExceeded,
    #[error("Sequence `{0}` already has a page table")]
    SequenceAlreadyRegistered(SeqId),
    #[error("Rollback target {keep} is below pinned prompt length {prompt_len}")]
    RollbackBelowPrompt { keep: usize, prompt_len: usize },
    #[error("Rollback target {keep} exceeds current token count {current}")]
    RollbackAboveCurrent { keep: usize, current: usize },
    #[error("Block I/O error: {0}")]
    Io(String),
    #[error("Invalid block operation: {0}")]
    InvalidBlock(String),
}

/// Explicit state machine for every physical block.
///
/// - `Free`: ref == 0, on the free stack, no trie node, not in LRU.
/// - `Active`: ref >= 1, owned by one or more page tables (or pins).
/// - `Cached`: ref == 0, trie-pinned, evictable through the LRU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockState {
    Free,
    Active,
    Cached,
}

/// Persistent tiering hook. `hash` is the trie content hash (FNV-1a chain),
/// which makes the store content-addressed without the block manager needing
/// to understand KV tensor layouts.
pub trait BlockIo: Send {
    fn save(&mut self, hash: u64, block: BlockId) -> Result<(), String>;
    fn load(&mut self, hash: u64, block: BlockId) -> Result<(), String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemEventKind {
    Alloc,
    Free,
    Hit,
    Fault,
    Evict,
    Fork,
    Cow,
    Rollback,
    Pin,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemEvent {
    pub tick: u64,
    pub kind: MemEventKind,
    pub seq: Option<SeqId>,
    pub blocks: Vec<BlockId>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemStats {
    pub lookups: u64,
    pub hits: u64,
    pub evictions: u64,
    pub page_faults: u64,
    pub cow_copies: u64,
}

/// Metadata for a physical page used by the dashboard and flight recorder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockInspectInfo {
    pub block_id: BlockId,
    pub ref_count: usize,
    pub is_free: bool,
    pub mapped_seq_id: Option<SeqId>,
    pub is_prefix_cached: bool,
    pub is_speculative: bool,
    pub state: String,
}

/// Sequence page table mapping logical sequence tokens to physical blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SequencePageTable {
    pub seq_id: SeqId,
    pub logical_tokens: usize,
    pub prompt_len: usize,
    pub physical_block_ids: Vec<BlockId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Res {
    Gpu(BlockId),
    Swapped,
}

#[derive(Debug, Clone)]
pub struct TrieNode {
    pub children: HashMap<Box<[u32]>, NodeId>,
    pub res: Option<Res>,
    pub parent: Option<NodeId>,
    pub key: Box<[u32]>,
    pub hash: u64,
}

fn hash_chunk(parent_hash: u64, chunk: &[u32]) -> u64 {
    // FNV-1a over the token chunk, seeded with the parent hash.
    let mut h = parent_hash ^ 0xcbf29ce484222325;
    for &token in chunk {
        h = (h ^ token as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn ceil_div(a: usize, b: usize) -> usize {
    if a == 0 { 0 } else { a.div_ceil(b) }
}

/// Shared mock tier store used by phase-1 tests and phase-6 integration tests.
#[derive(Clone, Default)]
pub struct TestIo {
    inner: Arc<Mutex<TestIoInner>>,
}

#[derive(Default)]
struct TestIoInner {
    store: HashMap<u64, Vec<u8>>,
    content: HashMap<BlockId, Vec<u8>>,
    fail_save: bool,
    fail_load: bool,
}

impl TestIo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_content(&self, block: BlockId, bytes: impl Into<Vec<u8>>) {
        self.inner
            .lock()
            .unwrap()
            .content
            .insert(block, bytes.into());
    }

    pub fn content(&self, block: BlockId) -> Option<Vec<u8>> {
        self.inner.lock().unwrap().content.get(&block).cloned()
    }

    pub fn copy_block(&self, src: BlockId, dst: BlockId) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(data) = inner.content.get(&src).cloned() {
            inner.content.insert(dst, data);
        }
    }

    pub fn saved(&self, hash: u64) -> Option<Vec<u8>> {
        self.inner.lock().unwrap().store.get(&hash).cloned()
    }

    pub fn fail_save(&self, fail: bool) {
        self.inner.lock().unwrap().fail_save = fail;
    }

    pub fn fail_load(&self, fail: bool) {
        self.inner.lock().unwrap().fail_load = fail;
    }
}

impl BlockIo for TestIo {
    fn save(&mut self, hash: u64, block: BlockId) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();
        if inner.fail_save {
            return Err("injected save failure".to_string());
        }
        let data = inner.content.get(&block).cloned().unwrap_or_default();
        inner.store.insert(hash, data);
        Ok(())
    }

    fn load(&mut self, hash: u64, block: BlockId) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();
        if inner.fail_load {
            return Err("injected load failure".to_string());
        }
        let data = inner
            .store
            .get(&hash)
            .cloned()
            .ok_or_else(|| format!("missing tier content for hash {hash:016x}"))?;
        inner.content.insert(block, data);
        Ok(())
    }
}

enum Acquisition {
    /// A cached or already-active block whose refcount was incremented.
    Existing(BlockId),
    /// A swapped block that was demand-paged into a fresh block.
    Loaded { block: BlockId, node: NodeId },
    /// A brand-new trie node + freshly allocated block.
    NewNode { block: BlockId, node: NodeId },
    /// A freshly allocated block with no trie node (partial tail / remainder).
    Fresh(BlockId),
}

/// Physical block allocator, radix prefix cache and paged KV page tables.
pub struct BlockSpaceManager {
    pub total_blocks: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,

    ref_count: Vec<usize>,
    pins: Vec<usize>,
    states: Vec<BlockState>,
    in_free: Vec<bool>,
    free_stack: Vec<BlockId>,
    mapped_seq: Vec<Option<SeqId>>,

    page_tables: HashMap<SeqId, SequencePageTable>,

    nodes: Vec<TrieNode>,
    node_of: Vec<Option<NodeId>>,

    lru: BTreeMap<u64, BlockId>,
    lru_key: Vec<Option<u64>>,
    next_lru_tick: u64,

    io: Option<Box<dyn BlockIo>>,
    stats: MemStats,
    sink: Option<Box<dyn FnMut(MemEvent) + Send>>,
    event_tick: u64,

    pending_copies: Vec<(BlockId, BlockId)>,
}

impl BlockSpaceManager {
    pub fn new(total_blocks: usize, block_size: usize, max_blocks_per_seq: usize) -> Self {
        Self::build(total_blocks, block_size, max_blocks_per_seq, None)
    }

    pub fn with_io(
        total_blocks: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        io: Box<dyn BlockIo>,
    ) -> Self {
        Self::build(total_blocks, block_size, max_blocks_per_seq, Some(io))
    }

    fn build(
        total_blocks: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        io: Option<Box<dyn BlockIo>>,
    ) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        let free_stack: Vec<BlockId> = (0..total_blocks).rev().collect();
        Self {
            total_blocks,
            block_size,
            max_blocks_per_seq,
            ref_count: vec![0; total_blocks],
            pins: vec![0; total_blocks],
            states: vec![BlockState::Free; total_blocks],
            in_free: vec![true; total_blocks],
            free_stack,
            mapped_seq: vec![None; total_blocks],
            page_tables: HashMap::new(),
            nodes: vec![TrieNode {
                children: HashMap::new(),
                res: None,
                parent: None,
                key: Box::new([]),
                hash: 0,
            }],
            node_of: vec![None; total_blocks],
            lru: BTreeMap::new(),
            lru_key: vec![None; total_blocks],
            next_lru_tick: 1,
            io,
            stats: MemStats::default(),
            sink: None,
            event_tick: 0,
            pending_copies: Vec::new(),
        }
    }

    pub fn set_event_sink(&mut self, sink: Box<dyn FnMut(MemEvent) + Send>) {
        self.sink = Some(sink);
    }

    fn emit(&mut self, kind: MemEventKind, seq: Option<SeqId>, blocks: Vec<BlockId>) {
        self.event_tick = self.event_tick.wrapping_add(1);
        if let Some(sink) = self.sink.as_mut() {
            sink(MemEvent {
                tick: self.event_tick,
                kind,
                seq,
                blocks,
            });
        }
    }

    // ------------------------------------------------------------------
    // Basic allocator primitives
    // ------------------------------------------------------------------

    /// O(1) allocation, with LRU eviction as the fallback under pressure.
    pub fn allocate(&mut self) -> Result<BlockId, MemoryError> {
        loop {
            if let Some(block) = self.free_stack.pop() {
                debug_assert!(self.in_free[block]);
                debug_assert_eq!(self.states[block], BlockState::Free);
                self.in_free[block] = false;
                self.ref_count[block] = 1;
                self.states[block] = BlockState::Active;
                self.mapped_seq[block] = None;
                self.node_of[block] = None;
                self.emit(MemEventKind::Alloc, None, vec![block]);
                return Ok(block);
            }
            if !self.evict_one()? {
                return Err(MemoryError::OutOfMemory);
            }
        }
    }

    fn make_free(&mut self, block: BlockId) {
        self.ref_count[block] = 0;
        self.states[block] = BlockState::Free;
        self.in_free[block] = true;
        self.mapped_seq[block] = None;
        self.free_stack.push(block);
    }

    fn lru_insert(&mut self, block: BlockId) {
        if self.lru_key[block].is_none() {
            let tick = self.next_lru_tick;
            self.next_lru_tick = self.next_lru_tick.wrapping_add(1);
            self.lru_key[block] = Some(tick);
            self.lru.insert(tick, block);
        }
    }

    fn lru_remove(&mut self, block: BlockId) {
        if let Some(tick) = self.lru_key[block].take() {
            self.lru.remove(&tick);
        }
    }

    fn node_is_leaf(&self, node: NodeId) -> bool {
        self.nodes[node].children.is_empty()
    }

    /// Decrement a reference. When the count reaches zero, a trie-pinned block
    /// becomes `Cached` (and evictable), otherwise it returns to the free stack.
    fn release(&mut self, block: BlockId, seq: Option<SeqId>) -> Result<(), MemoryError> {
        if block >= self.total_blocks || self.ref_count[block] == 0 {
            return Err(MemoryError::DoubleFree(block));
        }
        self.ref_count[block] -= 1;
        if self.ref_count[block] == 0 {
            self.mapped_seq[block] = None;
            match self.node_of[block] {
                Some(node) => {
                    self.states[block] = BlockState::Cached;
                    if self.io.is_some() || self.node_is_leaf(node) {
                        self.lru_insert(block);
                    }
                }
                None => self.make_free(block),
            }
        }
        self.emit(MemEventKind::Free, seq, vec![block]);
        Ok(())
    }

    fn acquire_existing(&mut self, block: BlockId, seq: SeqId) -> Result<(), MemoryError> {
        if block >= self.total_blocks || self.states[block] == BlockState::Free {
            return Err(MemoryError::InvalidBlock(format!(
                "attempted to acquire free block {block}"
            )));
        }
        self.ref_count[block] += 1;
        if self.states[block] == BlockState::Cached {
            self.lru_remove(block);
            self.states[block] = BlockState::Active;
        }
        if self.mapped_seq[block].is_none() {
            self.mapped_seq[block] = Some(seq);
        }
        Ok(())
    }

    fn detach_node(&mut self, node: NodeId) {
        let parent = self.nodes[node].parent;
        let key = self.nodes[node].key.clone();
        if let Some(parent) = parent {
            self.nodes[parent].children.remove(&key);
        }
        self.nodes[node].res = None;
    }

    fn drop_cached_node(&mut self, block: BlockId, node: NodeId) {
        let parent = self.nodes[node].parent;

        self.detach_node(node);
        self.node_of[block] = None;
        self.lru_remove(block);
        self.make_free(block);

        // A parent that just became a childless cached GPU leaf becomes evictable.
        if let Some(parent_node) = parent
            && self.node_is_leaf(parent_node)
            && let Some(Res::Gpu(parent_block)) = self.nodes[parent_node].res
            && parent_block < self.total_blocks
            && self.states[parent_block] == BlockState::Cached
            && self.ref_count[parent_block] == 0
            && self.node_of[parent_block] == Some(parent_node)
        {
            self.lru_insert(parent_block);
        }
    }

    /// Evict the oldest cached block. Returns `Ok(false)` when nothing is evictable.
    ///
    /// Tier mode (`BlockIo` present): save the block, keep the trie node as
    /// `Swapped`, and free the physical page. Any cached block is evictable.
    /// Drop mode: only leaves are evictable; evicting a leaf may promote its
    /// parent into the LRU once it becomes a leaf.
    pub fn evict_one(&mut self) -> Result<bool, MemoryError> {
        if self.lru.is_empty() {
            return Ok(false);
        }
        let candidates: Vec<(u64, BlockId)> = self
            .lru
            .iter()
            .map(|(tick, block)| (*tick, *block))
            .collect();

        for (tick, block) in candidates {
            if self.lru_key[block] != Some(tick) {
                continue;
            }
            if self.states[block] != BlockState::Cached || self.ref_count[block] != 0 {
                continue;
            }
            let node = match self.node_of[block] {
                Some(node) => node,
                None => {
                    self.lru_remove(block);
                    continue;
                }
            };

            if self.io.is_some() {
                let hash = self.nodes[node].hash;
                let save_result = self.io.as_mut().unwrap().save(hash, block);
                match save_result {
                    Ok(()) => {
                        self.nodes[node].res = Some(Res::Swapped);
                        self.node_of[block] = None;
                        self.lru_remove(block);
                        self.make_free(block);
                        self.stats.evictions += 1;
                        self.emit(MemEventKind::Evict, None, vec![block]);
                        return Ok(true);
                    }
                    Err(_) if self.node_is_leaf(node) => {
                        // Save failure: dropping the leaf is the safe fallback.
                        self.drop_cached_node(block, node);
                        self.stats.evictions += 1;
                        self.emit(MemEventKind::Evict, None, vec![block]);
                        return Ok(true);
                    }
                    Err(_) => {
                        // Interior node: keep trying younger candidates.
                        continue;
                    }
                }
            } else {
                if !self.node_is_leaf(node) {
                    // Defensive: drop-mode LRU should only contain leaves.
                    self.lru_remove(block);
                    continue;
                }
                self.drop_cached_node(block, node);
                self.stats.evictions += 1;
                self.emit(MemEventKind::Evict, None, vec![block]);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn rollback_acquisitions(&mut self, acquisitions: &mut Vec<Acquisition>, seq: SeqId) {
        for acquisition in acquisitions.drain(..).rev() {
            match acquisition {
                Acquisition::Existing(block) | Acquisition::Fresh(block) => {
                    let _ = self.release(block, Some(seq));
                }
                Acquisition::NewNode { block, node } => {
                    self.detach_node(node);
                    self.node_of[block] = None;
                    let _ = self.release(block, Some(seq));
                }
                Acquisition::Loaded { block, node } => {
                    self.node_of[block] = None;
                    self.nodes[node].res = Some(Res::Swapped);
                    let _ = self.release(block, Some(seq));
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Prefix matching / allocation
    // ------------------------------------------------------------------

    /// Match full prompt blocks against the radix trie and allocate the rest.
    ///
    /// Transactional: every fallible allocation happens before any refcount is
    /// committed to a page table. On error, all acquisitions (including trie
    /// nodes created by this call and demand-paged blocks) are undone.
    pub fn match_or_allocate_prefix(
        &mut self,
        seq: SeqId,
        prompt_tokens: &[u32],
    ) -> Result<usize, MemoryError> {
        if self.page_tables.contains_key(&seq) {
            return Err(MemoryError::SequenceAlreadyRegistered(seq));
        }
        let full_blocks = prompt_tokens.len() / self.block_size;
        let remainder = prompt_tokens.len() % self.block_size;
        let needed_blocks = full_blocks + usize::from(remainder > 0);
        if needed_blocks > self.max_blocks_per_seq {
            return Err(MemoryError::CapacityExceeded);
        }

        let mut blocks: Vec<BlockId> = Vec::with_capacity(needed_blocks);
        let mut acquisitions: Vec<Acquisition> = Vec::new();
        let mut hits = 0usize;
        let mut chain_node = 0usize;

        macro_rules! fail {
            ($err:expr) => {{
                self.rollback_acquisitions(&mut acquisitions, seq);
                return Err($err);
            }};
        }

        for chunk_idx in 0..full_blocks {
            self.stats.lookups += 1;
            let start = chunk_idx * self.block_size;
            let chunk = &prompt_tokens[start..start + self.block_size];
            let child = self.nodes[chain_node].children.get(chunk).copied();

            if let Some(node) = child {
                match self.nodes[node].res {
                    Some(Res::Gpu(block)) => {
                        if let Err(err) = self.acquire_existing(block, seq) {
                            fail!(err);
                        }
                        blocks.push(block);
                        acquisitions.push(Acquisition::Existing(block));
                        hits += 1;
                        self.emit(MemEventKind::Hit, Some(seq), vec![block]);
                        chain_node = node;
                        continue;
                    }
                    Some(Res::Swapped) => {
                        if self.io.is_none() {
                            fail!(MemoryError::InvalidBlock(
                                "swapped trie node without BlockIo".to_string()
                            ));
                        }
                        let block = match self.allocate() {
                            Ok(block) => block,
                            Err(err) => fail!(err),
                        };
                        let hash = self.nodes[node].hash;
                        if let Err(err) = self.io.as_mut().unwrap().load(hash, block) {
                            let _ = self.release(block, Some(seq));
                            fail!(MemoryError::Io(err));
                        }
                        self.node_of[block] = Some(node);
                        self.nodes[node].res = Some(Res::Gpu(block));
                        self.mapped_seq[block] = Some(seq);
                        blocks.push(block);
                        acquisitions.push(Acquisition::Loaded { block, node });
                        self.stats.page_faults += 1;
                        hits += 1;
                        self.emit(MemEventKind::Fault, Some(seq), vec![block]);
                        chain_node = node;
                        continue;
                    }
                    None => {
                        fail!(MemoryError::InvalidBlock(format!(
                            "reachable trie node {node} has no resource"
                        )));
                    }
                }
            }

            let block = match self.allocate() {
                Ok(block) => block,
                Err(err) => fail!(err),
            };
            let parent = chain_node;
            let hash = hash_chunk(self.nodes[parent].hash, chunk);
            let node = self.nodes.len();
            self.nodes.push(TrieNode {
                children: HashMap::new(),
                res: Some(Res::Gpu(block)),
                parent: Some(parent),
                key: Box::from(chunk),
                hash,
            });
            self.nodes[parent].children.insert(Box::from(chunk), node);
            self.node_of[block] = Some(node);
            self.mapped_seq[block] = Some(seq);
            blocks.push(block);
            acquisitions.push(Acquisition::NewNode { block, node });
            chain_node = node;
        }

        if remainder > 0 {
            let block = match self.allocate() {
                Ok(block) => block,
                Err(err) => fail!(err),
            };
            self.mapped_seq[block] = Some(seq);
            blocks.push(block);
            acquisitions.push(Acquisition::Fresh(block));
        }

        self.page_tables.insert(
            seq,
            SequencePageTable {
                seq_id: seq,
                logical_tokens: prompt_tokens.len(),
                prompt_len: prompt_tokens.len(),
                physical_block_ids: blocks,
            },
        );
        self.stats.hits += hits as u64;
        Ok(hits)
    }

    // ------------------------------------------------------------------
    // Sequence page-table operations
    // ------------------------------------------------------------------

    /// Append logical tokens, performing copy-on-write for a shared partial tail.
    /// All allocations are attempted before the page table is mutated.
    pub fn append_tokens(&mut self, seq: SeqId, num_new_tokens: usize) -> Result<(), MemoryError> {
        let (old_tokens, current_blocks) = {
            let table = self
                .page_tables
                .get(&seq)
                .ok_or(MemoryError::SequenceNotFound(seq))?;
            (table.logical_tokens, table.physical_block_ids.len())
        };
        let new_tokens = old_tokens + num_new_tokens;
        let needed_blocks = ceil_div(new_tokens, self.block_size);
        if needed_blocks > self.max_blocks_per_seq {
            return Err(MemoryError::CapacityExceeded);
        }
        let num_new_blocks = needed_blocks.saturating_sub(current_blocks);

        let cow_needed = old_tokens % self.block_size != 0
            && current_blocks > 0
            && self.ref_count[self.page_tables[&seq].physical_block_ids[current_blocks - 1]] > 1;

        let mut fresh: Vec<BlockId> = Vec::with_capacity(num_new_blocks + usize::from(cow_needed));
        if cow_needed {
            let block = self.allocate()?;
            fresh.push(block);
        }
        for _ in 0..num_new_blocks {
            match self.allocate() {
                Ok(block) => fresh.push(block),
                Err(err) => {
                    for block in fresh.drain(..) {
                        let _ = self.release(block, Some(seq));
                    }
                    return Err(err);
                }
            }
        }

        let mut fresh_index = 0usize;
        if cow_needed {
            let table = self.page_tables.get_mut(&seq).unwrap();
            let last = table.physical_block_ids.len() - 1;
            let old_tail = table.physical_block_ids[last];
            let new_tail = fresh[fresh_index];
            fresh_index += 1;

            self.mapped_seq[new_tail] = Some(seq);
            table.physical_block_ids[last] = new_tail;
            let _ = self.release(old_tail, Some(seq));
            self.stats.cow_copies += 1;
            self.pending_copies.push((old_tail, new_tail));
            self.emit(MemEventKind::Cow, Some(seq), vec![old_tail, new_tail]);
        }

        let table = self.page_tables.get_mut(&seq).unwrap();
        for &block in &fresh[fresh_index..] {
            self.mapped_seq[block] = Some(seq);
            table.physical_block_ids.push(block);
        }
        table.logical_tokens = new_tokens;
        Ok(())
    }

    /// Fork a sequence with copy-on-write semantics. Full blocks are shared by
    /// refcount; a partial tail block is copied so the child can append safely.
    pub fn fork(&mut self, parent: SeqId, child: SeqId) -> Result<(), MemoryError> {
        let parent_table = self
            .page_tables
            .get(&parent)
            .cloned()
            .ok_or(MemoryError::SequenceNotFound(parent))?;
        if self.page_tables.contains_key(&child) {
            return Err(MemoryError::SequenceAlreadyRegistered(child));
        }

        let tokens = parent_table.logical_tokens;
        let last_is_partial =
            tokens % self.block_size != 0 && !parent_table.physical_block_ids.is_empty();
        let parent_tail = parent_table.physical_block_ids.last().copied();

        // Fallible allocation happens *before* any refcount is bumped.
        let new_tail = if last_is_partial {
            Some(self.allocate()?)
        } else {
            None
        };

        let mut child_blocks = parent_table.physical_block_ids.clone();
        if let (Some(new_tail), Some(old_tail)) = (new_tail, parent_tail) {
            child_blocks.pop();
            child_blocks.push(new_tail);
            self.mapped_seq[new_tail] = Some(child);
            self.stats.cow_copies += 1;
            self.pending_copies.push((old_tail, new_tail));
            self.emit(MemEventKind::Cow, Some(child), vec![old_tail, new_tail]);
        }

        let shared_prefix_len = if last_is_partial {
            child_blocks.len() - 1
        } else {
            child_blocks.len()
        };
        for &block in &child_blocks[..shared_prefix_len] {
            self.acquire_existing(block, child)?;
        }

        self.page_tables.insert(
            child,
            SequencePageTable {
                seq_id: child,
                logical_tokens: tokens,
                prompt_len: parent_table.prompt_len,
                physical_block_ids: child_blocks.clone(),
            },
        );
        self.emit(MemEventKind::Fork, Some(child), child_blocks);
        Ok(())
    }

    /// Roll back rejected speculative tokens. The prompt is always retained.
    pub fn rollback(&mut self, seq: SeqId, keep: usize) -> Result<(), MemoryError> {
        let (current, prompt_len) = {
            let table = self
                .page_tables
                .get(&seq)
                .ok_or(MemoryError::SequenceNotFound(seq))?;
            (table.logical_tokens, table.prompt_len)
        };
        if keep < prompt_len {
            return Err(MemoryError::RollbackBelowPrompt { keep, prompt_len });
        }
        if keep > current {
            return Err(MemoryError::RollbackAboveCurrent { keep, current });
        }
        let needed_blocks = ceil_div(keep, self.block_size);
        let mut freed = Vec::new();
        {
            let table = self.page_tables.get_mut(&seq).unwrap();
            while table.physical_block_ids.len() > needed_blocks {
                freed.push(table.physical_block_ids.pop().unwrap());
            }
            table.logical_tokens = keep;
        }
        for block in &freed {
            // Decode blocks are never trie-inserted; this must remain a plain release.
            self.release(*block, Some(seq))?;
        }
        self.emit(MemEventKind::Rollback, Some(seq), freed);
        Ok(())
    }

    /// Compatibility alias for the phase-3/4 call sites.
    pub fn rollback_sequence(&mut self, seq: SeqId, keep: usize) -> Result<(), MemoryError> {
        self.rollback(seq, keep)
    }

    /// Release a sequence. Blocks with trie nodes become `Cached`, not destroyed.
    pub fn free_sequence(&mut self, seq: SeqId) -> Result<(), MemoryError> {
        if let Some(table) = self.page_tables.remove(&seq) {
            for block in table.physical_block_ids {
                self.release(block, Some(seq))?;
            }
        }
        Ok(())
    }

    /// Pin a committed block: the refcount is not backed by a page table entry.
    pub fn pin(&mut self, block: BlockId) -> Result<(), MemoryError> {
        if block >= self.total_blocks || self.states[block] == BlockState::Free {
            return Err(MemoryError::InvalidBlock(format!(
                "cannot pin free block {block}"
            )));
        }
        self.ref_count[block] += 1;
        self.pins[block] += 1;
        if self.states[block] == BlockState::Cached {
            self.lru_remove(block);
            self.states[block] = BlockState::Active;
        }
        self.emit(MemEventKind::Pin, None, vec![block]);
        Ok(())
    }

    pub fn unpin(&mut self, block: BlockId) -> Result<(), MemoryError> {
        if block >= self.total_blocks || self.pins[block] == 0 {
            return Err(MemoryError::InvalidBlock(format!(
                "block {block} is not pinned"
            )));
        }
        self.pins[block] -= 1;
        self.release(block, None)
    }

    /// Attach a committed snapshot to a new sequence (checkout path).
    /// Infallible refcount bumps only: committed blocks are guaranteed resident.
    pub fn attach_snapshot(
        &mut self,
        seq: SeqId,
        blocks: &[BlockId],
        tokens: usize,
        prompt_len: usize,
    ) {
        assert!(
            !self.page_tables.contains_key(&seq),
            "attach_snapshot: sequence {seq} already has a page table"
        );
        assert_eq!(
            blocks.len(),
            ceil_div(tokens, self.block_size),
            "attach_snapshot: block count does not match token count"
        );
        assert!(
            prompt_len <= tokens,
            "attach_snapshot: prompt_len exceeds token count"
        );
        for &block in blocks {
            assert!(block < self.total_blocks, "attach_snapshot: bad block");
            assert!(
                self.states[block] != BlockState::Free,
                "attach_snapshot: block {block} is free"
            );
            self.ref_count[block] += 1;
            if self.states[block] == BlockState::Cached {
                self.lru_remove(block);
                self.states[block] = BlockState::Active;
            }
            if self.mapped_seq[block].is_none() {
                self.mapped_seq[block] = Some(seq);
            }
        }
        self.page_tables.insert(
            seq,
            SequencePageTable {
                seq_id: seq,
                logical_tokens: tokens,
                prompt_len,
                physical_block_ids: blocks.to_vec(),
            },
        );
        self.emit(MemEventKind::Pin, Some(seq), blocks.to_vec());
    }

    pub fn take_pending_copies(&mut self) -> Vec<(BlockId, BlockId)> {
        std::mem::take(&mut self.pending_copies)
    }

    // ------------------------------------------------------------------
    // Read-only API / telemetry
    // ------------------------------------------------------------------

    pub fn page_table(&self, seq: SeqId) -> Option<&SequencePageTable> {
        self.page_tables.get(&seq)
    }

    /// Alias matching the phase-1 API name.
    pub fn table(&self, seq: SeqId) -> Option<&SequencePageTable> {
        self.page_table(seq)
    }

    pub fn blocks_of(&self, seq: SeqId) -> Vec<BlockId> {
        self.page_tables
            .get(&seq)
            .map(|table| table.physical_block_ids.clone())
            .unwrap_or_default()
    }

    pub fn logical_tokens(&self, seq: SeqId) -> Option<usize> {
        self.page_tables.get(&seq).map(|table| table.logical_tokens)
    }

    pub fn prompt_len(&self, seq: SeqId) -> Option<usize> {
        self.page_tables.get(&seq).map(|table| table.prompt_len)
    }

    pub fn full_prompt_blocks(&self, seq: SeqId) -> usize {
        self.page_tables
            .get(&seq)
            .map(|table| table.prompt_len / self.block_size)
            .unwrap_or(0)
    }

    /// Logical positions `[start, start+len)` mapped to flat KV-pool slots.
    pub fn slot_mapping(
        &self,
        seq: SeqId,
        start: usize,
        len: usize,
    ) -> Result<Vec<i64>, MemoryError> {
        let table = self
            .page_tables
            .get(&seq)
            .ok_or(MemoryError::SequenceNotFound(seq))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| MemoryError::InvalidBlock("slot range overflow".to_string()))?;
        if end > table.logical_tokens {
            return Err(MemoryError::InvalidBlock(format!(
                "slot range {start}..{end} exceeds {} logical tokens",
                table.logical_tokens
            )));
        }
        let mut slots = Vec::with_capacity(len);
        for position in start..end {
            let logical_block = position / self.block_size;
            let offset = position % self.block_size;
            let block = *table.physical_block_ids.get(logical_block).ok_or_else(|| {
                MemoryError::InvalidBlock(format!(
                    "position {position} has no physical block in sequence {seq}"
                ))
            })?;
            slots.push((block * self.block_size + offset) as i64);
        }
        Ok(slots)
    }

    pub fn export_block_tables(&self, active_seq_ids: &[SeqId]) -> (Vec<i32>, Vec<i32>) {
        let mut flat = vec![-1i32; active_seq_ids.len() * self.max_blocks_per_seq];
        let mut seq_lens = Vec::with_capacity(active_seq_ids.len());
        for (i, &seq) in active_seq_ids.iter().enumerate() {
            if let Some(table) = self.page_tables.get(&seq) {
                seq_lens.push(table.logical_tokens as i32);
                for (j, &block) in table.physical_block_ids.iter().enumerate() {
                    if j < self.max_blocks_per_seq {
                        flat[i * self.max_blocks_per_seq + j] = block as i32;
                    }
                }
            } else {
                seq_lens.push(0);
            }
        }
        (flat, seq_lens)
    }

    pub fn inspect(&self) -> Vec<BlockInspectInfo> {
        (0..self.total_blocks)
            .map(|block| BlockInspectInfo {
                block_id: block,
                ref_count: self.ref_count[block],
                is_free: self.states[block] == BlockState::Free,
                mapped_seq_id: self.mapped_seq[block],
                is_prefix_cached: self.node_of[block].is_some(),
                is_speculative: false,
                state: match self.states[block] {
                    BlockState::Free => "free".to_string(),
                    BlockState::Active => "active".to_string(),
                    BlockState::Cached => "cached".to_string(),
                },
            })
            .collect()
    }

    pub fn is_free(&self, block: BlockId) -> bool {
        self.states[block] == BlockState::Free
    }

    pub fn ref_count(&self, block: BlockId) -> usize {
        self.ref_count[block]
    }

    pub fn pin_count(&self, block: BlockId) -> usize {
        self.pins[block]
    }

    pub fn state_of(&self, block: BlockId) -> BlockState {
        self.states[block]
    }

    pub fn trie_reachable_nodes(&self) -> usize {
        let mut stack = vec![0usize];
        let mut seen = 0usize;
        while let Some(node) = stack.pop() {
            seen += 1;
            for child in self.nodes[node].children.values() {
                stack.push(*child);
            }
        }
        seen
    }

    pub fn lru_len(&self) -> usize {
        self.lru.len()
    }

    pub fn lru_oldest(&self) -> Option<BlockId> {
        self.lru.values().next().copied()
    }

    pub fn lru_contains(&self, block: BlockId) -> bool {
        self.lru_key[block].is_some()
    }

    pub fn allocated(&self) -> usize {
        self.total_blocks.saturating_sub(self.free_stack.len())
    }

    pub fn allocated_blocks_count(&self) -> usize {
        self.allocated()
    }

    pub fn free_blocks_count(&self) -> usize {
        self.free_stack.len()
    }

    pub fn active_blocks(&self) -> usize {
        self.states
            .iter()
            .filter(|state| **state == BlockState::Active)
            .count()
    }

    pub fn cached_blocks(&self) -> usize {
        self.states
            .iter()
            .filter(|state| **state == BlockState::Cached)
            .count()
    }

    pub fn total(&self) -> usize {
        self.total_blocks
    }

    pub fn memory_usage_ratio(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            (self.allocated() as f64 / self.total_blocks as f64).clamp(0.0, 1.0)
        }
    }

    pub fn prefix_cache_hit_ratio(&self) -> f64 {
        if self.stats.lookups == 0 {
            0.0
        } else {
            (self.stats.hits as f64 / self.stats.lookups as f64).clamp(0.0, 1.0)
        }
    }

    /// Alias matching the phase-1 API name.
    pub fn hit_ratio(&self) -> f64 {
        self.prefix_cache_hit_ratio()
    }

    pub fn stats(&self) -> MemStats {
        self.stats
    }

    // ------------------------------------------------------------------
    // Invariants
    // ------------------------------------------------------------------

    /// Panics if the allocator's state machine is inconsistent. Tests call this
    /// after every operation; debug builds call it at tick boundaries.
    pub fn assert_invariants(&self) {
        let mut table_refs = vec![0usize; self.total_blocks];
        for table in self.page_tables.values() {
            assert_eq!(
                table.physical_block_ids.len(),
                ceil_div(table.logical_tokens, self.block_size),
                "sequence {} page-table length does not match token count",
                table.seq_id
            );
            assert!(
                table.prompt_len <= table.logical_tokens,
                "prompt_len exceeds logical tokens for sequence {}",
                table.seq_id
            );
            for &block in &table.physical_block_ids {
                assert!(block < self.total_blocks, "out-of-range block id");
                table_refs[block] += 1;
            }
        }

        let mut free_count = 0usize;
        for (block, expected_table_refs) in table_refs.iter().enumerate() {
            assert_eq!(
                self.ref_count[block],
                *expected_table_refs + self.pins[block],
                "refcount mismatch for block {block}"
            );
            match self.states[block] {
                BlockState::Free => {
                    free_count += 1;
                    assert_eq!(self.ref_count[block], 0);
                    assert!(self.in_free[block]);
                    assert!(self.node_of[block].is_none());
                    assert!(self.lru_key[block].is_none());
                }
                BlockState::Cached => {
                    assert_eq!(self.ref_count[block], 0);
                    assert!(!self.in_free[block]);
                    assert!(self.node_of[block].is_some());
                }
                BlockState::Active => {
                    assert!(self.ref_count[block] >= 1);
                    assert!(!self.in_free[block]);
                    assert!(self.lru_key[block].is_none());
                }
            }

            if let Some(node) = self.node_of[block] {
                match self.nodes[node].res {
                    Some(Res::Gpu(resident)) => {
                        assert_eq!(resident, block, "node {node} points at wrong block");
                    }
                    _ => panic!("node_of[{block}] maps to a non-resident node"),
                }
                assert!(
                    matches!(self.states[block], BlockState::Active | BlockState::Cached),
                    "trie-mapped block {block} must be active or cached"
                );
            }
        }

        assert_eq!(
            self.free_stack.len(),
            free_count,
            "free stack length mismatch"
        );
        let mut seen = vec![false; self.total_blocks];
        for &block in &self.free_stack {
            assert!(block < self.total_blocks);
            assert!(!seen[block], "duplicate free block {block}");
            seen[block] = true;
            assert!(self.in_free[block]);
            assert_eq!(self.states[block], BlockState::Free);
        }
        assert_eq!(
            seen.iter().filter(|present| **present).count(),
            self.free_stack.len()
        );

        for (tick, &block) in &self.lru {
            assert_eq!(self.states[block], BlockState::Cached);
            assert_eq!(self.lru_key[block], Some(*tick));
            assert_eq!(self.ref_count[block], 0);
            if self.io.is_none() {
                let node = self.node_of[block].expect("LRU block without trie node");
                assert!(self.node_is_leaf(node), "drop-mode LRU contains non-leaf");
            }
        }

        if self.io.is_some() {
            for block in 0..self.total_blocks {
                if self.states[block] == BlockState::Cached {
                    assert!(
                        self.lru_key[block].is_some(),
                        "tier-mode cached block {block} missing from LRU"
                    );
                }
            }
        } else {
            for block in 0..self.total_blocks {
                if let Some(node) = self.node_of[block]
                    && self.states[block] == BlockState::Cached
                    && self.node_is_leaf(node)
                {
                    assert!(
                        self.lru_key[block].is_some(),
                        "cached leaf {block} missing from LRU"
                    );
                }
            }
        }

        for node in 1..self.nodes.len() {
            match self.nodes[node].res {
                Some(Res::Gpu(block)) => {
                    assert_eq!(self.node_of[block], Some(node));
                    assert!(matches!(
                        self.states[block],
                        BlockState::Active | BlockState::Cached
                    ));
                }
                Some(Res::Swapped) => {
                    assert!(
                        !self.node_of.contains(&Some(node)),
                        "swapped node still owns a block"
                    );
                }
                None => {}
            }
        }
    }
}
