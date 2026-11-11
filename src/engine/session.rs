// src/engine/session.rs
use crate::memory::{BlockId, BlockSpaceManager, SeqId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A committed checkpoint in a session branch.
///
/// `pinned` contains only full-token blocks (block-boundary granularity). The
/// complete token history is kept so checkout can resume the logical sequence
/// and re-prefill just the partial tail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitNode {
    pub id: u64,
    pub parent: Option<u64>,
    pub label: String,
    pub session: String,
    pub tokens: Vec<u32>,
    pub pinned: Vec<BlockId>,
}

impl CommitNode {
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionNode {
    pub id: u64,
    pub parent: Option<u64>,
    pub label: String,
    pub tokens: usize,
    pub pinned_blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTree {
    pub session: String,
    pub nodes: Vec<SessionNode>,
}

#[derive(Debug, Clone)]
pub struct CheckoutSnapshot {
    pub tokens: Vec<u32>,
    /// Number of tokens whose KV is already resident (full-block boundary,
    /// capped at `tokens.len() - 1` so the actor can obtain first-token logits).
    pub computed: usize,
}

/// Actor-owned session store. There is no lock: the engine thread is the only
/// writer and HTTP routes ask it through request messages.
#[derive(Debug, Default)]
pub struct SessionStore {
    next_id: u64,
    nodes: HashMap<u64, CommitNode>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            nodes: HashMap::new(),
        }
    }

    /// Pin the sequence's full blocks and record its complete token history.
    /// Pins are transactional: a failure unpins everything acquired so far.
    pub fn commit(
        &mut self,
        session: &str,
        parent: Option<u64>,
        label: &str,
        seq: SeqId,
        tokens: Vec<u32>,
        block_manager: &mut BlockSpaceManager,
    ) -> Result<u64, String> {
        if tokens.is_empty() {
            return Err("cannot commit an empty sequence".to_string());
        }
        if let Some(parent) = parent
            && !self.nodes.contains_key(&parent)
        {
            return Err(format!("parent commit node {parent} does not exist"));
        }
        if block_manager.page_table(seq).is_none() {
            return Err(format!("sequence {seq} has no resident page table"));
        }

        let full_blocks = tokens.len() / block_manager.block_size;
        let blocks = block_manager.blocks_of(seq);
        if blocks.len() < full_blocks {
            return Err(format!(
                "sequence {seq} has {} blocks but {full_blocks} full blocks are required",
                blocks.len()
            ));
        }
        let pinned = blocks[..full_blocks].to_vec();
        let mut acquired = Vec::with_capacity(pinned.len());
        for &block in &pinned {
            if let Err(error) = block_manager.pin(block) {
                for block in acquired.drain(..) {
                    let _ = block_manager.unpin(block);
                }
                return Err(error.to_string());
            }
            acquired.push(block);
        }

        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            CommitNode {
                id,
                parent,
                label: label.to_string(),
                session: session.to_string(),
                tokens,
                pinned,
            },
        );
        Ok(id)
    }

    /// Attach a committed snapshot to a new sequence. The attached prefix is an
    /// O(1) refcount operation; only the partial tail can allocate.
    pub fn checkout(
        &self,
        node_id: u64,
        seq: SeqId,
        block_manager: &mut BlockSpaceManager,
    ) -> Result<CheckoutSnapshot, String> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| format!("commit node {node_id} does not exist"))?;
        if node.tokens.is_empty() {
            return Err(format!("commit node {node_id} has no tokens"));
        }
        let block_size = block_manager.block_size;
        let full_tokens = node.pinned.len() * block_size;

        block_manager.attach_snapshot(seq, &node.pinned, full_tokens, full_tokens);
        let tail = node.tokens.len().saturating_sub(full_tokens);
        if tail > 0
            && let Err(error) = block_manager.append_tokens(seq, tail)
        {
            let _ = block_manager.free_sequence(seq);
            return Err(error.to_string());
        }

        // Recompute the last resident token when the checkpoint is block
        // aligned; otherwise the scheduler re-prefills the partial tail. Both
        // paths need at least one forward pass to produce first-token logits.
        let computed = full_tokens.min(node.tokens.len().saturating_sub(1));
        Ok(CheckoutSnapshot {
            tokens: node.tokens.clone(),
            computed,
        })
    }

    /// Remove a checkpoint and release its pins. Checked-out sequences keep
    /// their own page-table references, so this never corrupts live KV.
    pub fn forget(
        &mut self,
        node_id: u64,
        block_manager: &mut BlockSpaceManager,
    ) -> Result<(), String> {
        let node = self
            .nodes
            .remove(&node_id)
            .ok_or_else(|| format!("commit node {node_id} does not exist"))?;
        for &block in &node.pinned {
            if let Err(error) = block_manager.unpin(block) {
                return Err(format!(
                    "commit node {node_id} left a partially unpinned block: {error}"
                ));
            }
        }
        Ok(())
    }

    pub fn tree(&self, session: &str) -> SessionTree {
        let mut nodes: Vec<SessionNode> = self
            .nodes
            .values()
            .filter(|node| node.session == session)
            .map(|node| SessionNode {
                id: node.id,
                parent: node.parent,
                label: node.label.clone(),
                tokens: node.tokens.len(),
                pinned_blocks: node.pinned.len(),
            })
            .collect();
        nodes.sort_by_key(|node| node.id);
        SessionTree {
            session: session.to_string(),
            nodes,
        }
    }

    pub fn node(&self, node_id: u64) -> Option<&CommitNode> {
        self.nodes.get(&node_id)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> BlockSpaceManager {
        BlockSpaceManager::new(32, 4, 16)
    }

    #[test]
    fn commit_pins_full_blocks_and_survives_release() {
        let mut block_manager = manager();
        let prompt: Vec<u32> = (1..=10).collect();
        block_manager.match_or_allocate_prefix(1, &prompt).unwrap();
        block_manager.append_tokens(1, 2).unwrap();
        let mut tokens = prompt.clone();
        tokens.extend_from_slice(&[7, 8]);

        let mut sessions = SessionStore::new();
        let node_id = sessions
            .commit(
                "main",
                None,
                "checkpoint",
                1,
                tokens.clone(),
                &mut block_manager,
            )
            .unwrap();
        let node = sessions.node(node_id).unwrap();
        assert_eq!(node.pinned.len(), 3);
        assert!(
            node.pinned
                .iter()
                .all(|block| block_manager.pin_count(*block) == 1)
        );

        block_manager.free_sequence(1).unwrap();
        block_manager.assert_invariants();
        for &block in &node.pinned {
            assert_eq!(
                block_manager.state_of(block),
                crate::memory::BlockState::Active
            );
            assert_eq!(block_manager.ref_count(block), 1);
        }

        // Eviction pressure must not touch pinned blocks.
        for seq in 10..20 {
            let prompt: Vec<u32> = (0..4).map(|index| (seq * 10 + index) as u32).collect();
            block_manager
                .match_or_allocate_prefix(seq, &prompt)
                .unwrap();
            block_manager.free_sequence(seq).unwrap();
        }
        for seq in 100..130 {
            let prompt: Vec<u32> = (0..4).map(|index| (seq * 10 + index) as u32).collect();
            let _ = block_manager.match_or_allocate_prefix(seq, &prompt);
        }
        for &block in &node.pinned {
            assert_eq!(
                block_manager.state_of(block),
                crate::memory::BlockState::Active
            );
            assert_eq!(block_manager.pin_count(block), 1);
        }
        block_manager.assert_invariants();
    }

    #[test]
    fn checkout_is_o1_and_diverging_branches_have_private_tails() {
        let mut block_manager = manager();
        let prompt: Vec<u32> = (1..=12).collect();
        block_manager.match_or_allocate_prefix(1, &prompt).unwrap();
        let mut sessions = SessionStore::new();
        let node = sessions
            .commit(
                "main",
                None,
                "aligned",
                1,
                prompt.clone(),
                &mut block_manager,
            )
            .unwrap();
        block_manager.free_sequence(1).unwrap();

        let before = block_manager.allocated();
        let first = sessions.checkout(node, 2, &mut block_manager).unwrap();
        assert_eq!(first.computed, prompt.len() - 1);
        assert_eq!(
            block_manager.allocated(),
            before,
            "checkout must not allocate"
        );

        block_manager.append_tokens(2, 1).unwrap();
        let first_tail = *block_manager.blocks_of(2).last().unwrap();
        let second = sessions.checkout(node, 3, &mut block_manager).unwrap();
        assert_eq!(second.tokens, prompt);
        block_manager.append_tokens(3, 1).unwrap();
        let second_tail = *block_manager.blocks_of(3).last().unwrap();
        assert_ne!(
            first_tail, second_tail,
            "branches must not share append tails"
        );

        block_manager.assert_invariants();
        assert_eq!(block_manager.pin_count(node_pin(&sessions, node, 0)), 1);
    }

    fn node_pin(sessions: &SessionStore, node: u64, index: usize) -> BlockId {
        sessions.node(node).unwrap().pinned[index]
    }

    #[test]
    fn unaligned_checkout_allocates_only_the_tail() {
        let mut block_manager = manager();
        let prompt: Vec<u32> = (1..=10).collect();
        block_manager.match_or_allocate_prefix(1, &prompt).unwrap();
        let mut sessions = SessionStore::new();
        let node = sessions
            .commit(
                "main",
                None,
                "unaligned",
                1,
                prompt.clone(),
                &mut block_manager,
            )
            .unwrap();
        block_manager.free_sequence(1).unwrap();

        let before = block_manager.allocated();
        let snapshot = sessions.checkout(node, 2, &mut block_manager).unwrap();
        assert_eq!(snapshot.computed, 8);
        let after = block_manager.allocated();
        assert_eq!(after, before + 1, "only the partial tail may allocate");
        assert_eq!(block_manager.logical_tokens(2), Some(10));
        block_manager.assert_invariants();
    }

    #[test]
    fn forget_releases_pins_and_tree_lists_branches() {
        let mut block_manager = manager();
        let prompt: Vec<u32> = (1..=8).collect();
        block_manager.match_or_allocate_prefix(1, &prompt).unwrap();
        let mut sessions = SessionStore::new();
        let root = sessions
            .commit("main", None, "root", 1, prompt.clone(), &mut block_manager)
            .unwrap();
        let child = sessions
            .commit(
                "main",
                Some(root),
                "child",
                1,
                prompt.clone(),
                &mut block_manager,
            )
            .unwrap();
        block_manager.free_sequence(1).unwrap();

        let tree = sessions.tree("main");
        assert_eq!(tree.nodes.len(), 2);
        assert_eq!(tree.nodes[1].parent, Some(root));
        assert_eq!(tree.nodes[1].label, "child");

        sessions.forget(child, &mut block_manager).unwrap();
        assert_eq!(sessions.tree("main").nodes.len(), 1);
        sessions.forget(root, &mut block_manager).unwrap();
        for block in 0..8 {
            assert_eq!(block_manager.ref_count(block), 0);
        }
        block_manager.assert_invariants();
    }
}
