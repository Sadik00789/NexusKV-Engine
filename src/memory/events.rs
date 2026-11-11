// src/memory/events.rs
use super::{BlockInspectInfo, MemEvent, MemEventKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

pub const DEFAULT_EVENT_CAPACITY: usize = 50_000;
pub const DEFAULT_MAX_KEYFRAMES: usize = 256;
pub const KEYFRAME_INTERVAL: u64 = 256;

/// A full allocator snapshot at a tick boundary. Replays start here and apply
/// only the events in `(keyframe.tick, target_tick]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Keyframe {
    pub tick: u64,
    pub blocks: Vec<BlockInspectInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replay {
    pub keyframe: Keyframe,
    pub events: Vec<MemEvent>,
}

/// Bounded flight recorder for the time-travel debugger.
///
/// Events live in a 50k ring buffer. Full allocator keyframes are kept at most
/// 256 ticks apart and capped at 256 entries, so replay can always start from a
/// state no older than one keyframe interval.
#[derive(Debug)]
pub struct FlightRecorder {
    events: VecDeque<MemEvent>,
    keyframes: BTreeMap<u64, Keyframe>,
    event_capacity: usize,
    max_keyframes: usize,
    latest_tick: u64,
}

impl FlightRecorder {
    pub fn new(event_capacity: usize, max_keyframes: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(event_capacity.min(1024)),
            keyframes: BTreeMap::new(),
            event_capacity: event_capacity.max(1),
            max_keyframes: max_keyframes.max(1),
            latest_tick: 0,
        }
    }

    pub fn record_event(&mut self, event: MemEvent) {
        self.latest_tick = self.latest_tick.max(event.tick);
        self.events.push_back(event);
        while self.events.len() > self.event_capacity {
            self.events.pop_front();
        }
    }

    /// Record a full keyframe if the last one is at least one interval old.
    /// The tick label is the actual allocator tick, so replay boundaries remain
    /// exact even when a batch produces several events at once.
    pub fn maybe_keyframe(&mut self, tick: u64, blocks: Vec<BlockInspectInfo>) {
        self.latest_tick = self.latest_tick.max(tick);
        if let Some((&last_tick, _)) = self.keyframes.iter().next_back()
            && tick < last_tick + KEYFRAME_INTERVAL
        {
            return;
        }
        self.keyframes.insert(tick, Keyframe { tick, blocks });
        while self.keyframes.len() > self.max_keyframes {
            if let Some(first) = self.keyframes.keys().next().copied() {
                self.keyframes.remove(&first);
            }
        }
    }

    /// Nearest keyframe at or before `tick`, plus events in `(keyframe, tick]`.
    pub fn replay_to(&self, tick: u64) -> Option<Replay> {
        let (&keyframe_tick, keyframe) = self.keyframes.range(..=tick).next_back()?;
        let events = self
            .events
            .iter()
            .filter(|event| event.tick > keyframe_tick && event.tick <= tick)
            .cloned()
            .collect();
        Some(Replay {
            keyframe: keyframe.clone(),
            events,
        })
    }

    pub fn latest_tick(&self) -> u64 {
        self.latest_tick
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

impl Default for FlightRecorder {
    fn default() -> Self {
        Self::new(DEFAULT_EVENT_CAPACITY, DEFAULT_MAX_KEYFRAMES)
    }
}

/// Apply the dashboard replay reducer on the Rust side. This is used by the
/// determinism property test and keeps the visualization semantics honest.
///
/// `Cow` and `Fault` events are intentional no-ops: allocation/release events
/// already describe the reference changes, and a fault loads content into a
/// block that an `Alloc` event created.
pub fn replay_state(replay: &Replay) -> Vec<BlockInspectInfo> {
    let mut blocks = replay.keyframe.blocks.clone();
    for event in &replay.events {
        for &block in &event.blocks {
            let Some(entry) = blocks.get_mut(block) else {
                continue;
            };
            match event.kind {
                MemEventKind::Alloc | MemEventKind::Hit | MemEventKind::Fork => {
                    entry.ref_count += 1;
                    entry.is_free = false;
                    entry.state = "active".to_string();
                }
                MemEventKind::Pin => {
                    entry.ref_count += 1;
                    entry.is_free = false;
                    entry.state = "active".to_string();
                }
                MemEventKind::Free => {
                    entry.ref_count = entry.ref_count.saturating_sub(1);
                    if entry.ref_count == 0 {
                        if event.cached {
                            entry.is_free = false;
                            entry.state = "cached".to_string();
                        } else {
                            entry.is_free = true;
                            entry.state = "free".to_string();
                        }
                    } else {
                        entry.is_free = false;
                        entry.state = "active".to_string();
                    }
                }
                MemEventKind::Evict => {
                    entry.ref_count = 0;
                    entry.is_free = true;
                    entry.state = "free".to_string();
                }
                MemEventKind::Cow | MemEventKind::Rollback | MemEventKind::Fault => {}
            }
        }
    }
    blocks
}

/// Convenience used by tests and the dashboard contract: the free/active/cached
/// counts reconstructed from a replay.
pub fn replay_counts(blocks: &[BlockInspectInfo]) -> (usize, usize, usize) {
    let free = blocks.iter().filter(|block| block.is_free).count();
    let cached = blocks
        .iter()
        .filter(|block| block.state == "cached")
        .count();
    (free, cached, blocks.len() - free - cached)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(block_id: usize, state: &str, ref_count: usize) -> BlockInspectInfo {
        BlockInspectInfo {
            block_id,
            ref_count,
            is_free: state == "free",
            mapped_seq_id: None,
            is_prefix_cached: state == "cached",
            is_speculative: false,
            state: state.to_string(),
        }
    }

    fn event(tick: u64, kind: MemEventKind, blocks: Vec<usize>, cached: bool) -> MemEvent {
        MemEvent {
            tick,
            kind,
            seq: None,
            blocks,
            cached,
        }
    }

    #[test]
    fn replay_starts_at_the_nearest_keyframe() {
        let mut recorder = FlightRecorder::new(100, 4);
        recorder.maybe_keyframe(0, vec![block(0, "free", 0), block(1, "free", 0)]);
        recorder.record_event(event(1, MemEventKind::Alloc, vec![0], false));
        recorder.maybe_keyframe(300, vec![block(0, "active", 1), block(1, "free", 0)]);
        recorder.record_event(event(301, MemEventKind::Alloc, vec![1], false));

        let replay = recorder.replay_to(301).expect("replay");
        assert_eq!(replay.keyframe.tick, 300);
        assert_eq!(replay.events.len(), 1);
        let blocks = replay_state(&replay);
        assert_eq!(blocks[1].ref_count, 1);
        assert!(!blocks[1].is_free);
    }

    #[test]
    fn free_to_cache_is_distinguishable_from_free() {
        let replay = Replay {
            keyframe: Keyframe {
                tick: 0,
                blocks: vec![block(0, "active", 1)],
            },
            events: vec![event(1, MemEventKind::Free, vec![0], true)],
        };
        let blocks = replay_state(&replay);
        assert_eq!(blocks[0].state, "cached");
        assert_eq!(blocks[0].ref_count, 0);
        assert!(!blocks[0].is_free);
    }

    #[test]
    fn ring_budget_is_bounded() {
        let mut recorder = FlightRecorder::new(4, 2);
        for tick in 0..10 {
            recorder.record_event(event(tick, MemEventKind::Alloc, vec![0], false));
            recorder.maybe_keyframe(tick * KEYFRAME_INTERVAL, vec![block(0, "free", 0)]);
        }
        assert_eq!(recorder.event_count(), 4);
        assert_eq!(recorder.keyframe_count(), 2);
        assert_eq!(recorder.latest_tick(), 9 * KEYFRAME_INTERVAL);
    }
}
