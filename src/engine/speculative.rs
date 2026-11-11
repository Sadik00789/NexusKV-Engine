// src/engine/speculative.rs
use serde::{Deserialize, Serialize};

/// Honest speculative-decoding counters.
///
/// Before phase 4 there is no drafter, so `proposed` stays zero and the
/// acceptance rate is exactly 0.0. There is deliberately no default estimate.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
