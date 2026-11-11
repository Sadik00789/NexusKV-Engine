// tests/speculative.rs
//! End-to-end speculative decoding proofs against the deterministic mock
//! executor: greedy output identity, real acceptance counters, and rollback
//! accounting under bounded memory.

mod common;

use common::test_config;
use nexuskv::engine::{FinishReason, GenerateJob, Request, TokenMsg};
use nexuskv::model::MockTokenizer;
use nexuskv::server::AppState;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

fn spec_config() -> nexuskv::config::EngineConfig {
    let mut config = test_config();
    config.total_physical_blocks = 256;
    config.max_num_seqs = 2;
    config.max_blocks_per_seq = 48;
    config.max_num_batched_tokens = 512;
    config
}

async fn generate(state: &AppState, prompt: &[u8], max_tokens: usize, spec: bool) -> Vec<u32> {
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
            spec,
            tx,
        }))
        .unwrap();

    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("timed out waiting for speculative sequence")
        {
            Some(TokenMsg::Token { id, .. }) => tokens.push(id),
            Some(TokenMsg::Done { reason }) => {
                assert!(
                    matches!(
                        reason,
                        FinishReason::MaxTokensReached | FinishReason::StopTokenEncountered
                    ),
                    "unexpected finish reason {reason:?}"
                );
                break;
            }
            Some(TokenMsg::Error { message }) => panic!("speculative run failed: {message}"),
            None => panic!("token channel closed before Done"),
        }
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn greedy_speculative_matches_plain_decoding_on_twenty_prompts() {
    let state = AppState::with_executor(
        spec_config(),
        Box::new(nexuskv::model::MockExecutor::new()),
        Arc::new(MockTokenizer),
        "nexuskv-mock-v1",
    );

    for index in 0..20 {
        let prompt = format!("abcabcabc{index}abcabcabc");
        let plain = generate(&state, prompt.as_bytes(), 18, false).await;
        let speculative = generate(&state, prompt.as_bytes(), 18, true).await;
        assert_eq!(
            plain, speculative,
            "speculative greedy output diverged for prompt {index}"
        );
        assert_eq!(plain.len(), 18);
    }

    let mut snapshot = state.snapshot();
    for _ in 0..200 {
        snapshot = state.snapshot();
        if snapshot.speculative_proposed > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        snapshot.speculative_proposed > 0,
        "the bandit must actually propose drafts"
    );
    assert!(
        snapshot.speculative_accepted <= snapshot.speculative_proposed,
        "acceptance cannot exceed proposals"
    );
    assert_eq!(
        snapshot.acceptance,
        snapshot.speculative_accepted as f64 / snapshot.speculative_proposed as f64
    );
    assert!(snapshot.adaptive_k > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_speculative_requests_report_zero_acceptance() {
    let state = AppState::mock(test_config());
    let _ = generate(&state, b"no drafts here", 6, false).await;
    let snapshot = state.snapshot();
    assert_eq!(snapshot.speculative_proposed, 0);
    assert_eq!(snapshot.speculative_accepted, 0);
    assert_eq!(snapshot.acceptance, 0.0);
    assert_eq!(snapshot.adaptive_k, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn speculative_runs_under_a_small_pool_stay_consistent() {
    let mut config = test_config();
    config.total_physical_blocks = 16;
    config.max_num_seqs = 1;
    config.max_blocks_per_seq = 4;
    config.max_num_batched_tokens = 64;
    let state = AppState::with_executor(
        config,
        Box::new(nexuskv::model::MockExecutor::new()),
        Arc::new(MockTokenizer),
        "nexuskv-mock-v1",
    );

    for _ in 0..5 {
        let tokens = generate(&state, b"pool", 8, true).await;
        assert_eq!(tokens.len(), 8);
    }

    for _ in 0..500 {
        let snapshot = state.snapshot();
        if snapshot.running_sequences == 0 && snapshot.active_blocks == 0 {
            assert_eq!(snapshot.free_blocks + snapshot.cached_blocks, 16);
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("engine retained active blocks after bounded speculative runs");
}
