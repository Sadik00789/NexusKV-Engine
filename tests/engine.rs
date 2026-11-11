// tests/engine.rs
mod common;

use common::{SlowExecutor, test_config};
use nexuskv::engine::{FinishReason, GenerateJob, Request, TokenMsg};
use nexuskv::model::{MockTokenizer, Tokenizer};
use nexuskv::server::AppState;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

async fn wait_for_idle(state: &AppState, baseline_active: usize) {
    for _ in 0..500 {
        if state.snapshot().active_blocks == baseline_active {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "engine did not return to baseline active blocks: {:?}",
        state.snapshot().active_blocks
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_stream_and_finish() {
    let state = AppState::mock(test_config());
    let seq = state.next_sequence_id();
    let (tx, mut rx) = unbounded_channel();
    state
        .request_tx
        .send(Request::Generate(GenerateJob {
            seq,
            prompt: b"hello engine"
                .iter()
                .map(|&byte| u32::from(byte))
                .collect(),
            max_tokens: 12,
            temperature: 0.0,
            top_p: 1.0,
            spec: false,
            tx,
        }))
        .unwrap();

    let mut tokens = Vec::new();
    let mut done = None;
    for _ in 0..100 {
        let message = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out waiting for token")
            .expect("channel closed");
        match message {
            TokenMsg::Token { id, step, text } => {
                assert_eq!(step, tokens.len() + 1);
                assert!(!text.is_empty());
                tokens.push(id);
            }
            TokenMsg::Done { reason } => {
                done = Some(reason);
                break;
            }
            TokenMsg::Error { message } => panic!("unexpected error: {message}"),
        }
    }

    assert_eq!(tokens.len(), 12);
    assert_eq!(done, Some(FinishReason::MaxTokensReached));
    for _ in 0..200 {
        if state.snapshot().total_tokens_generated >= 12 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("actor did not publish generated-token telemetry");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_mid_stream_releases_active_blocks() {
    let state = AppState::with_executor(
        test_config(),
        Box::new(SlowExecutor::new(5)),
        Arc::new(MockTokenizer),
        "nexuskv-mock-v1",
    );
    let baseline = state.snapshot().active_blocks;
    let seq = state.next_sequence_id();
    let (tx, mut rx) = unbounded_channel();
    state
        .request_tx
        .send(Request::Generate(GenerateJob {
            seq,
            prompt: b"abort this long generation"
                .iter()
                .map(|&byte| u32::from(byte))
                .collect(),
            max_tokens: 10_000,
            temperature: 0.0,
            top_p: 1.0,
            spec: false,
            tx,
        }))
        .unwrap();

    for _ in 0..3 {
        let message = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert!(matches!(message, TokenMsg::Token { .. }));
    }

    state.request_tx.send(Request::Abort { seq }).unwrap();

    let mut aborted = false;
    for _ in 0..100 {
        match tokio::time::timeout(Duration::from_secs(2), rx.recv()).await {
            Ok(Some(TokenMsg::Done { reason })) => {
                assert_eq!(reason, FinishReason::Aborted);
                aborted = true;
                break;
            }
            Ok(Some(TokenMsg::Token { .. })) => continue,
            Ok(Some(TokenMsg::Error { message })) => panic!("unexpected error: {message}"),
            Ok(None) => break,
            Err(_) => break,
        }
    }
    assert!(aborted, "abort must produce Done(Aborted)");

    wait_for_idle(&state, baseline).await;
    let snapshot = state.snapshot();
    assert_eq!(snapshot.active_blocks, baseline);
    assert_eq!(snapshot.running_sequences, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_sequences_share_prefix_blocks() {
    let mut config = test_config();
    config.total_physical_blocks = 128;
    config.max_num_seqs = 2;
    let state = AppState::with_executor(
        config,
        Box::new(SlowExecutor::new(5)),
        Arc::new(MockTokenizer),
        "nexuskv-mock-v1",
    );

    let prompt: Vec<u32> = b"shared system prompt: you are a deterministic echo model."
        .iter()
        .map(|&byte| u32::from(byte))
        .collect();
    let mut receivers = Vec::new();
    let mut ids = Vec::new();
    for _ in 0..2 {
        let seq = state.next_sequence_id();
        let (tx, rx) = unbounded_channel();
        state
            .request_tx
            .send(Request::Generate(GenerateJob {
                seq,
                prompt: prompt.clone(),
                max_tokens: 100,
                temperature: 0.0,
                top_p: 1.0,
                spec: false,
                tx,
            }))
            .unwrap();
        ids.push(seq);
        receivers.push(rx);
    }

    let mut shared = None;
    let mut ref_count = 0usize;
    for _ in 0..500 {
        let snapshot = state.snapshot();
        if let (Some(first), Some(second)) = (
            snapshot.seq_blocks.get(&ids[0]),
            snapshot.seq_blocks.get(&ids[1]),
        ) && let Some(&block) = first.iter().find(|block| second.contains(*block))
        {
            shared = Some(block);
            ref_count = snapshot
                .block_inspect_info
                .iter()
                .find(|info| info.block_id == block)
                .map(|info| info.ref_count)
                .unwrap_or(0);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let shared = shared.expect("concurrent shared-prefix sequences must share a block");
    assert!(
        ref_count >= 2,
        "shared physical block {shared} must have refcount >= 2, got {ref_count}"
    );

    for seq in ids {
        state.request_tx.send(Request::Abort { seq }).unwrap();
    }
    for mut rx in receivers {
        while let Ok(Some(_)) = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {}
    }
}

async fn generate_tokens(state: &AppState, prompt: Vec<u32>, max_tokens: usize) -> Vec<u32> {
    let seq = state.next_sequence_id();
    let (tx, mut rx) = unbounded_channel();
    state
        .request_tx
        .send(Request::Generate(GenerateJob {
            seq,
            prompt,
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            spec: false,
            tx,
        }))
        .unwrap();
    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out")
        {
            Some(TokenMsg::Token { id, .. }) => tokens.push(id),
            Some(TokenMsg::Done { .. }) => break,
            Some(TokenMsg::Error { message }) => panic!("unexpected error: {message}"),
            None => break,
        }
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repeated_block_aligned_prompt_skips_all_prefill_compute() {
    let state = AppState::mock(test_config());
    // 48 tokens = 12 blocks at the test block size of 4, leaving room for the
    // output within max_blocks_per_seq=16.
    let prompt: Vec<u32> = (0..48).map(|index| (index % 251) as u32).collect();

    let first = generate_tokens(&state, prompt.clone(), 12).await;
    for _ in 0..200 {
        if state.snapshot().prefill_tokens >= prompt.len() as u64 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let before = state.snapshot();
    assert_eq!(before.prefill_tokens, prompt.len() as u64);

    let second = generate_tokens(&state, prompt.clone(), 12).await;
    for _ in 0..200 {
        if state.snapshot().total_tokens_generated >= 24 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let after = state.snapshot();
    assert_eq!(
        first, second,
        "greedy output must be identical with prefill skipped"
    );
    assert_eq!(
        after.prefill_tokens, before.prefill_tokens,
        "second full-prefix request must not recompute prefill tokens"
    );
    assert!(
        after.prefix_skip_tokens >= before.prefix_skip_tokens + prompt.len() as u64,
        "second request must report the reused prompt tokens: before={} after={}",
        before.prefix_skip_tokens,
        after.prefix_skip_tokens
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn greedy_decoding_is_deterministic_across_requests() {
    let state = AppState::mock(test_config());
    let prompt: Vec<u32> = b"deterministic greedy prompt"
        .iter()
        .map(|&byte| u32::from(byte))
        .collect();
    let mut runs = Vec::new();

    for _ in 0..2 {
        let seq = state.next_sequence_id();
        let (tx, mut rx) = unbounded_channel();
        state
            .request_tx
            .send(Request::Generate(GenerateJob {
                seq,
                prompt: prompt.clone(),
                max_tokens: 10,
                temperature: 0.0,
                top_p: 1.0,
                spec: false,
                tx,
            }))
            .unwrap();
        let mut ids = Vec::new();
        loop {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("timed out")
            {
                Some(TokenMsg::Token { id, .. }) => ids.push(id),
                Some(TokenMsg::Done { .. }) => break,
                Some(TokenMsg::Error { message }) => panic!("unexpected error: {message}"),
                None => break,
            }
        }
        runs.push(ids);
    }

    assert_eq!(runs[0], runs[1], "temperature 0 must be deterministic");
    assert_eq!(runs[0].len(), 10);
}

#[test]
fn mock_tokenizer_is_byte_based() {
    let tokenizer = MockTokenizer;
    let tokens = tokenizer.encode("AB");
    assert_eq!(tokens, vec![65, 66]);
    assert_eq!(tokenizer.decode(&tokens), "AB");
}
