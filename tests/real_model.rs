#![cfg(feature = "candle-model")]
// tests/real_model.rs
//
// These tests require real model weights and are ignored by default. Set
// NEXUSKV_TEST_MODEL to a dense Qwen3-0.6B (or Qwen2.5-0.5B) directory
// containing config.json, tokenizer.json, and one or more *.safetensors shards,
// then run:
//
//   cargo test --features candle-model --test real_model -- --ignored --nocapture
//
// No GPU is required (candle runs on CPU), but a realistic model makes the test
// slow. The test proves greedy decoding is token-identical across two requests.

use nexuskv::config::EngineConfig;
use nexuskv::engine::{GenerateJob, Request, TokenMsg};
use nexuskv::model::qwen::load_model_and_tokenizer;
use nexuskv::server::AppState;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

fn real_config() -> EngineConfig {
    EngineConfig {
        total_physical_blocks: 192,
        block_size: 16,
        num_heads: 16,
        head_dim: 128,
        max_num_seqs: 8,
        max_blocks_per_seq: 24,
        num_layers: 28,
        max_num_batched_tokens: 256,
    }
}

async fn run_once(state: &AppState, prompt: Vec<u32>) -> Vec<u32> {
    let seq = state.next_sequence_id();
    let (tx, mut rx) = unbounded_channel();
    state
        .request_tx
        .send(Request::Generate(GenerateJob {
            seq,
            prompt,
            max_tokens: 8,
            temperature: 0.0,
            top_p: 1.0,
            spec: false,
            tx,
        }))
        .expect("actor request channel");

    let mut ids = Vec::new();
    loop {
        let message = tokio::time::timeout(Duration::from_secs(300), rx.recv())
            .await
            .expect("real model generation timed out")
            .expect("token channel closed");
        match message {
            TokenMsg::Token { id, .. } => ids.push(id),
            TokenMsg::Done { .. } => break,
            TokenMsg::Error { message } => panic!("model error: {message}"),
        }
    }
    ids
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires NEXUSKV_TEST_MODEL=<Qwen3/Qwen2 dir with config.json, tokenizer.json, *.safetensors>"]
async fn real_model_greedy_decoding_is_deterministic() {
    let directory = std::env::var("NEXUSKV_TEST_MODEL")
        .expect("set NEXUSKV_TEST_MODEL to a Qwen3/Qwen2 model directory");
    let config = real_config();
    let (executor, tokenizer) =
        load_model_and_tokenizer(&PathBuf::from(&directory), &config).expect("load model");
    let prompt = tokenizer.encode("The capital of France is");
    assert!(!prompt.is_empty());
    let state = AppState::with_executor(config, Box::new(executor), tokenizer, "qwen-test");

    let first = run_once(&state, prompt.clone()).await;
    let second = run_once(&state, prompt).await;
    assert!(!first.is_empty());
    assert_eq!(first, second, "temperature 0 must be deterministic");
}
