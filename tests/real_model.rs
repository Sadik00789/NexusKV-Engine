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
use nexuskv::model::Executor;
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

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU, CUDA build, and NEXUSKV_TEST_MODEL=<Qwen3 dir>"]
fn real_weights_custom_kernel_matches_gather_logits_and_greedy_tokens() {
    use nexuskv::memory::BlockSpaceManager;

    let directory = std::env::var("NEXUSKV_TEST_MODEL")
        .expect("set NEXUSKV_TEST_MODEL to a Qwen3/Qwen2 model directory");
    let config = real_config();
    let (mut executor, tokenizer) =
        load_model_and_tokenizer(&PathBuf::from(&directory), &config).expect("load model");
    let long_prompt = "The capital of France is Paris. ".repeat(16);
    let mut prompt = tokenizer.encode(&long_prompt);
    // Block-align the prompt so both sequences share only immutable full
    // blocks; decode-only attention is what we are comparing.
    prompt.truncate((prompt.len() / config.block_size) * config.block_size);
    assert!(prompt.len() >= config.block_size * 2, "prompt too short");

    let mut block_manager =
        BlockSpaceManager::new(config.total_physical_blocks, config.block_size, 24);
    block_manager
        .match_or_allocate_prefix(1, &prompt)
        .expect("seq 1 prompt");
    block_manager
        .match_or_allocate_prefix(2, &prompt)
        .expect("seq 2 prompt");

    let gather_prefill = executor
        .forward_with_cuda_attention(&mut block_manager, 1, &prompt, 0, false, false)
        .expect("gather prefill");
    let kernel_prefill = executor
        .forward_with_cuda_attention(&mut block_manager, 2, &prompt, 0, false, false)
        .expect("kernel-side prefill");
    assert_eq!(
        gather_prefill, kernel_prefill,
        "prefill is gather on both paths and must be identical"
    );
    let mut token = argmax(&gather_prefill[0]);

    for step in 0..8 {
        block_manager.append_tokens(1, 1).expect("seq 1 append");
        block_manager.append_tokens(2, 1).expect("seq 2 append");
        for (src, dst) in block_manager.take_pending_copies() {
            executor.copy_block(src, dst).expect("copy block");
        }
        let start = prompt.len() + step;
        let gather = executor
            .forward_with_cuda_attention(&mut block_manager, 1, &[token], start, false, false)
            .expect("gather decode");
        let kernel = executor
            .forward_with_cuda_attention(&mut block_manager, 2, &[token], start, false, true)
            .expect("custom kernel decode");
        let mut worst_absolute = 0f32;
        let mut squared_difference = 0f64;
        let mut squared_reference = 0f64;
        for (a, b) in gather[0].iter().zip(kernel[0].iter()) {
            let difference = f64::from(a - b);
            squared_difference += difference * difference;
            squared_reference += f64::from(*b) * f64::from(*b);
            worst_absolute = worst_absolute.max((a - b).abs());
        }
        // Relative L2 over the full vocab vector is the standard logit-parity
        // metric; per-element relative error near zero is dominated by f16
        // rounding and carries no token-selection information.
        let relative_l2 = (squared_difference / squared_reference.max(1e-12)).sqrt();
        assert!(
            relative_l2 < 2e-2,
            "step {step}: logit relative L2 {relative_l2} exceeds 2e-2 \
             (max absolute {worst_absolute})"
        );
        let gather_token = argmax(&gather[0]);
        let kernel_token = argmax(&kernel[0]);
        assert_eq!(
            gather_token, kernel_token,
            "step {step}: greedy token diverged (relative L2 {relative_l2}, \
             max absolute {worst_absolute})"
        );
        token = gather_token;
    }
}

#[cfg(feature = "cuda")]
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
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
