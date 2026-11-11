// tests/api.rs
mod common;

use common::{SlowExecutor, test_config};
use futures_util::StreamExt;
use nexuskv::model::MockTokenizer;
use nexuskv::server::{AppState, router};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

async fn spawn_server(state: AppState) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let address = listener.local_addr().unwrap();
    let app = router(state);
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), handle)
}

fn sse_json(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_models_and_metrics_are_honest() {
    let (url, _server) = spawn_server(AppState::mock(test_config())).await;
    let client = reqwest::Client::new();

    let health = client.get(format!("{url}/health")).send().await.unwrap();
    assert_eq!(health.text().await.unwrap(), "OK");

    let models: Value = client
        .get(format!("{url}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["object"], "list");
    assert_eq!(models["executor"], "mock");
    assert_eq!(models["data"][0]["executor"], "mock");

    let metrics: Value = client
        .get(format!("{url}/v1/metrics"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(metrics["executor"], "mock");
    assert!(metrics["total_blocks"].as_u64().unwrap() > 0);
    assert_eq!(metrics["total_blocks"], metrics["blocks"]);
    assert_eq!(metrics["allocated_blocks"], 0);
    assert_eq!(metrics["speculative_acceptance_rate"], json!(0.0));
    assert_eq!(metrics["acceptance"], json!(0.0));
    assert_eq!(metrics["adaptive_k"], 0);
    assert!(!metrics["block_inspect_info"].as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generate_stream_event_schema_and_telemetry() {
    let (url, _server) = spawn_server(AppState::mock(test_config())).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/generate/stream"))
        .json(&json!({
            "prompt": "stream schema test",
            "max_tokens": 6,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let body = response.text().await.unwrap();
    let events = sse_json(&body);
    assert_eq!(events.len(), 7, "6 tokens plus a final done event");

    for (index, event) in events[..6].iter().enumerate() {
        assert_eq!(event["sequence_id"], events[0]["sequence_id"]);
        assert_eq!(event["step"], index + 1);
        assert!(event["token"].as_str().is_some());
        assert_eq!(event["is_finished"], false);
        assert_eq!(event["executor"], "mock");
        assert!(event["allocated_blocks"].as_u64().unwrap() <= 64);
        assert!(event["active_block_ids"].as_array().is_some());
    }
    assert!(
        events[..6]
            .iter()
            .any(|event| !event["active_block_ids"].as_array().unwrap().is_empty()),
        "active block ids must be real"
    );
    let last = events.last().unwrap();
    assert_eq!(last["is_finished"], true);
    assert_eq!(last["finish_reason"], "length");
    assert!(events[0]["ttft_ms"].as_f64().unwrap() >= 0.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn different_prompts_produce_different_token_streams() {
    let (url, _server) = spawn_server(AppState::mock(test_config())).await;
    let client = reqwest::Client::new();

    let mut outputs = Vec::new();
    for prompt in ["alpha", "beta"] {
        let body = client
            .post(format!("{url}/v1/generate/stream"))
            .json(&json!({"prompt": prompt, "max_tokens": 5, "temperature": 0.0}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let text: String = sse_json(&body)
            .iter()
            .filter_map(|event| event["token"].as_str())
            .collect();
        assert!(!text.is_empty());
        outputs.push(text);
    }
    assert_ne!(outputs[0], outputs[1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chat_streaming_ends_with_done_and_finish_reason() {
    let (url, _server) = spawn_server(AppState::mock(test_config())).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "nexuskv-engine-v1",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let body = response.text().await.unwrap();
    assert!(
        body.trim_end().ends_with("[DONE]"),
        "missing [DONE]: {body}"
    );

    let chunks = sse_json(&body);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert!(!content.is_empty());
    let finish_reason = chunks
        .iter()
        .find_map(|chunk| chunk["choices"][0]["finish_reason"].as_str())
        .expect("finish reason chunk");
    assert_eq!(finish_reason, "length");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chat_non_streaming_returns_usage() {
    let (url, _server) = spawn_server(AppState::mock(test_config())).await;
    let response: Value = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({
            "model": "nexuskv-engine-v1",
            "messages": [{"role": "user", "content": "usage please"}],
            "max_tokens": 7,
            "temperature": 0.0,
            "stream": false
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["finish_reason"], "length");
    assert_eq!(response["usage"]["completion_tokens"], 7);
    let prompt_tokens = response["usage"]["prompt_tokens"].as_u64().unwrap();
    let total_tokens = response["usage"]["total_tokens"].as_u64().unwrap();
    assert!(prompt_tokens > 0);
    assert_eq!(total_tokens, prompt_tokens + 7);
    assert!(
        !response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_disconnect_triggers_abort() {
    let mut config = test_config();
    config.max_num_seqs = 2;
    let state = AppState::with_executor(
        config,
        Box::new(SlowExecutor::new(10)),
        Arc::new(MockTokenizer),
        "nexuskv-mock-v1",
    );
    let baseline = state.snapshot().active_blocks;
    let (url, _server) = spawn_server(state).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{url}/v1/generate/stream"))
        .json(&json!({
            "prompt": "this request will be disconnected mid-stream",
            "max_tokens": 100_000,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());

    let mut stream = response.bytes_stream();
    let first = stream.next().await.expect("first SSE chunk").unwrap();
    assert!(!first.is_empty());
    drop(stream);

    let mut released = false;
    for _ in 0..500 {
        let metrics: Value = client
            .get(format!("{url}/v1/metrics"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if metrics["active_blocks"].as_u64() == Some(baseline as u64) {
            released = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(released, "disconnect must release all active blocks");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_prompt_is_rejected_not_fabricated() {
    let (url, _server) = spawn_server(AppState::mock(test_config())).await;
    let response = reqwest::Client::new()
        .post(format!("{url}/v1/generate/stream"))
        .json(&json!({"prompt": "", "max_tokens": 4}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
}
