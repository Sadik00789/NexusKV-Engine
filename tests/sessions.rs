// tests/sessions.rs
//! Git-for-KV sessions, flight-recorder replay, and time-travel API proofs.

mod common;

use common::{SlowExecutor, test_config};
use futures_util::StreamExt;
use nexuskv::memory::{
    BlockInspectInfo, BlockSpaceManager, FlightRecorder, MemEvent, replay_state,
};
use nexuskv::model::MockTokenizer;
use nexuskv::server::{AppState, router};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, upper: u64) -> u64 {
        if upper == 0 { 0 } else { self.next() % upper }
    }
}

fn compare_states(actual: &[BlockInspectInfo], replayed: &[BlockInspectInfo], tick: u64) {
    assert_eq!(actual.len(), replayed.len());
    for (index, (actual, replayed)) in actual.iter().zip(replayed.iter()).enumerate() {
        assert_eq!(
            actual.ref_count, replayed.ref_count,
            "tick {tick} block {index} refcount"
        );
        assert_eq!(
            actual.is_free, replayed.is_free,
            "tick {tick} block {index} free flag"
        );
        assert_eq!(
            actual.state, replayed.state,
            "tick {tick} block {index} state"
        );
    }
}

#[test]
fn replay_reconstructs_allocator_state_at_recorded_ticks() {
    let recorder = Arc::new(Mutex::new(FlightRecorder::new(50_000, 256)));
    let mut block_manager = BlockSpaceManager::new(24, 4, 10);
    {
        let recorder = Arc::clone(&recorder);
        block_manager.set_event_sink(Box::new(move |event: MemEvent| {
            recorder.lock().unwrap().record_event(event);
        }));
    }
    recorder
        .lock()
        .unwrap()
        .maybe_keyframe(0, block_manager.inspect());

    let mut rng = Lcg::new(20260501);
    let mut next_seq = 1usize;
    let mut live: Vec<usize> = Vec::new();
    let mut actual: HashMap<u64, Vec<BlockInspectInfo>> = HashMap::new();
    actual.insert(0, block_manager.inspect());

    for _ in 0..400 {
        match rng.below(100) {
            0..=29 => {
                let length = 1 + rng.below(16) as usize;
                let prompt: Vec<u32> = (0..length).map(|_| rng.below(32) as u32).collect();
                if block_manager
                    .match_or_allocate_prefix(next_seq, &prompt)
                    .is_ok()
                {
                    live.push(next_seq);
                }
                next_seq += 1;
            }
            30..=59 => {
                if let Some(&seq) = live.get(rng.below(live.len().max(1) as u64) as usize) {
                    let _ = block_manager.append_tokens(seq, 1 + rng.below(4) as usize);
                }
            }
            60..=74 => {
                if let Some(&parent) = live.get(rng.below(live.len().max(1) as u64) as usize) {
                    let child = next_seq;
                    next_seq += 1;
                    if block_manager.fork(parent, child).is_ok() {
                        live.push(child);
                    }
                }
            }
            75..=89 => {
                if !live.is_empty() {
                    let index = rng.below(live.len() as u64) as usize;
                    let seq = live.swap_remove(index);
                    let _ = block_manager.free_sequence(seq);
                }
            }
            _ => {
                if let Some(&seq) = live.get(rng.below(live.len().max(1) as u64) as usize)
                    && let Some(table) = block_manager.page_table(seq)
                {
                    let current = table.logical_tokens;
                    let prompt = table.prompt_len;
                    if current > prompt {
                        let keep = prompt + rng.below((current - prompt) as u64 + 1) as usize;
                        let _ = block_manager.rollback(seq, keep);
                    }
                }
            }
        }

        block_manager.assert_invariants();
        let tick = block_manager.event_tick();
        let inspect = block_manager.inspect();
        recorder
            .lock()
            .unwrap()
            .maybe_keyframe(tick, inspect.clone());
        actual.insert(tick, inspect);
    }

    let ticks: Vec<u64> = actual.keys().copied().collect();
    assert!(ticks.len() > 50);
    for _ in 0..50 {
        let tick = ticks[rng.below(ticks.len() as u64) as usize];
        let replay = recorder
            .lock()
            .unwrap()
            .replay_to(tick)
            .unwrap_or_else(|| panic!("no replay for tick {tick}"));
        assert!(replay.keyframe.tick <= tick);
        assert!(replay.events.iter().all(|event| event.tick <= tick));
        let replayed = replay_state(&replay);
        compare_states(&actual[&tick], &replayed, tick);
    }
}

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

fn first_event(bytes: &[u8]) -> Value {
    let text = String::from_utf8_lossy(bytes);
    let data = text
        .lines()
        .find_map(|line| line.strip_prefix("data:"))
        .expect("SSE data line")
        .trim();
    serde_json::from_str(data).expect("SSE JSON")
}

async fn wait_active_baseline(state: &AppState, expected: usize) -> usize {
    for _ in 0..500 {
        let active = state.snapshot().active_blocks;
        if active == expected {
            return active;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    state.snapshot().active_blocks
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_commit_checkout_forget_cycle_keeps_pins_alive() {
    let mut config = test_config();
    config.total_physical_blocks = 128;
    config.max_num_seqs = 4;
    config.max_blocks_per_seq = 32;
    let state = AppState::with_executor(
        config,
        Box::new(SlowExecutor::new(5)),
        Arc::new(MockTokenizer),
        "nexuskv-mock-v1",
    );
    let (url, _server) = spawn_server(state.clone()).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{url}/v1/generate/stream"))
        .json(&json!({
            "prompt": "session commit prompt that is long enough for several blocks",
            "max_tokens": 100_000,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let mut stream = response.bytes_stream();
    let mut seq = 0u64;
    let mut active_blocks = 0usize;
    for _ in 0..10 {
        let chunk = stream.next().await.expect("SSE chunk").unwrap();
        let event = first_event(&chunk);
        if let Some(id) = event["sequence_id"].as_u64() {
            seq = id;
        }
        active_blocks = event["active_block_ids"]
            .as_array()
            .map(|blocks| blocks.len())
            .unwrap_or(0);
        if active_blocks > 1 {
            break;
        }
    }
    assert!(active_blocks > 1, "telemetry must expose physical blocks");

    let commit: Value = client
        .post(format!("{url}/v1/sessions/commit"))
        .json(&json!({
            "session": "alpha",
            "parent": null,
            "label": "root",
            "seq": seq
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let node = commit["node"].as_u64().expect("commit node id");

    drop(stream);
    let mut pinned_baseline = 0usize;
    for _ in 0..500 {
        let snapshot = state.snapshot();
        if snapshot.running_sequences == 0 {
            pinned_baseline = snapshot.active_blocks;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        pinned_baseline > 0,
        "committed blocks must stay pinned after the producer is aborted"
    );

    let checkout = client
        .post(format!("{url}/v1/sessions/checkout"))
        .json(&json!({
            "node": node,
            "max_tokens": 100_000,
            "temperature": 0.0
        }))
        .send()
        .await
        .unwrap();
    assert!(checkout.status().is_success());
    let mut checkout_stream = checkout.bytes_stream();
    let first_checkout = checkout_stream
        .next()
        .await
        .expect("checkout SSE chunk")
        .unwrap();
    let checkout_event = first_event(&first_checkout);
    assert_ne!(checkout_event["sequence_id"], json!(seq));
    drop(checkout_stream);

    let after_checkout = wait_active_baseline(&state, pinned_baseline).await;
    assert_eq!(
        after_checkout, pinned_baseline,
        "checkout abort must release table refs but not commit pins"
    );

    let tree: Value = client
        .get(format!("{url}/v1/sessions/alpha/tree"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(tree["session"], "alpha");
    assert_eq!(tree["nodes"].as_array().unwrap().len(), 1);
    assert_eq!(tree["nodes"][0]["id"], node);

    let forgotten: Value = client
        .post(format!("{url}/v1/sessions/forget"))
        .json(&json!({"node": node}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(forgotten["forgotten"], node);

    let released = wait_active_baseline(&state, 0).await;
    assert_eq!(released, 0, "forget must release the committed pins");

    let tree: Value = client
        .get(format!("{url}/v1/sessions/alpha/tree"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(tree["nodes"].as_array().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timetravel_route_returns_keyframe_and_events() {
    let state = AppState::mock(test_config());
    let (url, _server) = spawn_server(state.clone()).await;
    let client = reqwest::Client::new();

    let body = client
        .post(format!("{url}/v1/generate/stream"))
        .json(&json!({"prompt": "time travel telemetry", "max_tokens": 5}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("is_finished"));

    let latest = state.snapshot().event_tick;
    let replay: Value = client
        .get(format!("{url}/v1/timetravel"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(replay["keyframe"]["tick"].as_u64().unwrap() <= latest);
    assert!(!replay["keyframe"]["blocks"].as_array().unwrap().is_empty());
    assert!(replay["events"].as_array().is_some());

    let at_zero: Value = client
        .get(format!("{url}/v1/timetravel?tick=0"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(at_zero["keyframe"]["tick"], 0);
    assert_eq!(
        at_zero["keyframe"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|block| block["is_free"] == json!(true))
            .count(),
        at_zero["keyframe"]["blocks"].as_array().unwrap().len()
    );
}
