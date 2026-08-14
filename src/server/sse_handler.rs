// src/server/sse_handler.rs
use crate::config::EngineConfig;
use crate::engine::{ContinuousScheduler, Sequence, SpeculativeVerifier};
use crate::memory::{BlockInspectInfo, BlockSpaceManager};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
};
use futures_util::stream::Stream;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Shared thread-safe application state utilizing parking_lot::RwLock
#[derive(Clone)]
pub struct AppState {
    pub config: EngineConfig,
    pub block_manager: Arc<RwLock<BlockSpaceManager>>,
    pub scheduler: Arc<RwLock<ContinuousScheduler>>,
    pub verifier: Arc<RwLock<SpeculativeVerifier>>,
    pub sequence_counter: Arc<AtomicUsize>,
}

impl AppState {
    pub fn new(config: EngineConfig) -> Self {
        let block_manager = Arc::new(RwLock::new(BlockSpaceManager::new(
            config.total_physical_blocks,
            config.block_size,
            config.max_blocks_per_seq,
        )));

        let scheduler = Arc::new(RwLock::new(ContinuousScheduler::new(
            config.max_num_seqs,
            Arc::clone(&block_manager),
        )));

        let verifier = Arc::new(RwLock::new(SpeculativeVerifier::new(Arc::clone(
            &block_manager,
        ))));

        Self {
            config,
            block_manager,
            scheduler,
            verifier,
            sequence_counter: Arc::new(AtomicUsize::new(1)),
        }
    }
}

/// RAII Drop Guard that triggers sequence cancellation and block deallocation
/// if the client disconnects or the SSE stream future drops mid-flight.
pub struct AbortDropGuard {
    pub seq_id: usize,
    pub scheduler: Arc<RwLock<ContinuousScheduler>>,
    pub completed: Arc<AtomicBool>,
}

impl Drop for AbortDropGuard {
    fn drop(&mut self) {
        if !self.completed.load(Ordering::SeqCst) {
            tracing::warn!(
                "Client connection dropped/aborted for sequence #{}. Immediate VRAM page reclamation triggered.",
                self.seq_id
            );
            let mut sched = self.scheduler.write();
            let _ = sched.abort_sequence(self.seq_id);
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[allow(dead_code)]
fn default_max_tokens() -> usize {
    64
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TelemetryPayload {
    pub sequence_id: usize,
    pub token: String,
    pub step: usize,
    pub tokens_per_sec: f64,
    pub ttft_ms: f64,
    pub itl_ms: f64,
    pub allocated_blocks: usize,
    pub total_blocks: usize,
    pub memory_usage_ratio: f64,
    pub prefix_cache_hit_ratio: f64,
    pub speculative_acceptance_rate: f64,
    pub active_block_ids: Vec<usize>,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemMetricsResponse {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub memory_usage_percent: f64,
    pub running_sequences: usize,
    pub waiting_sequences: usize,
    pub prefix_cache_hit_ratio: f64,
    pub prefix_cache_lookups: usize,
    pub prefix_cache_hits: usize,
    pub speculative_acceptance_rate: f64,
    pub active_sequence_ids: Vec<usize>,
    pub block_inspect_info: Vec<BlockInspectInfo>,
}

// ---------------------------------------------------------------------------
// OpenAI API Compatibility Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model_name")]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

#[allow(dead_code)]
fn default_model_name() -> String {
    "nexuskv-engine-v1".to_string()
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatChoiceDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatChoiceDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatChoiceMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatChoiceMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

// ---------------------------------------------------------------------------
// Route Handlers
// ---------------------------------------------------------------------------

/// Health check probe endpoint
pub async fn health_check() -> &'static str {
    "OK"
}

/// Models list endpoint conforming to OpenAI specification
pub async fn list_models() -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![
            ModelCard {
                id: "nexuskv-engine-v1".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "nexuskv".to_string(),
            },
            ModelCard {
                id: "nexuskv-speculative-v1".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "nexuskv".to_string(),
            },
        ],
    })
}

/// Real-time engine telemetry snapshot endpoint with active block mapping
pub async fn get_metrics(State(state): State<AppState>) -> Json<SystemMetricsResponse> {
    let block_mgr = state.block_manager.read();
    let scheduler = state.scheduler.read();
    let verifier = state.verifier.read();

    let total = state.config.total_physical_blocks;
    let allocated = block_mgr.allocated_blocks_count();
    let free = block_mgr.free_blocks_count();
    let prefix_ratio = block_mgr.prefix_cache_hit_ratio();
    let inspect_info = block_mgr.get_block_inspect_info();

    Json(SystemMetricsResponse {
        total_blocks: total,
        allocated_blocks: allocated,
        free_blocks: free,
        memory_usage_percent: (allocated as f64 / total.max(1) as f64) * 100.0,
        running_sequences: scheduler.running_count(),
        waiting_sequences: scheduler.waiting_count(),
        prefix_cache_hit_ratio: prefix_ratio,
        prefix_cache_lookups: block_mgr.prefix_cache_lookups,
        prefix_cache_hits: block_mgr.prefix_cache_hits,
        speculative_acceptance_rate: verifier.overall_acceptance_rate(),
        active_sequence_ids: scheduler.active_sequence_ids(),
        block_inspect_info: inspect_info,
    })
}

/// Server-Sent Events (SSE) streaming token generation & VRAM telemetry with AbortDropGuard
pub async fn stream_generation(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let seq_id = state.sequence_counter.fetch_add(1, Ordering::SeqCst);
    let max_tokens = payload.max_tokens;

    // Convert prompt to token IDs
    let prompt_tokens: Vec<u32> = if payload.prompt.is_empty() {
        vec![1, 2, 3]
    } else {
        payload.prompt.bytes().map(|b| b as u32).collect()
    };

    let sequence = Sequence::new(seq_id, prompt_tokens, max_tokens);

    {
        let mut scheduler = state.scheduler.write();
        scheduler.submit_sequence(sequence);
        // Run prefill schedule step
        let _ = scheduler.schedule_iteration();
    }

    let completed_flag = Arc::new(AtomicBool::new(false));
    let drop_guard = AbortDropGuard {
        seq_id,
        scheduler: Arc::clone(&state.scheduler),
        completed: Arc::clone(&completed_flag),
    };

    let scheduler_clone = Arc::clone(&state.scheduler);
    let block_mgr_clone = Arc::clone(&state.block_manager);
    let verifier_clone = Arc::clone(&state.verifier);
    let total_physical_blocks = state.config.total_physical_blocks;

    let stream = async_stream::stream! {
        let _guard = drop_guard; // Bound to async stream lifetime
        let start_time = Instant::now();
        let mut step = 0;
        let mut first_token_time: Option<Instant> = None;
        let mut last_token_time: Option<Instant> = None;

        let dummy_words = [
            "NexusKV", "executes", "PagedAttention", "with", "lock-free", "continuous",
            "batching", "and", "O(1)", "speculative", "tree", "page", "rollbacks", "at", "hardware", "line-rate", "."
        ];

        loop {
            tokio::time::sleep(Duration::from_millis(16)).await; // Simulation of GPU decode step
            step += 1;
            let now = Instant::now();

            if first_token_time.is_none() {
                first_token_time = Some(now);
            }
            let itl_ms = if let Some(last) = last_token_time {
                now.duration_since(last).as_secs_f64() * 1000.0
            } else {
                16.0
            };
            last_token_time = Some(now);

            let word = dummy_words[(step - 1) % dummy_words.len()];
            let token_str = format!("{} ", word);
            let is_last = step >= max_tokens;

            // Update scheduler & physical memory manager
            let (allocated, usage_ratio, acceptance_rate, prefix_hit_ratio, active_blocks) = {
                let mut sched = scheduler_clone.write();
                let _ = sched.step_sequence(seq_id, (step + 100) as u32, is_last);

                let block_mgr = block_mgr_clone.read();
                let verifier = verifier_clone.read();

                let active_b = block_mgr.get_sequence_blocks(seq_id).unwrap_or_default();

                (
                    block_mgr.allocated_blocks_count(),
                    block_mgr.memory_usage_ratio(),
                    verifier.overall_acceptance_rate(),
                    block_mgr.prefix_cache_hit_ratio(),
                    active_b,
                )
            };

            let elapsed_sec = start_time.elapsed().as_secs_f64();
            let tokens_per_sec = if elapsed_sec > 0.0 {
                (step as f64) / elapsed_sec
            } else {
                0.0
            };

            let ttft_ms = first_token_time
                .map(|t| t.duration_since(start_time).as_secs_f64() * 1000.0)
                .unwrap_or(16.0);

            let telemetry = TelemetryPayload {
                sequence_id: seq_id,
                token: token_str,
                step,
                tokens_per_sec: (tokens_per_sec * 10.0).round() / 10.0,
                ttft_ms: (ttft_ms * 10.0).round() / 10.0,
                itl_ms: (itl_ms * 10.0).round() / 10.0,
                allocated_blocks: allocated,
                total_blocks: total_physical_blocks,
                memory_usage_ratio: usage_ratio,
                prefix_cache_hit_ratio: (prefix_hit_ratio * 100.0).round() / 100.0,
                speculative_acceptance_rate: (acceptance_rate * 100.0).round() / 100.0,
                active_block_ids: active_blocks,
                is_finished: is_last,
                finish_reason: if is_last {
                    Some("MaxTokensReached".into())
                } else {
                    None
                },
            };

            if let Ok(json_data) = serde_json::to_string(&telemetry) {
                yield Ok(Event::default().data(json_data));
            }

            if is_last {
                completed_flag.store(true, Ordering::SeqCst);
                break;
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// OpenAI-Compatible `/v1/chat/completions` endpoint handling streaming and non-streaming requests
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    let max_tokens = payload.max_tokens.unwrap_or(64);
    let seq_id = state.sequence_counter.fetch_add(1, Ordering::SeqCst);
    let model = payload.model.clone();

    // Flatten chat messages into a single prompt string
    let mut prompt = String::new();
    for msg in &payload.messages {
        prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
    }
    prompt.push_str("<|assistant|>\n");

    let prompt_tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
    let prompt_token_count = prompt_tokens.len();

    let sequence = Sequence::new(seq_id, prompt_tokens, max_tokens);

    {
        let mut scheduler = state.scheduler.write();
        scheduler.submit_sequence(sequence);
        let _ = scheduler.schedule_iteration();
    }

    let created_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let completion_id = format!("chatcmpl-nexus-{}", seq_id);

    if payload.stream {
        // Streaming Mode: Return SSE event stream with ChatCompletionChunk objects
        let completed_flag = Arc::new(AtomicBool::new(false));
        let drop_guard = AbortDropGuard {
            seq_id,
            scheduler: Arc::clone(&state.scheduler),
            completed: Arc::clone(&completed_flag),
        };

        let scheduler_clone = Arc::clone(&state.scheduler);
        let stream = async_stream::stream! {
            let _guard = drop_guard;
            let mut step = 0;
            let dummy_words = [
                "NexusKV", "delivers", "high-throughput", "continuous", "batching",
                "with", "PagedAttention", "and", "Radix", "prefix", "caching", "."
            ];

            // 1. Initial role chunk
            let role_chunk = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: created_timestamp,
                model: model.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatChoiceDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            if let Ok(json_str) = serde_json::to_string(&role_chunk) {
                yield Ok::<Event, Infallible>(Event::default().data(json_str));
            }

            // 2. Token generation chunks
            loop {
                tokio::time::sleep(Duration::from_millis(16)).await;
                step += 1;
                let is_last = step >= max_tokens;

                let word = dummy_words[(step - 1) % dummy_words.len()];
                let token_content = format!("{} ", word);

                {
                    let mut sched = scheduler_clone.write();
                    let _ = sched.step_sequence(seq_id, (step + 100) as u32, is_last);
                }

                let chunk = ChatCompletionChunk {
                    id: completion_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created_timestamp,
                    model: model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatChoiceDelta {
                            role: None,
                            content: Some(token_content),
                        },
                        finish_reason: if is_last { Some("stop".to_string()) } else { None },
                    }],
                };

                if let Ok(json_str) = serde_json::to_string(&chunk) {
                    yield Ok::<Event, Infallible>(Event::default().data(json_str));
                }

                if is_last {
                    completed_flag.store(true, Ordering::SeqCst);
                    break;
                }
            }

            // 3. Final [DONE] event
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        // Non-Streaming Mode: Generate full text and return JSON
        let mut generated_words = Vec::new();
        let dummy_words = [
            "NexusKV", "delivers", "high-throughput", "continuous", "batching",
            "with", "PagedAttention", "and", "Radix", "prefix", "caching", "."
        ];

        for step in 1..=max_tokens {
            let is_last = step >= max_tokens;
            let word = dummy_words[(step - 1) % dummy_words.len()];
            generated_words.push(word);

            let mut sched = state.scheduler.write();
            let _ = sched.step_sequence(seq_id, (step + 100) as u32, is_last);
        }

        let full_text = generated_words.join(" ");

        let response = ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created: created_timestamp,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatChoiceMessage {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageInfo {
                prompt_tokens: prompt_token_count,
                completion_tokens: max_tokens,
                total_tokens: prompt_token_count + max_tokens,
            },
        };

        (StatusCode::OK, Json(response)).into_response()
    }
}