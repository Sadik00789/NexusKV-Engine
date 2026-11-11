// src/server/sse_handler.rs
use crate::config::EngineConfig;
use crate::engine::{
    EngineActor, GenerateJob, Request, Scheduler, Snapshot, TelemetrySink, TokenMsg,
};
use crate::memory::{BlockIo, BlockSpaceManager, FlightRecorder, MemEvent};
use crate::model::{Executor, MockExecutor, MockTokenizer, Tokenizer};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;

/// Shared application state. There are no scheduler or allocator locks: the
/// actor thread is the only writer, and telemetry is published as a snapshot.
#[derive(Clone)]
pub struct AppState {
    pub config: EngineConfig,
    pub request_tx: UnboundedSender<Request>,
    pub snapshot: Arc<Mutex<Snapshot>>,
    pub recorder: Arc<Mutex<FlightRecorder>>,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub next_seq: Arc<AtomicUsize>,
    pub model_id: String,
    pub executor_name: String,
}

impl AppState {
    pub fn new(config: EngineConfig) -> Self {
        Self::mock(config)
    }

    pub fn mock(config: EngineConfig) -> Self {
        Self::with_executor(
            config,
            Box::new(MockExecutor::new()),
            Arc::new(MockTokenizer),
            "nexuskv-mock-v1",
        )
    }

    pub fn mock_with_io(config: EngineConfig, io: Option<Box<dyn BlockIo>>) -> Self {
        Self::with_executor_and_io(
            config,
            Box::new(MockExecutor::new()),
            Arc::new(MockTokenizer),
            "nexuskv-mock-v1",
            io,
        )
    }

    pub fn with_executor(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        tokenizer: Arc<dyn Tokenizer>,
        model_id: impl Into<String>,
    ) -> Self {
        Self::with_executor_and_io(config, executor, tokenizer, model_id, None)
    }

    pub fn with_executor_and_io(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        tokenizer: Arc<dyn Tokenizer>,
        model_id: impl Into<String>,
        io: Option<Box<dyn BlockIo>>,
    ) -> Self {
        let model_id = model_id.into();
        let executor_name = executor.name().to_string();
        if executor.is_mock() {
            tracing::warn!(
                "MOCK EXECUTOR ACTIVE — token streams are synthetic; no model weights are loaded. \
                 This mode is labeled \"executor\": \"mock\" in /v1/models and /v1/metrics."
            );
            if io.is_some() {
                tracing::warn!(
                    "MOCK TIERING ACTIVE — block contents are synthetic bytes; the paging, host \
                     LRU, and NVMe spill machinery are real."
                );
            }
        }

        let snapshot = Arc::new(Mutex::new(Snapshot::initial(
            &model_id,
            &executor_name,
            executor.attention_backend(),
            config.total_physical_blocks,
            config.block_size,
        )));
        let recorder = Arc::new(Mutex::new(FlightRecorder::default()));
        let (event_tx, event_rx) = std::sync::mpsc::sync_channel::<MemEvent>(8192);
        let recorder_thread = Arc::clone(&recorder);
        std::thread::Builder::new()
            .name("nexuskv-flight-recorder".to_string())
            .spawn(move || {
                while let Ok(event) = event_rx.recv() {
                    if let Ok(mut recorder) = recorder_thread.lock() {
                        recorder.record_event(event);
                    }
                }
            })
            .expect("failed to start the flight recorder thread");

        let (request_tx, request_rx) = unbounded_channel();
        let scheduler = Scheduler::new(config.max_num_seqs, config.max_num_batched_tokens);
        let mut block_manager = match io {
            Some(io) => BlockSpaceManager::with_io(
                config.total_physical_blocks,
                config.block_size,
                config.max_blocks_per_seq,
                io,
            ),
            None => BlockSpaceManager::new(
                config.total_physical_blocks,
                config.block_size,
                config.max_blocks_per_seq,
            ),
        };
        // Non-blocking telemetry: dropping an event under extreme pressure is
        // preferable to stalling the allocator; keyframes keep replay bounded.
        block_manager.set_event_sink(Box::new(move |event| {
            let _ = event_tx.try_send(event);
        }));
        let actor = EngineActor::new(
            request_rx,
            scheduler,
            block_manager,
            executor,
            Arc::clone(&tokenizer),
            TelemetrySink {
                snapshot: Arc::clone(&snapshot),
                recorder: Arc::clone(&recorder),
            },
            model_id.clone(),
        );
        std::thread::Builder::new()
            .name("nexuskv-engine".to_string())
            .spawn(move || actor.run())
            .expect("failed to start the NexusKV engine thread");

        Self {
            config,
            request_tx,
            snapshot,
            recorder,
            tokenizer,
            next_seq: Arc::new(AtomicUsize::new(1)),
            model_id,
            executor_name,
        }
    }

    pub fn next_sequence_id(&self) -> usize {
        self.next_seq.fetch_add(1, Ordering::SeqCst)
    }

    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

/// RAII guard: if the client disconnects before a sequence finishes, tell the
/// actor to abort it. No locks, no scheduler access from the HTTP task.
pub struct AbortDropGuard {
    pub seq_id: usize,
    pub request_tx: UnboundedSender<Request>,
    pub completed: Arc<AtomicBool>,
}

impl Drop for AbortDropGuard {
    fn drop(&mut self) {
        if !self.completed.load(Ordering::SeqCst) {
            tracing::warn!(
                "Client disconnected for sequence #{}. Sending abort to the engine actor.",
                self.seq_id
            );
            let _ = self.request_tx.send(Request::Abort { seq: self.seq_id });
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub spec: bool,
}

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
    pub adaptive_k: usize,
    pub executor: String,
    pub attention_backend: String,
    pub prefill_tokens: u64,
    pub prefix_skip_tokens: u64,
    pub active_block_ids: Vec<usize>,
    pub event_tick: u64,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAI API compatibility structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Clone)]
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
    pub executor: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ModelListResponse {
    pub object: String,
    pub executor: String,
    pub data: Vec<ModelCard>,
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

pub async fn health_check() -> &'static str {
    "OK"
}

pub async fn list_models(State(state): State<AppState>) -> axum::Json<ModelListResponse> {
    let snapshot = state.snapshot();
    let executor = snapshot.executor.clone();
    let mut data = vec![ModelCard {
        id: state.model_id.clone(),
        object: "model".to_string(),
        created: 1700000000,
        owned_by: "nexuskv".to_string(),
        executor: executor.clone(),
    }];
    if state.model_id != "nexuskv-engine-v1" {
        data.push(ModelCard {
            id: "nexuskv-engine-v1".to_string(),
            object: "model".to_string(),
            created: 1700000000,
            owned_by: "nexuskv".to_string(),
            executor: executor.clone(),
        });
    }

    axum::Json(ModelListResponse {
        object: "list".to_string(),
        executor,
        data,
    })
}

pub async fn get_metrics(State(state): State<AppState>) -> axum::Json<Snapshot> {
    axum::Json(state.snapshot())
}

fn generation_id(seq_id: usize) -> String {
    format!("chatcmpl-nexus-{seq_id}")
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn make_guard(state: &AppState, seq_id: usize, completed: Arc<AtomicBool>) -> AbortDropGuard {
    AbortDropGuard {
        seq_id,
        request_tx: state.request_tx.clone(),
        completed,
    }
}

#[allow(clippy::too_many_arguments)]
fn telemetry_payload(
    state: &AppState,
    seq_id: usize,
    token: String,
    step: usize,
    started: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    is_finished: bool,
    finish_reason: Option<String>,
) -> TelemetryPayload {
    let snapshot = state.snapshot();
    let now = Instant::now();
    let ttft_ms = first_token_at
        .map(|first| first.duration_since(started).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let itl_ms = last_token_at
        .map(|last| now.duration_since(last).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let tokens_per_sec = if step == 0 {
        0.0
    } else {
        step as f64 / started.elapsed().as_secs_f64().max(1e-6)
    };
    let active_block_ids = snapshot
        .seq_blocks
        .get(&seq_id)
        .cloned()
        .unwrap_or_default();

    TelemetryPayload {
        sequence_id: seq_id,
        token,
        step,
        tokens_per_sec,
        ttft_ms,
        itl_ms,
        allocated_blocks: snapshot.allocated_blocks,
        total_blocks: snapshot.total_blocks,
        memory_usage_ratio: snapshot.memory_usage_ratio,
        prefix_cache_hit_ratio: snapshot.prefix_cache_hit_ratio,
        speculative_acceptance_rate: snapshot.speculative_acceptance_rate,
        adaptive_k: snapshot.adaptive_k,
        executor: snapshot.executor,
        attention_backend: snapshot.attention_backend,
        prefill_tokens: snapshot.prefill_tokens,
        prefix_skip_tokens: snapshot.prefix_skip_tokens,
        active_block_ids,
        event_tick: snapshot.event_tick,
        is_finished,
        finish_reason,
    }
}

pub async fn stream_generation(
    State(state): State<AppState>,
    axum::Json(payload): axum::Json<GenerateRequest>,
) -> Response {
    let prompt_tokens = state.tokenizer.encode(&payload.prompt);
    if prompt_tokens.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "prompt must not be empty"})),
        )
            .into_response();
    }

    let seq_id = state.next_sequence_id();
    let (tx, rx) = unbounded_channel::<TokenMsg>();
    let job = GenerateJob {
        seq: seq_id,
        prompt: prompt_tokens,
        max_tokens: payload.max_tokens.max(1),
        temperature: payload.temperature.unwrap_or(0.7).max(0.0),
        top_p: payload.top_p.unwrap_or(1.0),
        spec: payload.spec,
        tx,
    };
    if state.request_tx.send(Request::Generate(job)).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"error": "engine actor is not running"})),
        )
            .into_response();
    }

    sse_generation_response(state, seq_id, rx)
}

/// Relay one sequence's `TokenMsg` channel into the engine telemetry SSE
/// schema. Shared by normal generation and session checkout.
fn sse_generation_response(
    state: AppState,
    seq_id: usize,
    mut rx: UnboundedReceiver<TokenMsg>,
) -> Response {
    let completed = Arc::new(AtomicBool::new(false));
    let guard = make_guard(&state, seq_id, Arc::clone(&completed));
    let stream_state = state;

    let stream = async_stream::stream! {
        let _guard = guard;
        let started = Instant::now();
        let mut first_token_at: Option<Instant> = None;
        let mut last_token_at: Option<Instant> = None;
        let mut step = 0usize;

        while let Some(message) = rx.recv().await {
            match message {
                TokenMsg::Token { text, step: message_step, .. } => {
                    let now = Instant::now();
                    if first_token_at.is_none() {
                        first_token_at = Some(now);
                    }
                    let previous = last_token_at;
                    last_token_at = Some(now);
                    step = message_step;
                    let payload = telemetry_payload(
                        &stream_state,
                        seq_id,
                        text,
                        step,
                        started,
                        first_token_at,
                        previous,
                        false,
                        None,
                    );
                    yield Ok::<Event, Infallible>(
                        Event::default().data(serde_json::to_string(&payload).unwrap_or_default()),
                    );
                }
                TokenMsg::Done { reason } => {
                    completed.store(true, Ordering::SeqCst);
                    let payload = telemetry_payload(
                        &stream_state,
                        seq_id,
                        String::new(),
                        step,
                        started,
                        first_token_at,
                        last_token_at,
                        true,
                        Some(reason.as_str().to_string()),
                    );
                    yield Ok::<Event, Infallible>(
                        Event::default().data(serde_json::to_string(&payload).unwrap_or_default()),
                    );
                    break;
                }
                TokenMsg::Error { message } => {
                    completed.store(true, Ordering::SeqCst);
                    let error = serde_json::json!({"error": message});
                    yield Ok::<Event, Infallible>(
                        Event::default().event("error").data(error.to_string()),
                    );
                    break;
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct CommitRequest {
    pub session: String,
    pub parent: Option<u64>,
    #[serde(default)]
    pub label: String,
    pub seq: usize,
}

#[derive(Debug, Deserialize)]
pub struct ForgetRequest {
    pub node: u64,
}

#[derive(Debug, Deserialize)]
pub struct CheckoutRequest {
    pub node: u64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub spec: bool,
}

#[derive(Debug, Deserialize)]
pub struct TimeTravelQuery {
    pub tick: Option<u64>,
}

async fn oneshot_result<T>(
    receiver: oneshot::Receiver<Result<T, String>>,
) -> Result<T, Box<Response>>
where
    T: Send + 'static,
{
    match tokio::time::timeout(Duration::from_secs(30), receiver).await {
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(error))) => Err(Box::new(
            (
                StatusCode::CONFLICT,
                axum::Json(serde_json::json!({"error": error})),
            )
                .into_response(),
        )),
        Ok(Err(_)) => Err(Box::new(
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "engine dropped the request"})),
            )
                .into_response(),
        )),
        Err(_) => Err(Box::new(
            (
                StatusCode::GATEWAY_TIMEOUT,
                axum::Json(serde_json::json!({"error": "engine request timed out"})),
            )
                .into_response(),
        )),
    }
}

pub async fn commit_session(
    State(state): State<AppState>,
    axum::Json(payload): axum::Json<CommitRequest>,
) -> Response {
    let (reply, receiver) = oneshot::channel();
    let label = if payload.label.is_empty() {
        format!("checkpoint-{}", payload.seq)
    } else {
        payload.label
    };
    if state
        .request_tx
        .send(Request::Commit {
            session: payload.session,
            parent: payload.parent,
            label,
            seq: payload.seq,
            reply,
        })
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"error": "engine actor is not running"})),
        )
            .into_response();
    }
    match oneshot_result(receiver).await {
        Ok(node) => (
            StatusCode::OK,
            axum::Json(serde_json::json!({"node": node})),
        )
            .into_response(),
        Err(response) => *response,
    }
}

pub async fn checkout_session(
    State(state): State<AppState>,
    axum::Json(payload): axum::Json<CheckoutRequest>,
) -> Response {
    let seq_id = state.next_sequence_id();
    let (tx, rx) = unbounded_channel::<TokenMsg>();
    let job = GenerateJob {
        seq: seq_id,
        prompt: Vec::new(),
        max_tokens: payload.max_tokens.max(1),
        temperature: payload.temperature.unwrap_or(0.7).max(0.0),
        top_p: payload.top_p.unwrap_or(1.0),
        spec: payload.spec,
        tx,
    };
    if state
        .request_tx
        .send(Request::Checkout {
            node: payload.node,
            job,
        })
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"error": "engine actor is not running"})),
        )
            .into_response();
    }
    sse_generation_response(state, seq_id, rx)
}

pub async fn forget_session(
    State(state): State<AppState>,
    axum::Json(payload): axum::Json<ForgetRequest>,
) -> Response {
    let (reply, receiver) = oneshot::channel();
    if state
        .request_tx
        .send(Request::Forget {
            node: payload.node,
            reply,
        })
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"error": "engine actor is not running"})),
        )
            .into_response();
    }
    match oneshot_result(receiver).await {
        Ok(()) => (
            StatusCode::OK,
            axum::Json(serde_json::json!({"forgotten": payload.node})),
        )
            .into_response(),
        Err(response) => *response,
    }
}

pub async fn session_tree(State(state): State<AppState>, Path(session): Path<String>) -> Response {
    let (reply, receiver) = oneshot::channel();
    if state
        .request_tx
        .send(Request::SessionTree {
            session: session.clone(),
            reply,
        })
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"error": "engine actor is not running"})),
        )
            .into_response();
    }
    match tokio::time::timeout(Duration::from_secs(30), receiver).await {
        Ok(Ok(tree)) => axum::Json(tree).into_response(),
        Ok(Err(_)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": "engine dropped the request"})),
        )
            .into_response(),
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            axum::Json(serde_json::json!({"error": "engine request timed out"})),
        )
            .into_response(),
    }
}

pub async fn timetravel(
    State(state): State<AppState>,
    Query(query): Query<TimeTravelQuery>,
) -> Response {
    let recorder = state
        .recorder
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let tick = query.tick.unwrap_or_else(|| recorder.latest_tick());
    match recorder.replay_to(tick) {
        Some(replay) => axum::Json(replay).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(serde_json::json!({
                "error": format!("no keyframe at or before tick {tick}")
            })),
        )
            .into_response(),
    }
}

pub async fn chat_completions(
    State(state): State<AppState>,
    axum::Json(payload): axum::Json<ChatCompletionRequest>,
) -> Response {
    let messages: Vec<(String, String)> = payload
        .messages
        .iter()
        .map(|message| (message.role.clone(), message.content.clone()))
        .collect();
    let prompt_tokens = state.tokenizer.encode_chat(&messages);
    if prompt_tokens.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "messages must not be empty"})),
        )
            .into_response();
    }

    let seq_id = state.next_sequence_id();
    let max_tokens = payload.max_tokens.unwrap_or(64).max(1);
    let temperature = payload.temperature.unwrap_or(0.7).max(0.0);
    let top_p = payload.top_p.unwrap_or(1.0);
    let model = payload.model.clone();
    let completion_id = generation_id(seq_id);
    let created = unix_timestamp();
    let (tx, mut rx) = unbounded_channel::<TokenMsg>();
    let job = GenerateJob {
        seq: seq_id,
        prompt: prompt_tokens.clone(),
        max_tokens,
        temperature,
        top_p,
        spec: false,
        tx,
    };
    if state.request_tx.send(Request::Generate(job)).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({"error": "engine actor is not running"})),
        )
            .into_response();
    }

    let completed = Arc::new(AtomicBool::new(false));
    let guard = make_guard(&state, seq_id, Arc::clone(&completed));

    if payload.stream {
        let stream = async_stream::stream! {
            let _guard = guard;

            let role_chunk = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
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
            yield Ok::<Event, Infallible>(
                Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default()),
            );

            while let Some(message) = rx.recv().await {
                match message {
                    TokenMsg::Token { text, .. } => {
                        let chunk = ChatCompletionChunk {
                            id: completion_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: ChatChoiceDelta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                        yield Ok::<Event, Infallible>(
                            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                        );
                    }
                    TokenMsg::Done { reason } => {
                        completed.store(true, Ordering::SeqCst);
                        let chunk = ChatCompletionChunk {
                            id: completion_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: ChatChoiceDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(reason.as_str().to_string()),
                            }],
                        };
                        yield Ok::<Event, Infallible>(
                            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                        );
                        break;
                    }
                    TokenMsg::Error { message } => {
                        completed.store(true, Ordering::SeqCst);
                        let error = serde_json::json!({"error": message});
                        yield Ok::<Event, Infallible>(
                            Event::default().event("error").data(error.to_string()),
                        );
                        break;
                    }
                }
            }

            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let _guard = guard;
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = None;
    let mut error = None;

    while let Some(message) = rx.recv().await {
        match message {
            TokenMsg::Token {
                text: token_text, ..
            } => {
                text.push_str(&token_text);
                completion_tokens += 1;
            }
            TokenMsg::Done { reason } => {
                completed.store(true, Ordering::SeqCst);
                finish_reason = Some(reason.as_str().to_string());
                break;
            }
            TokenMsg::Error { message } => {
                completed.store(true, Ordering::SeqCst);
                error = Some(message);
                break;
            }
        }
    }

    if let Some(message) = error {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": message})),
        )
            .into_response();
    }
    let Some(finish_reason) = finish_reason else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": "engine closed the stream unexpectedly"})),
        )
            .into_response();
    };

    let response = ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatChoiceMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason,
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: prompt_tokens.len() + completion_tokens,
        },
    };
    (StatusCode::OK, axum::Json(response)).into_response()
}
