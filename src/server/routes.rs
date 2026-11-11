// src/server/routes.rs
use super::sse_handler::{
    AppState, chat_completions, get_metrics, health_check, list_models, stream_generation,
};
use axum::{
    Router,
    http::{Method, header::CONTENT_TYPE},
    routing::{get, post},
};
use tower_http::cors::{Any, CorsLayer};

pub fn router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([CONTENT_TYPE]);

    Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/metrics", get(get_metrics))
        .route("/v1/generate/stream", post(stream_generation))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(cors)
        .with_state(state)
}
