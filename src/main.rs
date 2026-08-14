// src/main.rs
mod config;
mod engine;
mod kernels;
mod memory;
mod server;

use config::EngineConfig;
use server::{
    chat_completions, get_metrics, health_check, list_models, stream_generation, AppState,
};

use axum::{
    http::{header::CONTENT_TYPE, Method},
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize structured console logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("==================================================");
    tracing::info!("  NexusKV Engine: Rust 2024 + CUDA 13.1 Runtime   ");
    tracing::info!("==================================================");

    // 2. Initialize engine configuration & shared application state
    let config = EngineConfig::default();
    tracing::info!(
        "Allocated Physical Memory Pool: {} Blocks ({} tokens/block, Head Dim: {})",
        config.total_physical_blocks,
        config.block_size,
        config.head_dim
    );
    tracing::info!(
        "Max Concurrent Seqs: {}, Max Seq Length: {} tokens",
        config.max_num_seqs,
        config.max_seq_token_length()
    );

    let state = AppState::new(config);

    // 3. Configure CORS policy for Next.js 15 dashboard communication
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([CONTENT_TYPE]);

    // 4. Build Axum Router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/metrics", get(get_metrics))
        .route("/v1/generate/stream", post(stream_generation))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(cors)
        .with_state(state);

    // 5. Start TCP Listener
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;

    tracing::info!("🚀 NexusKV Engine listening on http://{}", addr);
    tracing::info!("   - OpenAI Chat API : POST http://{}/v1/chat/completions", addr);
    tracing::info!("   - OpenAI Models   : GET  http://{}/v1/models", addr);
    tracing::info!("   - Telemetry Stream: POST http://{}/v1/generate/stream", addr);
    tracing::info!("   - System Metrics  : GET  http://{}/v1/metrics", addr);
    tracing::info!("   - Health Probe    : GET  http://{}/health", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // 6. Run server with graceful shutdown signal
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("NexusKV Engine shut down successfully.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install Unix terminate signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received Ctrl+C signal. Initiating shutdown..."),
        _ = terminate => tracing::info!("Received SIGTERM signal. Initiating shutdown..."),
    }
}