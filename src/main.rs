// src/main.rs
use nexuskv::config::EngineConfig;
use nexuskv::server::{AppState, router};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("==================================================");
    tracing::info!("  NexusKV Engine: Rust 2024 Inference Runtime      ");
    tracing::info!("==================================================");

    let config = EngineConfig::default();
    tracing::info!(
        "Physical KV pool: {} blocks ({} tokens/block)",
        config.total_physical_blocks,
        config.block_size
    );
    tracing::info!(
        "Max concurrent sequences: {}, max sequence tokens: {}",
        config.max_num_seqs,
        config.max_seq_token_length()
    );

    let state = build_state(config, parse_model_dir())?;
    tracing::info!(
        "Executor: {} | model: {} | mock: {}",
        state.executor_name,
        state.model_id,
        state.snapshot().executor == "mock"
    );
    if state.snapshot().executor == "mock" {
        tracing::warn!(
            "=== MOCK MODE: responses are synthetic and labeled executor=mock in the API. ==="
        );
    }

    let app = router(state);
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    tracing::info!("NexusKV listening on http://{addr}");
    tracing::info!("  - OpenAI Chat API : POST http://{addr}/v1/chat/completions");
    tracing::info!("  - Models          : GET  http://{addr}/v1/models");
    tracing::info!("  - Telemetry       : POST http://{addr}/v1/generate/stream");
    tracing::info!("  - Metrics         : GET  http://{addr}/v1/metrics");
    tracing::info!("  - Health          : GET  http://{addr}/health");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("NexusKV shut down.");
    Ok(())
}

fn build_state(config: EngineConfig, model_dir: Option<PathBuf>) -> anyhow::Result<AppState> {
    #[cfg(feature = "candle-model")]
    if let Some(directory) = model_dir {
        tracing::info!("Loading real model from {}", directory.display());
        let (executor, tokenizer) =
            nexuskv::model::qwen2::load_model_and_tokenizer(&directory, &config)?;
        return Ok(AppState::with_executor(
            config,
            Box::new(executor),
            tokenizer,
            "nexuskv-qwen2",
        ));
    }

    #[cfg(not(feature = "candle-model"))]
    if model_dir.is_some() {
        anyhow::bail!(
            "--model requires a build with `--features candle-model` (and weights/config/tokenizer)"
        );
    }

    Ok(AppState::mock(config))
}

fn parse_model_dir() -> Option<PathBuf> {
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--model" {
            return arguments.next().map(PathBuf::from);
        }
    }
    None
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl+C received, shutting down"),
        _ = terminate => tracing::info!("SIGTERM received, shutting down"),
    }
}
