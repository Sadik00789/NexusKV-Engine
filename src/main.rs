// src/main.rs
use nexuskv::config::EngineConfig;
use nexuskv::memory::{BlockIo, MemoryBridge, TierStore};
use nexuskv::server::{AppState, router};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const DEFAULT_HOST_BUDGET: usize = 512 * 1024 * 1024;

#[derive(Debug, Default)]
struct Cli {
    model: Option<PathBuf>,
    swap_dir: Option<PathBuf>,
    host_budget: usize,
}

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

    let cli = parse_args()?;
    #[allow(unused_mut)]
    let mut config = EngineConfig::default();
    #[cfg(feature = "cuda")]
    if cli.model.is_some() && nexuskv::model::qwen::cuda_f32_requested() {
        config.clamp_for_f32_cuda();
        tracing::warn!(
            "NEXUSKV_DTYPE=f32 on CUDA: clamping to 96 blocks / 4 sequences to stay under 4 GB VRAM"
        );
    }
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

    let state = build_state(config, &cli)?;
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
    if let Some(directory) = &cli.swap_dir {
        tracing::info!(
            "Tier store: {} (host budget {} MiB, NVMe spill enabled)",
            directory.display(),
            cli.host_budget / (1024 * 1024)
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
    tracing::info!("  - Sessions        : POST http://{addr}/v1/sessions/commit|checkout|forget");
    tracing::info!("  - Time travel     : GET  http://{addr}/v1/timetravel?tick=N");
    tracing::info!("  - Health          : GET  http://{addr}/health");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("NexusKV shut down.");
    Ok(())
}

fn build_state(config: EngineConfig, cli: &Cli) -> anyhow::Result<AppState> {
    #[cfg(feature = "candle-model")]
    if let Some(directory) = &cli.model {
        tracing::info!("Loading real model from {}", directory.display());
        let (executor, tokenizer) =
            nexuskv::model::qwen::load_model_and_tokenizer(directory, &config)?;
        let io: Option<Box<dyn BlockIo>> = match &cli.swap_dir {
            Some(swap_dir) => {
                let bridge = executor.gpu_bridge();
                Some(Box::new(
                    TierStore::new(bridge, swap_dir, cli.host_budget)
                        .map_err(anyhow::Error::msg)?,
                ))
            }
            None => None,
        };
        return Ok(AppState::with_executor_and_io(
            config,
            Box::new(executor),
            tokenizer,
            "nexuskv-qwen3-0.6b",
            io,
        ));
    }

    #[cfg(not(feature = "candle-model"))]
    if cli.model.is_some() {
        anyhow::bail!(
            "--model requires a build with `--features candle-model` (and weights/config/tokenizer)"
        );
    }

    let io: Option<Box<dyn BlockIo>> = match &cli.swap_dir {
        Some(swap_dir) => {
            let block_bytes = config.block_size * config.num_heads * config.head_dim * 4 * 2;
            Some(Box::new(
                TierStore::new(MemoryBridge::new(block_bytes), swap_dir, cli.host_budget)
                    .map_err(anyhow::Error::msg)?,
            ))
        }
        None => None,
    };
    Ok(AppState::mock_with_io(config, io))
}

fn parse_args() -> anyhow::Result<Cli> {
    let mut cli = Cli {
        host_budget: DEFAULT_HOST_BUDGET,
        ..Cli::default()
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                cli.model =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--model requires a directory")
                    })?));
            }
            "--swap-dir" => {
                cli.swap_dir =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        anyhow::anyhow!("--swap-dir requires a directory")
                    })?));
            }
            "--host-budget" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--host-budget requires a byte count"))?;
                cli.host_budget = value
                    .parse()
                    .map_err(|error| anyhow::anyhow!("invalid --host-budget {value}: {error}"))?;
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    Ok(cli)
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
