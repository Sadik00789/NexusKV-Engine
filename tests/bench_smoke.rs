// tests/bench_smoke.rs
//! Smoke test for the e2e benchmark harness. Skips cleanly when Python is not
//! installed; CI images with Python exercise the real script against a mock
//! server.

mod common;

use common::test_config;
use nexuskv::server::{AppState, router};
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

fn python_executable() -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|&candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bench_e2e_script_produces_sane_json_against_mock_server() {
    let Some(python) = python_executable() else {
        eprintln!("python not available; skipping e2e benchmark smoke test");
        return;
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let address = listener.local_addr().unwrap();
    let url = format!("http://{address}");
    let app = router(AppState::mock(test_config()));
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("bench_e2e.py");
    let output = tokio::task::spawn_blocking(move || {
        Command::new(python)
            .arg(script)
            .arg("--url")
            .arg(url)
            .arg("--clients")
            .arg("2")
            .arg("--max-tokens")
            .arg("8")
            .output()
            .expect("run bench_e2e.py")
    })
    .await
    .expect("benchmark task");

    server.abort();
    assert!(
        output.status.success(),
        "benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value =
        serde_json::from_slice(&output.stdout).expect("benchmark stdout must be JSON");
    assert!(summary["tokens_per_sec"].as_f64().unwrap() > 0.0);
    assert_eq!(summary["total_tokens"], 16);
    assert!(
        summary["ttft_ms"]["p50"].as_f64().unwrap() <= summary["ttft_ms"]["p99"].as_f64().unwrap()
    );
    assert!(
        summary["itl_ms"]["p50"].as_f64().unwrap() <= summary["itl_ms"]["p99"].as_f64().unwrap()
    );
}
