// tests/honesty.rs
//! Build-failing regressions against fabricated output and fake telemetry.

use std::fs;
use std::path::{Path, PathBuf};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    let path = manifest().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()))
}

fn collect_rs(directory: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).expect("read source directory") {
        let entry = entry.expect("directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, output);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            output.push(path);
        }
    }
}

#[test]
fn no_canned_serving_vocabulary() {
    for file in ["src/server/sse_handler.rs", "src/server/routes.rs"] {
        let source = read(file);
        for marker in [
            "dynamic_tokens",
            "(step + 100)",
            "(step+100)",
            "virtualizes",
            "wire-speed",
            "lock-free",
            "\"PagedAttention\",",
        ] {
            assert!(
                !source.contains(marker),
                "{file} still contains fabricated serving marker {marker:?}"
            );
        }
    }
}

#[test]
fn no_fabricated_metrics_defaults() {
    let mut files = Vec::new();
    collect_rs(&manifest().join("src"), &mut files);
    for file in files {
        let source = fs::read_to_string(&file).expect("read Rust source");
        assert!(
            !source.contains("0.85"),
            "{} contains the old fabricated 0.85 acceptance default",
            file.display()
        );
        assert!(
            !source.contains("simulateLocalStream"),
            "{} contains fake local playback",
            file.display()
        );
    }
}

#[test]
fn no_pacing_sleeps_in_serving_paths() {
    for file in ["src/server/sse_handler.rs", "src/server/routes.rs"] {
        let source = read(file);
        assert!(
            !source.contains("sleep(") && !source.contains("Duration::from_millis"),
            "{file} contains pacing sleeps in the serving path"
        );
    }
}

#[test]
fn dashboard_has_no_fake_playback_or_defaults() {
    let page = read("dashboard/src/app/page.tsx");
    for marker in ["simulateLocalStream", "0.42", "0.85", "Local Simulator"] {
        assert!(
            !page.contains(marker),
            "dashboard page still contains fabricated marker {marker:?}"
        );
    }
    let metrics_panel = read("dashboard/src/components/MetricsPanel.tsx");
    assert!(
        !metrics_panel.contains("85.0"),
        "MetricsPanel still fabricates an 85% acceptance default"
    );
    let canvas = read("dashboard/src/components/VramBlockCanvas.tsx");
    for marker in ["0.42", "0.85", "ref_count: isActive ? 1 : 0"] {
        assert!(
            !canvas.contains(marker),
            "VramBlockCanvas still fabricates block telemetry: {marker:?}"
        );
    }
}

#[test]
fn readme_has_no_fabricated_throughput_claims() {
    let readme = read("README.md");
    for marker in [
        "50.88",
        "Million Speculative Tokens/sec",
        "45.06 Million",
        "63.09 Million",
        "1.18 Million",
    ] {
        assert!(
            !readme.contains(marker),
            "README still claims fabricated benchmark result {marker:?}"
        );
    }
}

#[test]
fn mock_mode_is_explicit_in_server_code() {
    let source = read("src/server/sse_handler.rs");
    assert!(source.contains("MOCK EXECUTOR ACTIVE"));
    assert!(source.contains("executor"));
}
