# Windows 11 Setup — Native GPU Bring-Up

NexusKV runs **natively on Windows 11** with the MSVC toolchain. WSL is not
used anywhere in the build, test, or runtime path. All Windows scripts are
PowerShell (`.ps1`); the `.sh` helpers exist only for Linux CI.

Target machine: RTX 3050 Laptop (4 GB VRAM, compute capability 8.6), 8 GB RAM,
Windows 11, CUDA Toolkit 12.6, Qwen3-0.6B.

---

## 1. Prerequisites

Install these yourself (the scripts verify them, they never install a driver):

| # | Component | Why | Verification |
| --- | --- | --- | --- |
| 1 | NVIDIA Windows display driver | `nvcuda.dll` + `nvidia-smi`; never install the driver bundled with CUDA | `nvidia-smi` |
| 2 | Visual Studio 2022 Build Tools, workload **Desktop development with C++** | `cl.exe`, `lib.exe`, Windows SDK; nvcc uses `cl` as host compiler | `scripts/check_env.ps1` |
| 3 | **CUDA Toolkit 12.6 for Windows** | nvcc, cudart, cublas/cublasLt, curand, nvrtc | `nvcc --version` |
| 4 | `rustup default stable-x86_64-pc-windows-msvc` | GNU toolchain cannot link MSVC/nvcc objects | `rustc -vV` |
| 5 | Python 3 + `pip install -U huggingface_hub` | weight download | `python --version` |
| 6 | Node 22 + pnpm (`corepack enable`) | dashboard | `pnpm --version` |

CUDA 12.6 download: <https://developer.nvidia.com/cuda-12-6-0-download-archive>
→ Windows → x86_64 → 11 → exe. Do **not** use CUDA 13.x with candle 0.9 /
cudarc 0.17+ unless you verify cudarc support and record the deviation in
`DECISIONS.md`.

### Per-user toolkit layout (used when the installer cannot elevate)

The NVIDIA silent installer requires Administrator. A non-admin user can build
an equivalent toolkit root from the official redistributable archives:

```
%USERPROFILE%\cuda-12.6\
  bin\nvcc.exe, cudart64_12.dll, cublas64_12.dll, curand64_10.dll, ...
  include\...
  lib\x64\cudart.lib, cublas.lib, curand.lib, ...
  nvvm\...
```

`scripts/windows_env.ps1` resolves `CUDA_HOME`, then `CUDA_PATH`, then
`%USERPROFILE%\cuda-12.6`, then `C:\Program Files\NVIDIA GPU Computing
Toolkit\CUDA\v12.6`. A normal Administrator install needs no special layout.

---

## 2. Environment variables

`scripts/windows_env.ps1` sets these for every script; a plain shell only needs
them if you invoke `cargo` directly.

```powershell
$env:CUDA_COMPUTE_CAP = "86"   # target sm_86 only; keeps candle-kernels fast
$env:CARGO_BUILD_JOBS = "2"    # 8 GB RAM OOMs at full parallelism
$env:CUDA_PATH = "C:\path\to\cuda-12.6"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler -Xcompiler /MD -cudart=shared"  # MSVC 14.44 + dynamic CRT
$env:NVCC_CCBIN = (Get-Command cl.exe).Source            # candle-kernels forwards the flag only with this set
```

`CUDA_PATH\bin` must be on `PATH` at **runtime** for `cudart64_12.dll`,
`cublas64_12.dll`, and `curand64_10.dll`. `scripts/run_windows.ps1` does this.

The build script imports the Visual Studio developer environment itself via
`vswhere` + `vcvars64.bat`, so `cl.exe` does not need to be on your default
`PATH` and you do not need a "Developer PowerShell" window.

---

## 3. Install model weights

```powershell
hf download Qwen/Qwen3-0.6B --local-dir models\qwen3-0.6b
```

Fallback: `Qwen/Qwen2.5-0.5B-Instruct`. The directory must contain
`config.json`, `tokenizer.json`, and at least one `*.safetensors` shard.
`models/` is gitignored and never committed.

Hybrid Qwen3.5/3.6/3.8 (Gated DeltaNet) checkpoints are rejected on purpose.

---

## 4. Preflight

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
```

Prints a PASS/WARN/FAIL table: MSVC host, `cl.exe`, CUDA 12.6, `lib\x64`,
`nvidia-smi`, pnpm, Python, free disk (>10 GB), weights (warn only). Non-zero on
hard failures with exact fixes. Self-test without CUDA:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1 -SelfTest
```

---

## 5. Build

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

This imports the VS environment, runs the preflight, runs
`cargo build --release --features cuda`, then `pnpm build` for the dashboard.
The default-feature build (no CUDA) is:

```powershell
cargo build --release
```

`build.rs` only invokes nvcc when `--features cuda` is set; the default build
never requires the toolkit.

---

## 6. Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1
# Engine:    http://127.0.0.1:8080
# Health:    GET /health
# Metrics:   GET /v1/metrics   (executor=real when weights are loaded)
# Dashboard: cd dashboard; pnpm dev  ->  http://localhost:3000
```

The dashboard is live telemetry only; it never simulates offline playback. A
mock server is labeled `MOCK` in the UI and `"executor": "mock"` in the API.

---

## 7. GPU tests

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_gpu.ps1
```

Runs `cargo test --features cuda -- --ignored --test-threads=1` and then
`scripts/sanitize.ps1` (`compute-sanitizer.exe` from `%CUDA_PATH%\bin`). The
GPU-only tests are `#[ignore]`d so the default `cargo test` stays GPU-free.

**One-time administrator step for compute-sanitizer on WDDM.** Windows requires
the sanitizer debugger interface to be enabled before it can instrument a
kernel. Without it, compute-sanitizer prints
`Failed to initialize WDDM debugger interface` and `Device not supported`, and
the test runs **uninstrumented**. Run once from an Administrator PowerShell:

```powershell
& "$env:USERPROFILE\cuda-12.6\compute-sanitizer\EnableDebuggerInterface.bat"
# or, for a standard install:
& "$env:CUDA_PATH\compute-sanitizer\EnableDebuggerInterface.bat"
```

Reboot if prompted, then re-run `scripts/sanitize.ps1`. The script treats the
WDDM error as a hard blocker (exit 2) instead of reporting a false clean run.

---

## 8. VRAM budget and Sysmem Fallback

| Item | f16 |
| --- | --- |
| Qwen3-0.6B weights | ≈1.2 GB |
| KV pool (192 blocks × 16 tokens) | ≈344 MB |
| Activations / workspace | ≈0.3 GB |
| Windows display | ≈0.3–0.5 GB |
| **Total** | **≈2.4 GB of 4 GB** |

f32 weights do not fit with 192 blocks; the CUDA f32 fallback clamps to 96
blocks / 4 sequences with a warning.

Recommended (optional) NVIDIA Control Panel setting for `nexuskv.exe`:
**CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback**. VRAM overflow
then fails loudly instead of silently thrashing the 8 GB of system RAM.

Close browsers and heavy apps during the first CUDA build; candle-kernels
compilation is memory-hungry on an 8 GB host.

---

## 9. Benchmarks

```powershell
python scripts/bench_e2e.py --url http://127.0.0.1:8080 `
  --clients 4 --max-tokens 128 --spec --prefix-probe `
  --output benches/results/rtx3050-qwen3-0.6b-f16.json
```

`--prefix-probe` submits a unique prompt twice and reports the measured
cold-vs-warm TTFT speedup; the JSON also carries prefill/skipped tokens, swap
bytes, evictions, page faults, acceptance, and adaptive K. Files under
`benches/results/` are local evidence and are gitignored. Every number in the
README names the hardware, model, dtype, and executor mode.

---

## 10. Troubleshooting

- **`nvcc fatal: unsupported Microsoft Visual Studio version`** — `build.rs`
  adds `-allow-unsupported-compiler` on MSVC and logs a cargo warning. The host
  compiler is still the installed `cl.exe`.
- **`cudart64_12.dll not found` at runtime** — `%CUDA_PATH%\bin` is not on
  `PATH`; run through `scripts/run_windows.ps1` or `scripts/test_gpu.ps1`.
- **Linker cannot find `cudart.lib`** — Windows uses `lib\x64`, not `lib64`.
  `scripts/check_env.ps1` verifies this exact path.
- **Build OOM on 8 GB** — keep `CARGO_BUILD_JOBS=2` (set by
  `.cargo/config.toml` and the scripts) and close other applications.
