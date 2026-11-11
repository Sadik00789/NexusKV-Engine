# Windows 11 Setup — Native GPU Bring-Up

NexusKV runs **natively on Windows 11** with the MSVC toolchain. WSL is not
used anywhere in the build, test, or runtime path. All Windows scripts are
PowerShell (`.ps1`); the `.sh` helpers exist only for Linux CI.

Target machine: RTX 3050 Laptop (4 GB VRAM, compute capability 8.6), 8 GB RAM,
Windows 11, CUDA Toolkit 13.4, Qwen3-0.6B.

> **CUDA version ceiling.** candle 0.9.x depends on cudarc **0.19.10**, whose
> build script hard-panics on any toolkit newer than **13.4**. CUDA 13.5 is the
> current release and will **not** build this checkout. Use 13.4 (or 12.6+).
> Raising the ceiling requires moving candle to a cudarc that knows the newer
> version.

---

## 1. Prerequisites

Install these yourself (the scripts verify them, they never install a driver):

| # | Component | Why | Verification |
| --- | --- | --- | --- |
| 1 | NVIDIA Windows display driver | `nvcuda.dll` + `nvidia-smi`; never install the driver bundled with CUDA | `nvidia-smi` |
| 2 | Visual Studio 2022 Build Tools, workload **Desktop development with C++** | `cl.exe`, `lib.exe`, Windows SDK; nvcc uses `cl` as host compiler | `scripts/check_env.ps1` |
| 3 | **CUDA Toolkit 13.4 for Windows** | nvcc, cudart, cublas/cublasLt, curand, nvrtc | `nvcc --version` |
| 4 | `rustup default stable-x86_64-pc-windows-msvc` | GNU toolchain cannot link MSVC/nvcc objects | `rustc -vV` |
| 5 | Python 3 + `pip install -U huggingface_hub` | weight download | `python --version` |
| 6 | Node 22 + pnpm (`corepack enable`) | dashboard | `pnpm --version` |

CUDA 13.4.3 download: <https://developer.nvidia.com/cuda-13-4-0-download-archive>
→ Windows → x86_64 → 11 → exe. Do **not** use CUDA 13.5 with candle 0.9 /
cudarc 0.19.10: the cudarc build script panics on an unrecognized version.

### Per-user toolkit layout (used when the installer cannot elevate)

The NVIDIA silent installer requires Administrator. A non-admin user can build
an equivalent toolkit root from the official redistributable archives:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_cuda_user.ps1
# installs CUDA 13.4.3 into %USERPROFILE%\cuda-13.4
```

The script downloads NVIDIA's published component zips (`cuda_nvcc`,
`cuda_cudart`, `cuda_crt`, `cccl`, `libcublas`, `libcurand`, `cuda_nvrtc`,
`libnvvm`, `libnvjitlink`, `cuda_sanitizer_api`, `cuda_cupti`), verifies the
resulting layout, and reports the `nvcc` version. It is idempotent and refuses
unsupported toolkit versions.

The resulting tree is:

```
%USERPROFILE%\cuda-13.4\
  bin\nvcc.exe, nvcc tools
  bin\x64\cudart64_13.dll, cublas64_13.dll, curand64_10.dll, nvrtc64_130_0.dll
  include\...
  lib\x64\cudart.lib, cublas.lib, curand.lib, nvrtc.lib, ...
  nvvm\...
  compute-sanitizer\compute-sanitizer.exe
```

CUDA 13 moved the **runtime DLLs into `bin\x64`** while `bin` keeps the tools;
`scripts/windows_env.ps1` adds both directories to `PATH`.

`scripts/windows_env.ps1` resolves `CUDA_HOME`, then `CUDA_PATH`, then the
per-user `cuda-13.4`/`cuda-13.2`/`cuda-12.6` trees, then
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`. A normal
Administrator install needs no special layout.

---

## 2. Environment variables

`scripts/windows_env.ps1` sets these for every script; a plain shell only needs
them if you invoke `cargo` directly.

```powershell
$env:CUDA_COMPUTE_CAP = "86"   # target sm_86 only; keeps candle-kernels fast
$env:CARGO_BUILD_JOBS = "2"    # 8 GB RAM OOMs at full parallelism
$env:CUDA_PATH = "C:\path\to\cuda-13.4"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:PATH"
# MSVC 14.44 + dynamic CRT. /Zc:preprocessor is required by CUDA 13's CCCL
# headers and harmless on CUDA 12.6.
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler -Xcompiler /MD -Xcompiler /Zc:preprocessor -cudart=shared"
$env:NVCC_CCBIN = (Get-Command cl.exe).Source            # candle-kernels forwards the flag only with this set
```

`CUDA_PATH\bin` and `CUDA_PATH\bin\x64` must be on `PATH` at **runtime** for
`cudart64_13.dll`, `cublas64_13.dll`, and `curand64_10.dll` (CUDA 13 keeps the
runtime DLLs in `bin\x64`). `scripts/run_windows.ps1` does this.

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

Prints a PASS/WARN/FAIL table: MSVC host, `cl.exe`, CUDA 12.6–13.4, `lib\x64`,
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
`scripts/sanitize.ps1` (`compute-sanitizer.exe` from the toolkit). The GPU-only
tests are `#[ignore]`d so the default `cargo test` stays GPU-free.

**One-time administrator step for compute-sanitizer on WDDM.** Windows requires
the sanitizer debugger interface to be enabled before it can instrument a
kernel. Without it, compute-sanitizer prints
`Failed to initialize WDDM debugger interface` and `Device not supported`, and
the test runs **uninstrumented**. Run once from an Administrator PowerShell:

```powershell
& "$env:USERPROFILE\cuda-13.4\compute-sanitizer\EnableDebuggerInterface.bat"
# or, for a standard install:
& "$env:CUDA_PATH\compute-sanitizer\EnableDebuggerInterface.bat"
```

That batch file is a single `REG ADD` under
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger`, so it cannot be performed from
a filtered (non-elevated) administrator token — the write is denied. It must be
a genuinely elevated shell.

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

Single runs on a laptop GPU are noisy. The headline numbers are medians of three
runs, and the raw runs stay on disk:

```powershell
# Median of 3 runs; each raw run is written under benches\results\.
python scripts/bench_median.py --runs 3 --label rtx3050-4c `
  --output benches\results\rtx3050-4c-median.json `
  -- --clients 4 --max-tokens 64 --spec

# One run with the cold/warm prefix probe.
python scripts/bench_e2e.py --url http://127.0.0.1:8080 `
  --clients 4 --max-tokens 128 --spec --prefix-probe `
  --output benches/results/rtx3050-prefix-512.json

# Why batching matters: 8 clients share one fused forward per step.
python scripts/bench_e2e.py --url http://127.0.0.1:8080 --clients 8 --max-tokens 64 --spec

# Speculative acceptance on a task where prompt lookup actually helps.
python scripts/spec_acceptance_probe.py --url http://127.0.0.1:8080 --max-tokens 64

# Force tiering: shared prefix, fillers to evict it, then re-request to page in.
python scripts/tiering_probe.py --url http://127.0.0.1:8080 `
  --clients 4 --fillers 12 --filler-blocks 6 --remeasure 4
```

`--prefix-probe` submits a unique prompt twice and reports the measured
cold-vs-warm TTFT speedup; the JSON also carries prefill/skipped tokens, swap
bytes, evictions, page faults, acceptance, adaptive K, and the batched-decode
counters. Files under `benches/results/` are local evidence and are gitignored.
Every number in the README names the hardware, model, dtype, and executor mode.

`/v1/metrics` reports `batched_decode`, `decode_batches`,
`decode_batch_sequences`, and `max_decode_batch`, so you can confirm the fusion
is happening instead of trusting a token rate.

---

## 10. Troubleshooting

- **`nvcc fatal: unsupported Microsoft Visual Studio version`** — `build.rs`
  adds `-allow-unsupported-compiler` on MSVC and logs a cargo warning. The host
  compiler is still the installed `cl.exe`.
- **`nvcc fatal: Cannot find compiler 'cl.exe' in PATH`** — nvcc needs the VS
  environment. Run through the scripts (they import `vcvars64.bat`) or set
  `NVCC_CCBIN` to the full `cl.exe` path.
- **`fatal error C1189: MSVC/cl.exe with traditional preprocessor is used`** —
  CUDA 13's CCCL headers require `/Zc:preprocessor`. It is set by
  `NVCC_PREPEND_FLAGS` in `scripts/windows_env.ps1`.
- **`cudart64_13.dll not found` at runtime** — CUDA 13 keeps runtime DLLs in
  `%CUDA_PATH%\bin\x64`. Run through `scripts/run_windows.ps1`; on a manual
  shell add both `bin` and `bin\x64` to `PATH`.
- **`Unsupported cuda toolkit version` during `cargo build`** — cudarc 0.19.10
  only knows CUDA up to 13.4. Install 13.4
  (`scripts/install_cuda_user.ps1`) instead of 13.5.
- **Linker cannot find `cudart.lib`** — Windows uses `lib\x64`, not `lib64`.
  `scripts/check_env.ps1` verifies this exact path.
- **Build OOM on 8 GB** — keep `CARGO_BUILD_JOBS=2` (set by
  `.cargo/config.toml` and the scripts) and close other applications.
