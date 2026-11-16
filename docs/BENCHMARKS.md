# NexusKV Engine — `v1.0.0` Measured Benchmarks

Every benchmark number in this document was **measured directly on real hardware and software** (`v1.0.0`) with zero synthetic counters or simulated streams. End-to-end inference throughput is always reported separately from CPU bookkeeping microbenchmarks.

---

## 1. Test Environment

| Component | Exact Configuration |
| :--- | :--- |
| **OS / Host** | Windows 11 Native (`x86_64-pc-windows-msvc`, PowerShell) |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (`4,096 MiB` VRAM, CUDA `13.4`) |
| **Model** | `Qwen/Qwen3-0.6B` (`f16` weights on CUDA, `28` layers, `16` tokens/block) |
| **Engine Build** | `target\release\nexuskv.exe` (`v1.0.0`, `--features cuda`, fused `qkv_proj` + `gate_up_proj`, hoisted RoPE/masks) |
| **Attention Backend** | Custom CUDA f16 decode kernel (`NEXUSKV_CUDA_ATTENTION=1` -> `custom-cuda-f16`) |
| **Executor Mode** | `real` (prefill gather attention + custom CUDA f16 continuous-batched decode) |

---

## 2. End-to-End Inference Benchmarks (RTX 3050 Laptop 4 GB, Qwen3-0.6B f16)

### 2.1 Multi-Client Continuous Batching (`192` KV Blocks, `spec=true`)
Measured using `python scripts/bench_median.py --runs 3 -- --clients <N> --max-tokens 64 --spec` against `nexuskv.exe --model models\qwen3-0.6b --blocks 192`:

| Workload | Aggregate Throughput (tok/s) | Per-Client Median (tok/s) | TTFT `p50` / `p99` (ms) | ITL `p50` / `p99` (ms) | Batching / Notes | Raw Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4 clients × 64 tokens** | **85.01** median (`[65.41, 86.19, 85.01]`) | **22.01** | **120.29 / 189.14** | **40.83 / 128.14** | 3/3 runs `0` errors; steady runs 2–3 at `85.0–86.2 tok/s` | `rtx3050-4c-v1-median.json` |
| **8 clients × 64 tokens** | **136.25** median (`[139.15, 136.25, 132.65]`) | **17.83** | **222.79 / 386.04** | **45.56 / 180.15** | `380` decode batches, `2,241` batch seqs (`~5.9` avg, `max 8`), `adaptive_k=4` | `rtx3050-8c-v1-median.json` |
| **Dashboard single request (64 tok)** | **15.8** | **15.8** | **619.1** | **42.6** | `1.0x` decode batch (`56 fused`), `19.4%` Spec Alpha, live browser capture | `docs/dashboard-live.png` |

> **VRAM Footprint (`192` KV blocks = `3,072` tokens):** Engine logs (`nexuskv::model::qwen`) and `nvidia-smi` measured **`75 MiB` idle before load -> `1,739 MiB / 4,096 MiB`** after allocating Qwen3-0.6B `f16` weights plus the 192-block CUDA KV pool.

### 2.2 Radix Prefix Caching (`512`-Token Aligned Re-Hit)
Measured with `python scripts/bench_e2e.py --prefix-probe-only`:

- **Cold request (`512` prompt tokens):** `252.89 ms` TTFT (`520.31 ms` total for 8 generated tokens), `prefill_delta = 512`, `skip_delta = 0`.
- **Warm re-hit (`512` identical prompt tokens):** `21.47 ms` TTFT (`303.37 ms` total for 8 generated tokens), `prefill_delta = 0`, `skip_delta = 512`.
- **Measured TTFT speedup:** **`11.78×`** (`100%` of the 512-token prefix skipped via Radix Trie block reuse).
- **Source artifact:** `benches/results/rtx3050-prefix-512-v1.json`.

### 2.3 Speculative Decoding (`LookupDrafter` + `AdaptiveK` UCB1)
- **Quote/echo self-referential prompt** (`python scripts/spec_acceptance_probe.py --max-tokens 64`): **`57.5%` acceptance** (`0.575`), `adaptive_k = 4`, `45` prefill tokens, `executor = real`, `attention_backend = custom-cuda-f16`.
- **Diverse technical prompts (4c / 8c median benches):** `2.2%` / `3.1%` acceptance (as expected, N-gram suffix drafting shines on repetitive/quoted/code continuation prompts and drops back gracefully via UCB1 arm selection when prompts are non-repetitive).
- **Single-prompt mixed prose (`docs/dashboard-live.png`):** `19.4%` Spec Alpha at `adaptive_k = 2`.

### 2.4 Forced 3-Tier Offloading & Demand Page-In (`16` Physical Blocks)
Measured on a constrained pool (`--blocks 16 --max-seqs 4 --swap-dir $env:TEMP\nexuskv-tier-v1`) using:
`python scripts/tiering_probe.py --url http://127.0.0.1:8080 --clients 4 --fillers 12 --filler-blocks 6 --remeasure 4 --max-tokens 32`:

- **Tokens generated:** `640` (`0` errors, `executor = real`, `attention_backend = custom-cuda-f16`)
- **Evictions:** `35` (`22` before re-measure)
- **Swapped out:** `35` blocks (`64,225,280` bytes / `61.25 MiB`)
- **Demand page-ins (`page_faults`):** `1` block swapped back in (`1,835,008` bytes / `1.75 MiB`)
- **Prefix skip tokens preserved across spill/page-in:** `304` tokens
- **Source artifact:** `benches/results/rtx3050-tiering-v1.json`.

---

## 3. CPU Bookkeeping Microbenchmarks (`benches/bookkeeping.rs`)

> **Important:** These Criterion benchmarks isolate `BlockSpaceManager` CPU bookkeeping in memory (`4096` blocks, `block_size=16`, `max_blocks_per_seq=64`). They **do not run a neural network** and must never be conflated with inference throughput.

Measured with `cargo bench --bench bookkeeping`:

| Criterion Benchmark | Low Bound | Point Estimate | High Bound |
| :--- | :--- | :--- | :--- |
| `bookkeeping/allocator/match_and_free_unique_16_token_prompts` | `4.6253 µs` | **`4.8977 µs`** | `5.1091 µs` |
| `bookkeeping/trie/resident_prefix_hit` (`256`-token shared prompt) | `2.4247 µs` | **`2.4325 µs`** | `2.4424 µs` |
| `bookkeeping/rollback/rollback_64_decode_tokens` (`O(1)` spec discard) | `593.03 ns` | **`598.92 ns`** | `605.97 ns` |

---

## 4. Reproducing These Numbers

```powershell
# 1. Build release CUDA binary (Windows)
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1

# 2. Start the CUDA engine with custom decode attention (shell 1)
. .\scripts\windows_env.ps1
Initialize-NexusKvEnvironment | Out-Null
$env:NEXUSKV_CUDA_ATTENTION = '1'
.\target\release\nexuskv.exe --model models\qwen3-0.6b --blocks 192

# 3. Run multi-client, prefix, and acceptance benchmarks (shell 2)
python scripts/bench_median.py --url http://127.0.0.1:8080 --runs 3 -- --clients 4 --max-tokens 64 --spec
python scripts/bench_median.py --url http://127.0.0.1:8080 --runs 3 -- --clients 8 --max-tokens 64 --spec
python scripts/bench_e2e.py --url http://127.0.0.1:8080 --prefix-probe-only
python scripts/spec_acceptance_probe.py --url http://127.0.0.1:8080 --max-tokens 64

# 4. Run constrained forced-tiering probe (restart engine with --blocks 16 --max-seqs 4 --swap-dir ...)
.\target\release\nexuskv.exe --model models\qwen3-0.6b --swap-dir $env:TEMP\nexuskv-tier-v1 --blocks 16 --max-seqs 4
python scripts/tiering_probe.py --url http://127.0.0.1:8080 --clients 4 --fillers 12 --filler-blocks 6 --remeasure 4 --max-tokens 32

# 5. Run CPU bookkeeping benchmarks (then rebuild CUDA binary if you need nexuskv.exe with GPU support)
cargo bench --bench bookkeeping
```
