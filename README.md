# NexusKV Engine

<div align="center">

![Rust](https://img.shields.io/badge/Rust-2024_Edition-DEA584?style=for-the-badge&logo=rust&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15.0_App_Router-000000?style=for-the-badge&logo=next.js&logoColor=white)
![Axum](https://img.shields.io/badge/Axum-0.7_Async_Web-3B82F6?style=for-the-badge&logo=tokio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)

**Paged KV-cache, radix prefix caching, continuous batching, speculative decoding,
Git-like KV sessions, time travel, and tiered KV memory — in Rust 2024.**

</div>

NexusKV is an inference runtime for transformer decoding. It treats the KV cache
as a paged, reference-counted resource: blocks move through an explicit state
machine, shared prefixes live in a radix trie, and evicted blocks can spill to
host memory and NVMe without changing model semantics. A single-writer actor
owns the allocator, scheduler, executor, and session store, while HTTP/SSE
handlers relay measured telemetry to clients and the dashboard.

> **Measurement policy.** The default `cargo run` starts a deterministic **mock
> executor**. Mock mode is logged at startup, reported as `"executor": "mock"`
> by `/v1/models` and `/v1/metrics`, and shown as a MOCK banner in the
> dashboard. Real dense Qwen inference runs on CPU with the `candle-model`
> feature or on an NVIDIA GPU with `cuda`; the custom CUDA kernel is part of the
> GPU path. Allocator microbenchmarks and mock-mode
> token rates are never presented as inference throughput. Every benchmark in
> this document names its hardware, model, executor mode, and metric.

---

## Contents

- [Screenshots](#screenshots)
- [Current capabilities](#current-capabilities)
- [Target hardware and defaults](#target-hardware-and-defaults)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [API](#api)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Screenshots

Live dashboard captured on the target machine with real Qwen3-0.6B f16 weights,
`executor=real`, and the custom CUDA f16 decode kernel. The KV pipeline tiles show
measured prefill/skipped tokens, adaptive K, evictions, tier swap bytes, decode
batch size, and the attention backend:

![NexusKV live dashboard](docs/dashboard-live.png)

Time-travel replay at tick 66 of the same run: a recorded keyframe plus memory
events reconstructs the 6 active pages exactly. The HUD above keeps the last
measured live performance values; only allocator state is replayed:

![NexusKV time-travel dashboard](docs/dashboard-time-travel.png)

---

## Current capabilities

| Area | State |
| --- | --- |
| Memory core | `Free`/`Active`/`Cached` state machine, radix trie, transactional prefix match/append/fork, copy-on-write tails, O(1) free-list guard, LRU eviction, pinned commits |
| Engine | Single-writer actor, continuous batching, chunked prefill, decodes-first scheduling, starvation-free admission, preemption |
| Batched decode | One fused model forward per scheduler step for every decoding sequence: all projections, QK-norm, RoPE, MLP, and lm_head run as batched GEMMs; paged attention runs per sequence, or as one multi-sequence custom-kernel launch when each sequence decodes one token |
| Serving | OpenAI-compatible chat completions, SSE generation, measured TTFT/ITL/tokens-per-second, client-disconnect aborts |
| Speculative decoding | Prompt-lookup n-gram drafter, distribution-exact rejection sampling, greedy verification, UCB1 adaptive `K` over `{0,2,4,8,16}` |
| Sessions | Commit pins full-block KV prefixes, checkout resumes in O(1) and branches independently, forget releases pins |
| Time travel | 50k-event ring buffer, keyframes at most 256 ticks apart, `GET /v1/timetravel?tick=N` |
| Tiering | Content-addressed host LRU plus NVMe spill, demand page-in on `Swapped` trie hits, fault-injection coverage, NKV2 dtype-tagged spill files |
| Real model | Dense Qwen3-0.6B (primary) and Qwen2.5-0.5B-Instruct (fallback) behind `candle-model` on CPU (f32) or `cuda` on GPU (f16); QK-norm, GQA, RoPE, SwiGLU, f32 norm/softmax accumulation |
| CUDA | f16 `paged_decode_attention` with f32 accumulation and GQA-correct head mapping, enabled for decode by `NEXUSKV_CUDA_ATTENTION=1`, launched on candle's stream with automatic gather fallback; the kernel's `grid.x` covers the whole decode batch; plus `reshape_and_cache` and `copy_block` |
| Prefix hits | KV-valid trie blocks skip prefill compute; a block-aligned full re-hit computes zero prefill tokens (measured `prefill_tokens` / `prefix_skip_tokens`) |
| Execution targets | CPU and NVIDIA GPU: `candle-model` runs dense Qwen on CPU; `cuda` adds the GPU backend and custom decode kernel |
| Default build | Mock executor; `cargo test` passes without a GPU, CUDA toolkit, or model weights |

### Feature flags

| Feature set | What it enables |
| --- | --- |
| `default` | Mock executor, memory/scheduler/session engine, HTTP/SSE server, CPU attention reference |
| `candle-model` | Dense Qwen3/Qwen2 inference on CPU through candle (f32 weights and KV pools) |
| `cuda` | Implies `candle-model`; runs inference on an NVIDIA GPU (f16 weights and KV pools), enables candle's CUDA backend, and compiles `csrc/paged_attention.cu` with nvcc; falls back to CPU if no CUDA device is available |

### Target hardware and defaults

The default engine configuration targets an **RTX 3050 Laptop with 4 GB VRAM
and 8 GB system RAM on native Windows 11** (no WSL):

- `total_physical_blocks = 192`
- `block_size = 16`
- `max_num_seqs = 8`
- `max_blocks_per_seq = 40` — a per-sequence cap, not a reservation; supports a
  512-token prompt
- `max_num_batched_tokens = 256`
- CUDA weights and KV pools are f16; the CPU path stays f32
- f16 KV pool ≈ 344 MB
- Tier host budget defaults to `512 MiB`

The same block and sequence defaults apply to both backends. On CUDA, f16 is the
default; `NEXUSKV_DTYPE=f32` forces f32 and automatically clamps the pool to 96
blocks and 4 sequences with a warning. CPU builds always use f32. On 8 GB hosts,
build with `CARGO_BUILD_JOBS=2` (already set in `.cargo/config.toml`) to avoid
OOM while compiling candle.

Native Windows setup, PowerShell build/run/test scripts, and the system-memory
fallback note live in [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

---

## Architecture

```
 HTTP / SSE (Axum)
        │  Generate · Abort · Commit · Checkout · Forget · SessionTree
        ▼                              ▲ TokenMsg → SSE
 ┌────────────────────────────────────┴──────────────────────────────┐
 │ Engine actor thread (single writer)                               │
 │  Scheduler · BlockSpaceManager · Executor · SessionStore          │
 │  LookupDrafter · AdaptiveK bandit · FlightRecorder keyframes      │
 └──────────────┬───────────────────────────────┬────────────────────┘
                │ paged slots / block tables     │ non-blocking MemEvents
                ▼                                ▼
 ┌──────────────────────────────┐      ┌──────────────────────────────┐
 │ Executor                      │      │ FlightRecorder               │
 │  Mock | Qwen dense (candle)   │      │  50k-event ring buffer       │
 │  Paged K/V pools              │      │  keyframes ≤256 ticks apart  │
 └──────────────┬───────────────┘      └──────────────────────────────┘
                │ GpuBridge (dtype-tagged K/V block bytes)
                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TierStore: content-addressed host LRU → NVMe {hash}.kv (NKV2)    │
 └──────────────────────────────────────────────────────────────────┘
```

### 1. Paged KV memory

Physical blocks move through `Free → Active → Cached → Free` with explicit
refcounts, pins, and an O(1) free-list guard. Cached blocks are never on the
free stack. Prefix matching, appends, forks, rollbacks, and copy-on-write are
transactional: a late OOM undoes every acquisition. Tests assert the full
invariant set after every operation, including deterministic fuzzing and
property-based tests.

### 2. Continuous batching actor

One engine thread owns the allocator, scheduler, executor, and sessions. HTTP
handlers submit channel requests and relay per-sequence token channels. The
scheduler runs decodes first, chunks running prefills, and admits waiting
sequences while others are decoding, so a long-running decode cannot starve new
work. An admission OOM preempts the newest non-progressing sequence and requeues
it.

Each scheduler step decodes every ready sequence in **one fused forward**: the
actor prepares all inputs (appends KV slots, materializes copy-on-write tails),
calls `Executor::forward_decode_batch` once, then verifies and emits per
sequence. Batched decode is a trait method with a safe per-sequence default, so
the mock executor stays correct while `Qwen2Executor` fuses the real model.

### 3. Speculative decoding

- `LookupDrafter` indexes 3-grams over the prompt and committed output, and
  proposes the continuation after the most recent suffix match.
- For one-hot draft distributions, rejection sampling is distribution-exact:
  accept with `p_target(draft)`; otherwise sample a correction from the target
  with the draft token removed and stop the chain. Fully accepted drafts earn
  one bonus token.
- Temperature below `1e-3` switches to greedy verification.
- `AdaptiveK` is a UCB1 bandit over `{0,2,4,8,16}` with reward
  `accepted_tokens / verification_ms`.

### 4. Sessions and time travel

`POST /v1/sessions/commit` pins the sequence's full blocks and stores its token
history. `POST /v1/sessions/checkout` attaches the pinned prefix with O(1)
refcount bumps and re-prefills only the partial tail. `POST
/v1/sessions/forget` releases the pins. Pinned blocks are never evictable.

Every allocator event is recorded in a bounded flight recorder. Full allocator
keyframes are kept at most 256 ticks apart, so `GET /v1/timetravel?tick=N`
returns a keyframe plus the exact event suffix to replay. The dashboard applies
the same reducer client-side and never fabricates offline state.

### 5. Tiered KV

`TierStore` implements the allocator's `BlockIo` trait, content-addressed by the
trie hash. Eviction reads a block through `GpuBridge`, keeps it in a
byte-budgeted host LRU, and spills the oldest entries to
`{swap-dir}/{hash:016x}.kv`. Spill files use the NKV2 container with a dtype tag,
so f16 and f32 blocks can never be decoded as the wrong type. A `Swapped` trie
hit allocates a block, pages the bytes back in, and increments `page_faults`.
Missing or corrupt files are loud errors; save failures fall back to safe leaf
drops or the next candidate.

---

## Quickstart

Prerequisites:

- Rust stable (2024 edition)
- Optional: CUDA Toolkit **13.4** (or 12.6+), Visual Studio 2022 Build Tools, and
  an NVIDIA GPU for the `cuda` feature
- Node 22 and pnpm for the dashboard

> **Toolkit ceiling.** candle 0.9.x depends on cudarc 0.19.10, whose build script
> panics on any toolkit newer than **13.4**. CUDA 13.5 (current latest) cannot
> build this checkout. `scripts/install_cuda_user.ps1` installs 13.4.3 without
> Administrator rights.

### Mock mode (no GPU, no weights)

```bash
cargo test
cargo run --release
```

The engine listens on `http://127.0.0.1:8080`. In a second terminal, start the
dashboard:

```bash
cd dashboard
pnpm install
pnpm dev          # http://localhost:3000
```

Mock mode is deterministic and explicitly labeled. It is intended for memory,
scheduler, and session testing, and is never presented as model inference.

### Real dense Qwen inference (CPU or GPU)

```bash
# Primary target: Qwen/Qwen3-0.6B (dense, 28 layers, 8 KV heads, head_dim 128)
# Fallback: Qwen/Qwen2.5-0.5B-Instruct
export CARGO_BUILD_JOBS=2

# CPU backend
cargo run --release --features candle-model -- --model /path/to/Qwen3-0.6B

# NVIDIA GPU backend (requires an MSVC host toolchain and CUDA 13.4)
cargo run --release --features cuda -- --model /path/to/Qwen3-0.6B
```

`candle-model` selects the CPU device; `cuda` selects the GPU and is the only
feature set that compiles the custom kernel. With `cuda`, a missing CUDA device
falls back to CPU with a warning. On Windows, use
`scripts/install_cuda_user.ps1` once, then `scripts/build_windows.ps1` and
`scripts/run_windows.ps1`.

Pool sizing is adjustable for measurement and demos:

```bash
--blocks 24        # physical KV blocks (default 192)
--block-size 16    # tokens per block
--max-seqs 6       # concurrent sequences
--swap-dir DIR     # enables host/NVMe tiering
--host-budget N    # tier host budget in bytes (default 512 MiB)
```

The model directory must contain `config.json`, `tokenizer.json`, and at least
one `*.safetensors` shard. Split checkpoints are loaded one shard at a time; a
single-shard checkpoint takes the low-overhead mmap path. Qwen3 QK-norm tensors
are detected and applied before RoPE; Qwen2 checkpoints use the same code path
without them. Qwen3.5/3.6/3.8 Gated DeltaNet hybrid architectures are rejected
with an explicit error.

### CUDA decode kernel (native Windows)

```powershell
# One time: installs CUDA 13.4.3 under %USERPROFILE%\cuda-13.4 (no admin needed).
powershell -ExecutionPolicy Bypass -File scripts/install_cuda_user.ps1

# Imports the VS x64 environment, resolves CUDA, and builds the engine.
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1

$env:NEXUSKV_CUDA_ATTENTION = "1"
powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1
```

The custom f16 kernel handles decode (`seq_len == 1`) on candle's device stream;
prefill keeps gather attention. Its `grid.x` is the sequence count, so a batched
decode step is one kernel launch for the whole batch. Without the environment
flag, or on any dtype or contiguity mismatch, the engine falls back to gather
attention with a warning. GPU parity and compute-sanitizer tests are `#[ignore]`d
for ordinary `cargo test`; run them on the target machine with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_gpu.ps1
```

The CPU f32 softmax reference, including the GQA head mapping, is always
available and tested without a GPU.

### Tiered KV (host + NVMe)

```bash
cargo run --release --features candle-model -- \
  --model /path/to/Qwen3-0.6B \
  --swap-dir /var/tmp/nexuskv-kv \
  --host-budget 536870912
```

`--host-budget` is in bytes (default `512 MiB`). In mock mode the paging
machinery is real, but block bytes are synthetic; startup logs a warning.

To watch tiering actually fire, shrink the pool below the working set and drive
both eviction and a demand page-in:

```bash
# 16 blocks, 4 sequences, tiering enabled
cargo run --release --features cuda -- \
  --model /path/to/Qwen3-0.6B --swap-dir /var/tmp/nexuskv-kv \
  --blocks 16 --max-seqs 4

# Shared prefix first, judged fillers to force eviction, then the same prefix
# again so its swapped trie nodes are paged back in.
python scripts/tiering_probe.py --clients 4 --fillers 12 --filler-blocks 6 --remeasure 4
```

The probe prints measured deltas: on this machine it produced 35 evictions
(61.3 MiB written to the tier), one demand page-in (1.75 MiB), and 304
prefix-skip tokens.

### Dashboard

```bash
cd dashboard
pnpm install
pnpm dev          # development
pnpm build        # production build
pnpm typecheck
pnpm test         # replay reducer tests
```

The dashboard reads live telemetry from `http://127.0.0.1:8080`, provides a
time-travel scrubber, and exposes commit/checkout/forget controls with a session
branch tree. If the engine is offline it says so; it does not simulate playback.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `GET` | `/v1/models` | Model and executor identity |
| `GET` | `/v1/metrics` | Live allocator, scheduler, speculative, and tier counters |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (streaming or buffered) |
| `POST` | `/v1/generate/stream` | Engine SSE stream with per-token telemetry |
| `POST` | `/v1/sessions/commit` | Pin a sequence's KV prefix and token history |
| `POST` | `/v1/sessions/checkout` | Resume a committed prefix and stream tokens |
| `POST` | `/v1/sessions/forget` | Release a commit's pins |
| `GET` | `/v1/sessions/:id/tree` | Session branch tree |
| `GET` | `/v1/timetravel?tick=N` | Keyframe plus event suffix for allocator replay |

### OpenAI Chat Completions

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexuskv-engine-v1",
    "messages": [{"role": "user", "content": "Explain paged attention."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": true
  }'
```

Streaming ends with `[DONE]`; non-streaming responses return OpenAI `usage`.
Existing DTO shapes are preserved and new fields are additive.

### Engine stream with speculative decoding

```bash
curl -N -X POST http://127.0.0.1:8080/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain continuous batching", "max_tokens": 64, "spec": true}'
```

Each SSE event carries measured `ttft_ms`, `itl_ms`, `tokens_per_sec`,
`event_tick`, physical block ids, acceptance, and adaptive `K`.

### Sessions

```bash
# Commit a live sequence (the sequence id comes from SSE telemetry).
curl -X POST http://127.0.0.1:8080/v1/sessions/commit \
  -H "Content-Type: application/json" \
  -d '{"session":"main","label":"root","seq":7}'

# Checkout a commit (SSE token stream, same telemetry schema).
curl -N -X POST http://127.0.0.1:8080/v1/sessions/checkout \
  -H "Content-Type: application/json" \
  -d '{"node":1,"max_tokens":64,"temperature":0.7,"spec":true}'

# Unpin and forget.
curl -X POST http://127.0.0.1:8080/v1/sessions/forget \
  -H "Content-Type: application/json" -d '{"node":1}'

# Branch tree.
curl http://127.0.0.1:8080/v1/sessions/main/tree
```

### Time travel and telemetry

```bash
curl 'http://127.0.0.1:8080/v1/timetravel?tick=256'
curl http://127.0.0.1:8080/v1/metrics
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/health
```

`/v1/metrics` includes `event_tick`, live/cached/free block counts, prefix hit
ratio, acceptance, adaptive `K`, evictions, page faults, CoW copies, preemptions,
tier counters (`swapped_out`, `swapped_in`, `disk_spills`, `disk_loads`), and
batched-decode counters (`batched_decode`, `decode_batches`,
`decode_batch_sequences`, `max_decode_batch`).

---

## Benchmarks

Two tools measure different things:

**Bookkeeping only** — allocator, trie, and rollback; no model and no tokens:

```bash
cargo bench --bench bookkeeping
```

The benchmark prints `BOOKKEEPING THROUGHPUT — NOT INFERENCE`. Criterion results
must never be quoted as serving throughput.

**End-to-end serving** — real concurrent SSE clients and measured tokens:

```bash
python3 scripts/bench_e2e.py \
  --url http://127.0.0.1:8080 \
  --clients 8 --max-tokens 128 --spec
```

The script reports aggregate tokens/s, TTFT p50/p99, ITL p50/p99, per-client
throughput, prefix hit rate, prefill/skipped tokens, acceptance, adaptive `K`,
evictions, page faults, and tier counters. `--prefix-probe` measures cold versus
warm TTFT on a unique prompt re-hit. `--compare` prints the same-GPU/same-model
llama.cpp comparison protocol; it does not assert a result on its own.

**Medians, not single runs.** Laptop GPUs move between runs (clocks, thermals,
Windows scheduling), so every headline number below is the median of three runs:

```bash
python scripts/bench_median.py --runs 3 -- --clients 4 --max-tokens 64 --spec
```

Each raw run is kept under `benches/results/` (gitignored local evidence).

### Measured on this machine (2026-11-11)

| Item | Value |
| --- | --- |
| Hardware | NVIDIA RTX 3050 Laptop 4 GB, driver 621.01, 8 GB RAM, Windows 11 |
| Toolchain | Rust 1.99.1 MSVC, CUDA Toolkit 13.4 (per-user redistributable), Visual Studio 2022 Build Tools (MSVC 14.44) |
| Model | Qwen/Qwen3-0.6B, f16 weights and f16 KV pools |
| Engine | 192 blocks × 16 tokens, max 8 sequences, batched decode, custom CUDA f16 decode + gather prefill |
| VRAM (nvidia-smi) | **1,675 MiB / 4,096 MiB** after model + KV pool |
| 4 clients × 64 tokens, spec on | **76.3 tok/s** aggregate median (24.6 tok/s per client), range 75.0–76.7 |
| TTFT p50 / p99 (median of runs) | **131.4 ms / 196.9 ms** |
| ITL p50 / p99 (median of runs) | **42.6 ms / 106.4 ms** |
| 8 clients × 64 tokens, spec on | **111.4 tok/s** aggregate, 7.8 sequences per fused decode forward |
| Speculative acceptance | **46.7% on quote/echo** (`scripts/spec_acceptance_probe.py`), **25.3% median** on the benchmark prompt; adaptive `K=4` |
| Evictions / page faults / swap (192-block run) | 0 / 0 / 0 bytes (pool not under pressure) |
| Forced tiering (16 blocks, 4 clients ÷ 4) | **35 evictions, 61.3 MiB swapped out, 1 demand page-in (1.75 MiB), 304 prefix-skip tokens**, no client errors |
| 512-token aligned prefix re-hit | TTFT **243.6 ms → 3.01 ms (80.9×)**; cold prefill 512 tokens, warm prefill **0**, 512 skipped |
| Dashboard single request (warm prefix) | 22.0 tok/s, TTFT 59.1 ms, ITL 45.2 ms (median of 6 warm requests) |

Batched decode replaced one forward pass per sequence with one fused forward per
scheduler step. On the same GPU, model, and prompt, 4 clients went from
**23.76 tok/s** (one forward per sequence, recorded in the previous revision) to
**76.3 tok/s** median — a **3.2×** aggregate gain. The measurement is visible in
telemetry: `/v1/metrics` reports `batched_decode`, `decode_batches`,
`decode_batch_sequences`, and `max_decode_batch`, so the fusion is countable
rather than asserted.

Raw JSON is written to `benches/results/` (gitignored local evidence), including
`rtx3050-4c-median.json` for the median summary above.

Mock-mode token rates measure bookkeeping and transport only and are never
presented as model inference. These numbers are single-machine measurements on
this GPU and are not a comparison against another engine; run `--compare` with
llama.cpp on the same device for that.

---

## Testing

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test                                      # no GPU or weights required
cargo test --features candle-model
cd dashboard && pnpm typecheck && pnpm test && pnpm build
```

On Windows, `scripts/check_env.ps1 -SelfTest` verifies the toolchain resolution
logic used by CI.

Coverage highlights:

- B1/B2/B5 regressions, LRU/CoW/rollback units, 8×8000 deterministic fuzz runs,
  12k-operation pressure runs, and proptest refcount conservation.
- Scheduler starvation/preemption units and actor lifecycle, abort, and
  shared-prefix integration.
- Batched decode parity: fused logits match per-sequence decode on synthetic
  weights (three sequences, distinct lengths and positions) and on real Qwen3
  weights on the GPU, including greedy token identity across four steps.
- HTTP/SSE/OpenAI schema, client disconnect, session commit/checkout/forget, and
  time-travel replay determinism.
- Rejection-sampling statistics (400k seeded draws), greedy speculative/plain
  equivalence over 20 prompts, bandit convergence, and speculative KV accounting
  under OOM.
- Tier roundtrip, host budget spill, disk/missing/corrupt-file behavior,
  failure-injection fuzz, and an 8-block demand-paging end-to-end test.
- Anti-fake build regressions: no canned vocabularies, pacing sleeps, `0.85`
  acceptance defaults, or dashboard playback simulations.

GPU tests are the only `#[ignore]`d tests. Each carries a hardware comment and is
run on the target machine with `scripts/test_gpu.ps1` (f32/f16 kernel parity, f16
GQA parity against the CPU oracle, real-weight logit L2 parity plus greedy token
identity, and real-model determinism). `scripts/sanitize.ps1` runs
compute-sanitizer; on Windows WDDM it requires the one-time Administrator step
documented in [WINDOWS_SETUP.md](WINDOWS_SETUP.md#7-gpu-tests). If that interface
is not enabled, the script exits 2 rather than claiming a clean run.

---

## Project layout

```
nexuskv/
├── Cargo.toml                  # Features: default | candle-model | cuda
├── build.rs                    # CUDA build orchestration (feature-gated)
├── src/build_support.rs        # CUDA root/flag resolution, unit-tested
├── DECISIONS.md                # Engineering decisions and recorded deviations
├── WINDOWS_SETUP.md            # Native Windows 11 setup
├── LICENSE                     # MIT
├── csrc/
│   └── paged_attention.cu      # Online-softmax decode + KV kernels
├── src/
│   ├── config.rs
│   ├── memory/
│   │   ├── block_manager.rs    # States, trie, CoW, LRU, BlockIo, invariants
│   │   ├── events.rs           # FlightRecorder keyframes + replay reducer
│   │   ├── tier.rs             # TierStore, GpuBridge, host/NVMe tiering
│   │   ├── tests.rs
│   │   └── fuzz.rs
│   ├── engine/
│   │   ├── actor.rs            # Single-writer loop + Snapshot telemetry
│   │   ├── scheduler.rs        # Continuous batching + preemption
│   │   ├── speculative.rs      # Drafter, rejection sampling, AdaptiveK
│   │   └── session.rs          # Commit/checkout/forget branch store
│   ├── model/
│   │   ├── mod.rs              # Executor/Tokenizer + MockExecutor
│   │   └── qwen.rs             # cfg(candle-model): dense Qwen3/Qwen2
│   ├── kernels/paged_attention.rs
│   └── server/{mod.rs,routes.rs,sse_handler.rs}
├── tests/{api,bench_smoke,engine,honesty,real_model,sessions,speculative,tiering}.rs
├── benches/bookkeeping.rs
├── scripts/
│   ├── bench_e2e.py            # End-to-end SSE benchmark
│   ├── bench_median.py         # N-run medians of the above
│   ├── tiering_probe.py        # Forces eviction + demand page-in
│   ├── spec_acceptance_probe.py# Quote/echo acceptance measurement
│   ├── install_cuda_user.ps1   # Per-user CUDA 13.4 (no admin)
│   ├── build.sh, sanitize.sh   # Linux/macOS helpers
│   └── *.ps1                   # Windows environment/build/run/test scripts
└── dashboard/                  # Next.js 15 telemetry and time-travel UI
```

---

## Known limitations

- The custom f16 CUDA kernel is wired into **decode only**
  (`NEXUSKV_CUDA_ATTENTION=1`, `seq_len == 1`). Prefill uses candle's gather
  attention; the kernel path falls back to gather on any dtype, contiguity, or
  launch mismatch with a one-time warning.
- Batched decode fuses every weight GEMM, but paged attention is one gather
  call per sequence unless the custom kernel is active and each sequence decodes
  exactly one token, in which case the whole batch is a single kernel launch.
  Speculative drafts therefore use the per-sequence gather path.
- CUDA weights and KV pools are f16. `NEXUSKV_DTYPE=f32` is supported but
  automatically clamps the pool to 96 blocks and 4 sequences because f32 weights
  plus 192 f32 blocks do not fit in 4 GB.
- The toolkit ceiling is CUDA **13.4**: candle 0.9.x → cudarc 0.19.10 panics on
  newer majors. CUDA 13.5 cannot build this checkout until candle moves to a
  newer cudarc.
- compute-sanitizer needs a one-time **Administrator** step
  (`EnableDebuggerInterface.bat`) to attach on WDDM. Until that is done the
  script exits 2 and reports the run as uninstrumented rather than clean; see
  [WINDOWS_SETUP.md](WINDOWS_SETUP.md#7-gpu-tests).
- A single sequence cannot exceed the resident block pool: active blocks are not
  evictable. The 8-block/200-token tiering end-to-end test completes 200 tokens
  across sequences with a shared prefix, forcing swap-out and page-in between
  rounds.
- Mock tiering stores synthetic block bytes; the paging machinery is real and
  labeled at startup.
- The Qwen3.5/3.6/3.8 hybrid architectures are rejected; only dense Qwen3/Qwen2
  checkpoints are supported.
- Tier spill files use the NKV2 dtype-tagged format. Legacy NKV1 files are
  rejected loudly; delete an old swap directory before reusing it.
- Full-prefix re-hits serve first-token logits from a bounded 8-entry cache
  measured at prefill time. A cache miss performs one last-token forward and
  counts it honestly in `prefill_tokens`.

---

## License

Licensed under the [MIT License](LICENSE).
