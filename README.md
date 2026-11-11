# ⚡ NexusKV Engine

<div align="center">

![Rust](https://img.shields.io/badge/Rust-2024_Edition-DEA584?style=for-the-badge&logo=rust&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15.0_App_Router-000000?style=for-the-badge&logo=next.js&logoColor=white)
![Axum](https://img.shields.io/badge/Axum-0.7_Async_Web-3B82F6?style=for-the-badge&logo=tokio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)

**Paged KV-Cache, Radix Prefix Caching, Continuous Batching, Speculative Decoding,
Git-for-KV Sessions, Time Travel, and Tiered KV Memory — in Rust 2024.**

</div>

> **Honesty notice.** The default `cargo run` starts a deterministic **mock
> executor**. Mock mode is loudly logged at startup, reports
> `"executor": "mock"` from `/v1/models` and `/v1/metrics`, and the dashboard
> shows a MOCK banner. Real dense Qwen inference lives behind the
> `candle-model` feature; the custom CUDA kernel lives behind `cuda`. This
> README does not claim inference throughput from allocator microbenchmarks or
> mock-mode token rates. Every benchmark must name its hardware, model, executor
> mode, and what it measures.

---

## 📸 Screenshots

Live dashboard (mock executor, explicitly labeled):

![NexusKV live dashboard](docs/dashboard-live.png)

Time-travel replay driven by flight-recorder keyframes and memory events:

![NexusKV time-travel dashboard](docs/dashboard-time-travel.png)

---

## 🧭 Current status

| Area | State |
| --- | --- |
| Memory core | `Free`/`Active`/`Cached` state machine, radix trie, transactional prefix match/append/fork, copy-on-write tails, O(1) free-list guard, LRU eviction, pinned commits |
| Engine | Single-writer actor, continuous batching, chunked prefill, decodes-first scheduling, starvation-free admission, preemption |
| Serving | OpenAI-compatible chat, SSE generation, measured TTFT/ITL/tok-s, client-disconnect aborts |
| Speculative decoding | Prompt-lookup n-gram drafter, distribution-exact rejection sampling, greedy verification, UCB1 adaptive `K` over `{0,2,4,8,16}` |
| Sessions | Commit pins full-block KV prefixes, checkout resumes in O(1) and branches independently, forget unpins |
| Time travel | 50k-event ring buffer, keyframes at most 256 ticks apart, `GET /v1/timetravel?tick=N` |
| Tiering | Content-addressed host LRU + NVMe spill, demand page-in on `Swapped` trie hits, fault-injection coverage |
| Real model | Dense Qwen3-0.6B (primary) / Qwen2.5-0.5B-Instruct fallback behind `candle-model`; QK-norm, GQA, RoPE, SwiGLU, paged gather attention |
| CUDA | `paged_decode_attention` with online softmax, `reshape_and_cache`, `copy_block`; GPU parity tests are `#[ignore]`d because this build machine has no NVIDIA GPU/nvcc |
| Default build | CPU-only, no GPU, no CUDA toolkit, no model weights, all tests green |

### Target hardware and defaults

The default engine configuration targets an **RTX 3050 4GB + 8GB system RAM
(WSL2)** host:

- `total_physical_blocks = 192`
- `block_size = 16`
- `max_num_seqs = 8`
- `max_blocks_per_seq = 24`
- KV pool ≈ 344 MB at f16 (the current candle path stores f32; see limitations)
- Tier host budget defaults to `512 MiB`

Builds on 8GB hosts should use `CARGO_BUILD_JOBS=2` (also set in
`.cargo/config.toml`) to avoid OOM while linking candle.

---

## 🏛️ Architecture

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
                │ GpuBridge (f32 block bytes)
                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ TierStore: content-addressed host LRU → NVMe {hash}.kv           │
 └──────────────────────────────────────────────────────────────────┘
```

### 1. Paged KV memory

Physical blocks move through `Free → Active → Cached → Free` with explicit
refcounts, pins, and an O(1) free-list guard. Cached blocks are never on the
free stack. Prefix matching, appends, forks, rollbacks, and CoW are
transactional: a late OOM undoes every acquisition. Tests assert a full
invariant set after every operation, including deterministic fuzzing and
proptest.

### 2. Continuous batching actor

One engine thread owns the allocator, scheduler, executor, and sessions.
HTTP handlers submit channel requests and relay per-sequence token channels.
The scheduler runs decodes first, chunks running prefills, and admits waiting
sequences while others are decoding, so a long-running decode cannot starve
new work. Admission OOM preempts the newest non-progressing sequence and
requeues it.

### 3. Speculative decoding

- `LookupDrafter` indexes 3-grams over prompt + committed output and proposes
  the continuation after the most recent suffix match.
- For one-hot draft distributions, rejection sampling is distribution-exact:
  accept with `p_target(draft)`, otherwise sample a correction from the target
  with the draft token removed and stop the chain. All accepted drafts earn one
  bonus token.
- Temperature < 1e-3 switches to greedy verification.
- `AdaptiveK` is a UCB1 bandit over `{0,2,4,8,16}` with reward
  `accepted_tokens / verification_ms`.

### 4. Git-for-KV sessions and time travel

`POST /v1/sessions/commit` pins the sequence's full blocks and stores its token
history. `POST /v1/sessions/checkout` attaches the pinned prefix with O(1)
refcount bumps and re-prefills only the partial tail. `POST
/v1/sessions/forget` releases the pins. Pinned blocks are never evictable.

Every allocator event is recorded in a bounded flight recorder. Full allocator
keyframes are kept at most 256 ticks apart, so `GET /v1/timetravel?tick=N`
returns a keyframe plus the exact event suffix to replay. The dashboard
applies the same reducer client-side and never fabricates offline state.

### 5. Tiered KV

`TierStore` implements the allocator's `BlockIo` trait, content-addressed by the
trie hash. Eviction reads a block through `GpuBridge`, keeps it in a
byte-budgeted host LRU, and spills the oldest entries to
`{swap-dir}/{hash:016x}.kv`. A `Swapped` trie hit allocates a block, pages the
bytes back in, and increments `page_faults`. Missing or corrupt files are loud
errors; save failures fall back to safe leaf drops or the next candidate.

---

## 🚀 Quickstart

### Mock mode (no GPU, no weights)

```bash
cargo test
cargo run --release
# POST http://127.0.0.1:8080/v1/generate/stream
# Dashboard: http://localhost:3000
```

Mock mode is deterministic and labeled. It is useful for memory/scheduler/
session tests and never presented as model inference.

### Real dense Qwen inference

```bash
# Primary target: Qwen/Qwen3-0.6B (dense, 28 layers, 8 KV heads, head_dim 128)
# Fallback: Qwen/Qwen2.5-0.5B-Instruct
export CARGO_BUILD_JOBS=2
cargo run --release --features candle-model -- --model /path/to/Qwen3-0.6B
```

The model directory must contain `config.json`, `tokenizer.json`, and at least
one `*.safetensors` shard. Split checkpoints are loaded one shard at a time; a
single-shard checkpoint takes the low-overhead mmap path. Qwen3 QK-norm tensors
are detected and applied before RoPE; Qwen2 checkpoints use the same code path
without them. Qwen3.5/3.6/3.8 Gated DeltaNet hybrid architectures are rejected
with an explicit error.

### CUDA decode kernel

```bash
# Requires nvcc and an NVIDIA GPU (sm_86 SASS + compute_86 PTX emitted)
cargo run --release --features cuda -- --model /path/to/Qwen3-0.6B
```

The correct CPU softmax reference is always available and tested without a GPU.
CUDA parity and compute-sanitizer tests carry `#[ignore]` with hardware notes.

### Tiered KV (host + NVMe)

```bash
cargo run --release --features candle-model -- \
  --model /path/to/Qwen3-0.6B \
  --swap-dir /var/tmp/nexuskv-kv \
  --host-budget 536870912
```

`--host-budget` is in bytes (default `512 MiB`). In mock mode the paging
machinery is real but the block bytes are synthetic; startup logs a warning.

### Dashboard

```bash
cd dashboard
pnpm install
pnpm dev          # development
pnpm build        # production build
pnpm typecheck
pnpm test         # replay reducer tests
```

The dashboard defaults to live telemetry, adds a time-travel scrubber, and
shows commit/checkout/forget controls with a session branch tree. If the engine
is offline it says so; it does not simulate playback.

---

## 🔌 API

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

Streaming ends with `[DONE]`; non-streaming returns OpenAI `usage`. Existing
DTO shapes are preserved; new fields are additive.

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
# Commit a live sequence (seq comes from the SSE telemetry)
curl -X POST http://127.0.0.1:8080/v1/sessions/commit \
  -H "Content-Type: application/json" \
  -d '{"session":"main","label":"root","seq":7}'

# Checkout a commit (SSE token stream, same telemetry schema)
curl -N -X POST http://127.0.0.1:8080/v1/sessions/checkout \
  -H "Content-Type: application/json" \
  -d '{"node":1,"max_tokens":64,"temperature":0.7,"spec":true}'

# Unpin and forget
curl -X POST http://127.0.0.1:8080/v1/sessions/forget \
  -H "Content-Type: application/json" -d '{"node":1}'

# Branch tree
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
and tier counters (`swapped_out`, `swapped_in`, `disk_spills`, `disk_loads`).

---

## 📊 Benchmarks that stay honest

Two tools, with different meanings:

**Bookkeeping only** (allocator, trie, rollback; no model, no tokens):

```bash
cargo bench --bench bookkeeping
```

The benchmark prints `BOOKKEEPING THROUGHPUT — NOT INFERENCE`. Criterion results
must never be quoted as serving throughput.

**End-to-end serving** (real concurrent SSE clients, real measured tokens):

```bash
python3 scripts/bench_e2e.py \
  --url http://127.0.0.1:8080 \
  --clients 8 --max-tokens 128 --spec
```

The script reports aggregate tok/s, TTFT p50/p99, ITL p50/p99, per-client
throughput, prefix hit rate, acceptance, adaptive K, evictions, page faults, and
tier counters. `--compare` prints the same-GPU/same-model llama.cpp comparison
protocol; it does not assert a result on its own.

No throughput, latency, or acceptance number is presented as a project result
in this README. Mock-mode token rates measure bookkeeping and transport only.

---

## 🧪 Testing

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test                       # no GPU, no weights required
cargo test --features candle-model
cd dashboard && pnpm typecheck && pnpm test && pnpm build
```

Coverage highlights:

- B1/B2/B5 regressions, LRU/CoW/rollback units, 8×8000 deterministic fuzz runs,
  12k-op pressure runs, and proptest refcount conservation.
- Scheduler starvation/preemption units and actor lifecycle/abort/shared-prefix
  integration.
- HTTP/SSE/OpenAI schema, client disconnect, session commit/checkout/forget,
  and time-travel replay determinism.
- Rejection-sampling statistics (400k seeded draws), greedy spec-on/off
  equivalence over 20 prompts, bandit convergence, and speculative KV
  accounting under OOM.
- Tier roundtrip, host budget spill, disk/missing/corrupt-file behavior,
  failure-injection fuzz, and an 8-block demand-paging e2e.
- Anti-fake build regressions: no canned vocabularies, pacing sleeps, `0.85`
  acceptance defaults, or dashboard playback simulations.

GPU tests are the only `#[ignore]`d tests; each carries a hardware comment and
is run with `scripts/sanitize.sh` on a CUDA machine.

---

## 📂 Directory structure

```
nexuskv/
├── Cargo.toml                  # default | candle-model | cuda
├── build.rs                    # compiles .cu only with feature=cuda
├── DECISIONS.md                # engineering decisions + recorded deviations
├── LICENSE                     # MIT
├── csrc/
│   └── paged_attention.cu      # online-softmax decode + KV kernels
├── src/
│   ├── config.rs
│   ├── memory/
│   │   ├── block_manager.rs    # states, trie, CoW, LRU, BlockIo, invariants
│   │   ├── events.rs           # FlightRecorder keyframes + replay reducer
│   │   ├── tier.rs             # TierStore, GpuBridge, host/NVMe tiering
│   │   ├── tests.rs
│   │   └── fuzz.rs
│   ├── engine/
│   │   ├── actor.rs            # single-writer loop + Snapshot telemetry
│   │   ├── scheduler.rs        # continuous batching + preemption
│   │   ├── speculative.rs      # drafter, rejection sampling, AdaptiveK
│   │   └── session.rs          # commit/checkout/forget branch store
│   ├── model/
│   │   ├── mod.rs              # Executor/Tokenizer + MockExecutor
│   │   └── qwen.rs             # cfg(candle-model): dense Qwen3/Qwen2
│   ├── kernels/paged_attention.rs
│   └── server/{routes.rs,sse_handler.rs}
├── tests/{api,engine,sessions,speculative,tiering,honesty,bench_smoke}.rs
├── benches/bookkeeping.rs
├── scripts/{bench_e2e.py,sanitize.sh,build.sh}
└── dashboard/                  # Next.js 15 telemetry / time-travel UI
```

---

## ⚠️ Known limitations

- The custom CUDA kernel is parity-tested against the CPU oracle but is not yet
  wired to candle's persistent pool tensors; the candle path uses gather
  attention on the selected device. This is recorded in `DECISIONS.md`.
- Weights load as f32, so memory use is higher than an f16 deployment. The tier
  bridge also uses an f32 byte container; raw device-to-host copies and dtype-
  aware serialization are the documented optimization path.
- v1 recomputes prefill even on resident trie hits. Hits save memory, not
  compute yet; the code carries `// TODO(kv-valid)` at the exact spot.
- A single sequence cannot exceed the resident block pool: active blocks are not
  evictable. The 8-block/200-token tiering e2e completes 200 tokens across
  sequences with a shared prefix, forcing swap-out/page-in between rounds.
- Mock tiering stores synthetic block bytes; the paging machinery is real and
  labeled at startup.
- The Qwen3.5/3.6/3.8 hybrid architectures are rejected; only dense Qwen3/Qwen2
  checkpoints are supported.

---

## 📄 License

Licensed under the [MIT License](LICENSE).
