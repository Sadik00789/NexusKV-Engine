# ⚡ NexusKV Engine

<div align="center">

![Rust](https://img.shields.io/badge/Rust-2024_Edition-DEA584?style=for-the-badge&logo=rust&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.1_Ampere%2FHopper-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15.0_App_Router-000000?style=for-the-badge&logo=next.js&logoColor=white)
![Axum](https://img.shields.io/badge/Axum-0.7_Async_Web-3B82F6?style=for-the-badge&logo=tokio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)

**Paged KV-Cache, Radix Prefix Caching & Continuous Batching in Rust 2024.**

> **Implementation status.** The default build is CPU-only and starts a
> deterministic **mock executor**. Mock mode is loudly logged at startup and
> labeled `"executor": "mock"` in `/v1/models` and `/v1/metrics`; the dashboard
> shows a MOCK banner. Real Qwen2 inference is behind the `candle-model` feature,
> and the custom CUDA kernels are behind the `cuda` feature. This repository
> does **not** claim inference throughput from allocator/bookkeeping
> microbenchmarks. See [Current status](#-current-status) for details.

</div>

---

## 📖 Table of Contents
- [Architectural Overview](#-architectural-overview)
  - [1. PagedAttention Memory Virtualization](#1-pagedattention-memory-virtualization)
  - [2. Radix Tree Prefix Cache](#2-radix-tree-prefix-cache)
  - [3. Copy-on-Write (CoW) Page Rollbacks](#3-copy-on-write-cow-page-rollbacks)
  - [4. Axum Continuous Batching & SSE Pipeline](#4-axum-continuous-batching--sse-pipeline)
- [Core Features Matrix](#-core-features-matrix)
- [Current status](#-current-status)
- [Prerequisites & Quickstart](#-prerequisites--quickstart)
  - [1. CUDA & Linux / WSL2 Setup](#1-cuda--linux--wsl2-setup)
  - [2. Rust Toolchain (2024 Edition)](#2-rust-toolchain-2024-edition)
  - [3. Telemetry Dashboard (Next.js 15)](#3-telemetry-dashboard-nextjs-15)
- [API Reference & `curl` Examples](#-api-reference--curl-examples)
  - [OpenAI Chat Completions (`POST /v1/chat/completions`)](#openai-chat-completions-post-v1chatcompletions)
  - [Engine Telemetry Stream (`POST /v1/generate/stream`)](#engine-telemetry-stream-post-v1generatestream)
  - [Model Metadata (`GET /v1/models`)](#model-metadata-get-v1models)
  - [System Metrics (`GET /v1/metrics`)](#system-metrics-get-v1metrics)
  - [Health Probe (`GET /health`)](#health-probe-get-health)
- [Directory Structure](#-directory-structure)
- [License](#-license)

---

## 🏛️ Architectural Overview

NexusKV addresses memory fragmentation and scheduler overhead in LLM serving by combining virtual-memory paging, Trie-based prefix caching, and a single-writer continuous-batching engine actor.

### 1. PagedAttention Memory Virtualization
Standard LLM serving pre-allocates contiguous memory buffers for maximum sequence lengths, wasting 60%–80% of VRAM due to internal and external fragmentation. NexusKV divides the Key-Value (KV) cache into non-contiguous physical pages (16 tokens per block) gathered on-the-fly inside thread-local GPU SRAM.

```
Logical Sequence 1 (Tokens 0..47)
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  Logical Block 0     │  Logical Block 1     │  Logical Block 2     │
│  [Tokens 00..15]     │  [Tokens 16..31]     │  [Tokens 32..47]     │
└──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌────────────────────────────────────────────────────────────────────┐
│ Sequence Page Table (Seq #1)                                       │
│ [ Logical Block 0 -> Phys #12 ]                                    │
│ [ Logical Block 1 -> Phys #04 ]                                    │
│ [ Logical Block 2 -> Phys #89 ]                                    │
└────────────────────────────────────────────────────────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
Physical GPU VRAM Pool (512 Physical Blocks @ 16 tokens/block)
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Block 04 │ │ Block 12 │ │ Block 89 │ │ Block 02 │ │ Block 15 │
│ (Seq #1) │ │ (Seq #1) │ │ (Seq #1) │ │ (Free)   │ │ (Seq #2) │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

---

### 2. Radix Tree Prefix Cache
Incoming sequences sharing common prefixes (e.g., system instructions, few-shot examples, multi-turn dialogues) are matched against a Radix Trie node index. Shared token chunks increment the physical block's `ref_count` without duplicate allocations or GPU memory copies.

```
                       [ Radix Trie Root ]
                                │
                 Token Chunk: "System: You are an AI..."
                                │
                      [ Physical Block #08 ] (ref_count = 3)
                                ├───┬───┐
                ┌───────────────┘   │   └───────────────┐
   User A: "Summarize..."           │      User C: "Translate..."
                │                   │                   │
      [ Phys Block #24 ]            │         [ Phys Block #67 ]
      (ref_count = 1)               │         (ref_count = 1)
                                    │
                         User B: "Explain..."
                                    │
                          [ Phys Block #41 ]
                          (ref_count = 1)
```

---

### 3. Copy-on-Write (CoW) Page Rollbacks
Foundation for speculative decoding in phase 4.
Speculative decoding generates multiple candidate draft tokens in parallel. When a draft branch diverges from target model logits, NexusKV triggers an $O(1)$ page rollback by popping excess block pointers and decrementing physical `ref_count`, completely avoiding expensive $O(N)$ tensor copies.

```
Target Verification:
Draft Candidates: [ Token 100, Token 101, Token 102 (Divergence!), Token 103 ]
Target Logits   : [ Token 100, Token 101, Token 999 (Accepted Correction)   ]

Rollback Action:
1. Retain 3 Tokens -> [ 100, 101, 999 ]
2. Rollback Page Table -> Drop Block #52 back to Free Stack in O(1)
3. Zero Memory Copies (page-table pointer reclamation)
```

---

### 4. Axum Continuous Batching & SSE Pipeline

```
 HTTP Requests (OpenAI API / SSE)
               │
               ▼
┌────────────────────────────────┐
│ Axum handlers (no engine locks)│
└──────────────┬─────────────────┘
               │ Generate / Abort (mpsc)        ▲ TokenMsg -> SSE
               ▼                                │
┌──────────────────────────────────────────────┴─────────────────────┐
│ Engine actor thread (single writer)                                │
│  Continuous scheduler · BlockSpaceManager · Executor · Telemetry    │
│  waiting/running queues · prefix trie · LRU/tiering · rollbacks     │
└──────────────┬─────────────────────────────────────────────────────┘
               │ paged slots / block tables
               ▼
┌──────────────────────────────┐       ┌─────────────────────────────┐
│ Executor: Mock | Qwen2+CUDA  │ <===> │ Paged KV pools + allocator  │
└──────────────┬───────────────┘       └─────────────────────────────┘
               │
               ▼
  Snapshot + SSE telemetry -> Next.js dashboard (MOCK banner in mock mode)
```

---

## 🚀 Core Features Matrix

| Feature | Standard Contiguous KV Cache | NexusKV Engine |
| :--- | :--- | :--- |
| **Memory Allocation** | Static maximum pre-allocation | **Dynamic physical paging (configurable block size)** |
| **VRAM Fragmentation** | Fixed reservation regardless of use | **Paged allocation without a fixed reservation** |
| **Prefix Caching** | Duplicate prompt KV computation | **Radix Tree sharing with refcounts** |
| **Rollbacks** | Rewriting tensors | **Page-table rollback (decode blocks are never trie-inserted)** |
| **Concurrency Locking** | Cross-thread scheduler locks | **Single-writer engine actor (no hot-path locks)** |
| **Client Disconnections** | Memory leaked until sequence end | **RAII `AbortDropGuard` sends an actor abort** |
| **API Compatibility** | Custom proprietary endpoints | **OpenAI-compatible (`/v1/chat/completions`)** |
| **VRAM Observability** | Blind black-box memory | **Interactive Next.js block heatmap** |
| **Speculative decoding** | — | **Planned (phase 4); telemetry reports 0.0 until then** |

---

## 📊 Current status

- **Phase 1 — memory core:** block state machine (`Free`/`Active`/`Cached`),
  persistent trie nodes, transactional prefix match/append/fork, copy-on-write
  partial tails, O(1) free-list guard, LRU eviction (leaf-only in drop mode,
  any cached block in tier mode), and an injectable `BlockIo` tier interface.
  `BlockSpaceManager::assert_invariants()` is exercised by deterministic fuzzing
  and a proptest suite.
- **Phase 2 — engine:** one actor thread owns the allocator, scheduler, executor
  and session state. HTTP handlers submit work through channels and relay tokens
  through SSE. The scheduler performs decodes first, chunks running prefills, and
  admits waiting sequences while other sequences are decoding (no starvation).
- **Phase 3 — real inference:** Qwen2 forward pass, tokenizer, paged KV pool, and
  a correct CPU attention reference live behind `candle-model`. The optimized
  CUDA decode kernel is behind `cuda` and was compiled only when `nvcc` exists.

No throughput, latency, or acceptance numbers are asserted as project results in
this README. Any benchmark must identify the hardware, the model, the executor
mode, and whether it measures bookkeeping or inference.

## 🛠️ Prerequisites & Quickstart

### 1. CUDA & Linux / WSL2 Setup
- NVIDIA GPU (Ampere, Ada Lovelace, Hopper, or Blackwell).
- NVIDIA Driver $\ge 535.x$ and CUDA Toolkit 12.x or 13.x.
- Set your `CUDA_HOME`:
  ```bash
  export CUDA_HOME=/usr/local/cuda-13.1  # or /usr/local/cuda
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  ```

### 2. Rust Toolchain (2024 Edition)
Ensure you have the latest stable Rust toolchain:
```bash
# Verify Rust 2024 Edition compatibility
rustc --version
cargo --version

# Default build: CPU-only, deterministic mock executor, no weights required
cargo test
cargo run --release

# Real Qwen2 inference (CPU or CUDA-backed candle device)
cargo run --release --features candle-model -- --model /path/to/Qwen2.5-0.5B-Instruct

# Custom CUDA decode kernels (requires nvcc / CUDA toolkit)
cargo run --release --features cuda -- --model /path/to/Qwen2.5-0.5B-Instruct
```
The model directory must contain `config.json`, `tokenizer.json`, and one
safetensors shard. Without `--model`, the engine starts in mock mode and labels
itself as such in the API and dashboard.

### 3. Telemetry Dashboard (Next.js 15)
```bash
cd dashboard

# Install dependencies with pnpm
pnpm install

# Start Next.js 15 dev server
pnpm dev

# Or compile optimized production build
pnpm build && pnpm start
```
Open **http://localhost:3000** in your browser.

---

## 🔌 API Reference & `curl` Examples

### OpenAI Chat Completions (`POST /v1/chat/completions`)

#### A. Streaming Request (`stream: true`):
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexuskv-engine-v1",
    "messages": [
      {"role": "system", "content": "You are NexusKV, an ultra-fast inference assistant."},
      {"role": "user", "content": "Explain PagedAttention memory pooling."}
    ],
    "max_tokens": 32,
    "temperature": 0.7,
    "stream": true
  }'
```

#### B. Non-Streaming Request (`stream: false`):
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexuskv-engine-v1",
    "messages": [
      {"role": "user", "content": "How does Radix prefix caching eliminate prefill duplicate work?"}
    ],
    "max_tokens": 48,
    "stream": false
  }'
```

---

### Engine Telemetry Stream (`POST /v1/generate/stream`)
Streams real-time physical VRAM block IDs, TTFT, ITL, and speculative acceptance metrics:
```bash
curl -N -X POST http://127.0.0.1:8080/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Continuous batching decode iteration",
    "max_tokens": 32
  }'
```

---

### Model Metadata (`GET /v1/models`)
```bash
curl -X GET http://127.0.0.1:8080/v1/models
```

---

### System Metrics (`GET /v1/metrics`)
```bash
curl -X GET http://127.0.0.1:8080/v1/metrics
```

---

### Health Probe (`GET /health`)
```bash
curl -X GET http://127.0.0.1:8080/health
```

---

## 📂 Directory Structure

```
nexuskv/
├── Cargo.toml                  # Rust 2024 features: default | candle-model | cuda
├── build.rs                    # Compiles the .cu kernel ONLY with feature=cuda
├── README.md
├── DECISIONS.md                # Engineering decisions and recorded deviations
├── LICENSE                     # MIT
│
├── csrc/
│   └── paged_attention.cu      # Online-softmax decode attention + KV kernels
│
├── src/
│   ├── lib.rs                  # Library target used by integration tests
│   ├── main.rs                 # CLI (--model) + Axum server entry point
│   ├── config.rs
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── block_manager.rs    # States, trie, CoW, LRU, BlockIo, invariants
│   │   ├── tests.rs            # B1/B2/B5 regressions, rollback/LRU/tier tests
│   │   └── fuzz.rs             # Deterministic LCG fuzz + proptest
│   ├── engine/
│   │   ├── mod.rs
│   │   ├── actor.rs            # Single-writer engine loop + Snapshot telemetry
│   │   ├── scheduler.rs        # Decodes-first continuous batching + preemption
│   │   └── speculative.rs      # Honest acceptance counters (phase 4 adds drafter)
│   ├── model/
│   │   ├── mod.rs              # Executor/Tokenizer traits + MockExecutor
│   │   └── qwen2.rs            # cfg(candle-model): Qwen2 forward, paged KV
│   ├── kernels/
│   │   ├── mod.rs
│   │   └── paged_attention.rs  # CPU softmax oracle + cfg(cuda) FFI
│   └── server/
│       ├── mod.rs
│       ├── routes.rs
│       └── sse_handler.rs      # OpenAI DTOs, SSE relay, AbortDropGuard
│
├── tests/
│   ├── api.rs                  # HTTP/SSE/schema/disconnect integration
│   ├── engine.rs               # actor lifecycle/abort/shared-prefix integration
│   ├── honesty.rs              # anti-fake build regressions
│   └── real_model.rs           # #[ignore] weights-required determinism test
│
├── dashboard/                  # Next.js 15 telemetry & VRAM visualizer
│   ├── package.json
│   └── src/{app,components}/
│
└── scripts/
    ├── sanitize.sh             # compute-sanitizer run for the GPU parity test
    └── build.sh                # CUDA/WSL build helper
```

---

## 📄 License
Licensed under the [MIT License](LICENSE).
<img width="1920" height="1080" alt="Screenshot 2026-08-14 132320" src="https://github.com/user-attachments/assets/36f2a1f2-02d5-49c9-bd20-65ad551dd931" />
<img width="1920" height="1080" alt="Screenshot 2026-08-14 173607" src="https://github.com/user-attachments/assets/3da908e9-9e82-4962-b6dc-01b0d1020f5e" />
