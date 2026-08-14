# ⚡ NexusKV Engine

<div align="center">

![Rust](https://img.shields.io/badge/Rust-2024_Edition-DEA584?style=for-the-badge&logo=rust&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.1_Ampere%2FHopper-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15.0_App_Router-000000?style=for-the-badge&logo=next.js&logoColor=white)
![Axum](https://img.shields.io/badge/Axum-0.7_Async_Web-3B82F6?style=for-the-badge&logo=tokio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)

**High-Throughput Paged KV-Cache, Radix Prefix Caching & Speculative Inference Engine in Pure Rust 2024 and Custom CUDA 13.1 Kernels.**

</div>

---

## 📖 Table of Contents
- [Architectural Overview](#-architectural-overview)
  - [1. PagedAttention Memory Virtualization](#1-pagedattention-memory-virtualization)
  - [2. Radix Tree Prefix Cache](#2-radix-tree-prefix-cache)
  - [3. Copy-on-Write (CoW) Speculative Tree Decodes](#3-copy-on-write-cow-speculative-tree-decodes)
  - [4. Axum Continuous Batching & SSE Pipeline](#4-axum-continuous-batching--sse-pipeline)
- [Core Features Matrix](#-core-features-matrix)
- [System Architecture & Benchmarks](#-system-architecture--benchmarks)
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

NexusKV addresses memory fragmentation and lock contention in Large Language Model (LLM) serving by combining virtual memory paging, Trie-based prefix caching, and fine-grained concurrent scheduling.

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

### 3. Copy-on-Write (CoW) Speculative Tree Decodes
Speculative decoding generates multiple candidate draft tokens in parallel. When a draft branch diverges from target model logits, NexusKV triggers an $O(1)$ page rollback by popping excess block pointers and decrementing physical `ref_count`, completely avoiding expensive $O(N)$ tensor copies.

```
Target Verification:
Draft Candidates: [ Token 100, Token 101, Token 102 (Divergence!), Token 103 ]
Target Logits   : [ Token 100, Token 101, Token 999 (Accepted Correction)   ]

Rollback Action:
1. Retain 3 Tokens -> [ 100, 101, 999 ]
2. Rollback Page Table -> Drop Block #52 back to Free Stack in O(1)
3. Zero Memory Copies -> 50.88 Million Speculative Tokens/sec
```

---

### 4. Axum Continuous Batching & SSE Pipeline

```
 HTTP Requests (OpenAI API / SSE)
               │
               ▼
┌──────────────────────────────┐
│ Axum 0.7 Web Routing Layer   │
└──────────────┬───────────────┘
               │ (Client Drop Detection via RAII AbortDropGuard)
               ▼
┌────────────────────────────────────────────────────────────────────┐
│ ContinuousScheduler (parking_lot::RwLock)                          │
│                                                                    │
│  Waiting Queue:  [ Seq 3 ] ──> Admitted via match_or_allocate_prefix│
│  Running Pool :  [ Seq 1 (Decode) ] [ Seq 2 (Decode) ]             │
│  Finished Pool:  [ Seq 0 ]                                         │
└──────────────┬─────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐       ┌─────────────────────────────┐
│ PagedAttention CUDA Kernel   │ <===> │ BlockSpaceManager           │
│ (csrc/paged_attention.cu)    │       │ Physical Allocator + Radix  │
└──────────────┬───────────────┘       └─────────────────────────────┘
               │
               ▼
 Server-Sent Events (SSE) Telemetry Stream -> Next.js 15 Canvas Visualizer
```

---

## 🚀 Core Features Matrix

| Feature | Standard Contiguous KV Cache | NexusKV Engine |
| :--- | :--- | :--- |
| **Memory Allocation** | Static maximum pre-allocation | **Dynamic Physical Paging (16 tokens/block)** |
| **VRAM Fragmentation** | 60% – 80% wasted memory | **< 3% (Zero external fragmentation)** |
| **Prefix Caching** | Duplicate prompt KV computation | **Radix Tree Multi-Tenant Sharing ($O(1)$ CoW)** |
| **Speculative Rollbacks** | Expensive $O(N)$ GPU `cudaMemcpy` | **Instant $O(1)$ Page Pointer Reclamation** |
| **Concurrency Locking** | `std::sync::Mutex` decode stalls | **`parking_lot::RwLock` Lock-Free Read Concurrency** |
| **Client Disconnections** | Memory leaked until sequence end | **RAII `AbortDropGuard` Instant Block Freeing** |
| **API Compatibility** | Custom proprietary endpoints | **OpenAI Standard (`/v1/chat/completions`)** |
| **VRAM Observability** | Blind black-box memory | **Interactive 512-Block Next.js 15 Heatmap** |

---

## 📊 System Architecture & Benchmarks

Benchmarked on AMD Ryzen / NVIDIA RTX 3050 (Ampere sm_86) & Host CPU:

```
================================================================================
  NexusKV Engine: Physical Paging & Speculative Verification Benchmark Suite   
================================================================================

--- 1. Physical Block Allocator Throughput ---
  • Allocated & Freed : 65,536 physical blocks (16 tokens/block)
  • Allocation Time   : 1.454 ms (45.06 Million allocs/sec)
  • Deallocation Time : 1.039 ms (63.09 Million frees/sec)
  • Aggregate Rate    : 52.57 Million O(1) Ops/sec

--- 2. Radix Tree Prefix Cache Lookups ---
  • Total Invocations : 100,000 prompt block queries
  • Trie Hits / Misses: 20.0% hits / 80.0% misses
  • Average Latency   : 844.22 ns per lookup
  • Lookup Throughput : 1.18 Million Lookups/sec

--- 3. Speculative Verification & O(1) Page Rollback ---
  • Verified Iterations : 500,000 speculative batches (K = 5)
  • Tokens Verified     : 2,500,000 candidate tokens
  • Verification Latency: 98.26 ns / speculative branch
  • Verification Rate   : 50.88 Million Speculative Tokens/sec
```

---

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

# Run test suite
cargo test

# Build and run the engine server
cargo run --release
```

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
nexuskv-rust/
├── Cargo.toml                  # Rust 2024 dependencies & crate metadata
├── build.rs                    # Auto-compiles CUDA .cu kernels via cc & nvcc
├── README.md                   # System architecture & benchmark documentation
├── .gitignore                  # Production ignore rules (Rust, CUDA, Next.js)
│
├── csrc/                       # Custom CUDA 13.1 Kernels
│   └── paged_attention.cu      # PagedAttention block-gather CUDA kernel
│
├── src/                        # Pure Rust Engine Core (Edition 2024)
│   ├── main.rs                 # Axum web server & CLI entry point
│   ├── config.rs               # Engine runtime configuration & capacity limits
│   ├── memory/
│   │   ├── mod.rs
│   │   └── block_manager.rs    # Thread-Safe Physical Block Allocator & Radix Trie
│   ├── engine/
│   │   ├── mod.rs
│   │   ├── scheduler.rs        # Continuous batching iteration scheduler (RwLock)
│   │   └── speculative.rs      # Speculative tree verification & O(1) page rollbacks
│   ├── kernels/
│   │   ├── mod.rs
│   │   └── paged_attention.rs  # Safe FFI wrapper (Rust 2024 `unsafe extern` syntax)
│   └── server/
│       ├── mod.rs
│       └── sse_handler.rs      # OpenAI chat completions & Axum SSE drop guards
│
├── dashboard/                  # Next.js 15 Telemetry & VRAM Visualizer
│   ├── package.json
│   ├── next.config.mjs
│   ├── tailwind.config.js
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   └── page.tsx        # Main VRAM Heatmap & Metrics Dashboard
│       └── components/
│           ├── VramBlockCanvas.tsx # Interactive HTML5 Canvas with hover inspection
│           └── MetricsPanel.tsx    # Live TTFT, ITL, Tokens/sec & Alpha rate HUD
│
└── scripts/
    ├── build.sh                # Cargo build & CUDA compilation script
    └── run_benchmarks.rs       # High-throughput benchmark suite
```

---

## 📄 License
Licensed under the [MIT License](LICENSE).
