# NexusKV Engine — Architecture & Internals (`v1.0.0`)

NexusKV Engine is a Rust continuous-batching LLM inference server and telemetry workspace built around **Paged KV-Cache memory**, a **Radix Prefix Trie**, **3-tier VRAM → Host RAM → NVMe paging**, **adaptive speculative decoding**, **Git-for-KV branching sessions**, and a **keyframe + event-sourced Flight Recorder**.

> **Scope note:** This document describes verified symbols, modules, and HTTP endpoints in `v1.0.0`.

---

## 1. End-to-End Request Lifecycle

```text
HTTP / SSE Clients & Next.js Dashboard (localhost:3000)
   │  POST /v1/chat/completions  |  POST /v1/generate/stream
   │  POST /v1/sessions/commit   |  POST /v1/sessions/checkout | forget
   │  GET  /v1/metrics           |  GET  /v1/timetravel?tick=N
   ▼
Axum Router & SSE Layer (src/server/routes.rs, src/server/sse_handler.rs)
   │  mpsc::Sender<Request> + watch::Receiver<Snapshot> + AbortDropGuard
   ▼
EngineActor (src/engine/actor.rs) — single owner of Scheduler, BlockSpaceManager, SessionStore, FlightRecorder
   ├─► Scheduler (src/engine/scheduler.rs)
   │     • Admits waiting & checkout sequences within max_num_seqs / block capacity
   │     • Builds chunked PrefillChunk work + fused multi-sequence Decode batches
   │     • Preempts & swaps/frees victim blocks if VRAM allocation hits OOM, rolling back unconsumed skip tokens
   ├─► BlockSpaceManager + Radix Trie + TierStore (src/memory/block_manager.rs, src/memory/tier.rs)
   │     • Allocates 16-token physical KV blocks, matches aligned prefix blocks via Radix Trie
   │     • Evicts ref_count=0 tail blocks LRU -> Host RAM LRU -> NVMe .bin files
   │     • Preserves host-cached copies across transactional OOM rollback on mid-prefill swap-in
   ├─► LookupDrafter + AdaptiveK (src/engine/speculative.rs)
   │     • Observes prompt tail + every emitted token (including first generated token)
   │     • 3-gram drafter; proposes k in {0, 2, 4, 8, 16} via UCB1; O(1) tail rollback
   ├─► SessionStore + FlightRecorder (src/engine/session.rs, src/memory/events.rs)
   │     • Commits active OR recently completed sequences (64-entry completed_sequences history)
   │     • Publishes synchronous Snapshots on Commit/Forget and appends delta MemEvents + Keyframes
   ▼
Executor Backend (src/model/mod.rs, src/model/qwen.rs, src/kernels/paged_attention.rs, csrc/paged_attention.cu)
   • Qwen2Executor + Qwen2Model (Candle CUDA f16/f32 or CPU fallback; MockExecutor when --model omitted)
   • Fused qkv_proj & gate_up_proj weights + hoisted per-step RoPE cos/sin & causal masks
   • Custom CUDA f16/f32 paged decode attention kernel (NEXUSKV_CUDA_ATTENTION=1)
```

---

## 2. Memory Subsystem (`src/memory/`)

### 2.1 Physical Paged KV Pool & `BlockSpaceManager`
Implemented in [block_manager.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/memory/block_manager.rs).

- **Logical-to-physical paging:** Each active sequence maintains a `Vec<BlockId>` page table (`16` tokens/block by default, configurable with `--block-size`). Blocks are mapped independently per sequence rather than contiguously in VRAM.
- **Radix Prefix Trie & Sharing:** Committed/freed full blocks whose token hashes match previously populated KV state are indexed in the prefix trie (`match_or_allocate_prefix`). Cache hits pin shared blocks, increment `ref_count`, and advance `prefix_skip_tokens` so those prefix tokens are not re-forwarded through attention.
- **Copy-on-Write / Partial Tail Forking:** Full shared blocks stay read-only (`ref_count > 1`). Appending tokens into a sequence allocates a fresh mutable block (or duplicates only when mutating a shared tail), avoiding whole-context copies.
- **O(1) Rollback:** `BlockSpaceManager::rollback(seq, target_tokens)` drops speculative tail blocks from `seq` back to `target_tokens` in microseconds without copying KV tensors.

### 2.2 3-Tier KV Offloading (`TierStore` & `QwenGpuBridge`)
Implemented in [tier.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/memory/tier.rs) and [qwen.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/model/qwen.rs).

1. **Tier 1 (GPU VRAM):** Physical `[num_blocks, num_kv_heads, block_size, head_dim]` tensors held inside `Qwen2Model`. `QwenGpuBridge` implements `GpuBridge` for real CUDA/CPU tensors, while `MemoryBridge` backs non-model unit/integration tests.
2. **Tier 2 (Host RAM LRU):** Evicted `ref_count == 0` blocks are downloaded via `save_block` into an in-memory host cache bounded by `--host-budget` (default `536870912` bytes / `512 MiB`).
3. **Tier 3 (NVMe Spill):** Enabled when `--swap-dir <DIR>` is supplied. Blocks overflowing `--host-budget` spill to `<DIR>/block_<id>.bin` using atomic write-and-rename and demand-page back on cache re-hit (`page_faults`, `swapped_in`, `disk_loads`).
4. **Transactional OOM Rollback Safety:** When loading a swapped-out prefix block during allocation, `TierStore::load` retains and touches the host LRU copy instead of removing it eagerly. If a subsequent block allocation in the same `match_or_allocate_prefix` / `checkout` / `swap_in` call hits VRAM OOM, rolling back `Acquisition::Loaded` blocks to `Res::Swapped` preserves every byte on host RAM/disk with zero data loss.

### 2.3 Event-Sourced `FlightRecorder`
Implemented in [events.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/memory/events.rs).

- Every allocation, append, eviction, swap, rollback, and free emits a `MemEvent` into a `50,000`-event bounded ring buffer (`FlightRecorder`), with periodic full-pool `Keyframe` snapshots every `256` ticks.
- `GET /v1/timetravel?tick=N` finds the nearest `Keyframe <= tick` and deterministically replays subsequent `MemEvent` entries via `replay_state` so the dashboard reconstructs exact per-page state without storing per-tick snapshots or fabricating telemetry.

---

## 3. Scheduling, Speculative Decoding & Sessions (`src/engine/`)

### 3.1 `EngineActor` & Continuous-Batching `Scheduler`
Implemented in [actor.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/engine/actor.rs) and [scheduler.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/engine/scheduler.rs).

- **`EngineActor`** runs in a dedicated OS thread with lock-free ownership of `Scheduler`, `BlockSpaceManager`, `SessionStore`, `FlightRecorder`, `LookupDrafter`, and `AdaptiveK`.
- **Chunked Prefill + Continuous Decode:** `Scheduler::schedule` limits prefill chunks by `max_num_batched_tokens` while merging all ready sequences into a single `Decode` batch.
- **Preemption Rollback:** If VRAM allocation fails while extending a running sequence, `Scheduler` preempts a lower-priority running sequence and rolls back any `unconsumed_skip_tokens` state cleanly so resumed execution preserves exact token positions.
- **Synchronous Telemetry & Sampling Sanitization:** HTTP requests clamp invalid/NaN/out-of-range temperature, `top_p`, and `max_tokens`. Session `Commit` and `Forget` requests trigger an immediate synchronous `publish()` of `Snapshot` so polling clients and SSE consumers observe updates with zero lag.

### 3.2 Adaptive Speculative Decoding
Implemented in [speculative.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/engine/speculative.rs).

- **`LookupDrafter`** builds an `N`-gram suffix index (`DRAFT_NGRAM = 3`) over active context tokens, observing the prompt and **every emitted token** (including the first sampled token produced right after prefill) so early self-repetition or quoted echoes can be drafted immediately.
- **`AdaptiveK`** selects draft length `k` from `DRAFT_ARMS = [0, 2, 4, 8, 16]` using UCB1 exploration/exploitation weighted by recent acceptance reward.
- Drafted tokens are verified in a single multi-token forward pass against target logits; mismatched tails are discarded and their KV blocks reclaimed in `O(1)` via `BlockSpaceManager::rollback`.

### 3.3 Git-for-KV Branching Sessions
Implemented in [session.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/engine/session.rs) and wired into [actor.rs](file:///c:/Users/sadik/Downloads/NexusKV-Engine/src/engine/actor.rs) + [dashboard/src/app/page.tsx](file:///c:/Users/sadik/Downloads/NexusKV-Engine/dashboard/src/app/page.tsx).

- **`SessionStore`** stores immutable `CommitNode` snapshots holding committed token sequences and pinned KV block references (`parent_id` linking child checkpoints for DAG branching).
- **Post-Completion Commits:** `EngineActor` keeps a `64`-entry `completed_sequences` ring history so `POST /v1/sessions/commit` succeeds both mid-stream and after generation finishes (such as when a user clicks **Commit** in the dashboard after reading a reply).
- **`POST /v1/sessions/checkout`** restores a checkpoint's prefix blocks in `O(1)` (with partial-tail prompt length alignment via `set_prompt_len`) and resumes generation independently on any named branch.

---

## 4. Model Execution & Custom CUDA Kernels (`src/model/`, `src/kernels/`, `csrc/`)

### 4.1 Qwen3 / Qwen2 Execution (`src/model/qwen.rs`)
- Loaded via `--model <DIR>` (`candle-model` or `cuda` features). Supports Candle CUDA (`f16` default, `NEXUSKV_DTYPE=f32` opt-out) and CPU execution.
- **Weight Fusion:** At load time (`load_fused_linear`), per-layer `q_proj`/`k_proj`/`v_proj` weights are concatenated into a single `qkv_proj` linear projection, and `gate_proj`/`up_proj` are concatenated into `gate_up_proj`, reducing linear kernel launches across all `28` Qwen3-0.6B transformer layers. Tied/untied `lm_head` shares one unified `Linear` path.
- **Loop-Hoisted Tensors:** Per-step RoPE `cos`/`sin` slices, KV slot indices (`slot_tensor`), and `causal_mask` tensors are built/uploaded **once per step** outside the 28-layer loop (and all-zero `causal_mask` construction is skipped entirely when `q_len == 1`).

### 4.2 Custom CUDA Paged Decode Attention (`csrc/paged_attention.cu`, `src/kernels/paged_attention.rs`)
- Built automatically by `build.rs` when compiling with `--features cuda`.
- Enabled at runtime by setting `NEXUSKV_CUDA_ATTENTION=1` (`forward_with_cuda_attention`).
- `CudaDecodeContext::new` receives exact written KV lengths (`active_lens = start + len`) rather than block-table capacities that include an unwritten pre-allocated slot during batched decode, ensuring exact numerical parity with the Candle gather reference.
- Exposes `launch_paged_attention` (f32) and `launch_paged_attention_f16` (f16, stream-aware), tested against the CPU scalar reference in `src/kernels/paged_attention.rs` and verified end-to-end in `tests/real_model.rs`.

---

## 5. HTTP & SSE API Surface (`src/server/routes.rs`, `src/server/sse_handler.rs`)

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Engine readiness & model/mock health check |
| `GET` | `/v1/models` | OpenAI-compatible model card list (`ModelListResponse`) |
| `GET` | `/v1/metrics` | Live `Snapshot` telemetry (KV utilization, prefix reuse, tier bytes, batching, adaptive K, executor & attention backend) |
| `POST` | `/v1/generate/stream` | Streaming generation SSE with per-token text + live `TelemetryPayload` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible Chat Completions (streaming SSE or JSON response) |
| `POST` | `/v1/sessions/commit` | Pin current or completed sequence KV blocks into a named branch checkpoint (`CommitRequest`) |
| `POST` | `/v1/sessions/checkout` | Fork & resume generation from a `CommitNode` (`CheckoutRequest`) |
| `POST` | `/v1/sessions/forget` | Unpin a checkpoint and release unreferenced KV blocks (`ForgetRequest`) |
| `GET` | `/v1/sessions/:id/tree` | Return checkpoint DAG (`SessionTree`) for session `:id` |
| `GET` | `/v1/timetravel?tick=N` | Deterministic `FlightRecorder` replay of KV pool state at tick `N` |

---

## 6. Test & Verification Architecture (`tests/`, `scripts/`)

- **Unit + Integration Suite (`cargo test`):** Exercises allocator/trie/LRU invariants + proptest fuzzing (`src/memory/tests.rs`, `src/memory/fuzz.rs`), CPU vs CUDA kernel parity (`src/kernels/paged_attention.rs`), speculative rollback & first-token draft observation (`tests/speculative.rs`), tiering OOM rollback & NVMe spill (`tests/tiering.rs`), session commit/checkout/forget (`tests/sessions.rs`), engine end-to-end streaming (`tests/engine.rs`), and automated anti-fabrication checks (`tests/honesty.rs`).
- **GPU Parity & Sanitizer Suite (`scripts/test_gpu.ps1`, `scripts/sanitize.ps1`):** Runs `cargo test --features cuda --lib --test real_model -- --ignored --test-threads=1` against local `models/qwen3-0.6b` weights plus `compute-sanitizer --tool memcheck` & `racecheck`.
- **Dashboard Suite (`dashboard/`):** `pnpm typecheck`, `pnpm test` (`vitest` replay & telemetry state tests), and `pnpm build` (Next.js 15).
