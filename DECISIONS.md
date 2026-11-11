# NexusKV Engineering Decisions

This file records decisions, deviations, and the evidence behind them. Phases
1–6 are implemented in the current tree.

## Phase 1 — Memory core

- **Block state machine is explicit.** `BlockState::{Free, Active, Cached}` is
  stored per block in addition to refcounts. This lets `assert_invariants()`
  check the state machine directly rather than inferring it from two booleans.
- **Trie nodes survive `free_sequence` in tier/drop modes.** A block at refcount
  zero with a trie node becomes `Cached` and is never simultaneously on the free
  stack. In drop mode a cached interior node is *not* placed in the LRU until it
  becomes a leaf; `evict_one` never selects an interior node in drop mode.
- **Allocation auto-evicts.** `allocate()` pops the free stack, then calls
  `evict_one()` under pressure. This keeps callers free of eviction policy and
  makes demand paging work through the same path.
- **Transactional prefix match.** Newly created trie nodes, demand-paged blocks,
  shared refcounts, and fresh remainder/CoW blocks are recorded in an
  acquisition log. Any later failure undoes the log in reverse order (including
  restoring a loaded node to `Swapped` and detaching newly created nodes).
  The page table is inserted last.
- **CoW is pointer-level in the allocator and content-level in the executor.**
  `append_tokens` and `fork` allocate a private partial tail before touching any
  shared refcount, record a pending `(src, dst)` copy, and emit a `Cow` event.
  The block manager does not understand KV tensor layout; the actor drains
  `take_pending_copies()` and asks the executor to copy the block. This keeps the
  allocator testable without a GPU and makes the copy explicit in the real model
  path.
- **A bare `allocate()` is a transient caller-owned reference.** Invariant 1
  (`ref == Σ page-table occurrences + pins`) is only asserted at operation
  boundaries. Tests that call `allocate()` directly deliberately do not assert
  invariants until the reference is attached or released.
- **Preemption progress rule.** The scheduler preempts only running sequences
  that have not produced output in their current residency. Without this, two
  equal-sized sequences that each need the whole pool preempt each other forever.
  Sequences that have made progress are allowed to finish; the waiting sequence
  is admitted afterwards. This is a deliberate interpretation of "preempt the
  newest running sequence" that preserves liveness.

## Phase 2 — Engine actor and serving

- **Single-writer actor, no `parking_lot`.** `parking_lot` was removed. The actor
  thread owns the allocator, scheduler, executor, and speculative counters. HTTP
  handlers only hold channels and an `Arc<Mutex<Snapshot>>` for telemetry.
- **Actor blocking.** The actor drains requests with `try_recv`; when idle it
  calls `blocking_recv` on the Tokio unbounded receiver from a plain
  `std::thread` (not inside a runtime context). No pacing sleeps exist in serving
  paths.
- **Mock tokenizer/model are explicit.** The mock tokenizer is byte-level; the
  mock model is a deterministic echo (`next = 3*last + 7*previous + 1 +
  prompt_offset` mod 256) with a fixed vocabulary and no EOS. `prompt_offset` is
  an FNV hash of the first prompt chunk, so identical prompts are deterministic
  across request IDs and different prompts diverge. The rule depends only on the
  last two tokens plus that offset, so an n-gram lookup drafter sees consistent
  continuations and greedy speculative/non-speculative decoding can be
  token-identical. History is recorded per sequence but is not used to alter the
  continuation rule.
- **Mock mode is labeled everywhere.** Startup logs a warning, `/v1/models` and
  `/v1/metrics` return `"executor": "mock"`, and the dashboard shows a MOCK
  banner. The dashboard's fake offline playback (`simulateLocalStream`) and its
  hardcoded 0.42/0.85 defaults were deleted.
- **Telemetry is measured.** `tokens_per_sec` is a 2-second sliding window of
  actually emitted tokens; TTFT p50 is computed from measured first-token times;
  acceptance is exactly 0.0 while no proposals exist (never a default estimate).
  Per-token TTFT/ITL/tok-s values in the SSE payload are measured in the HTTP
  relay, not fabricated by the model.
- **Abort semantics.** `AbortDropGuard` sends `Request::Abort`; the actor frees
  the sequence's blocks. Blocks with trie nodes become cached rather than
  destroyed. Tests and metrics therefore use `active_blocks` (and account for
  cached blocks) rather than assuming `allocated == 0` after an abort.
- **Scheduler budget.** Decodes consume one token each, running prefills are
  chunked with `min(remaining, budget)`, and waiting sequences are admitted
  while `running < max_num_seqs` and budget remains. This is the starvation fix:
  admission is no longer gated on an empty running pool.
- **Preemption batch hygiene.** If a sequence admitted earlier in the same
  scheduling pass is later preempted to make room, its decode/prefill entries are
  removed from the current batch before execution.

## Phase 3 — Real inference and kernels

- **Candle 0.9 pinned** as required. `candle-model` enables the model stack;
  `cuda` implies `candle-model`, enables candle's CUDA backend, and compiles
  `csrc/paged_attention.cu` with nvcc. Without the feature, `build.rs` returns
  immediately, so a machine with no CUDA toolkit still builds and tests.
- **Weights are loaded as F32.** Qwen checkpoints are commonly f16/bf16; loading
  as F32 costs memory but guarantees CPU correctness and matches the directive's
  f32 accumulation requirements. A dtype/quantization path is an optimization,
  not a correctness requirement, for this phase.
- **RoPE table size is capped by engine capacity.** The table is built for
  `min(config.max_position_embeddings, max_blocks_per_seq * block_size)`
  positions because a sequence cannot legally exceed that many tokens. This
  avoids allocating a 128k-position table for a small paged pool.
- **Unsupported Qwen variants are logged, not silently misinterpreted.**
  `rope_scaling` is not implemented; the loader warns and uses default RoPE.
  `use_sliding_window` is not implemented (the engine uses full causal
  attention). `hidden_act` values other than SiLU/Swish are warned about.
  These limitations are recorded rather than silently changing semantics.
- **Paged storage is honest gather attention.** Each layer owns flat
  `[total_slots, kv_heads, head_dim]` K/V tensors. New tokens are written with
  coalesced `slice_set` runs (one per contiguous slot run); attention gathers the
  sequence's slots with `index_select`. The CUDA kernel is the optimized decode
  path, not the reference implementation.
- **KV content copies are explicit.** The Qwen executor implements `copy_block`,
  copying all layers for allocator CoW events. Mock mode implements a no-op.
- **CPU reference attention is the test oracle.** `paged_attention.rs` always
  ships a real softmax reference (scale, max subtraction, weighted value sum).
  CUDA parity tests are `#[ignore]` GPU-only tests with hardware comments.
- **CUDA kernel errors are checked, not synchronized.** Every launch checks
  `cudaGetLastError()` and returns the error code; the hot path does not call
  `cudaDeviceSynchronize`. The kernel uses 128 threads/block, so it cannot exceed
  the 1024-thread limit. Both sm_86 SASS and compute_86 PTX are emitted.
- **The CUDA kernel is GQA-correct.** Query head `h` maps to KV head
  `h / (num_q_heads / num_kv_heads)`, matching candle's consecutive
  `repeat_kv` expansion. The host launcher validates the head divisibility.
- **Safe CUDA wrappers copy through device buffers.** The public host-slice
  functions allocate device memory, copy inputs in, launch, and copy results
  back. This makes the GPU parity tests valid without requiring callers to own
  raw pointers. The performance path is expected to keep persistent device
  buffers and call the `launch_*` functions directly; that integration is not
  exercised here.
- **The misleading standalone benchmark was deleted early.**
  `scripts/run_benchmarks.rs` measured the old fake verifier and printed
  "speculative tokens/sec" as if it were inference. It was removed rather than
  left as a fabricated result; phase 6 adds `criterion` bookkeeping benchmarks
  explicitly labeled "BOOKKEEPING THROUGHPUT — NOT INFERENCE".
- **Remaining phase-3 integration gap.** The Qwen2 executor uses candle's
  gather-attention kernels on the selected device (CPU here, CUDA when
  available). The standalone custom kernel is parity-tested but not yet wired to
  candle's persistent pool storage, because this machine has no GPU on which to
  validate raw device-pointer integration. The CPU oracle, candle gather path,
  and custom kernel all implement the same softmax semantics.
- **Hardware deviation (recorded).** The implementation machine has no `nvcc`
  and no NVIDIA GPU. Therefore:
  - the CUDA kernel could not be compiled or executed here;
  - GPU parity/sanitizer tests are present but `#[ignore]`d with explicit
    hardware comments;
  - `cargo clippy --all-targets --all-features` cannot be run because `cuda`
    requires nvcc. Default-feature clippy and `--features candle-model` clippy
    both pass.
- **Verification evidence on this machine.**
  - `cargo test` (default): all memory, scheduler, actor, API, honesty, and CPU
    attention tests passed; no GPU or weights.
  - `cargo test --features candle-model`: the same suite plus 25 library tests
    including attention-vs-naive (20 seeded trials, T=137, d=64, <1e-5), candle
    gather attention vs the CPU oracle (<1e-3), RoPE chunking (<1e-6), full-vs-
    chunked Qwen2 prefill logits (<1e-4 on a randomly initialized tiny model),
    GQA expansion, and tokenizer round-trip/chat-template tests.
  - `cargo fmt --all -- --check` passes.
  - `cargo clippy --all-targets -- -D warnings` and the same with
    `--features candle-model` pass.
  - `pnpm install`, `pnpm typecheck`, and `pnpm build` pass for the dashboard.
  - The CUDA feature was not built; this is the only gate that could not run.

## Phase 4 — Speculative decoding

- **The drafter is a prompt-lookup 3-gram index, not a model.** It observes only
  committed tokens (prompt + accepted output) and proposes the continuation
  after the most recent suffix match. Rejected drafts never enter the index.
- **Rejection sampling is distribution-exact for one-hot drafts.** Acceptance is
  `p_target(x)`; rejection samples the residual distribution with `x` removed and
  stops the chain. When all drafts are accepted, the bonus token is sampled from
  the final target row. The 400k-draw test measures max absolute deviation
  against the target and exercises the residual branch.
- **Greedy verification is separate.** Temperature below 1e-3 accepts a draft iff
  it equals the target argmax; rejection emits the argmax correction and stops.
  This keeps temperature-0 behavior deterministic and token-identical to plain
  decoding.
- **Actor KV accounting.** A speculative step appends `[last_token] + drafts`,
  runs one all-logits forward, verifies, rolls the page table back to
  `prompt + emitted`, then sets `computed = min(kv_valid, logical - 1)`. The
  correction/bonus token's KV is written by the next step. On append OOM or
  capacity exhaustion the actor retries with zero drafts before failing the
  sequence.
- **Adaptive K is UCB1 over `{0,2,4,8,16}`.** Untried arms are explored first,
  then `mean + sqrt(2 ln t / n)`. Reward is accepted tokens per verification
  millisecond. The telemetry fields are the bandit's current arm and the real
  proposed/accepted counters; acceptance stays exactly 0.0 when nothing was
  proposed.
- **Prefill drafters are reset on preemption.** `Batch::preempted` tells the
  actor which sequences were requeued so their output history does not leak into
  future proposals.

## Phase 5 — Sessions and time travel

- **Commits pin only full blocks.** Block-boundary granularity avoids pinning a
  partial tail that later appends would CoW. The complete token history is kept
  in the commit node for checkout.
- **Aligned checkouts re-run the final token.** The directive's
  `computed = pinned.len() * block_size` would leave no forward pass to obtain
  first-token logits when the commit is block-aligned. Checkout therefore uses
  `computed = min(full_tokens, tokens.len() - 1)`: unaligned commits re-prefill
  the partial tail, aligned commits re-prefill exactly the last token. This is a
  deliberate, recorded deviation that preserves semantics (no KV is reused for
  an uncomputed token).
- **Checkout attaches before appending the tail.** `attach_snapshot` bumps refs
  for the pinned prefix; tail blocks are then allocated transactionally. If the
  tail allocation fails, `free_sequence` releases the attached refs and returns
  an error. O(1) checkout means no prefix copies, not zero allocations for an
  unaligned tail.
- **Replay needs a cache bit.** A `Free` event that leaves a block trie-cached is
  indistinguishable from a free-stack return otherwise. `MemEvent.cached`
  carries that bit; fork events list only shared-prefix blocks so the replay
  reducer does not double-count the freshly allocated CoW tail.
- **Flight-recorder delivery is non-blocking.** The allocator emits through a
  bounded `SyncSender` with `try_send`; dropping an event under extreme pressure
  is preferable to stalling allocation. Keyframes capture full state at least
  every 256 ticks, so bounded loss cannot corrupt a replay that starts from a
  keyframe.
- **The dashboard reducer mirrors Rust.** `replay_state` and the TypeScript
  reducer use the same rules; cow/fault/rollback are intentional no-ops because
  alloc/free events already describe the reference changes. The determinism test
  reconstructs recorded operation-boundary states exactly (refcounts, free flag,
  and state string).

## Phase 6 — Tiering, benchmarks, docs, CI

- **Tier files are self-validating.** Every spill writes `NKV1`, the content
  hash, payload length, and an FNV-1a checksum, then atomically renames a temp
  file. Missing, truncated, wrong-hash, or checksum-mismatched files return loud
  errors; the allocator never silently exposes wrong KV.
- **The GPU bridge is an f32 byte container.** `QwenGpuBridge` shares the same
  `Arc<Mutex<QwenKvPools>>` tensors as the candle executor and serializes each
  layer's K/V block. This is DAW-level honest but memory-heavy; dtype-aware
  device-to-host copies are the documented optimization path. The mutex is only
  held during forward/copy/read/write and the engine actor is the sole writer.
- **Mock tiering is labeled.** With `--swap-dir` and the mock executor, the
  paging, host LRU, checksums, and NVMe spill are real but block contents are
  synthetic zero-filled bytes. Startup logs `MOCK TIERING ACTIVE`.
- **The 8-block/200-token e2e is aggregate demand paging, not one monolithic
  sequence.** Active page-table blocks cannot be evicted, so a single 200-token
  sequence cannot fit an 8-block pool. The test completes 200 generated tokens
  across five 40-token rounds with a shared 60-token prefix; a unique 100-token
  filler between rounds forces the prefix to swap out, and later rounds demand-
  page it back in. This preserves the intent (long-context generation through a
  tiny pool with real tier I/O) without pretending active KV can be evicted.
- **Bookkeeping benchmarks are labeled in output.** `benches/bookkeeping.rs`
  measures allocator/trie/rollback only and prints
  `BOOKKEEPING THROUGHPUT — NOT INFERENCE`. `scripts/bench_e2e.py` measures real
  SSE streams and refuses to invent comparison numbers; `--compare` prints the
  same-GPU/same-model protocol instead.
- **CI runs the default feature set.** CUDA requires nvcc and is a
  commented/self-hosted job. The dashboard job runs typecheck, reducer tests, and
  a production build.

## Addendum A — Qwen3 and RTX 3050 pin

- **Primary target is dense Qwen3-0.6B**, with Qwen2.5-0.5B-Instruct as the
  fallback. `qwen.rs` implements Qwen3 QK-norm (per-head RMSNorm on Q and K
  before RoPE), optional QKV biases (absent in Qwen3), tied embeddings, GQA,
  SwiGLU, and absolute-position RoPE.
- **Optional tensors are detected with `contains_tensor`.** A VarMap-backed
  builder silently creates zeros for missing names, so QK-norm, `lm_head`, and
  the prefixed/root embedding layout are checked explicitly. Without this, a
  Qwen2 checkpoint could have loaded zero QK-norm weights and zeroed attention.
- **Hybrid Qwen3.5/3.6/3.8 architectures are rejected.** Any `model_type` of
  `qwen3_5`, `qwen3_6`, `qwen3_8`, or `qwen3_next`, or an architecture name
  containing `Qwen3Next`/those versions, produces an explicit error. Gated
  DeltaNet layers are not implemented.
- **Defaults match the 4GB/8GB host:** 192 blocks, 16 tokens/block, 8 sequences,
  24 blocks/sequence, tier host budget 512 MiB. `.cargo/config.toml` sets
  `jobs = 2` for OOM-prone candle linking.
- **Shard mmap policy.** A single-shard checkpoint (the 0.6B case) uses one mmap.
  Split checkpoints are loaded in two passes (shape discovery, then copy) with
  exactly one shard mapped at a time. Peak RAM is higher for split checkpoints;
  this is a deliberate 8GB-host trade-off.
- **The f32 weight path is a memory trade-off.** Correctness and f32 accumulation
  were prioritized; the 344 MB f16 KV estimate in the README notes that the
  current pool is f32. f16 KV and weight quantization are optimization work, not
  silently claimed.
- **Hardware deviation remains.** This implementation machine has no NVIDIA
  GPU/nvcc, so the CUDA feature was not compiled or executed here. GPU parity and
  sanitizer tests are present, `#[ignore]`d with hardware comments, and wired into
  a commented self-hosted CI job.

## Verification evidence (Phases 4–6 on this machine)

- `cargo fmt --all -- --check` passes.
- `cargo clippy --all-targets -- -D warnings` passes (default features).
- `cargo test` passes with no GPU or weights: memory fuzz/proptest, scheduler,
  actor/API, sessions/replay, speculative statistics and greedy equivalence,
  tiering roundtrip/spill/failure-injection/paging, honesty, and bench smoke.
- `cargo test --features candle-model` compiles and passes the Qwen/QK-norm/
  bridge/tokenizer tests (no weights).
- `cargo check --all-targets --features candle-model` passes.
- `cargo bench --bench bookkeeping --no-run` compiles; the criterion suite is
  labeled bookkeeping-only.
- `python scripts/bench_e2e.py --clients 2 --max-tokens 8` produced sane JSON
  against the mock server (`tokens_per_sec > 0`, ordered percentiles), and
  `tests/bench_smoke.rs` exercises the same path when Python is available.
- `pnpm typecheck`, `pnpm test`, and `pnpm build` pass for the dashboard.
- Screenshots in `docs/` were captured from the running mock dashboard and show
  the MOCK banner and the time-travel scrubber.
- Coverage: `cargo llvm-cov` is not installed in this environment, so the ≥90%
  line target for memory/scheduler/speculative is not measured here. The tests
  exercise every public allocator operation, both eviction modes, transactional
  failures, all scheduler paths, speculative rollback accounting, session
  pin/checkout paths, and the actor/API/tier paths; the gap is tooling.



- The deterministic fuzz LCG is exactly the directive's constants
  (`6364136223846793005`, `1442695040888963407`, shift 33). Seeds 1337/42/7/99 ×
  8000 ops × tier on/off are supplemented by two 12,000-op pressure runs.
- Proptest is seeded from the generated test input (not wall-clock time), so
  failures are reproducible. `max_shrink_iters` is disabled to keep the oracle
  run bounded.
- No test was weakened or marked ignored except GPU-only tests and the
  weights-required real-model end-to-end test. Abort tests measure `active_blocks`
  because cached prefix blocks legitimately remain allocated after a sequence
  ends.
- The tokenizer test builds a minimal WordLevel `tokenizer.json` fixture with
  ChatML added tokens instead of committing a binary tokenizer. It verifies
  encode/decode round-trip, the ChatML rendering, and EOS discovery without
  network access.
- The real-model determinism test is `#[ignore]`d because this machine has no
  weights; it compiles under `candle-model` and is run with
  `NEXUSKV_TEST_MODEL=<dir> cargo test --features candle-model --test real_model
  -- --ignored --nocapture`.
- Coverage: `cargo llvm-cov` is not installed in this environment, so the ≥90%
  line target for memory/scheduler/speculative is not measured here. The test
  suite exercises every public allocator operation, both eviction modes,
  transactional failures, all scheduler paths, and the actor/API paths; the gap
  is tooling, not ignored tests.
