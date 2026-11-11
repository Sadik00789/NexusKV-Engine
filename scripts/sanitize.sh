#!/usr/bin/env bash
# Run the CUDA attention parity test under compute-sanitizer.
#
# Hardware requirement: an NVIDIA GPU with a working driver, the CUDA toolkit
# (nvcc), and compute-sanitizer on PATH. This script is intentionally not part
# of the default `cargo test` gate because the default build has no GPU
# dependency.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found; install the CUDA toolkit or set PATH" >&2
  exit 1
fi
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo "error: compute-sanitizer not found; it ships with the CUDA toolkit" >&2
  exit 1
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/sanitize}"
TEST_NAME="kernels::paged_attention::tests::cuda_attention_parity_with_cpu_reference"

cargo test --features cuda --lib --no-run --quiet

BIN="$(find "$CARGO_TARGET_DIR/debug/deps" -maxdepth 1 -type f -perm -111 -name 'nexuskv-*' | head -n 1)"
if [[ -z "${BIN}" ]]; then
  echo "error: could not locate the compiled test binary" >&2
  exit 1
fi

exec compute-sanitizer --tool memcheck --leak-check full "$BIN" \
  "$TEST_NAME" --ignored --nocapture
