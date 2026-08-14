#!/usr/bin/env bash
# =============================================================================
# NexusKV Engine: Automated Multi-Stage Build & Compilation Script
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}  NexusKV: Production Build & Toolchain Compiler      ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Resolve CUDA Environment
CUDA_PATH="${CUDA_HOME:-${CUDA_PATH:-}}"
if [ -z "$CUDA_PATH" ]; then
    if [ -d "/usr/local/cuda-13.1" ]; then
        CUDA_PATH="/usr/local/cuda-13.1"
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    fi
fi

if [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH" ]; then
    echo -e "${GREEN}✓ CUDA Toolchain located at:${NC} $CUDA_PATH"
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
else
    echo -e "${YELLOW}! CUDA Toolchain not explicitly found in standard paths. Fallback/build.rs path will be used.${NC}"
fi

# 2. Check Rust Toolchain
echo -e "\n${CYAN}[1/3] Validating Rust 2024 Toolchain...${NC}"
rustc --version
cargo --version

# 3. Compile Rust Engine & CUDA Kernel via Cargo
echo -e "\n${CYAN}[2/3] Compiling NexusKV Engine Core (Cargo)...${NC}"
cargo build --release
echo -e "${GREEN}✓ NexusKV Engine Binary built successfully at target/release/nexuskv${NC}"

# 4. Build Next.js 15 Telemetry Dashboard
echo -e "\n${CYAN}[3/3] Building Next.js 15 Telemetry Dashboard...${NC}"
cd dashboard
if command -v pnpm &> /dev/null; then
    pnpm install --frozen-lockfile || pnpm install
    pnpm build
elif command -v npm &> /dev/null; then
    npm install
    npm run build
else
    echo -e "${RED}Error: Neither pnpm nor npm found on system path.${NC}"
    exit 1
fi
cd ..

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}  ✓ NexusKV Build Pipeline Completed Successfully!    ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "To start the engine:    ${CYAN}./target/release/nexuskv${NC}"
echo -e "To start the dashboard: ${CYAN}cd dashboard && pnpm start${NC}"
