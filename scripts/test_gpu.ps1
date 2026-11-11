# scripts/test_gpu.ps1
# Runs the GPU-only (ignored) parity tests and the compute-sanitizer pass on
# this machine. Native Windows only; no WSL.
#
#   powershell -ExecutionPolicy Bypass -File scripts/test_gpu.ps1
param(
    [switch]$SkipSanitizer
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows_env.ps1')
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    $envInfo = Initialize-NexusKvEnvironment
    if (-not $envInfo.CudaHome) {
        Write-Output 'ERROR: CUDA toolkit not found. Run scripts/check_env.ps1 for the exact fix.'
        exit 1
    }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Output 'ERROR: nvidia-smi not found; a working NVIDIA driver is required.'
        exit 1
    }

    $modelDir = Join-Path $repo 'models\qwen3-0.6b'
    if (Test-Path (Join-Path $modelDir 'config.json')) {
        $env:NEXUSKV_TEST_MODEL = $modelDir
        Write-Output ("Real-weights parity enabled: " + $modelDir)
    } else {
        Write-Output 'Model weights not present; real-weights parity tests will skip loudly.'
    }

    Write-Output '--- GPU tests (--ignored --test-threads=1) ---'
    & cargo test --features cuda -- --ignored --test-threads=1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not $SkipSanitizer) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'sanitize.ps1')
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    Write-Output 'GPU test suite complete.'
} finally {
    Pop-Location
}
