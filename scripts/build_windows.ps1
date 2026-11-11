# scripts/build_windows.ps1
# One-command native Windows build: imports the VS x64 environment, resolves
# CUDA, runs the preflight, builds the engine with the custom kernels, and
# builds the dashboard. Run from a plain PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
param(
    [switch]$SkipDashboard,
    [switch]$NoCuda
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows_env.ps1')

$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    Write-Output '=== NexusKV Windows build ==='
    $envInfo = Initialize-NexusKvEnvironment
    Write-Output ("Visual Studio env: " + $envInfo.VisualStudio)
    Write-Output ("CUDA toolkit:      " + $envInfo.CudaHome)
    if (-not $NoCuda -and -not $envInfo.CudaHome) {
        Write-Output 'ERROR: CUDA toolkit not found. Run scripts/check_env.ps1 for the exact fix.'
        Write-Output 'No-admin install: powershell -ExecutionPolicy Bypass -File scripts/install_cuda_user.ps1'
        Write-Output 'It installs CUDA 13.4.3 under %USERPROFILE%\cuda-13.4 (cudarc supports 12.6-13.4).'
        exit 1
    }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'check_env.ps1')
    if ($LASTEXITCODE -ne 0) {
        Write-Output 'Preflight failed; build aborted.'
        exit $LASTEXITCODE
    }

    if ($NoCuda) {
        Write-Output '--- cargo build --release (default features) ---'
        & cargo build --release
    } else {
        Write-Output '--- cargo build --release --features cuda ---'
        & cargo build --release --features cuda
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not $SkipDashboard) {
        Write-Output '--- dashboard build ---'
        Push-Location (Join-Path $repo 'dashboard')
        try {
            if (-not (Test-Path 'node_modules')) {
                & pnpm install
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            & pnpm build
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        } finally {
            Pop-Location
        }
    }

    Write-Output ''
    Write-Output 'Build complete.'
    Write-Output ("Engine: " + (Join-Path $repo 'target\release\nexuskv.exe'))
    Write-Output 'Run:    powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1'
} finally {
    Pop-Location
}
