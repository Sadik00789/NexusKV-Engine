# scripts/run_windows.ps1
# Start the engine with real Qwen3-0.6B weights on the local GPU. Run from a
# plain PowerShell after scripts/build_windows.ps1:
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1
param(
    [string]$Model = 'models\qwen3-0.6b',
    [string]$SwapDir = '',
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows_env.ps1')
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    Initialize-NexusKvEnvironment | Out-Null
    $env:PORT = $Port.ToString()

    $exe = Join-Path $repo 'target\release\nexuskv.exe'
    if (-not (Test-Path $exe)) {
        Write-Output 'ERROR: target\release\nexuskv.exe not found. Run scripts/build_windows.ps1 first.'
        exit 1
    }

    $modelPath = Join-Path $repo $Model
    $config = Join-Path $modelPath 'config.json'
    if (-not (Test-Path $config)) {
        Write-Output ("ERROR: model directory " + $modelPath + " has no config.json.")
        Write-Output 'Manual step: hf download Qwen/Qwen3-0.6B --local-dir models\qwen3-0.6b'
        exit 1
    }

    $arguments = @('--model', $modelPath)
    if ($SwapDir) { $arguments += @('--swap-dir', (Join-Path $repo $SwapDir)) }

    Write-Output ("Engine:    http://127.0.0.1:" + $Port)
    Write-Output ("Dashboard: cd dashboard; pnpm dev  ->  http://localhost:3000")
    Write-Output ("Model:     " + $modelPath)
    Write-Output 'Press Ctrl+C to stop.'
    & $exe @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
