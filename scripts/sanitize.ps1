# scripts/sanitize.ps1
# Builds the CUDA test binary and runs the attention parity test under
# compute-sanitizer. Native Windows (compute-sanitizer.exe ships with the CUDA
# toolkit under %CUDA_PATH%\bin).
#
#   powershell -ExecutionPolicy Bypass -File scripts/sanitize.ps1
param(
    [string[]]$TestName = @(
        'kernels::paged_attention::tests::cuda_attention_parity_with_cpu_reference',
        'kernels::paged_attention::tests::cuda_f16_gqa_attention_parity_with_cpu_reference'
    )
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
    $sanitizer = Join-Path $envInfo.CudaHome 'bin\compute-sanitizer.exe'
    if (-not (Test-Path $sanitizer)) {
        Write-Output ("ERROR: compute-sanitizer.exe not found at " + $sanitizer)
        Write-Output 'Install the CUDA 12.6 sanitizer API component or use the full toolkit installer.'
        exit 1
    }

    Write-Output '--- building CUDA test binary ---'
    & cargo test --features cuda --lib --no-run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $deps = Join-Path $repo 'target\debug\deps'
    $binary = Get-ChildItem $deps -Filter 'nexuskv-*.exe' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $binary) {
        Write-Output 'ERROR: could not locate the compiled test binary under target\debug\deps.'
        exit 1
    }

    $debuggerInterface = Join-Path $envInfo.CudaHome 'compute-sanitizer\EnableDebuggerInterface.bat'
    foreach ($name in $TestName) {
        Write-Output ("--- compute-sanitizer: " + $binary.Name + " :: " + $name + " ---")
        $output = & $sanitizer --tool memcheck --leak-check full $binary.FullName $name --ignored --nocapture 2>&1
        $code = $LASTEXITCODE
        foreach ($line in $output) { Write-Output $line }
        $text = ($output | Out-String)
        if ($text -match 'Failed to initialize WDDM debugger interface' -or $text -match 'Device not supported') {
            Write-Output 'SANITIZER BLOCKED: compute-sanitizer could not attach on WDDM.'
            Write-Output 'The kernel test ran uninstrumented; this is NOT a clean sanitizer result.'
            Write-Output ("Fix once in an Administrator PowerShell: & `"" + $debuggerInterface + "`"")
            Write-Output 'Then reboot if prompted and re-run scripts/sanitize.ps1.'
            exit 2
        }
        if ($code -ne 0) {
            Write-Output ("compute-sanitizer reported errors for " + $name)
            exit $code
        }
        if ($text -match 'ERROR SUMMARY:\s*[1-9]') {
            Write-Output ("compute-sanitizer ERROR SUMMARY is non-zero for " + $name)
            exit 1
        }
    }
    Write-Output 'compute-sanitizer: clean.'
} finally {
    Pop-Location
}
