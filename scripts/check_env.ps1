# scripts/check_env.ps1
# Native-Windows preflight for the NexusKV CUDA build. Prints a PASS/WARN/FAIL
# table and exits non-zero on hard failures. It performs no installs.
#
#   powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1 -SelfTest
#   powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1 -NoGpuRequired
param(
    [switch]$SelfTest,
    [switch]$NoGpuRequired
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'windows_env.ps1')

function Test-RustHost {
    param([string]$HostString)
    return ($HostString -match 'windows-msvc')
}

function Find-ClExe {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) { return $cl.Source }
    $vswhere = Get-VsWherePath
    if ($vswhere) {
        $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($install) {
            $tools = Join-Path $install 'VC\Tools\MSVC'
            if (Test-Path $tools) {
                $version = Get-ChildItem $tools -Directory | Sort-Object Name -Descending | Select-Object -First 1
                if ($version) {
                    $candidate = Join-Path $version.FullName 'bin\Hostx64\x64\cl.exe'
                    if (Test-Path $candidate) { return $candidate }
                }
            }
        }
    }
    return $null
}

if ($SelfTest) {
    $failures = 0
    function Assert-True {
        param([bool]$Condition, [string]$Name)
        if ($Condition) {
            Write-Output ("PASS self-test: " + $Name)
        } else {
            Write-Output ("FAIL self-test: " + $Name)
            $script:failures++
        }
    }
    $fake = Join-Path $env:TEMP ("nexuskv-checkenv-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path (Join-Path $fake 'bin'), (Join-Path $fake 'lib\x64') | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $fake 'bin\nvcc.exe') | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $fake 'lib\x64\cudart.lib') | Out-Null
    Assert-True (Test-CudaLayout -Path $fake) 'detects a complete CUDA layout'
    Assert-True (-not (Test-CudaLayout -Path (Join-Path $fake 'missing'))) 'reports a missing CUDA layout'
    Assert-True (-not (Test-MsvcLayout $null)) 'reports a missing MSVC installation'
    Assert-True (Test-MsvcLayout (Join-Path $fake 'bin\nvcc.exe')) 'accepts an existing vcvars path'
    Assert-True (Test-RustHost 'x86_64-pc-windows-msvc') 'accepts the Windows MSVC host'
    Assert-True (-not (Test-RustHost 'x86_64-unknown-linux-gnu')) 'rejects a Linux or GNU host'
    Remove-Item -Recurse -Force $fake -ErrorAction SilentlyContinue
    if ($failures -gt 0) {
        Write-Output ("SELF-TEST FAILED (" + $failures + ")")
        exit 1
    }
    Write-Output 'SELF-TEST PASS'
    exit 0
}

$rows = New-Object System.Collections.Generic.List[object]
$fixes = New-Object System.Collections.Generic.List[string]

function Add-Row {
    param([string]$Status, [string]$Check, [string]$Detail, [string]$Fix)
    $rows.Add([pscustomobject]@{ Status = $Status; Check = $Check; Detail = $Detail })
    if ($Status -eq 'FAIL' -and $Fix) { $fixes.Add($Fix) }
}

$envInfo = Initialize-NexusKvEnvironment

# 1. Rust host
$rustc = Get-Command rustc -ErrorAction SilentlyContinue
if ($rustc) {
    $versionText = (& rustc -vV 2>&1 | Out-String)
    $hostMatch = [regex]::Match($versionText, 'host:\s*(\S+)')
    $rustHost = if ($hostMatch.Success) { $hostMatch.Groups[1].Value } else { 'unknown' }
    if (Test-RustHost $rustHost) {
        Add-Row 'PASS' 'rustc host' $rustHost ''
    } else {
        Add-Row 'FAIL' 'rustc host' $rustHost 'rustup default stable-x86_64-pc-windows-msvc'
    }
} else {
    Add-Row 'FAIL' 'rustc host' 'rustc not found' 'Install Rust from https://rustup.rs and select stable-x86_64-pc-windows-msvc'
}

# 2. MSVC cl.exe and vcvars
$cl = Find-ClExe
$vcvars = Find-VsDevShell
if ($cl -and $vcvars) {
    Add-Row 'PASS' 'MSVC cl.exe' $cl ''
} else {
    $detail = if (-not $vcvars) { 'vcvars64.bat not found' } else { 'cl.exe not found' }
    Add-Row 'FAIL' 'MSVC cl.exe' $detail 'Install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload'
}

# 3. CUDA toolkit and nvcc version
$cuda = $envInfo.CudaHome
if ($cuda) {
    Add-Row 'PASS' 'CUDA toolkit' $cuda ''
    if ($envInfo.CudaVersion -eq '12.6') {
        Add-Row 'PASS' 'nvcc version' '12.6' ''
    } elseif ($envInfo.CudaVersion) {
        Add-Row 'FAIL' 'nvcc version' ("found " + $envInfo.CudaVersion + ", require 12.6") 'Install CUDA Toolkit 12.6 and point CUDA_PATH at it (do not use 13.x with candle 0.9)'
    } else {
        Add-Row 'FAIL' 'nvcc version' 'could not parse nvcc --version' 'Reinstall CUDA Toolkit 12.6 or repair the toolkit'
    }
    $cudart = Join-Path $cuda 'lib\x64\cudart.lib'
    if (Test-Path $cudart) {
        Add-Row 'PASS' 'CUDA lib' $cudart ''
    } else {
        Add-Row 'FAIL' 'CUDA lib' ($cudart + ' missing') 'Install the CUDA 12.6 runtime/development libraries into the toolkit root'
    }
} else {
    Add-Row 'FAIL' 'CUDA toolkit' 'CUDA_HOME/CUDA_PATH and conventional paths were not found' 'Install CUDA Toolkit 12.6 for Windows, set CUDA_PATH to its root, then re-run this script'
    Add-Row 'FAIL' 'nvcc version' 'not found' 'Install CUDA Toolkit 12.6 for Windows'
    Add-Row 'FAIL' 'CUDA lib' 'lib\x64\cudart.lib not found' 'Install the CUDA 12.6 runtime/development libraries'
}

# 4. NVIDIA GPU and driver
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | Out-String).Trim()
    if ($gpu -and $LASTEXITCODE -eq 0) {
        if ($gpu -match 'RTX 3050') {
            Add-Row 'PASS' 'NVIDIA GPU' ($gpu -replace "`r?`n", '; ') ''
        } else {
            Add-Row 'WARN' 'NVIDIA GPU' ($gpu -replace "`r?`n", '; ') 'Expected the RTX 3050 Laptop target; tests may still run on another CUDA GPU.'
        }
    } else {
        Add-Row 'FAIL' 'NVIDIA GPU' 'nvidia-smi failed' 'Install/repair the NVIDIA Windows display driver; never install a driver from the CUDA toolkit bundle'
    }
} else {
    Add-Row 'FAIL' 'NVIDIA GPU' 'nvidia-smi not found' 'Install the NVIDIA Windows display driver for the RTX 3050'
}

# 5. pnpm and python
$pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
if ($pnpm) {
    Add-Row 'PASS' 'pnpm' ((& pnpm --version 2>&1 | Out-String).Trim()) ''
} else {
    Add-Row 'FAIL' 'pnpm' 'not found' 'corepack enable; corepack prepare pnpm@12.18.0 --activate'
}
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    Add-Row 'PASS' 'python' ((& python --version 2>&1 | Out-String).Trim()) ''
} else {
    Add-Row 'FAIL' 'python' 'not found' 'Install Python 3 and the Hugging Face CLI (pip install -U huggingface_hub)'
}

# 6. Disk space
$repo = Split-Path -Parent $PSScriptRoot
$free = Get-FreeDiskGb -Path $repo
if ($null -ne $free -and $free -gt 10) {
    Add-Row 'PASS' 'free disk' ($free.ToString() + ' GB') ''
} else {
    Add-Row 'FAIL' 'free disk' (($free.ToString() + ' GB') + ', require > 10 GB') 'Free at least 10 GB on the drive holding the repository'
}

# 7. Model weights (warn only: the default build needs none)
$modelDir = Join-Path $repo 'models\qwen3-0.6b'
$configJson = Join-Path $modelDir 'config.json'
$tokenizerJson = Join-Path $modelDir 'tokenizer.json'
$safetensors = @()
if (Test-Path $modelDir) { $safetensors = @(Get-ChildItem $modelDir -Filter '*.safetensors' -ErrorAction SilentlyContinue) }
if ((Test-Path $configJson) -and (Test-Path $tokenizerJson) -and $safetensors.Count -gt 0) {
    Add-Row 'PASS' 'Qwen3 weights' ($modelDir + ' (' + $safetensors.Count + ' shard(s))') ''
} else {
    Add-Row 'WARN' 'Qwen3 weights' 'missing config.json/tokenizer.json/*.safetensors' 'hf download Qwen/Qwen3-0.6B --local-dir models\qwen3-0.6b'
}

if ($NoGpuRequired) {
    $rows = @($rows | Where-Object { $_.Check -in @('rustc host', 'pnpm', 'python', 'free disk', 'Qwen3 weights') })
    $fixes.Clear()
}

Write-Output 'NexusKV environment check'
Write-Output '-------------------------'
foreach ($row in $rows) {
    Write-Output ("{0,-5} {1,-16} {2}" -f $row.Status, $row.Check, $row.Detail)
}
Write-Output '-------------------------'

$failCount = @($rows | Where-Object { $_.Status -eq 'FAIL' }).Count
if ($failCount -gt 0) {
    Write-Output ("HARD FAILURES: " + $failCount)
    foreach ($fix in $fixes) { Write-Output ("FIX: " + $fix) }
    exit 1
}
Write-Output 'OK: environment satisfies the native Windows CUDA build contract.'
exit 0
