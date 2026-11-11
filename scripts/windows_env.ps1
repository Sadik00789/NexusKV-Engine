# scripts/windows_env.ps1
# Shared native-Windows environment setup for NexusKV. Dot-source this file:
#   . "$PSScriptRoot\windows_env.ps1"
# It imports the Visual Studio x64 developer environment (cl.exe) and resolves
# the CUDA toolkit without requiring the caller to start a special shell.

function Get-VsWherePath {
    $candidate = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $candidate) { return $candidate }
    return $null
}

function Find-VsDevShell {
    $vswhere = Get-VsWherePath
    if (-not $vswhere) { return $null }
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $install) { return $null }
    $vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
    if (Test-Path $vcvars) { return $vcvars }
    return $null
}

function Import-VsDevEnvironment {
    $vcvars = Find-VsDevShell
    if (-not $vcvars) { return $false }
    $lines = & cmd.exe /c "call `"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            if ($name -and $name -notmatch '^=') {
                Set-Item -Path ("env:" + $name) -Value $value
            }
        }
    }
    return $true
}

function Resolve-CudaToolkit {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:CUDA_HOME) { $candidates.Add($env:CUDA_HOME) }
    if ($env:CUDA_PATH) { $candidates.Add($env:CUDA_PATH) }
    if ($env:USERPROFILE) { $candidates.Add((Join-Path $env:USERPROFILE 'cuda-12.6')) }
    $candidates.Add('C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6')
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-CudaLayout -Path $candidate)) { return $candidate }
    }
    return $null
}

function Test-CudaLayout {
    param([string]$Path)
    if (-not $Path) { return $false }
    $nvcc = Join-Path $Path 'bin\nvcc.exe'
    $cudart = Join-Path $Path 'lib\x64\cudart.lib'
    return ((Test-Path $nvcc) -and (Test-Path $cudart))
}

function Test-MsvcLayout {
    param([string]$VcVarsPath)
    if (-not $VcVarsPath) { return $false }
    return (Test-Path $VcVarsPath)
}

function Get-NvccVersion {
    param([string]$CudaHome)
    if (-not $CudaHome) { return $null }
    $nvcc = Join-Path $CudaHome 'bin\nvcc.exe'
    if (-not (Test-Path $nvcc)) { return $null }
    $output = & $nvcc --version 2>&1
    foreach ($line in $output) {
        if ($line -match 'release\s+([0-9]+\.[0-9]+)') { return $matches[1] }
    }
    return $null
}

function Initialize-NexusKvEnvironment {
    $vs = Import-VsDevEnvironment
    $cuda = Resolve-CudaToolkit
    $env:CUDA_COMPUTE_CAP = '86'
    if (-not $env:CARGO_BUILD_JOBS) { $env:CARGO_BUILD_JOBS = '2' }
    # CUDA 12.6 predates MSVC 14.44. Make every nvcc invocation (including
    # candle-kernels' bindgen_cuda, which only forwards the flag when
    # NVCC_CCBIN is set) relax the host-compiler version gate.
    # nvcc's Windows default for -c objects is /MT; -Xcompiler /MD matches
    # Rust's dynamic CRT (otherwise MSVC fails at link time with a
    # RuntimeLibrary fail-if-mismatch). -cudart=shared selects the CUDA runtime.
    $env:NVCC_PREPEND_FLAGS = '-allow-unsupported-compiler -Xcompiler /MD -cudart=shared'
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) { $env:NVCC_CCBIN = $cl.Source }
    if ($cuda) {
        $env:CUDA_PATH = $cuda
        $env:CUDA_HOME = $cuda
        $cudaBin = Join-Path $cuda 'bin'
        if (($env:PATH -split ';') -notcontains $cudaBin) {
            $env:PATH = $cudaBin + ';' + $env:PATH
        }
    }
    return [pscustomobject]@{
        VisualStudio = $vs
        CudaHome     = $cuda
        CudaVersion  = (Get-NvccVersion -CudaHome $cuda)
    }
}

function Get-FreeDiskGb {
    param([string]$Path)
    try {
        $full = [IO.Path]::GetFullPath($Path)
        $root = [IO.Path]::GetPathRoot($full)
        $drive = New-Object IO.DriveInfo($root)
        return [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    } catch {
        return $null
    }
}
