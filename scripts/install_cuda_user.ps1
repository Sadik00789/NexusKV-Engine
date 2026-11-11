# scripts/install_cuda_user.ps1
# Installs a per-user CUDA toolkit from NVIDIA's official redistributable
# archives, with no Administrator rights and no system-wide changes.
#
# Why not the normal installer: the NVIDIA silent installer requires elevation.
# NVIDIA publishes every toolkit component as a plain zip, so a non-admin user
# can assemble the same tree under %USERPROFILE%\cuda-<version>.
#
#   powershell -ExecutionPolicy Bypass -File scripts/install_cuda_user.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/install_cuda_user.ps1 -Version 13.4.3 -Force
#
# Version choice: candle 0.9.x depends on cudarc 0.19.10, whose build script
# panics on any toolkit newer than 13.4. Requesting a newer version fails here
# with that reason instead of at compile time.
param(
    [string]$Version = '13.4.3',
    [string]$Destination = '',
    [switch]$Force,
    [switch]$KeepDownloads
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$supported = @('13.4.3')
if ($supported -notcontains $Version) {
    Write-Output ("ERROR: CUDA $Version is not supported by cudarc 0.19.10 (candle 0.9.x).")
    Write-Output ("Supported versions for this checkout: " + ($supported -join ', '))
    Write-Output 'Newer toolkits need a newer candle/cudarc pair; see WINDOWS_SETUP.md.'
    exit 1
}

if (-not $Destination) {
    $majorMinor = ($Version -split '\.')[0..1] -join '.'
    $Destination = Join-Path $env:USERPROFILE ("cuda-" + $majorMinor)
}

$manifestUrl = "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_$Version.json"
$downloadBase = 'https://developer.download.nvidia.com/compute/cuda/redist'
$downloadDir = Join-Path $env:TEMP ("nexuskv-cuda-$Version")

# Components the engine, candle, and compute-sanitizer need.
#   cuda_nvcc      nvcc + host compiler glue (custom kernel, candle-kernels)
#   cuda_cudart    cudart headers/libs/dll
#   cuda_crt       device runtime builtins
#   cccl           C++ standard library for device code
#   libcublas      candle's GEMM backend
#   libcurand      candle's RNG
#   cuda_nvrtc     runtime compilation (candle JIT paths)
#   libnvvm        NVRTC's optimizer
#   libnvjitlink   device linker for NVRTC
#   cuda_sanitizer_api  compute-sanitizer.exe
#   cuda_cupti     sanitizer/profiler dependency
$components = @(
    'cuda_nvcc',
    'cuda_cudart',
    'cuda_crt',
    'cccl',
    'libcublas',
    'libcurand',
    'cuda_nvrtc',
    'libnvvm',
    'libnvjitlink',
    'cuda_sanitizer_api',
    'cuda_cupti'
)

Write-Output "--- NexusKV per-user CUDA $Version install ---"
Write-Output ("destination: " + $Destination)

if ((Test-Path (Join-Path $Destination 'bin\nvcc.exe')) -and -not $Force) {
    Write-Output 'A toolkit already exists there; nothing to do (use -Force to reinstall).'
    exit 0
}

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
$manifest = Invoke-RestMethod -Uri $manifestUrl -TimeoutSec 60

foreach ($component in $components) {
    $entry = $manifest.$component
    if (-not $entry) {
        Write-Output ("ERROR: component " + $component + " is missing from the manifest")
        exit 1
    }
    $windows = $entry.'windows-x86_64'
    if (-not $windows) {
        Write-Output ("ERROR: component " + $component + " has no windows-x86_64 archive")
        exit 1
    }
    $relative = $windows.relative_path
    $archive = Join-Path $downloadDir (Split-Path $relative -Leaf)
    if (-not (Test-Path $archive) -or (Get-Item $archive).Length -ne $windows.size) {
        Write-Output ("GET " + $component + " (" + [math]::Round($windows.size / 1MB, 1) + " MB)")
        & curl.exe -sSL --retry 3 -o $archive "$downloadBase/$relative"
        if ($LASTEXITCODE -ne 0) {
            Write-Output ("ERROR: download failed for " + $component)
            exit 1
        }
    }
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
foreach ($component in $components) {
    $relative = $manifest.$component.'windows-x86_64'.relative_path
    $archive = Join-Path $downloadDir (Split-Path $relative -Leaf)
    $extract = Join-Path $downloadDir ("x-" + $component)
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    & tar -xzf $archive -C $extract
    if ($LASTEXITCODE -ne 0) {
        Write-Output ("ERROR: extract failed for " + $component)
        exit 1
    }
    # Every archive contains one top-level folder holding bin/include/lib/etc.
    $root = Get-ChildItem $extract -Directory | Select-Object -First 1
    if (-not $root) {
        Write-Output ("ERROR: unexpected archive layout for " + $component)
        exit 1
    }
    Write-Output ("install " + $component)
    Get-ChildItem $root.FullName -Force | ForEach-Object {
        Copy-Item -Recurse -Force $_.FullName $Destination
    }
    Remove-Item -Recurse -Force $extract
}

if (-not $KeepDownloads) { Remove-Item -Recurse -Force $downloadDir }

$nvcc = Join-Path $Destination 'bin\nvcc.exe'
$cudart = Join-Path $Destination 'lib\x64\cudart.lib'
foreach ($required in @($nvcc, $cudart)) {
    if (-not (Test-Path $required)) {
        Write-Output ("ERROR: install incomplete, missing " + $required)
        exit 1
    }
}

$versionText = (& $nvcc --version 2>&1) -join ' '
if ($versionText -match 'release\s+([0-9]+\.[0-9]+)') {
    Write-Output ("nvcc reports CUDA " + $matches[1])
}

Write-Output ''
Write-Output 'Installed. Point this session at it with:'
Write-Output ('  $env:CUDA_HOME = "' + $Destination + '"')
Write-Output ('  $env:CUDA_PATH = "' + $Destination + '"')
Write-Output ('  $env:PATH = "' + $Destination + '\bin;' + $Destination + '\bin\x64;" + $env:PATH')
Write-Output ''
Write-Output 'scripts/windows_env.ps1 resolves this path automatically.'
