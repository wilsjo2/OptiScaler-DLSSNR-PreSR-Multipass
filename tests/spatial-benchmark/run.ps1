param(
    [Parameter(Mandatory)][string]$Driver,
    [Parameter(Mandatory)][string]$RuntimeDirectory,
    [string]$VcVars,
    [string]$OutputDirectory,
    [int]$Frames = 20,
    [switch]$ReverseOrder
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo 'x64/nr-spatial-benchmark' }
if (!$VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$installation) { throw 'Visual Studio C++ build tools are required.' }
    $VcVars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
}
$build = Join-Path $OutputDirectory 'build'
New-Item -ItemType Directory -Force $build | Out-Null
Set-Content -LiteralPath "$build/pch.h" -Value '// Standalone benchmark shim.'
Set-Content -LiteralPath "$build/Logger.h" -Value '// Logging supplied by the compatibility adapter.'
$cmd = @"
@echo off
call "$VcVars" >nul
if errorlevel 1 exit /b 1
cd /d "$repo"
cl /nologo /std:c++20 /EHsc /O2 /MD /W4 /FI"$repo/tests/nr_compatibility/Adapter.h" /I"$build" /I"$repo/external/nvngx_dlss_sdk" /I"$repo/OptiScaler" "$PSScriptRoot/spatial_benchmark.cpp" "$repo/OptiScaler/dlssnr/DlssNr_CompatibilityRuntime.cpp" /Fo"$build/" /Fe:"$build/spatial_benchmark.exe" /link d3d12.lib dxgi.lib d3dcompiler.lib
"@
Set-Content -LiteralPath "$build/build.cmd" -Value $cmd
& "$build/build.cmd"
if ($LASTEXITCODE) { throw 'Spatial benchmark compilation failed.' }
$driverPath = (Resolve-Path -LiteralPath $Driver).Path
$runtimePath = (Resolve-Path -LiteralPath $RuntimeDirectory).Path
$manifest = [ordered]@{
    gpu = (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
    git_head = (& git -C $repo rev-parse HEAD)
    frames_per_case = $Frames
    order = $(if ($ReverseOrder) { 'reverse' } else { 'forward' })
    driver = @{ path = $driverPath; sha256 = (Get-FileHash -LiteralPath $driverPath -Algorithm SHA256).Hash }
    nr_runtime = @{ path = (Join-Path $runtimePath 'nvngx_dlssnr.dll'); sha256 = (Get-FileHash -LiteralPath (Join-Path $runtimePath 'nvngx_dlssnr.dll') -Algorithm SHA256).Hash }
    spatial_color_blob_sha256 = (Get-FileHash -LiteralPath "$repo/OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_Shader.cso" -Algorithm SHA256).Hash
    spatial_guides_blob_sha256 = (Get-FileHash -LiteralPath "$repo/OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_guides_Shader.cso" -Algorithm SHA256).Hash
    benchmark_source_sha256 = (Get-FileHash -LiteralPath "$PSScriptRoot/spatial_benchmark.cpp" -Algorithm SHA256).Hash
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$OutputDirectory/manifest.json"
$arguments = @($driverPath, $runtimePath, $OutputDirectory, $Frames)
if ($ReverseOrder) { $arguments += '--reverse' }
& "$build/spatial_benchmark.exe" @arguments
if ($LASTEXITCODE) { throw 'Spatial benchmark run failed.' }
& python "$PSScriptRoot/analyze.py" $OutputDirectory
if ($LASTEXITCODE) { throw 'Image analysis failed.' }
