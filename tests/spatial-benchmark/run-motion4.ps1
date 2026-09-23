param(
    [Parameter(Mandatory)][string]$Driver,
    [Parameter(Mandatory)][string]$RuntimeDirectory,
    [string]$VcVars,
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo 'x64/nr-spatial-benchmark' }
if (!$VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $VcVars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
}
$build = Join-Path $OutputDirectory 'motion4-build'
New-Item -ItemType Directory -Force $build | Out-Null
Set-Content -LiteralPath "$build/pch.h" -Value '// Standalone compatibility seam.'
Set-Content -LiteralPath "$build/Logger.h" -Value '// Logging supplied by Adapter.h.'
$cmd = @"
@echo off
call "$VcVars" >nul
if errorlevel 1 exit /b 1
cd /d "$repo"
cl /nologo /std:c++20 /EHsc /O2 /MD /W4 /FI"$repo/tests/nr_compatibility/Adapter.h" /I"$build" /I"$repo/external/nvngx_dlss_sdk" "$PSScriptRoot/motion4_probe.cpp" "$repo/OptiScaler/dlssnr/DlssNr_CompatibilityRuntime.cpp" /Fo"$build/" /Fe:"$build/motion4_probe.exe" /link d3d12.lib dxgi.lib
"@
Set-Content -LiteralPath "$build/build.cmd" -Value $cmd
& "$build/build.cmd"
if ($LASTEXITCODE) { throw 'Motion-format probe compilation failed.' }
& "$build/motion4_probe.exe" (Resolve-Path -LiteralPath $Driver).Path (Resolve-Path -LiteralPath $RuntimeDirectory).Path --direct 2>&1 | Tee-Object -FilePath "$OutputDirectory/motion4-probe.log"
if ($LASTEXITCODE) { throw 'Motion-format probe failed.' }
