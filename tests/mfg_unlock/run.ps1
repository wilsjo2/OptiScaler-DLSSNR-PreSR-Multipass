param([string]$Runtime)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$out = Join-Path $repo 'x64/rtx40-mfg-validation'
New-Item -ItemType Directory -Force "$out/seams/misc", "$out/seams/proxies" | Out-Null
foreach ($header in @('pch.h', 'SysUtils.h', 'Config.h', 'Util.h', 'misc/IdentifyGpu.h', 'proxies/KernelBase_Proxy.h')) {
    Set-Content -LiteralPath "$out/seams/$header" -Value '// Supplied by Mocks.h; patching and scanning are production code.'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$build = @"
@echo off
call "$vs/VC/Auxiliary/Build/vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /DUNICODE /D_UNICODE /DOPTISCALER_RTX40_MFG /I "$out/seams" /I "$repo/OptiScaler" "$PSScriptRoot/PatchTests.cpp" /Fe:"$out/mfg-patch.exe" /Fo:"$out/mfg-patch.obj"
"@
Set-Content -LiteralPath "$out/build.cmd" -Value $build
& "$out/build.cmd"
if ($LASTEXITCODE) { throw 'MFG regression build failed' }
foreach ($case in @('disabled', 'blackwell', 'ampere', 'other-vendor', 'restart', 'missing-gate', 'duplicate-gate', 'unknown', 'no-kernel', 'malformed', 'legacy', '3109')) {
    & "$out/mfg-patch.exe" $case
    if ($LASTEXITCODE) { throw "MFG regression failed: $case" }
}
# Host checks of the option helpers (provider discovery, plugin ceiling, PTX rewrite, temporal method, flip
# metering). They compile alone against synthetic PE images; no NVIDIA code executes.
foreach ($smoke in @('mfg_provider_smoke', 'mfg_ceiling_smoke', 'mfg_ptx_smoke', 'mfg_method_smoke', 'mfg_flipmeter_smoke')) {
    $smokeBuild = @"
@echo off
call "$vs/VC/Auxiliary/Build/vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /EHsc /W4 "$repo/tests/$smoke.cpp" /Fe:"$out/$smoke.exe" /Fo:"$out/$smoke.obj"
"@
    Set-Content -LiteralPath "$out/$smoke.cmd" -Value $smokeBuild
    & "$out/$smoke.cmd"
    if ($LASTEXITCODE) { throw "$smoke build failed" }
    & "$out/$smoke.exe"
    if ($LASTEXITCODE) { throw "$smoke failed" }
}
if ($Runtime) {
    $before = (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash
    & "$out/mfg-patch.exe" runtime $Runtime
    if ($LASTEXITCODE) { throw 'Installed runtime patch smoke failed' }
    if ((Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash -ne $before) { throw 'Runtime file changed' }
}
