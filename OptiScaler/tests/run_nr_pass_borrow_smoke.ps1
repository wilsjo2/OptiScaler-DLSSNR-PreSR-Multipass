# Builds and runs the NrPassBorrow smoke test: fake NvAPI originals and table-synthesised kernel sequences on the CPU,
# plus the real D3D12 handoff copy on WARP. No NR runtime, NGX or game is used. Also builds NrPassBorrow.cpp with
# /W4 /WX, and checks NrPassBorrowTables.h against the recorded traces when they are present (they are gitignored).
param([string]$VcVars)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (!$VcVars) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$installation) { throw 'Install Visual Studio C++ build tools, or supply -VcVars.' }
    $VcVars = Join-Path $installation 'VC/Auxiliary/Build/vcvars64.bat'
}
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-pass-borrow-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
$includes = '/I "{0}/OptiScaler/dlssnr" /I "{0}/OptiScaler/include" /I "{0}/external/nvapi"' -f $repo
$build = @"
@echo off
call "$VcVars" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /W4 /WX /EHsc /permissive- /utf-8 /O2 /MD $includes /c "$repo/OptiScaler/dlssnr/NrPassBorrow.cpp" /Fo:"$out/NrPassBorrow.obj"
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /W4 /EHsc /permissive- /utf-8 /O2 /MD $includes "$PSScriptRoot/nr_pass_borrow_smoke.cpp" "$out/NrPassBorrow.obj" /Fe:"$out/nr_pass_borrow_smoke.exe" /Fo:"$out/nr_pass_borrow_smoke.obj" /link "$repo/OptiScaler/library/detours/detours.lib" d3d12.lib dxgi.lib
"@
Set-Content -LiteralPath (Join-Path $out 'build.cmd') -Value $build
Push-Location $repo
try {
    $traces = Join-Path $repo 'standalone/NrCapture/out-model-map/model-b-20261106/native'
    if (Test-Path -LiteralPath $traces) {
        & python (Join-Path $repo 'OptiScaler/dlssnr/tools/make_pass_borrow_tables.py') --check | Select-Object -Last 1
        if ($LASTEXITCODE) { throw 'NrPassBorrowTables.h does not match the recorded traces.' }
    } else {
        Write-Host 'Recorded traces not present: skipping the table check.'
    }
    & (Join-Path $out 'build.cmd')
    if ($LASTEXITCODE) { throw 'NR pass borrow smoke did not compile.' }
    & (Join-Path $out 'nr_pass_borrow_smoke.exe')
    if ($LASTEXITCODE) { throw 'NR pass borrow smoke failed.' }
} finally {
    Pop-Location
    Get-ChildItem -LiteralPath $out -File | Remove-Item -Force
    Remove-Item -LiteralPath $out -Force
}
