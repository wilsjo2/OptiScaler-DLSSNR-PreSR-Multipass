# Run from a Visual Studio x64 developer shell. Executes shader math on WARP.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/..").Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-raw-residual-' + [guid]::NewGuid())
New-Item -ItemType Directory $out | Out-Null
Push-Location $repo
try {
    $executable = Join-Path $out 'nr_raw_residual_smoke.exe'
    & cl.exe /nologo /std:c++20 /EHsc "/Fo$out/nr_raw_residual_smoke.obj" "/Fe$executable" `
        "$PSScriptRoot/nr_raw_residual_smoke.cpp" /link d3d11.lib d3dcompiler.lib
    if ($LASTEXITCODE) { throw 'NR raw residual smoke did not compile' }
    & $executable
    if ($LASTEXITCODE) { throw 'NR raw residual smoke failed' }
} finally {
    Pop-Location
    Get-ChildItem -LiteralPath $out -File | Remove-Item -Force
    Remove-Item -LiteralPath $out -Force
}
