# Run from a Visual Studio x64 developer shell. Executes shader math on WARP (D3D11 and D3D12).
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/..").Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-reapply-' + [guid]::NewGuid())
New-Item -ItemType Directory $out | Out-Null
Push-Location $repo
try {
    $executable = Join-Path $out 'nr_reapply_smoke.exe'
    & cl.exe /nologo /std:c++20 /EHsc "/Fo$out/nr_reapply_smoke.obj" "/Fe$executable" `
        "$PSScriptRoot/nr_reapply_smoke.cpp" /link d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib
    if ($LASTEXITCODE) { throw 'NR re-apply smoke did not compile' }
    & $executable
    if ($LASTEXITCODE) { throw 'NR re-apply smoke failed' }
} finally {
    Pop-Location
    Get-ChildItem -LiteralPath $out -File | Remove-Item -Force
    Remove-Item -LiteralPath $out -Force
}
