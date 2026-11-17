# Run from a Visual Studio developer PowerShell (cl.exe on PATH). No GPU, NR runtime or game is used.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$build = Join-Path ([System.IO.Path]::GetTempPath()) ('nr-nvfp4-host-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $build | Out-Null
try {
    & cl.exe /nologo /std:c++20 /EHsc /W4 /DNOMINMAX "/Fo$build/" "/Fe$build/HostTests.exe" `
        "$PSScriptRoot/HostTests.cpp" "$repo/OptiScaler/dlssnr/NrNvfp4.cpp"
    if ($LASTEXITCODE -ne 0) { throw 'NR 4-bit host test compilation failed.' }
    & "$build/HostTests.exe"
    if ($LASTEXITCODE -ne 0) { throw 'NR 4-bit host tests failed.' }
} finally {
    Get-ChildItem -LiteralPath $build -File | Remove-Item -Force
    Remove-Item -LiteralPath $build -Force
}
