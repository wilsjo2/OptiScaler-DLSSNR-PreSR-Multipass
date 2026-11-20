# Run from a Visual Studio x64 developer PowerShell (cl.exe on PATH). Uses WARP; no game, no NVIDIA runtime.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$build = Join-Path ([System.IO.Path]::GetTempPath()) ('nr-calibrate-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $build | Out-Null
try {
    Set-Content -LiteralPath (Join-Path $build 'pch.h') -Value '// Standalone production-helper build.'
    Set-Content -LiteralPath (Join-Path $build 'Util.h') -Value '#include <unknwn.h>
namespace Util { inline bool CheckForRealObject(const char*, IUnknown*, IUnknown**) { return false; } }'
    & cl.exe /nologo /std:c++20 /EHsc /W4 /O2 /DNOMINMAX "/I$build" "/Fo$build/" "/Fe$build/calibrate.exe" `
        "$PSScriptRoot/CalibrateTests.cpp" "$repo/OptiScaler/dlssnr/DlssNr_GpuLifetime.cpp" d3d12.lib dxgi.lib ole32.lib
    if ($LASTEXITCODE -ne 0) { throw 'NR calibrator test compilation failed.' }
    & "$build/calibrate.exe"
    if ($LASTEXITCODE -ne 0) { throw "NR calibrator test failed (exit $LASTEXITCODE)." }
} finally { Remove-Item -Recurse -Force -LiteralPath $build -ErrorAction SilentlyContinue }
