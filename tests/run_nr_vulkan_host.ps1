# Run after a Release x64 build from an x64 Visual Studio developer shell.
param(
    [Parameter(Mandatory)][string]$Assets,
    [ValidateSet(4, 21, 22, 25, 26, 27, 28)][int[]]$Preset = @(25, 26)
)
$ErrorActionPreference = 'Stop'
$modelFolder = (Resolve-Path -LiteralPath $Assets).Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-vulkan-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
Push-Location "$PSScriptRoot/.."
try {
    $exe = Join-Path $out 'nr_vulkan_host_smoke.exe'
    $objects = 'NrHost_Vk', 'NrNet4', 'NrSwin4', 'NrFfnV2' | ForEach-Object { "OptiScaler/x64/Release/$_.obj" }
    & cl.exe /nologo /std:c++20 /O2 /MD /EHsc /IOptiScaler /Iexternal/vulkan/include /Iexternal/nvngx_dlss_sdk /Iexternal/spdlog/include `
        "/Fo$out/test.obj" "/Fe$exe" tests/nr_vulkan_host_smoke.cpp @objects /link /LTCG `
        OptiScaler/library/vulkan/vulkan-1.lib bcrypt.lib version.lib dxguid.lib d3d12.lib
    if ($LASTEXITCODE) { throw 'Vulkan host smoke did not compile' }
    foreach ($model in $Preset) {
        & $exe $modelFolder $model
        if ($LASTEXITCODE) { throw "Vulkan host smoke failed for preset $model" }
    }
} finally {
    Pop-Location
    Get-ChildItem -LiteralPath $out -File | Remove-Item -Force
    Remove-Item -LiteralPath $out -Force
}
