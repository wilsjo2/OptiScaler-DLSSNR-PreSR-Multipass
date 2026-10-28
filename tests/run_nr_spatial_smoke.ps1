# Run from a Visual Studio x64 developer shell. Vulkan smoke uses the production SPIR-V blobs.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/..").Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-spatial-' + [guid]::NewGuid())
New-Item -ItemType Directory $out | Out-Null
Push-Location $repo
try {
    $mapping = Join-Path $out 'nr_spatial_mapping_smoke.exe'
    & cl.exe /nologo /std:c++20 /EHsc "/Fo$out/nr_spatial_mapping_smoke.obj" "/Fe$mapping" `
        "$PSScriptRoot/nr_spatial_mapping_smoke.cpp"
    if ($LASTEXITCODE) { throw 'NR spatial mapping smoke did not compile' }
    & $mapping
    if ($LASTEXITCODE) { throw 'NR spatial mapping smoke failed' }

    $vulkan = Join-Path $out 'nr_spatial_vulkan_smoke.exe'
    & cl.exe /nologo /std:c++20 /EHsc /IOptiScaler /Iexternal/vulkan/include `
        /Iexternal/nvngx_dlss_sdk /Iexternal/spdlog/include "/Fo$out/nr_spatial_vulkan_smoke.obj" "/Fe$vulkan" `
        "$PSScriptRoot/nr_spatial_vulkan_smoke.cpp" /link OptiScaler/library/vulkan/vulkan-1.lib
    if ($LASTEXITCODE) { throw 'NR spatial Vulkan smoke did not compile' }
    & $vulkan "$repo/OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_Shader_Vk.spv" `
              "$repo/OptiScaler/shaders/dlssnr/precompile/dlssnr_spatial_guides_Shader_Vk.spv"
    if ($LASTEXITCODE) { throw 'NR spatial Vulkan smoke failed' }
} finally {
    Pop-Location
    Get-ChildItem -LiteralPath $out -File | Remove-Item -Force
    Remove-Item -LiteralPath $out -Force
}
