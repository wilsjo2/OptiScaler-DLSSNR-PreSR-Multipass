# Run from a Visual Studio x64 developer shell. No game or NVIDIA NR runtime is loaded.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/..").Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-prerelease-' + [guid]::NewGuid())
New-Item -ItemType Directory $out | Out-Null
Push-Location $repo
try {
    & "$PSScriptRoot/run_nr_shutdown.ps1"
    & "$PSScriptRoot/run_nr_gpu_lifetime.ps1"
    & "$PSScriptRoot/dlssnr_proxy/run.ps1"
    & "$PSScriptRoot/mfg_unlock/run.ps1"
    & "$PSScriptRoot/run_nr_pipeline_capture.ps1"
    & "$PSScriptRoot/run_nr_streamline_hooks.ps1"
    function RunSmoke($name, $libraries, $arguments = @(), $includes = @()) {
        $exe = Join-Path $out "$name.exe"
        & cl.exe /nologo /std:c++20 /EHsc /DNOMINMAX "/Fo$out/$name.obj" "/Fe$exe" @includes "$PSScriptRoot/$name.cpp" @libraries
        if ($LASTEXITCODE) { throw "$name did not compile" }
        & $exe @arguments
        if ($LASTEXITCODE) { throw "$name failed" }
    }
    $shader = "$repo/OptiScaler/shaders/dlssnr/precompile/dlssnr.hlsl"
    RunSmoke 'nr_skin_shader_smoke' @('d3d11.lib', 'd3dcompiler.lib') @($shader) @('/IOptiScaler/include')
    RunSmoke 'nr_replace_detail_smoke' @('d3d11.lib', 'd3dcompiler.lib') @($shader)
    RunSmoke 'nr_resize_shader_smoke' @('d3d11.lib')
    RunSmoke 'nr_exposure_shader_smoke' @('d3d11.lib', 'd3dcompiler.lib')
    RunSmoke 'nr_residual_rr_smoke' @('d3d11.lib', 'd3dcompiler.lib')
    RunSmoke 'nr_active_color_smoke' @('d3d12.lib', 'dxgi.lib')
    RunSmoke 'nr_finished_queue_smoke' @('d3d12.lib', 'dxgi.lib')
    & "$PSScriptRoot/run_nr_spatial_smoke.ps1"
    RunSmoke 'nr_vulkan_shader_smoke' @('OptiScaler/library/vulkan/vulkan-1.lib') `
        @("$repo/OptiScaler/shaders/dlssnr/precompile/DlssNr_Shader_Vk.spv") `
        @('/IOptiScaler', '/Iexternal/vulkan/include', '/Iexternal/nvngx_dlss_sdk', '/Iexternal/spdlog/include')
    Write-Output 'NR prerelease checks passed.'
} finally { Pop-Location }
