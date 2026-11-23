# Compile the actual shutdown/destructor bodies with counting dependencies. No GPU/NGX loaded.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/..").Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-shutdown-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
function Extract($path, $signature) {
    $source = Get-Content -LiteralPath (Join-Path $repo $path) -Raw
    $start = $source.IndexOf($signature, [StringComparison]::Ordinal)
    if ($start -lt 0) { throw "Missing production function: $signature" }
    $first = $source.IndexOf('{', $start)
    $depth = 0
    for ($p = $first; $p -lt $source.Length; $p++) {
        if ($source[$p] -eq '{') { $depth++ }
        if ($source[$p] -eq '}') { $depth-- }
        if (!$depth) { return $source.Substring($start, $p - $start + 1) }
    }
    throw "Unterminated production function: $signature"
}
$accessors = @(
    (Extract 'OptiScaler/proxies/NVNGX_Proxy.h' 'static PFN_D3D12_Shutdown D3D12_Shutdown()'),
    (Extract 'OptiScaler/proxies/NVNGX_Proxy.h' 'static PFN_D3D12_DriverShutdown1 D3D12_Shutdown1()')
)
Set-Content -LiteralPath "$out/shutdown_accessors.inc" -Value ($accessors -join "`n")
$code = @(
    (Extract 'OptiScaler/shaders/dlssnr/DlssNr_Dx12.cpp' 'void DlssNr_Dx12::Retire('),
    (Extract 'OptiScaler/upscalers/IFeature_Dx12.cpp' 'IFeature_Dx12::~IFeature_Dx12()'),
    (Extract 'OptiScaler/upscalers/IFeature_Dx12.cpp' 'void IFeature_Dx12::RetireNeuralRendering()'),
    (Extract 'OptiScaler/upscalers/dlss/DLSSFeature_Dx12.cpp' 'void DLSSFeatureDx12::Shutdown('),
    (Extract 'OptiScaler/inputs/NVNGX_DLSS_Dx12.cpp' 'static NVSDK_NGX_Result ShutdownDx12('),
    (Extract 'OptiScaler/inputs/NVNGX_DLSS_Dx12.cpp' 'NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown(void)'),
    (Extract 'OptiScaler/inputs/NVNGX_DLSS_Dx12.cpp' 'NVSDK_NGX_API NVSDK_NGX_Result NVSDK_NGX_D3D12_Shutdown1(')
)
$dll = Get-Content -LiteralPath "$repo/OptiScaler/dllmain.cpp" -Raw
$start = $dll.IndexOf('    case DLL_PROCESS_DETACH:')
$end = $dll.IndexOf('    case DLL_THREAD_ATTACH:', $start)
$code += "void Detach(void* lpReserved) { switch (0) {`n" + $dll.Substring($start, $end - $start) + "`n} }"
Set-Content -LiteralPath "$out/shutdown_production.inc" -Value ($code -join "`n")
& cl.exe /nologo /std:c++20 /EHsc /W4 "/I$out" "/Fo$out/" "/Fe$out/shutdown.exe" "$PSScriptRoot/nr_shutdown_smoke.cpp"
if ($LASTEXITCODE) { throw 'Shutdown regression compilation failed' }
& "$out/shutdown.exe"
if ($LASTEXITCODE) { throw 'Shutdown regression failed' }
