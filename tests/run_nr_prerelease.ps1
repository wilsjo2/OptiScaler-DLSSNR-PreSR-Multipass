# ============================================================================
# tests/run_nr_prerelease.ps1 - NR prerelease gate for the upstream main port of
# the XeFG finished-picture NR handoff (T6-r2 expanded suite).
#
# Run from a Visual Studio x64 developer shell (cl.exe on PATH), from any
# working directory. No game is loaded; GPU smokes run on WARP. Each entry
# compiles its main-tree smoke source with cl.exe and runs the binary with a
# bounded process timeout.
#
# Coverage map (T6-r2 repair; see receipt for the full donor/main mapping):
#   nr_finished_color_smoke  finished-color transfer shader (compose path)
#   nr_residual_rr_smoke     shipped residual blob (compose path)
#   nr_skin_shader_smoke     main dlssnr.hlsl skin path (argv: repo hlsl)
#   nr_active_color_smoke    padded pre-SR colour on WARP D3D12
#   nr_gpu_time_smoke        GPU timing infra on WARP (Late Submitted hooks use it)
#   nr_guides_smoke          guide-format CPU contract
#   nr_ngx_routing_smoke     NGX feature-routing CPU contract
#   nr_seam_clock_smoke      seam-clock CPU contract
#   nr_vulkan_shader_smoke     shipped Vk SPV on a real Vulkan device (argv: repo spv;
#                            needs external/vulkan headers + shipped loader lib)
# Skipped with explicit rationale (see receipt): nr_residual_dlss/fg (need
# nvngx.dll + model argv at runtime, absent here), dlssg_sm86_ini (Ampere MFG INI
# subsystem, untouched by this port), donor-only files with no main-tree counterpart.
# Handoff decision/wiring coverage lives in run_xefg_handoff_smoke.ps1 and
# run_xefg_production_route_smoke.ps1, which run separately.
#
# Exit codes:
#   0 = every smoke passed
#   1 = a smoke failed to compile or failed
#   2 = the environment is broken (cl.exe missing, smoke hung)
# ============================================================================
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    Write-Host 'run_nr_prerelease: cl.exe is not on PATH - run from a Visual Studio x64 developer shell.'
    exit 2
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('nr-prerelease-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null

# Runs a smoke binary with a bounded process timeout (300 s). Returns the exit code.
function Invoke-NrSmoke([string]$exe, [string]$tag, [string[]]$runArgs)
{
    $stdout = Join-Path $out "$tag.stdout.txt"
    $stderr = Join-Path $out "$tag.stderr.txt"
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    if ($runArgs.Count -gt 0) { $info.Arguments = ($runArgs | ForEach-Object { '"' + $_ + '"' }) -join ' ' }
    $info.WorkingDirectory = $repo
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::Start($info)
    $outTask = $process.StandardOutput.ReadToEndAsync()
    $errTask = $process.StandardError.ReadToEndAsync()
    $exited = $process.WaitForExit(300000)
    # Terminate before awaiting pipe EOF; a hung child otherwise defeats the timeout.
    if (-not $exited) { $process.Kill() }
    $text = $outTask.Result
    $errText = $errTask.Result
    Set-Content -LiteralPath $stdout -Value $text
    Set-Content -LiteralPath $stderr -Value $errText
    if (-not $exited)
    {
        $text -split "`n" | ForEach-Object { Write-Host $_ }
        Write-Host "$tag timed out after 300s"
        exit 2
    }
    $text -split "`n" | ForEach-Object { Write-Host $_ }
    if ($errText.Length -gt 0) { $errText -split "`n" | ForEach-Object { Write-Host $_ } }
    return $process.ExitCode
}

function Invoke-NrCase([string]$name, [string[]]$libraries, [string[]]$includes, [string[]]$runArgs)
{
    $log = Join-Path $out "$name.compile.log"
    $exe = Join-Path $out "$name.exe"
    Write-Host ("cl.exe /nologo /std:c++20 /EHsc /DNOMINMAX <$name> -> $name.exe")
    & cl.exe /nologo /std:c++20 /EHsc /DNOMINMAX @includes "/Fo$out/$name.obj" "/Fe$exe" `
        (Join-Path $PSScriptRoot "$name.cpp") @libraries > $log 2>&1
    $code = $LASTEXITCODE
    Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
    if ($code -ne 0)
    {
        Write-Host ("{0}: cl.exe exit={1}" -f $name, $code)
        exit 1
    }
    $runCode = Invoke-NrSmoke $exe $name $runArgs
    Write-Host ("{0} exit={1}" -f $name, $runCode)
    if ($runCode -ne 0) { exit 1 }
}

$hlsl = Join-Path $repo 'OptiScaler/shaders/dlssnr/precompile/dlssnr.hlsl'
$vkSpv = Join-Path $repo 'OptiScaler/shaders/dlssnr/precompile/DlssNr_Shader_Vk.spv'
$vkLib = Join-Path $repo 'OptiScaler/library/vulkan/vulkan-1.lib'

Invoke-NrCase 'nr_finished_color_smoke' @('d3d11.lib', 'd3dcompiler.lib') @() @()
Invoke-NrCase 'nr_residual_rr_smoke' @('d3d11.lib', 'd3dcompiler.lib') @() @()
Invoke-NrCase 'nr_skin_shader_smoke' @('d3d11.lib', 'd3dcompiler.lib') @() @($hlsl)
Invoke-NrCase 'nr_active_color_smoke' @('d3d12.lib', 'dxgi.lib') @() @()
Invoke-NrCase 'nr_gpu_time_smoke' @('d3d12.lib', 'dxgi.lib') @() @()
Invoke-NrCase 'nr_guides_smoke' @() @() @()
Invoke-NrCase 'nr_ngx_routing_smoke' @() @("/I$repo/external/nvngx_dlss_sdk") @()
Invoke-NrCase 'nr_seam_clock_smoke' @() @() @()
Invoke-NrCase 'dxgi_window_size_smoke' @('d3d11.lib', 'dxgi.lib', 'user32.lib', 'dcomp.lib') @() @()
Invoke-NrCase 'nr_vulkan_shader_smoke' @($vkLib) @("/I$repo/external/vulkan/include") @($vkSpv)

Write-Output 'NR prerelease checks passed.'
exit 0
