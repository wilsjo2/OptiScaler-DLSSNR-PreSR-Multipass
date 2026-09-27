# ============================================================================
# tests/run_xefg_handoff_smoke.ps1 - runner for the XeFG handoff test seam.
#
# Run from a Visual Studio x64 developer shell (cl.exe on PATH), e.g. after
#   call "...\VC\Auxiliary\Build\vcvars64.bat"
# No GPU, no game, no NR runtime: the smoke drives proxy/SDK doubles only.
#
# The seam-owned contract surface is OptiScaler/dlssnr/DlssNr_XeFGHandoff.h
# (see tests/xefg_handoff_smoke.cpp for the full contract). The runner probes
# that surface: when it is absent it compiles the contract mirror and expects
# the smoke to exit 3 with the named HANDOFF_NOT_IMPLEMENTED marker (TDD RED).
#
# Exit codes:
#   0 = GREEN  - the production handoff core exists and every scenario passed
#   1 = the production core exists but the seam failed (compile, API mismatch,
#       scenario failure) or the environment is broken
#   2 = the environment/framework is broken (cl.exe missing, smoke hung)
#   3 = RED    - the production handoff core does not exist yet; marker printed
# ============================================================================
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    Write-Host 'run_xefg_handoff_smoke: cl.exe is not on PATH - run from a Visual Studio x64 developer shell.'
    exit 2
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$out = Join-Path ([IO.Path]::GetTempPath()) ('xefg-handoff-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
$smoke = Join-Path $PSScriptRoot 'xefg_handoff_smoke.cpp'
$core = 'OptiScaler/dlssnr/DlssNr_XeFGHandoff.h'
$coreHeader = Join-Path $repo $core
$coreSource = Join-Path $repo 'OptiScaler/dlssnr/DlssNr_XeFGHandoff.cpp'

# Runs the smoke with a bounded process timeout (120 s). The seam performs no
# waits; a hang here means a Present-side wait was added. Returns the exit code.
function Invoke-Smoke([string]$exe, [string]$tag)
{
    $stdout = Join-Path $out "$tag.stdout.txt"
    $stderr = Join-Path $out "$tag.stderr.txt"
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::Start($info)
    $outTask = $process.StandardOutput.ReadToEndAsync()
    $errTask = $process.StandardError.ReadToEndAsync()
    $exited = $process.WaitForExit(120000)
    # Terminate before awaiting pipe EOF; a hung child otherwise defeats the timeout.
    if (-not $exited) { $process.Kill() }
    $text = $outTask.Result
    $errText = $errTask.Result
    Set-Content -LiteralPath $stdout -Value $text
    Set-Content -LiteralPath $stderr -Value $errText
    if (-not $exited)
    {
        $text -split "`n" | ForEach-Object { Write-Host $_ }
        Write-Host "$tag timed out after 120s - a Present-side wait was observed"
        exit 2
    }
    $text -split "`n" | ForEach-Object { Write-Host $_ }
    if ($errText.Length -gt 0) { $errText -split "`n" | ForEach-Object { Write-Host $_ } }
    return $process.ExitCode
}

# Compiles via cl.exe and echoes the raw log. Returns the exe path or $null.
function Compile-Smoke([string[]]$defines, [string[]]$sources, [string]$exeName)
{
    $log = Join-Path $out "$exeName.compile.log"
    $exe = Join-Path $out "$exeName.exe"
    Write-Host ("cl.exe /nologo /std:c++20 /EHsc /W4 /DNOMINMAX " + ($defines -join ' ') +
                " <" + $sources.Count + " source(s)> -> $exeName.exe")
    & cl.exe /nologo /std:c++20 /EHsc /W4 /DNOMINMAX @defines "/Fo$out/" "/Fe$exe" @sources > $log 2>&1
    $code = $LASTEXITCODE
    Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
    if ($code -ne 0)
    {
        Write-Host "XEFG handoff seam: cl.exe exit=$code"
        return $null
    }
    return $exe
}

Write-Host "XEFG handoff seam: contract=$core"

if (-not (Test-Path -LiteralPath $coreHeader))
{
    Write-Host 'XEFG handoff seam: production core absent - compiling the contract mirror'
    $exe = Compile-Smoke @('/DNR_XEFG_HANDOFF_PRESENT=0') @($smoke) 'xefg_handoff_smoke_red'
    if ($null -eq $exe)
    {
        Write-Host 'XEFG handoff seam: RED-mode compilation failed'
        exit 1
    }
    $code = Invoke-Smoke $exe 'red'
    Write-Host "XEFG handoff seam: RED run exit=$code (expected 3)"
    if ($code -ne 3)
    {
        Write-Host "XEFG handoff seam: RED contract violation - smoke exited $code, expected 3"
        exit 1
    }
    $printed = Get-Content -LiteralPath (Join-Path $out 'red.stdout.txt') -Raw
    if ($printed -notmatch 'HANDOFF_NOT_IMPLEMENTED')
    {
        Write-Host 'XEFG handoff seam: RED marker HANDOFF_NOT_IMPLEMENTED missing from the smoke output'
        exit 1
    }
    Write-Host 'RED: HANDOFF_NOT_IMPLEMENTED - the seam is red until rows 9/10 implement the handoff core'
    exit 3
}

# The production core exists: prove the seam's API contract compiles and links
# before running the scenarios against it.
$probe = Join-Path $out 'xefg_handoff_probe.cpp'
@'
#include <dlssnr/DlssNr_XeFGHandoff.h>
int main()
{
    DlssNr::XeFGHandoff::Tracker tracker;
    tracker.Reset(1);
    const auto serial = tracker.OpenInterval(1);
    tracker.Submitted(serial);
    tracker.Ready(serial);
    DlssNr::XeFGHandoff::Identity identity {};
    identity.generation = 1;
    identity.frameId = 1;
    const auto outcome = tracker.Handoff(identity);
    return outcome.applied ? 0 : 1;
}
'@ | Set-Content -LiteralPath $probe -Encoding ascii

$probeExe = Compile-Smoke @("/I$repo/OptiScaler") @($probe) 'xefg_handoff_probe'
if ($null -eq $probeExe)
{
    Write-Host 'XEFG handoff seam: the production core exists but the seam API contract does not compile.'
    exit 1
}

$sources = @($smoke)
if (Test-Path -LiteralPath $coreSource) { $sources += $coreSource }
$exe = Compile-Smoke @('/DNR_XEFG_HANDOFF_PRESENT=1', "/I$repo/OptiScaler") $sources 'xefg_handoff_smoke'
if ($null -eq $exe)
{
    Write-Host 'XEFG handoff seam: GREEN-mode compilation failed'
    exit 1
}

$code = Invoke-Smoke $exe 'green'
Write-Host "XEFG handoff seam: GREEN run exit=$code"
if ($code -ne 0)
{
    exit $code
}

# NOTE (main port): the #106 wiring-double mirror (tests/xefg_wiring_regression.cpp) is
# intentionally not ported - it proves its own copy of the wiring, not the production
# sources. Present-site wiring is pinned instead by
# tests/run_xefg_production_route_smoke.ps1, which drives the actual Tracker headers and
# asserts the actual XeFG_Dx12::Present / FGHooks::FGPresent bodies with named failure seeds.

exit 0
