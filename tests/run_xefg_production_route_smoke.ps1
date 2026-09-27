# ============================================================================
# tests/run_xefg_production_route_smoke.ps1 - production-route pin for the XeFG
# finished-picture NR handoff.
#
# Run from a Visual Studio x64 developer shell (cl.exe on PATH). No GPU, no
# game, no NR runtime: tests/xefg_handoff_wiring_pin.cpp compiles the ACTUAL
# decision headers (DlssNr_XeFGHandoff.h, DlssNr_FinishedConsumer.h,
# DlssNr_FinishedReady.h) and asserts, function-bounded, the ACTUAL production
# wiring in OptiScaler/framegen/xefg/XeFG_Dx12.cpp (Present:
# PublishFinishedConsumerState(consumes)->OwnedNrHandoff) and
# OptiScaler/hooks/FG_Hooks.cpp (FGPresent: generic NR bypass + report).
#
# Sensitivity: the runner re-runs the same binary against disposable copies with
# exactly one defect each - the OwnedNrHandoff() call site removed, the
# !xefgOwnedHandoff bypass removed, the Late unwrap preserve removed at the submit
# site. Each seed MUST exit nonzero with its named marker; a seed that exits 0 means
# the pin no longer detects its defect.
#
# Exit codes:
#   0 = GREEN  - every pin held and every seed failed named
#   1 = a pin failed, a seed went green, or the compile failed
#   2 = the environment is broken (cl.exe missing, sources absent, seed missed)
# ============================================================================
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    Write-Host 'run_xefg_production_route_smoke: cl.exe is not on PATH - run from a Visual Studio x64 developer shell.'
    exit 2
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$xefg = Join-Path $repo 'OptiScaler/framegen/xefg/XeFG_Dx12.cpp'
$fgHooks = Join-Path $repo 'OptiScaler/hooks/FG_Hooks.cpp'
$late = Join-Path $repo 'OptiScaler/shaders/dlssnr/DlssNr_Late.inl'
$pin = Join-Path $PSScriptRoot 'xefg_handoff_wiring_pin.cpp'
foreach ($required in @($xefg, $fgHooks, $late, $pin))
{
    if (-not (Test-Path -LiteralPath $required))
    {
        Write-Host "run_xefg_production_route_smoke: required source absent: $required"
        exit 2
    }
}

$out = Join-Path ([IO.Path]::GetTempPath()) ('xefg-route-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null

# Runs the pin with a bounded process timeout (120 s). The pin performs no
# waits; a hang here means a wait was added. Returns the exit code.
function Invoke-Pin([string]$exe, [string[]]$sources, [string]$tag)
{
    $stdout = Join-Path $out "$tag.stdout.txt"
    $stderr = Join-Path $out "$tag.stderr.txt"
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    $info.Arguments = '"' + $sources[0] + '" "' + $sources[1] + '" "' + $sources[2] + '"'
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
        Write-Host "$tag timed out after 120s - a wait was observed"
        exit 2
    }
    $text -split "`n" | ForEach-Object { Write-Host $_ }
    if ($errText.Length -gt 0) { $errText -split "`n" | ForEach-Object { Write-Host $_ } }
    return $process.ExitCode
}

Write-Host 'XEFG production route pin: compiling the actual Tracker and wiring pin'
$log = Join-Path $out 'pin.compile.log'
$exe = Join-Path $out 'xefg_handoff_wiring_pin.exe'
& cl.exe /nologo /std:c++20 /EHsc /W4 /DNOMINMAX "/I$repo/OptiScaler" "/Fo$out/" "/Fe$exe" $pin > $log 2>&1
$code = $LASTEXITCODE
Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
if ($code -ne 0)
{
    Write-Host "XEFG production route pin: cl.exe exit=$code"
    exit 1
}

Write-Host 'XEFG production route pin: GREEN run against the actual production sources'
$green = Invoke-Pin $exe @($xefg, $fgHooks, $late) 'green'
Write-Host "XEFG production route pin: GREEN run exit=$green"
if ($green -ne 0)
{
    exit $green
}

# Seed A: the Present call site removed. The disposable copy must fail named.
$seedA = Join-Path $out 'XeFG_Dx12.seedA.cpp'
$text = Get-Content -LiteralPath $xefg -Raw
$hits = ([regex]::Matches($text, '(?m)^[ \t]*OwnedNrHandoff\(\);[ \t]*\r?$')).Count
if ($hits -ne 1)
{
    Write-Host "XEFG production route pin: call-site seed anchor found $hits times, expected exactly 1"
    exit 2
}
$text = [regex]::Replace($text, '(?m)^[ \t]*OwnedNrHandoff\(\);[ \t]*\r?\n?', '')
Set-Content -LiteralPath $seedA -Value $text -NoNewline
$seedACode = Invoke-Pin $exe @($seedA, $fgHooks, $late) 'seed_callsite'
Write-Host "XEFG production route pin: seed call-site run exit=$seedACode (expected nonzero PIN_XEFG_CALLSITE)"
if ($seedACode -eq 0)
{
    Write-Host 'XEFG production route pin: call-site seed went green - the pin is not failure-provable'
    exit 1
}
$printed = Get-Content -LiteralPath (Join-Path $out 'seed_callsite.stdout.txt') -Raw
if ($printed -notmatch 'PIN_XEFG_CALLSITE')
{
    Write-Host 'XEFG production route pin: call-site seed missed its named marker PIN_XEFG_CALLSITE'
    exit 1
}

# Seed B: the FGHooks generic-NR bypass removed. The disposable copy must fail named.
$seedB = Join-Path $out 'FG_Hooks.seedB.cpp'
$text = Get-Content -LiteralPath $fgHooks -Raw
$hits = ([regex]::Matches($text, 'if \(\!xefgOwnedHandoff\)')).Count
if ($hits -ne 1)
{
    Write-Host "XEFG production route pin: bypass seed anchor found $hits times, expected exactly 1"
    exit 2
}
$text = $text.Replace('if (!xefgOwnedHandoff)', 'if (true) /* SEED: bypass removed */')
Set-Content -LiteralPath $seedB -Value $text -NoNewline
$seedBCode = Invoke-Pin $exe @($xefg, $seedB, $late) 'seed_bypass'
Write-Host "XEFG production route pin: seed bypass run exit=$seedBCode (expected nonzero PIN_FG_BYPASS)"
if ($seedBCode -eq 0)
{
    Write-Host 'XEFG production route pin: bypass seed went green - the pin is not failure-provable'
    exit 1
}
$printed = Get-Content -LiteralPath (Join-Path $out 'seed_bypass.stdout.txt') -Raw
if ($printed -notmatch 'PIN_FG_BYPASS')
{
    Write-Host 'XEFG production route pin: bypass seed missed its named marker PIN_FG_BYPASS'
    exit 1
}

# Seed C: the Late unwrap preserve removed at the submit site. The production
# helper call appears once per unwrap site; removing the submit-site copy must fail named.
$seedC = Join-Path $out 'DlssNr_Late.seedC.inl'
$text = Get-Content -LiteralPath $late -Raw
$anchor = 'realQueue = static_cast<ID3D12CommandQueue*>(DlssNr::UnwrapOrPreserveQueue(queue, unwrapped, realQueue));'
$hits = ([regex]::Matches($text, [regex]::Escape($anchor))).Count
if ($hits -ne 2)
{
    Write-Host "XEFG production route pin: unwrap seed anchor found $hits times, expected exactly 2 (submit + facts sites)"
    exit 2
}
$index = $text.IndexOf($anchor)
$text = $text.Remove($index, $anchor.Length)
Set-Content -LiteralPath $seedC -Value $text -NoNewline
$seedCCode = Invoke-Pin $exe @($xefg, $fgHooks, $seedC) 'seed_unwrap'
Write-Host "XEFG production route pin: seed unwrap run exit=$seedCCode (expected nonzero PIN_B1_QUEUE)"
if ($seedCCode -eq 0)
{
    Write-Host 'XEFG production route pin: unwrap seed went green - the pin is not failure-provable'
    exit 1
}
$printed = Get-Content -LiteralPath (Join-Path $out 'seed_unwrap.stdout.txt') -Raw
if ($printed -notmatch 'PIN_B1_QUEUE')
{
    Write-Host 'XEFG production route pin: unwrap seed missed its named marker PIN_B1_QUEUE'
    exit 1
}

Write-Host 'XEFG production route pin: GREEN - every pin held and every seed failed named.'
exit 0
