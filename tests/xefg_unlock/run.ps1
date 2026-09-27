# ============================================================================
# tests/xefg_unlock/run.ps1 - opt-in native XeFG unlock + pacing pin.
#
# Run from a Visual Studio x64 developer shell (cl.exe on PATH). No GPU, no
# game, no provider DLL: tests/xefg_unlock/UnlockPacingTests.cpp compiles the
# ACTUAL headers (OptiScaler/proxies/XeFGUnlock.h, OptiScaler/proxies/XeFGPacing.h
# - stub include dir only shadows SysUtils.h/Logger.h/Config.h) and drives them
# against a synthetic PE image: config OFF, recognised build, bad build stamp
# refusal, partial-failure rollback, count-1 skip, pacing OFF/ON, corrupt-thunk
# refusal. It also asserts, function-bounded, the ACTUAL production wiring
# (HookXeFG Apply-before-exports, Apply->Install handover, Dispatch pacing
# feed) and the production Config.h/ini/menu defaults and limits.
#
# Sensitivity: the runner re-runs the same binary against disposable copies with
# exactly one defect each - the XeFGUnlock::Apply call removed from XeFG_Proxy.h,
# the XeFGPacing::Install call removed from XeFGUnlock.h, the RenderTimeMs feed
# removed from XeFG_Dx12.cpp. Each seed MUST exit nonzero with its named marker;
# a seed that exits 0 means the pin no longer detects its defect.
#
# Failure control flow: every failure below throws; the catch records a nonzero
# process exit and the finally always cleans the temp dir, so a logged failure
# can never surface as process exit 0. The final `exit $exitCode` is always
# reached because error branches throw instead of returning out of the try.
#
# Exit codes:
#   0 = GREEN  - every pin held and every seed failed named, temp cleaned
#   1 = a pin failed, a seed went green, or the compile failed
#   2 = the environment is broken (cl.exe missing, sources absent, seed missed,
#       pin timeout, temp cleanup failed)
# ============================================================================
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    Write-Host 'run_xefg_unlock: cl.exe is not on PATH - run from a Visual Studio x64 developer shell.'
    exit 2
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$proxy = Join-Path $repo 'OptiScaler/proxies/XeFG_Proxy.h'
$unlock = Join-Path $repo 'OptiScaler/proxies/XeFGUnlock.h'
$pacing = Join-Path $repo 'OptiScaler/proxies/XeFGPacing.h'
$dx12 = Join-Path $repo 'OptiScaler/framegen/xefg/XeFG_Dx12.cpp'
$configH = Join-Path $repo 'OptiScaler/Config.h'
$ini = Join-Path $repo 'OptiScaler.ini'
$menu = Join-Path $repo 'OptiScaler/menu/menu_common.cpp'
$test = Join-Path $PSScriptRoot 'UnlockPacingTests.cpp'
$stubs = Join-Path $PSScriptRoot 'stubs'
foreach ($required in @($proxy, $unlock, $pacing, $dx12, $configH, $ini, $menu, $test, $stubs))
{
    if (-not (Test-Path -LiteralPath $required))
    {
        Write-Host "run_xefg_unlock: required source absent: $required"
        exit 2
    }
}

$out = Join-Path ([IO.Path]::GetTempPath()) ('xefg-unlock-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
$exitCode = 0

# Runs the pin with a bounded process timeout (120 s). The pin performs no
# waits; a hang here means a wait was added. Returns the exit code; a timeout
# throws so the outer catch yields process exit 2 after cleanup.
function Invoke-Pin([string]$exe, [string[]]$sources, [string]$tag)
{
    $stdout = Join-Path $out "$tag.stdout.txt"
    $stderr = Join-Path $out "$tag.stderr.txt"
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    $info.Arguments = ($sources | ForEach-Object { '"' + $_ + '"' }) -join ' '
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
        $script:exitCode = 2
        throw "$tag timed out after 120s - a wait was observed"
    }
    $text -split "`n" | ForEach-Object { Write-Host $_ }
    if ($errText.Length -gt 0) { $errText -split "`n" | ForEach-Object { Write-Host $_ } }
    return $process.ExitCode
}

try
{
    Write-Host 'XEFG unlock/pacing pin: compiling the actual headers and pin'
    $log = Join-Path $out 'pin.compile.log'
    $exe = Join-Path $out 'UnlockPacingTests.exe'
    & cl.exe /nologo /std:c++20 /EHsc /W4 /DNOMINMAX "/I$stubs" "/I$repo/OptiScaler" "/Fo$out/" "/Fe$exe" $test > $log 2>&1
    $code = $LASTEXITCODE
    Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
    if ($code -ne 0)
    {
        $exitCode = 1
        throw "XEFG unlock/pacing pin: cl.exe exit=$code"
    }

    $realSources = @($proxy, $unlock, $dx12, $configH, $ini, $menu)
    Write-Host 'XEFG unlock/pacing pin: GREEN run against the actual production sources'
    $green = Invoke-Pin $exe $realSources 'green'
    Write-Host "XEFG unlock/pacing pin: GREEN run exit=$green"
    if ($green -ne 0)
    {
        $exitCode = $green
        throw "XEFG unlock/pacing pin: GREEN run failed with exit=$green"
    }

    # Seed A: the HookXeFG Apply call removed. The disposable copy must fail named.
    $seedA = Join-Path $out 'XeFG_Proxy.seedA.h'
    $text = Get-Content -LiteralPath $proxy -Raw
    $anchorA = 'XeFGUnlock::Apply(_dll);'
    $hits = ([regex]::Matches($text, [regex]::Escape($anchorA))).Count
    if ($hits -ne 1)
    {
        $exitCode = 2
        throw "XEFG unlock/pacing pin: Apply seed anchor found $hits times, expected exactly 1"
    }
    $text = $text.Replace($anchorA, '/* SEED: Apply removed */')
    Set-Content -LiteralPath $seedA -Value $text -NoNewline
    $seedACode = Invoke-Pin $exe @($seedA, $unlock, $dx12, $configH, $ini, $menu) 'seed_apply'
    Write-Host "XEFG unlock/pacing pin: seed Apply run exit=$seedACode (expected nonzero PIN_APPLY_ORDER)"
    if ($seedACode -eq 0)
    {
        $exitCode = 1
        throw 'XEFG unlock/pacing pin: Apply seed went green - the pin is not failure-provable'
    }
    $printed = Get-Content -LiteralPath (Join-Path $out 'seed_apply.stdout.txt') -Raw
    if ($printed -notmatch 'PIN_APPLY_ORDER')
    {
        $exitCode = 1
        throw 'XEFG unlock/pacing pin: Apply seed missed its named marker PIN_APPLY_ORDER'
    }

    # Seed B: the Apply->Install handover removed. The disposable copy must fail named.
    $seedB = Join-Path $out 'XeFGUnlock.seedB.h'
    $text = Get-Content -LiteralPath $unlock -Raw
    $anchorB = 'XeFGPacing::Install(base);'
    $hits = ([regex]::Matches($text, [regex]::Escape($anchorB))).Count
    if ($hits -ne 1)
    {
        $exitCode = 2
        throw "XEFG unlock/pacing pin: Install seed anchor found $hits times, expected exactly 1"
    }
    $text = $text.Replace($anchorB, '/* SEED: Install removed */')
    Set-Content -LiteralPath $seedB -Value $text -NoNewline
    $seedBCode = Invoke-Pin $exe @($proxy, $seedB, $dx12, $configH, $ini, $menu) 'seed_install'
    Write-Host "XEFG unlock/pacing pin: seed Install run exit=$seedBCode (expected nonzero PIN_UNLOCK_PACING)"
    if ($seedBCode -eq 0)
    {
        $exitCode = 1
        throw 'XEFG unlock/pacing pin: Install seed went green - the pin is not failure-provable'
    }
    $printed = Get-Content -LiteralPath (Join-Path $out 'seed_install.stdout.txt') -Raw
    if ($printed -notmatch 'PIN_UNLOCK_PACING')
    {
        $exitCode = 1
        throw 'XEFG unlock/pacing pin: Install seed missed its named marker PIN_UNLOCK_PACING'
    }

    # Seed C: the Dispatch pacing feed removed. The disposable copy must fail named.
    $seedC = Join-Path $out 'XeFG_Dx12.seedC.cpp'
    $text = Get-Content -LiteralPath $dx12 -Raw
    $anchorC = 'frameRenderTime = XeFGPacing::RenderTimeMs();'
    $hits = ([regex]::Matches($text, [regex]::Escape($anchorC))).Count
    if ($hits -ne 1)
    {
        $exitCode = 2
        throw "XEFG unlock/pacing pin: pacing-feed seed anchor found $hits times, expected exactly 1"
    }
    $text = $text.Replace($anchorC, '/* SEED: pacing feed removed */')
    Set-Content -LiteralPath $seedC -Value $text -NoNewline
    $seedCCode = Invoke-Pin $exe @($proxy, $unlock, $seedC, $configH, $ini, $menu) 'seed_pacing'
    Write-Host "XEFG unlock/pacing pin: seed pacing run exit=$seedCCode (expected nonzero PIN_DX12_PACING)"
    if ($seedCCode -eq 0)
    {
        $exitCode = 1
        throw 'XEFG unlock/pacing pin: pacing seed went green - the pin is not failure-provable'
    }
    $printed = Get-Content -LiteralPath (Join-Path $out 'seed_pacing.stdout.txt') -Raw
    if ($printed -notmatch 'PIN_DX12_PACING')
    {
        $exitCode = 1
        throw 'XEFG unlock/pacing pin: pacing seed missed its named marker PIN_DX12_PACING'
    }

    Write-Host 'XEFG unlock/pacing pin: GREEN - every pin held and every seed failed named.'
}
catch
{
    Write-Host "run_xefg_unlock: FAILED: $($_.Exception.Message)"
    if ($exitCode -eq 0) { $exitCode = 1 }
}
finally
{
    Remove-Item -LiteralPath $out -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $out)
    {
        Write-Host "run_xefg_unlock: temp cleanup failed: $out"
        if ($exitCode -eq 0) { $exitCode = 2 }
    }
    else
    {
        Write-Host 'run_xefg_unlock: TEMP_CLEAN - task-owned temp resources absent.'
    }
}

exit $exitCode
