# ============================================================================
# tests/run_fgpresent_owner_routing.ps1 - FGPresent incoming-owner routing regression.
#
# Run from a Visual Studio x64 developer shell (cl.exe on PATH). No GPU, no
# game, no window: tests/fgpresent_owner_routing.cpp supplies a stub world
# (State/Config/IFGFeature-double/spies) and this runner splices the ACTUAL
# FGHooks::FGPresent body text (brace-matched extraction from
# OptiScaler/hooks/FG_Hooks.cpp) into it at build time, then compiles. Fake
# swapchain pointers (the body only compares/forwards This) and an active
# IFGFeature double prove the routing invariant:
#   stale incoming chain A -> original Present/Present1 once, HRESULT kept,
#     live feature 0, generic NR 0 (PIN_FG_NONOWNER / PIN_FG_NONOWNER1);
#   registered chain B, native and Dx11wDx12 bridge -> feature once,
#     native generic bypass and bridge generic routing (PIN_FG_OWNER_NATIVE / PIN_FG_OWNER_BRIDGE).
#
# Sensitivity: the runner rebuilds the same harness against a disposable body
# with the non-owner guard removed (must fail PIN_FG_NONOWNER). A seed that
# exits 0 means the regression no longer guards the boundary. This is a
# routing test, not a GPU composition proof; interop/stale simulation beyond
# the caller-identity boundary is out of scope (see provenance row 3b).
#
# Failure control flow: every failure below throws; the catch records a nonzero
# process exit and the finally always cleans the temp dir. The final
# `exit $exitCode` is always reached because error branches throw instead of
# returning out of the try.
#
# Exit codes:
#   0 = GREEN - every pin held and the seed failed named, temp cleaned
#   1 = a pin failed, the seed went green, or the compile failed
#   2 = the environment is broken (cl.exe missing, sources absent, seed missed,
#       pin timeout, temp cleanup failed)
# ============================================================================
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue))
{
    Write-Host 'run_fgpresent_owner_routing: cl.exe is not on PATH - run from a Visual Studio x64 developer shell.'
    exit 2
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$hook = Join-Path $repo 'OptiScaler/hooks/FG_Hooks.cpp'
$template = Join-Path $PSScriptRoot 'fgpresent_owner_routing.cpp'
foreach ($required in @($hook, $template))
{
    if (-not (Test-Path -LiteralPath $required))
    {
        Write-Host "run_fgpresent_owner_routing: required source absent: $required"
        exit 2
    }
}

$out = Join-Path ([IO.Path]::GetTempPath()) ('fgpresent-owner-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $out | Out-Null
$exitCode = 0

# Extracts the inner body of the function whose definition starts at $signature.
# Strings, character literals and comments are skipped so braces inside them
# cannot corrupt the match. Throws when not found.
function Get-FunctionBody([string]$source, [string]$signature)
{
    $at = $source.IndexOf($signature, [StringComparison]::Ordinal)
    if ($at -lt 0) { throw "function signature not found: $signature" }
    $open = $source.IndexOf('{', $at)
    if ($open -lt 0) { throw "function body open brace not found: $signature" }
    $depth = 1
    $lineComment = $false; $blockComment = $false; $str = $false; $ch = $false
    for ($i = $open + 1; $i -lt $source.Length; $i++)
    {
        $c = $source[$i]
        $next = if ($i + 1 -lt $source.Length) { $source[$i + 1] } else { [char] 0 }
        if ($lineComment) { if ($c -eq "`n") { $lineComment = $false }; continue }
        if ($blockComment)
        {
            if ($c -eq '*' -and $next -eq '/') { $blockComment = $false; $i++ }
            continue
        }
        if ($str)
        {
            if ($c -eq '\') { $i++ } elseif ($c -eq '"') { $str = $false }
            continue
        }
        if ($ch)
        {
            if ($c -eq '\') { $i++ } elseif ($c -eq "'") { $ch = $false }
            continue
        }
        if ($c -eq '/' -and $next -eq '/') { $lineComment = $true }
        elseif ($c -eq '/' -and $next -eq '*') { $blockComment = $true }
        elseif ($c -eq '"') { $str = $true }
        elseif ($c -eq "'") { $ch = $true }
        elseif ($c -eq '{') { $depth++ }
        elseif ($c -eq '}')
        {
            $depth--
            if ($depth -eq 0) { return $source.Substring($open + 1, $i - $open - 1) }
        }
    }
    throw "function body close brace not found: $signature"
}

# Runs the test exe with a bounded process timeout (120 s). Returns exit code;
# a timeout throws so the outer catch yields process exit 2 after cleanup.
function Invoke-Routing([string]$exe, [string]$tag)
{
    $stdout = Join-Path $out "$tag.stdout.txt"
    $stderr = Join-Path $out "$tag.stderr.txt"
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    $info.Arguments = ''
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::Start($info)
    $outTask = $process.StandardOutput.ReadToEndAsync()
    $errTask = $process.StandardError.ReadToEndAsync()
    $exited = $process.WaitForExit(120000)
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

function Build-Harness([string]$body, [string]$tag)
{
    $tu = Join-Path $out "$tag.tu.cpp"
    $tpl = Get-Content -LiteralPath $template -Raw
    if (-not $tpl.Contains('/*__FG_PRESENT_BODY__*/'))
    {
        $script:exitCode = 2
        throw "template splice anchor missing in $template"
    }
    Set-Content -LiteralPath $tu -Value $tpl.Replace('/*__FG_PRESENT_BODY__*/', $body) -NoNewline
    $log = Join-Path $out "$tag.compile.log"
    $exe = Join-Path $out "$tag.exe"
    & cl.exe /nologo /std:c++20 /EHsc /W4 /DNOMINMAX "/Fo$out/$tag.obj" "/Fe$exe" $tu > $log 2>&1
    $code = $LASTEXITCODE
    Get-Content -LiteralPath $log | ForEach-Object { Write-Host $_ }
    if ($code -ne 0)
    {
        $script:exitCode = 1
        throw "$tag harness: cl.exe exit=$code"
    }
    return $exe
}

try
{
    $source = Get-Content -LiteralPath $hook -Raw
    $body = Get-FunctionBody $source 'FGHooks::FGPresent('
    if ([string]::IsNullOrWhiteSpace($body))
    {
        $exitCode = 2
        throw 'extracted FGPresent body is empty'
    }
    if ($body -notmatch 'nonOwnerXeFGPresent')
    {
        $exitCode = 2
        throw 'extracted body lacks the non-owner guard anchor (wrong function or stale source?)'
    }

    Write-Host 'FGPresent owner routing: building harness against the ACTUAL production body'
    $exe = Build-Harness $body 'green'

    Write-Host 'FGPresent owner routing: GREEN run (stale refused, owner + bridge admitted)'
    $green = Invoke-Routing $exe 'green'
    Write-Host "FGPresent owner routing: GREEN run exit=$green"
    if ($green -ne 0)
    {
        $exitCode = $green
        throw "FGPresent owner routing: GREEN run failed with exit=$green"
    }

    # Seed: the non-owner guard removed. The harness must fail named.
    $anchor = 'if (state.isShuttingDown || nonOwnerXeFGPresent)'
    $hits = ([regex]::Matches($body, [regex]::Escape($anchor))).Count
    if ($hits -ne 1)
    {
        $exitCode = 2
        throw "guard seed anchor found $hits times, expected exactly 1"
    }
    $seedBody = $body.Replace($anchor, 'if (state.isShuttingDown) /* SEED: owner guard removed */')
    $seedExe = Build-Harness $seedBody 'seed_guard'
    $seedCode = Invoke-Routing $seedExe 'seed_guard'
    Write-Host "FGPresent owner routing: seed guard run exit=$seedCode (expected nonzero PIN_FG_NONOWNER)"
    if ($seedCode -eq 0)
    {
        $exitCode = 1
        throw 'FGPresent owner routing: guard seed went green - the regression is not failure-provable'
    }
    $printed = Get-Content -LiteralPath (Join-Path $out 'seed_guard.stdout.txt') -Raw
    if ($printed -notmatch 'PIN_FG_NONOWNER')
    {
        $exitCode = 1
        throw 'FGPresent owner routing: guard seed missed its named marker PIN_FG_NONOWNER'
    }

    Write-Host 'FGPresent owner routing: GREEN - every pin held and the seed failed named.'
}
catch
{
    Write-Host "run_fgpresent_owner_routing: FAILED: $($_.Exception.Message)"
    if ($exitCode -eq 0) { $exitCode = 1 }
}
finally
{
    Remove-Item -LiteralPath $out -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $out)
    {
        Write-Host "run_fgpresent_owner_routing: temp cleanup failed: $out"
        if ($exitCode -eq 0) { $exitCode = 2 }
    }
    else
    {
        Write-Host 'run_fgpresent_owner_routing: TEMP_CLEAN - task-owned temp resources absent.'
    }
}

exit $exitCode
