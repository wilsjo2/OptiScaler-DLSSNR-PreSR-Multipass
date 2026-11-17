# Package OptiScaler and its ordinary dependencies, including the built-in NR backend.
# NVIDIA model/FG runtimes and unrelated optional payloads are never collected from build folders.
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$Version = 'nr-dev',
    [switch]$SkipBuild,
    [switch]$EnableRtx40Mfg,
    [string]$Nvfp4Assets = "$PSScriptRoot/standalone/NrCapture/out-model-map/nvfp4-20261113/v2/assets/ship-public"
)

$ErrorActionPreference = 'Stop'
$Nvfp4Assets = [IO.Path]::GetFullPath($Nvfp4Assets)
$root = Split-Path -Parent $PSCommandPath
$stage = Join-Path $root "release/$Version"
$zip = Join-Path $root "release/OptiScaler-NR-$Version.zip"
if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)) {
    throw 'Release output already exists. Choose a new -Version; existing packages are not overwritten.'
}

if (-not $SkipBuild) {
    $msbuild = (Get-Command MSBuild.exe -ErrorAction SilentlyContinue).Source
    if (-not $msbuild) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (Test-Path -LiteralPath $vswhere) {
            $msbuild = & $vswhere -latest -products '*' -requires Microsoft.Component.MSBuild -find 'MSBuild/Current/Bin/MSBuild.exe'
        }
    }
    if (-not $msbuild) { throw 'MSBuild.exe was not found. Use a Visual Studio developer PowerShell.' }
    & $msbuild (Join-Path $root 'OptiScaler.sln') /p:Configuration=Release /p:Platform=x64 /p:PostBuildEventUseInBuild=false "/p:OptiScalerRtx40Mfg=$($EnableRtx40Mfg.IsPresent.ToString().ToLowerInvariant())" /v:minimal /m
    if ($LASTEXITCODE -ne 0) { throw 'OptiScaler build failed.' }
}

$buildFolder = if ($EnableRtx40Mfg) { 'x64/Release-RTX40-MFG' } else { 'x64/Release' }
$buildRoot = Join-Path $root $buildFolder
# Also reject a stale/wrong-flavour DLL when using -SkipBuild.
$dllText = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes((Join-Path $buildRoot 'OptiScaler.dll')))
$hasUnlock = $dllText.Contains('RTX 40 MFG unlock (restart)')
if ($hasUnlock -ne $EnableRtx40Mfg.IsPresent) { throw 'DLL RTX 40 MFG feature does not match the requested package.' }
# Validate every source before creating the staging tree. An explicit manifest prevents stale
# Streamline/MFG, removed NR helpers or discarded experiment files entering this package.
$files = @{}
$files['OptiScaler.dll'] = Join-Path $buildRoot 'OptiScaler.dll'
foreach ($name in @('OptiScaler.ini', 'setup_windows.bat', 'setup_linux.sh')) {
    $files[$name] = Join-Path $root $name
}
foreach ($name in @('libxess.dll', 'libxess_dx11.dll', 'libxell.dll', 'libxess_fg.dll')) {
    $files["OptiScaler/$name"] = Join-Path $root "external/xess/bin/$name"
}
$files['OptiScaler/amd_fidelityfx_vk.dll'] = Join-Path $root 'external/FidelityFX-SDK/PrebuiltSignedDLL/amd_fidelityfx_vk.dll'
foreach ($name in @('amd_fidelityfx_loader_dx12.dll', 'amd_fidelityfx_upscaler_dx12.dll', 'amd_fidelityfx_framegeneration_dx12.dll')) {
    $files["OptiScaler/$name"] = Join-Path $root "external/FidelityFX-SDK-v2/Kits/FidelityFX/signedbin/$name"
}
$files['OptiScaler/D3D12_OptiScaler/D3D12Core.dll'] = Join-Path $root 'external/directx_agility_sdk/lib/D3D12Core.dll'
# The 4-bit (NVFP4) kernels and the two pass-in-one weight sets (Original - NVFP4, NVFP4 2xPass-in-one, NVFP4 3xPass-in-one).
foreach ($item in Get-ChildItem -LiteralPath $Nvfp4Assets -Recurse -File | Where-Object { $_.Name -notlike '*.md' }) {
    $files['OptiScaler/nvfp4/' + $item.FullName.Substring($Nvfp4Assets.Length).TrimStart('\', '/').Replace('\', '/')] = $item.FullName
}
# No documents or licence/attribution files in the package.
foreach ($key in @($files.Keys)) { if ($key -match '(?i)\.md$|licen[cs]e|attribution') { throw "Document in the release manifest: $key" } }
foreach ($entry in $files.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value -PathType Leaf)) {
        throw "Required release file is missing: $($entry.Value)"
    }
}

$ini = Get-Content -LiteralPath $files['OptiScaler.ini'] -Raw
if ($ini -match '(?mi)^Enabled=true\s*$') { throw 'A feature is enabled in the default INI.' }
foreach ($key in @('FinishedPicture', 'DeferredDLSS', 'UnlockPasses', 'AdaMfgUnlock', 'AdaFlipMeteringPatch', 'SpatialCompression')) {
    if ($ini -match "(?mi)^$key=true\s*$") { throw "Experimental option $key is enabled in the default INI." }
}
if ($ini -notmatch '(?mi)^TargetProcessName=auto\s*$') { throw 'The INI contains a game-specific process filter.' }

New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($entry in $files.GetEnumerator()) {
    $destination = Join-Path $stage $entry.Key
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $entry.Value -Destination $destination
}
if (-not $EnableRtx40Mfg) {
    $ini = $ini -replace '(?m)^; Experimental built-in RTX 40 MFG unlock[^\r\n]*\r?\n', ''
    $ini = $ini -replace '(?m)^AdaMfgUnlock=[^\r\n]*\r?\n', ''
    $ini = $ini -replace '(?ms)^; Frame timing fix for the extra frames.*?^AdaFlipMeteringPatch=[^\r\n]*\r?\n', ''
    [IO.File]::WriteAllText((Join-Path $stage 'OptiScaler.ini'), $ini, [Text.UTF8Encoding]::new($false))
}
[IO.File]::WriteAllText((Join-Path $stage '!! EXTRACT ALL FILES TO GAME FOLDER !!'), '')

$checksums = Get-ChildItem -LiteralPath $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.TrimEnd('\', '/').Length).TrimStart('\', '/').Replace('\', '/')
    '{0} *{1}' -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash, $relative
}
[IO.File]::WriteAllLines((Join-Path $stage 'SHA256SUMS.txt'), $checksums, [Text.UTF8Encoding]::new($false))
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Output "Created $zip"
