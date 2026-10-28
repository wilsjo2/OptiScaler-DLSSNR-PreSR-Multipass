param(
    [Parameter(Mandatory)][string]$Destination,
    [string]$Runtime,
    [string]$DlssRuntime = "$PSScriptRoot/runtime/nvngx_dlss.dll",
    [string]$FrameGenerationRuntime = "$PSScriptRoot/runtime/streamline",
    [string]$BuildDirectory = "$PSScriptRoot/out"
)
$ErrorActionPreference = 'Stop'
if (!$DlssRuntime -or !(Test-Path -LiteralPath $DlssRuntime -PathType Leaf)) {
    throw 'DLSS SR is required in standalone packages. Place nvngx_dlss.dll in runtime/ or supply -DlssRuntime.'
}
$fgFiles = @('sl.interposer.dll', 'sl.common.dll', 'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlssg.dll',
    'nvngx_dlss.license.txt', 'reflex.license.txt', 'streamline-license.txt')
foreach ($file in $fgFiles) {
    if (!(Test-Path -LiteralPath (Join-Path $FrameGenerationRuntime $file) -PathType Leaf)) {
        throw "Missing frame-generation distribution file: $file"
    }
}
$binary = Join-Path $BuildDirectory 'Release/display_filter.exe'
if (!(Test-Path -LiteralPath $binary)) { throw 'Run build.ps1 first.' }
$target = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $target) {
    if (!(Test-Path -LiteralPath (Join-Path $target 'external-nr-package.json'))) {
        throw 'Destination exists and is not an external NR package. Choose an empty destination.'
    }
}
New-Item -ItemType Directory -Path $target -Force | Out-Null
Copy-Item -LiteralPath $binary -Destination (Join-Path $target 'display_filter.exe') -Force
foreach ($file in @('README.md', 'VALIDATION.md', 'ALIGNMENT.md', 'PARITY.md', 'FRAMEGEN.md', 'MFG.md')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination (Join-Path $target $file) -Force
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../LICENSE') -Destination (Join-Path $target 'LICENSE') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Licenses/MFGUnlock_LICENSE.txt') -Destination (Join-Path $target 'MFGUnlock_LICENSE.txt') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../Licenses/RenoDX_ATTRIBUTION.txt') -Destination (Join-Path $target 'RenoDX_ATTRIBUTION.txt') -Force
$runtimeHash = $null
if ($Runtime) {
    $runtimeDirectory = Join-Path $target 'runtime'
    New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
    $destinationRuntime = Join-Path $runtimeDirectory 'nvngx_dlssnr.dll'
    Copy-Item -LiteralPath $Runtime -Destination $destinationRuntime -Force
    $runtimeHash = (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash
    if ((Get-FileHash -LiteralPath $destinationRuntime -Algorithm SHA256).Hash -ne $runtimeHash) {
        throw 'Packaged runtime hash mismatch.'
    }
}
$dlssHash = $null
if ($DlssRuntime) {
    $signature = Get-AuthenticodeSignature -LiteralPath $DlssRuntime
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
        throw 'Only an original NVIDIA-signed DLSS SR runtime may be packaged.'
    }
    $runtimeDirectory = Join-Path $target 'runtime'
    New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
    $destinationDlss = Join-Path $runtimeDirectory 'nvngx_dlss.dll'
    Copy-Item -LiteralPath $DlssRuntime -Destination $destinationDlss -Force
    $dlssHash = (Get-FileHash -LiteralPath $DlssRuntime -Algorithm SHA256).Hash
    if ((Get-FileHash -LiteralPath $destinationDlss -Algorithm SHA256).Hash -ne $dlssHash) {
        throw 'Packaged DLSS runtime hash mismatch.'
    }
}
$fgDestination = Join-Path $target 'runtime/streamline'
New-Item -ItemType Directory -Path $fgDestination -Force | Out-Null
$fgHashes = [ordered]@{}
foreach ($file in $fgFiles) {
    $source = Join-Path $FrameGenerationRuntime $file
    $copied = Join-Path $fgDestination $file
    Copy-Item -LiteralPath $source -Destination $copied -Force
    $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "FG runtime hash mismatch: $file" }
    $fgHashes[$file] = $hash
}
[ordered]@{
    application = 'Display Filter screen and window overlay'
    created = (Get-Date).ToString('o')
    executableSha256 = (Get-FileHash -LiteralPath (Join-Path $target 'display_filter.exe') -Algorithm SHA256).Hash
    runtimeSha256 = $runtimeHash
    dlssRuntimeSha256 = $dlssHash
    frameGenerationRuntimeSha256 = $fgHashes
    gameFilesChanged = $false
    multiplayerVerified = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'external-nr-package.json') -Encoding utf8
Write-Output $target
