param(
    [Parameter(Mandatory)][string]$Destination,
    [string]$Runtime,
    [string]$DlssRuntime = "$PSScriptRoot/runtime/nvngx_dlss.dll",
    [string]$FrameGenerationRuntime = "$PSScriptRoot/runtime/streamline",
    [string]$Nvfp4Assets = "$PSScriptRoot/runtime/nvfp4",
    [string]$BuildDirectory = "$PSScriptRoot/out"
)
$ErrorActionPreference = 'Stop'
# Copies unless the destination already holds the same bytes (a running Display Filter keeps its DLLs locked).
function Copy-Same([string]$source, [string]$copied) {
    if ((Test-Path -LiteralPath $copied -PathType Leaf) -and (Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash) { return }
    Copy-Item -LiteralPath $source -Destination $copied -Force
}
if (!$DlssRuntime -or !(Test-Path -LiteralPath $DlssRuntime -PathType Leaf)) {
    throw 'DLSS SR is required in standalone packages. Place nvngx_dlss.dll in runtime/ or supply -DlssRuntime.'
}
$fgFiles = @('sl.interposer.dll', 'sl.common.dll', 'sl.dlss_g.dll', 'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlssg.dll')
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
# The driver Smooth Motion helper the Pacing and latency tab names (tools/SmoothMotionProfile.cpp), when it was built.
$tool = Join-Path $BuildDirectory 'Release/smooth_motion_profile.exe'
if (Test-Path -LiteralPath $tool) { Copy-Item -LiteralPath $tool -Destination (Join-Path $target 'smooth_motion_profile.exe') -Force }
# No documents or licence/attribution files go into the package; ones an earlier build left are removed.
foreach ($item in @(Get-ChildItem -LiteralPath $target -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like '*.md' -or $_.Name -eq 'LICENSE' -or $_.Name -like '*license*.txt' -or $_.Name -like '*attribution*.txt' })) {
    Remove-Item -LiteralPath $item.FullName -Force
}
$runtimeHash = $null
if ($Runtime) {
    $runtimeDirectory = Join-Path $target 'runtime'
    New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
    $destinationRuntime = Join-Path $runtimeDirectory 'nvngx_dlssnr.dll'
    Copy-Same $Runtime $destinationRuntime
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
    Copy-Same $DlssRuntime $destinationDlss
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
    Copy-Same $source $copied
    $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "FG runtime hash mismatch: $file" }
    $fgHashes[$file] = $hash
}
# Native NR kernels and model weights when packaging the model pack.
$nvfp4Hashes = $null
if ($Nvfp4Assets -and (Test-Path -LiteralPath (Join-Path $Nvfp4Assets 'ffn_v2.cubin') -PathType Leaf)) {
    $nvfp4Hashes = [ordered]@{}
    foreach ($required in 'host/modelA_fp8.plan', 'host/amp_helpers_base.sm86.cubin', 'host4/modelA_fp8.plan', 'host4/fused_manifest.txt', 'host4/mixed/fused_manifest.txt', 'host4/mixed_set1.bin') {
        if (!(Test-Path -LiteralPath (Join-Path $Nvfp4Assets $required) -PathType Leaf)) {
            throw "Missing $required in the Display Filter model pack."
        }
    }
    # Model A's first set (modelA/ffn_v2.bin) and Model B's files are optional: a folder may leave them out.
    foreach ($stale in @('modelA/ffn_v2.bin', 'modelB/ffn_v2.bin', 'modelB/ffn_v2.txt')) {
        $old = Join-Path (Join-Path $target 'nvfp4') $stale
        if (!(Test-Path -LiteralPath (Join-Path $Nvfp4Assets $stale) -PathType Leaf) -and (Test-Path -LiteralPath $old -PathType Leaf)) { Remove-Item -LiteralPath $old -Force }
    }
    foreach ($file in @('ffn_v2.cubin', 'modelA/ffn_v2.bin', 'modelA/ffn_v2.txt', 'modelB/ffn_v2.bin', 'modelB/ffn_v2.txt')) {
        $source = Join-Path $Nvfp4Assets $file
        if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
            if ($file -in @('ffn_v2.cubin', 'modelA/ffn_v2.txt')) { throw "Missing NVFP4 asset: $file" }
            continue
        }
        $copied = Join-Path (Join-Path $target 'nvfp4') $file
        New-Item -ItemType Directory -Path (Split-Path -Parent $copied) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $copied -Force
        $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "NVFP4 asset hash mismatch: $file" }
        $nvfp4Hashes[$file] = $hash
    }
    # Optional second weight set of a model: trained 4-bit codes in the layout of ffn_v2.bin.
    foreach ($file in @('modelA/ffn_v2_trained.bin', 'modelB/ffn_v2_trained.bin')) {
        $source = Join-Path $Nvfp4Assets $file
        $copied = Join-Path (Join-Path $target 'nvfp4') $file
        if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
            if (Test-Path -LiteralPath $copied -PathType Leaf) { Remove-Item -LiteralPath $copied -Force }
            continue
        }
        Copy-Item -LiteralPath $source -Destination $copied -Force
        $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "NVFP4 asset hash mismatch: $file" }
        $nvfp4Hashes[$file] = $hash
    }
    # Optional: a second feed-forward kernel for small frames, named by a model's manifest
    # ("small_direct ... file <cubin>" / "small_contract ... file <cubin>", beside the manifest).
    foreach ($model in @('modelA', 'modelB')) {
        if (!(Test-Path -LiteralPath (Join-Path $Nvfp4Assets "$model/ffn_v2.txt") -PathType Leaf)) { continue }
        $kept = @()
        foreach ($line in Get-Content -LiteralPath (Join-Path $Nvfp4Assets "$model/ffn_v2.txt")) {
            if ($line -match '^\s*small_(direct|contract)\b.*\bfile\s+(\S+)') { $kept += $Matches[2] }
        }
        foreach ($item in @(Get-ChildItem -Path (Join-Path $target "nvfp4/$model") -Filter '*.cubin' -File -ErrorAction SilentlyContinue)) {
            if ($kept -notcontains $item.Name) { Remove-Item -LiteralPath $item.FullName -Force }
        }
        foreach ($name in $kept) {
            $file = "$model/$name"
            $source = Join-Path $Nvfp4Assets $file
            if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing NVFP4 asset: $file" }
            $copied = Join-Path (Join-Path $target 'nvfp4') $file
            Copy-Item -LiteralPath $source -Destination $copied -Force
            $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
            if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "NVFP4 asset hash mismatch: $file" }
            $nvfp4Hashes[$file] = $hash
        }
    }
    # Optional: what a numbered weight set keeps outside the family folders:
    # net4_set<n>.txt (bottleneck blocks left out, pooled transformer, split sides) and modelA/ffn_v2_set<n>.bin.
    $stale = @(Get-ChildItem -Path (Join-Path $target 'nvfp4') -Filter 'net4_set*.txt' -File -ErrorAction SilentlyContinue) +
             @(Get-ChildItem -Path (Join-Path $target 'nvfp4/modelA') -Filter 'ffn_v2_set*.bin' -File -ErrorAction SilentlyContinue)
    foreach ($item in $stale) { Remove-Item -LiteralPath $item.FullName -Force }
    $setFiles = @(Get-ChildItem -Path $Nvfp4Assets -Filter 'net4_set*.txt' -File -ErrorAction SilentlyContinue) +
                @(Get-ChildItem -Path (Join-Path $Nvfp4Assets 'modelA') -Filter 'ffn_v2_set*.bin' -File -ErrorAction SilentlyContinue)
    foreach ($item in $setFiles) {
        $relative = if ($item.Name -like 'net4_set*') { $item.Name } else { "modelA/$($item.Name)" }
        $copied = Join-Path (Join-Path $target 'nvfp4') $relative
        Copy-Item -LiteralPath $item.FullName -Destination $copied -Force
        $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
        if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "NVFP4 asset hash mismatch: $relative" }
        $nvfp4Hashes[$relative] = $hash
    }
    # Optional: the NVFP4 fused-block kernels and weights of Model A.
    # ... and of the rest of Model A's pass: split blocks, transformer
    # attention, input and output block, the FP8 models' records (fp8/fp8_set<n>.*) and the own pass host (host/: its
    # plan, kernel manifests and cubins; RTX 30 and 40 run the FP8 models on it). A family's folder is copied whole when
    # its manifest is there.
    foreach ($family in @('swin4', 'split4', 'vit4', 'prepost4', 'fp8', 'host', 'int8', 'host4', 'host4/mixed')) {
        $folder = Join-Path $Nvfp4Assets $family
        if (!(Test-Path -LiteralPath (Join-Path $folder "fused_manifest.txt") -PathType Leaf) -and
            !(Test-Path -LiteralPath (Join-Path $folder "$family.txt") -PathType Leaf) -and
            !(Test-Path -LiteralPath (Join-Path $folder 'modelA_fp8.plan') -PathType Leaf) -and
            !(Get-ChildItem -Path $folder -Filter "${family}_set*.txt" -File -ErrorAction SilentlyContinue)) { continue }
        $familyTarget = Join-Path $target "nvfp4/$family"
        New-Item -ItemType Directory -Path $familyTarget -Force | Out-Null
        # What an earlier build left there (a kernel that no longer exists) must not stay.
        Get-ChildItem -LiteralPath $familyTarget -File | Remove-Item -Force
        foreach ($item in Get-ChildItem -LiteralPath $folder -File | Sort-Object Name) {
            $copied = Join-Path $familyTarget $item.Name
            Copy-Item -LiteralPath $item.FullName -Destination $copied -Force
            $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
            if ((Get-FileHash -LiteralPath $copied -Algorithm SHA256).Hash -ne $hash) { throw "NVFP4 asset hash mismatch: $family/$($item.Name)" }
            $nvfp4Hashes["$family/$($item.Name)"] = $hash
        }
    }
}
[ordered]@{
    application = 'Display Filter screen and window overlay'
    created = (Get-Date).ToString('o')
    executableSha256 = (Get-FileHash -LiteralPath (Join-Path $target 'display_filter.exe') -Algorithm SHA256).Hash
    runtimeSha256 = $runtimeHash
    dlssRuntimeSha256 = $dlssHash
    frameGenerationRuntimeSha256 = $fgHashes
    nvfp4AssetsSha256 = $nvfp4Hashes
    gameFilesChanged = $false
    multiplayerVerified = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'external-nr-package.json') -Encoding utf8
Write-Output $target
