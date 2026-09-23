# Peripheral spatial compression

Enable **Peripheral compression** below Model resolution to reduce the number of pixels processed by Neural Rendering. The default enabled layout keeps an 80% centre band on each axis and compresses the periphery into a 90% working extent. At 3840×2160 and 100% Model resolution, NR processes 3456×1944: 81% of the original pixels. The GPU cost of packing and unpacking remains, so pixel savings are not an FPS prediction.

The feature is disabled by default. It runs in the shared DX12 and native Vulkan NR paths, including existing supported pre-upscale and finished-picture placements. DX11 processing through the DX12 bridge uses the same implementation. Existing backend restrictions still apply: native Vulkan does not gain private DLSS enlargement or deferred edit upscaling.

## Controls

Centre width/height and Working width/height are independent percentages of each image axis. Centre offsets move the protected band; Working region shifts redistribute the peripheral pixel budget between opposite sides. Offsets and shifts are signed percentages. Lower working sizes can soften edges or shimmer as objects cross the centre boundary.

Model resolution remains the uniform scale for the entire input. Below 100%, centre detail is also reduced. Above 100%, peripheral compression combines with the existing supersampling/downsampling path. The final compressed extent cannot fall below 25% of the original image on either axis. Settings commit when a slider is released. Invalid saved layouts fall back to ordinary NR and report the reason.

The optional cyan centre and orange working-region outlines are drawn by the overlay after NR; they are never baked into model input. The status reports the effective model dimensions or why compression is unavailable.

**Preview**, directly below Peripheral compression, selects the same view as **Debug view → Compressed model input** (`DebugView=4`). It displays the immutable input passed to NR before spatial unpacking, scaled to fill the screen, so peripheral squeezing is visible. The two controls stay in sync; unticking Preview turns debug view off. Without active compression it shows ordinary model input. Enable Apply model to see debug views. Existing Proxy and Model output views retain unpacked geometry.

Settings live under `[DlssNr]`:

| Key | Default |
| --- | --- |
| `SpatialCompression` | `false` |
| `SpatialCenterX`, `SpatialCenterY` | `80` |
| `SpatialWorkX`, `SpatialWorkY` | `90` |
| `SpatialOffsetX`, `SpatialOffsetY` | `0` |
| `SpatialShiftX`, `SpatialShiftY` | `0` |
| `SpatialShowCenter`, `SpatialShowWork` | `false` |

`WorkingScale` keeps its existing meaning and range. The packaged INI uses `auto` for these defaults.

## Processing and history

The pack stage samples the encoded colour proxy with an adaptive peripheral filter. Depth is point-sampled from its active region. Motion is converted to native pixel displacement and warped through both endpoints, rather than multiplied by a single scale; offscreen endpoints use the mapping's edge extension. Packed guides and colour share one geometry.

Each NR layer runs at the compressed extent with its own temporal history. The immutable packed input and final model answer are both unpacked to the ordinary Model-resolution grid before the existing composition or enlargement stage. Private DLSS receives ordinary image geometry and original guides. Changes to centre position, shifts or sizes reset model histories even if texture dimensions remain unchanged.

Spatial shaders have their own constant layout and use the existing backend descriptor infrastructure. No new presentation waits, inference queues or frame-skipping modes are introduced. A failed spatial evaluation leaves the original frame intact and switches subsequent evaluations to ordinary NR; changing the layout or using Retry permits another attempt. Packed output and stale model results are never used as a recovery image.

## Attribution

The layout and mapping are adapted from [Yuri Grib / BeliyG3's Optimizer FPS for DLSS5](https://github.com/BeliyG3/optimizer-fps-dlss5/tree/64902dd6a02460e5f6b778504ec2a4005faf4d9c), pinned at `64902dd6a02460e5f6b778504ec2a4005faf4d9c`. Its MIT notice is included with the source and candidate packages. This integration uses the spatial algorithm directly; it does not require the ReShade add-on or import its temporal scheduling.

## Validation

The [offscreen benchmark](../tests/spatial-benchmark/README.md) uses production spatial shaders and the actual NVIDIA NR runtime. Two runs on an RTX 5090 with driver 616.64, each with 20 evaluations per case and the first two excluded, measured 10.6–12.3% lower pack/model/unpack GPU time at 4K with the default 80/90 layout. These timings exclude game rendering and the host colour codec/composition; they are not an FPS gain prediction. The benchmark also records the combined 85% scale case, full image differences and runtime/source hashes.

CPU mapping checks cover asymmetric layouts, offsets, shifts, odd dimensions, supersampling, invalid values and identity bypass. The Vulkan shader fixture runs the production blobs on a device with no optional features enabled. Vulkan motion is stored as RGBA32F (xy used) to avoid requiring the optional extended-storage-format feature; DX12 uses RG32F. The NVIDIA runtime accepted RGBA32F motion in the DX12 hardware probe, but native Vulkan NR evaluation with that format still needs in-game verification.

The full NR prerelease suite, including the RDR2 finished-picture queue regression, passed. Synthetic fixtures and benchmarks do not establish in-game image quality or resolve the separate RDR2 HDR highlight flicker report. Live game coverage of reconstruction modes, HDR, frame hold, resize/layout changes, allocation failures and shutdown remains a candidate validation task. Compression stays opt-in.
