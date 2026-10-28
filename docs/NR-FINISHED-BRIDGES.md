# Finished-picture NR bridges

Finished-picture NR processes the presentation image, including HUD/effects. Unsupported or unready frames retain the game image.

## D3D12 with OptiFG / XeFG

Compose on the XeFG proxy's current app backbuffer before its Present, using the game queue retained by the XeFG feature. XeFG's internal DXGI swapchain can replace the globally captured presentation queue with its display queue. Using that global queue for the app buffer makes unfinished NR input appear cross-queue and can skip the edit indefinitely. When interpolation is inactive or paused, the proxy still owns the app buffer; the internal display Present must not consume the NR slot instead.

This follows [Intel's XeFG integration contract](https://github.com/intel/xess/blob/main/doc/xess_fg_developer_guide_english.md): the application uses the returned proxy swapchain, and XeFG uses the initialization queue for interpolation. Same-queue NR input is ordered before composition without requiring CPU-observed fence completion. The v0.8.7 readiness policy remains in force: unfinished input from a different queue is skipped, with no new queue wait.

The regression report and original investigation are [RDR2 issue #69](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/issues/69) and [LorisPicariello's PR #70](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/70). The WARP queue test verifies 16 saved-input transfers to a game-picture surrogate while producer completion is blocked on presentation, as well as rejection on the asynchronous display queue. RDR2 gameplay and the separate HDR highlight flicker in #57 are not validated by that test.

## D3D11

Copy the backbuffer into a dedicated shared texture, signal a shared fence, then let D3D12 wait and run NR. Return the texture to COMMON, signal completion, and let D3D11 wait and copy back on success. Consecutive reuse is ordered without per-frame CPU waits; resize/teardown drains work. These resources are separate from the upscaler cache.

The D3D11-to-D3D12 FG swapchain already has a D3D12 backbuffer. NR runs after its interop wait on the presentation queue, whether FG is active, paused or off. It creates no additional FG feature.

## Vulkan

Capture guides at the upscaler seam. Submission notifications track recordings; reset/free invalidate them. Presentation waits for game semaphores, uses fence-protected slots and signals image-indexed completion semaphores before the overlay. Vulkan-to-D3D12 upscalers use this same native presentation stage.

SDR UNORM/sRGB, HDR10 and scRGB are converted through a floating-point working image using shared HLSL. Warmup/failed model output stays hidden. The path requires one swapchain, the primary graphics/present queue and supported transfer/blit usage. Early generation with finished-picture application remains D3D12/D3D11 only.

## Checks

- RTX 5090 D3D11/D3D12 transfers passed unchanged/edited output, skipped copy-back, repeated reuse, resize and RGBA8/RGB10A2/RGBA16F cases.
- WARP checked HDR conversion, alpha, highlights and early residual composition.
- Production Vulkan objects presented 12 synthetic frames per SDR UNORM, sRGB, HDR10 and scRGB format, including guides and actual NR evaluation. The Khronos validation layer was unavailable; this does not validate a game.
- Earlier BG3 DX11 character-creation runs exceeded 900 frames with FG enabled and 2,100 with FG disabled. HDR captures clipped highlights and the latter logged frame-count warnings. See [subsequent game checks and remaining limits](NR-UPSTREAM-REVIEW.md).

Local harness/evidence: `x64/nr-dx11-finished/`. Native Vulkan gameplay remains unverified.
