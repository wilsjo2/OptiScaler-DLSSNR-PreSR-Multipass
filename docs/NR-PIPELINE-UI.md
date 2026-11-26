# NR menu

The Neural Rendering section is one list. The controls used to set a look are always shown; everything set once per game or used now and then is in collapsed sections below them.

| Part | Controls |
| --- | --- |
| Top | **Enable Neural Rendering** stops NR work; **Apply model** hides the edit while the model still runs. The status line gives NR's GPU time against the frame time. **Preset** loads a saved preset. |
| Look | **Model passes**; per pass (a tab each when there are two or more): **NR model**, **Style**, **Intensity**, **Local structure**, **Local tone**. For all passes: **Detail strength**, **Colour strength**, **Highlight guard**. |
| Skin and environment | The pass's **Skin structure** and **Auto skin mask**, and the separate skin / environment strengths of the final edit with the mask preview. |
| Resolution and upscaling | The pass's model resolution, peripheral compression and enlargement. The label shows the resolution in use. |
| Faster later passes | Re-apply and borrow options for pass 2 and later (shown with two or more passes). |
| Where NR runs | One choice of six placements, with the options of the chosen one. The header names the placement in use. |
| HDR and exposure | HDR mapping, white point source, exposure trim and paper white. |
| Artifact fixes | Halo suppression; stale-history rejection on the RR residual path; HDR brightness matching for early edits on the finished picture. |
| Compare and inspect | Hold, comparisons (with the original picture or another model), wipe/zoom, labels and debug views. |
| Presets and calibration | Save, overwrite, rename and delete presets; calibrate the sliders to a preset. |

## Where NR runs

| Choice | Behaviour |
| --- | --- |
| After the upscaler | NR edits the upscaled picture, before effects/HUD. |
| Before the upscaler | NR edits the active input before SR or RR+SR. |
| Before the upscaler, edit added after it | The game reconstructs its clean input; NR's edit is upscaled separately and joins after SR/RR. |
| On the finished picture | NR edits the final picture, after effects/HUD. |
| Before the upscaler, edit added to the finished picture | As the separate edit, joined at presentation. |
| Denoise first, then NR, then upscale | The game's upscaler runs once more at render resolution so NR sees a clean picture. Native D3D12. |

Depth/motion are guides, not colour stages; the placement does not configure FG.

Later passes inherit pass 1 except local tone (default zero); reset clears overrides. Model sliders commit on release. The pass tabs list the active passes.

Timing shows measured NR GPU elapsed time against the time between real DXGI frames rather than generated frames. Overlap can make NR time exceed the frame interval. It is not an additive workload measurement.

Hold freezes input and guides for early/deferred processing; early-to-finished mode also holds one clean final image across presentation slots. It does not pause the game. See [hold limits](../OptiScaler/dlssnr/design/frame-hold.md), [bridges](NR-FINISHED-BRIDGES.md) and [enlargement](NR-DLSS-ENLARGEMENT.md).

## Calibration

**Calibrate to a preset** holds the frame, renders the preset to match, and searches the chosen pass's Intensity, Local structure and Local tone and the Detail and Colour strength for the closest picture. With **Also set skin, environment and highlight guard** it reads the composition's own skin mask and compares skin and environment separately, each counting the same whatever its share of the frame; a frame with under 0.5 % skin leaves the four skin / environment sliders as they are. After a model slider changes, pictures are read until they stop changing before one is measured. D3D12 only.

GPU/proxy checks covered held inputs, rotating resources and restoration after failure. [Live results](NR-UPSTREAM-REVIEW.md) describe the narrower game coverage.
