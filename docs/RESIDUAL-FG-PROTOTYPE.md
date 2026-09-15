# Half-rate DLSS-NR investigation

Status: experimental source integration with standalone hardware tests. Not a public release.
The default DeferredDLSS mode still evaluates NR every rendered frame. The optional half-rate
mode is deliberately disabled by default.

## Experimental UI / INI controls

Under DLSS Neural Rendering, enable **Generate before SR, apply after SR (DLSS)**,
then **NR every second frame (current raster, experimental)**.

```ini
[DlssNr]
DeferredDLSS=true
ResidualFG=true
```

The current A3 path applies a fresh NR residual directly to the current clean SR
frame on anchors. On skipped frames it reprojects the preceding residual using
the current frame's current-to-previous motion. Invalid or offscreen motion keeps
the clean pixel. It does not use NVIDIA residual FG, approximate camera matrices,
or a delayed clean raster.

This requires low-resolution, non-jittered motion vectors for reprojection.
The current path is D3D12 or its D3D11 bridge, not native Vulkan or the RR path.
No full-game frame generation setting is changed by this control.

Watch the Residual DLSS status: it must report **NR every second frame; previous NR
reprojected onto CURRENT clean raster**. An inactive status means ordinary every-frame
NR is retained, not successful half-rate processing. Disable ResidualFG to return to
ordinary deferred NR.

### Missing-motion fallback: sample-and-hold

When the every-second-frame option is enabled and the motion texture is absent,
each successfully generated residual is applied to two consecutive **current**
SR images. The next pair gets a fresh sample. There is no raster delay, motion
warping or residual FG evaluation in this fallback. The status explicitly says
**sample-and-hold (motion unavailable)**; it does not claim interpolation.

Fresh private NR and residual-DLSS evaluations use an owned zero-motion texture
and reset history every time, rather than inventing motion continuity. This does
not repair missing inputs to the game's own upscaler: it must still produce a valid
clean SR image. Depth remains required. Cuts, frame gaps, failed composition,
resolution changes and returning motion vectors invalidate the held sample.
This may show a two-frame stepping effect in the NR edit, but does not freeze the raster.

## NVIDIA interface

The public [DLSS-FG Programming Guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/DLSS-FG%20Programming%20Guide.pdf)
(SDK 310.7, June 2026) documents a direct NGX FrameGeneration feature with an
application-owned `DLSSG.OutputInterpolated` resource. This is distinct from
Streamline's presentation-managed FG wrapper. Direct resource output is possible;
absence of a Streamline offscreen helper is not evidence otherwise.

`tests/nr_residual_fg_smoke.cpp` exercises that documented interface using the
existing NGX headers and public parameter names. It does not download, bundle,
patch or inject a runtime. Supply your own installed official DLL paths.

Build from an x64 Visual Studio developer prompt at the repository root:

```bat
cl /nologo /std:c++20 /EHsc /Iexternal\nvngx_dlss_sdk tests\nr_residual_fg_smoke.cpp /Fe:x64\nr_residual_fg_smoke.exe /Fo:x64\nr_residual_fg_smoke.obj /link d3d12.lib dxgi.lib
x64\nr_residual_fg_smoke.exe "ABSOLUTE_PATH_TO\nvngx.dll" "DIRECTORY_CONTAINING_OFFICIAL_FG_DLL"
```

The test queries FG capability, creates a separate feature, submits creation,
provides read-state depth/motion/color and UAV output, evaluates consecutive
anchors, waits for GPU completion before readback/reuse, and releases the feature
after completion. Its stationary camera transforms are accurate for the synthetic
scene, not proposed substitutes for real game camera data.

## Local results, 7 September 2026

RTX 5090, signed NVIDIA `nvngx_dlssg.dll` 310.8.0.0 already installed with BG3:

- RGBA16F neutral carrier: 0.500000 at all three sampled positions.
- Signed carrier regions: 0.250000 / 0.500000 / 0.750000.
- The production FG adapter was tested with a 4K RGBA16F residual and 1080p guides.
  Translating plane, 32-pixel anchor displacement: interpolated left edge 336;
  mathematical midpoint 336; current anchor 352. Five-pixel tolerance passed.
- The motion test uses current-to-previous vectors with the normalized scale
  documented in NVIDIA's public DLSSG header. Zero-motion-only tests would not
  establish this convention.

The signed driver fallback runtime 310.2.1.0 reported FG capability and allowed
feature creation, but its first evaluation failed with `BAD00005` and cached-size
validation errors. Explicit resource extents did not resolve it. This is an
observed compatibility failure, not proof of its underlying cause or a universal
minimum-version requirement. Capability checks alone do not prove evaluation works.

The synthetic pass does not validate NR-generated imagery, changing exposure,
disocclusions, moving cameras, nonuniform motion or real-game presentation.

## Temporal matching and remaining limitations

For anchors at game frames 0 and 2, FG produces residual 1 only after residual 2
exists. It must then be composed with clean game image 1, not clean image 2.

The current `DeferredSr::After` writes into the current game's SR output before
the game performs its remaining post-processing. Buffering clean SR image 1 and
putting it into frame 2's output would leave the downstream game effects using
frame 2's other resources. Merely delaying Present does not allow previously
recorded post-processing to be rerun with the newly available residual either.

A fully engine-matched integration would still need:

1. A composition/buffering point that preserves the matching frame's downstream
   effects and exposure, or engine integration to defer that work. The current
   pre-exposed linear residual cannot just be added to a tone-mapped backbuffer.
2. Full camera transforms. The current NR bridge does not carry them; this preview
   requires explicit consent to approximate guides instead.

The preview does implement displaced two-frame motion composition, matching clean
colour/exposure buffers, NR and private-SR cadence, generation completion markers,
cut/gap resets, and GPU-side handling of NVIDIA's suppression flag (see candidate a1 below).
Both NR and its private residual SR receive the anchor-to-anchor motion field.
Initial history repeats the first anchor while establishing the one-frame delay.
It is not a promise that halving NR work improves total FPS: interpolation itself
costs GPU time, and buffering increases VRAM use and latency.

Reprojecting the previous residual into the current frame would avoid waiting for
the next anchor, but would be a different algorithm. It must not be presented as
NVIDIA interpolation. Live-game validation is separate from the synthetic tests above.

## Local flicker candidate a1 (13 September 2026)

The Requiem log supplied for this investigation records 1,460 NR anchor frames
and 1,455 skipped frames across several activations. It does not record NVIDIA's
output-suppression flag. Cadence is working in that capture; its flicker cannot
be attributed to cadence failure from the log alone.

The original rejection path removes the entire NR contribution on a rejected
midpoint and restores it on the next anchor. The static-scene shader regression
reproduces this on/off difference. Candidate a1 changes only the half-rate
composition path: accepted NVIDIA output is unchanged; rejected output instead
uses the previous NR anchor reprojected by the **delayed midpoint's single-frame
motion**. Invalid/nonfinite/offscreen motion retains clean pixels. The fallback
never reads NVIDIA's rejected image. It is reprojection, NOT NVIDIA interpolation.

The previous residual is saved only after midpoint composition. Existing explicit
cut/reset, generation and GPU-completion guards remain in place. Suppression
readback uses the existing completion-marker slots without waiting on the CPU;
the log and status expose suppressed/observed counts (excluding initial anchors).
Counters are collected when completed slots are recycled and can lag a few frames.
Two-frame motion composition is omitted on skipped frames, where it was unused.

No resolution, WorkingScale, NR intensity, driver, MFG unlock, or game setting is
changed. Added memory is one RGBA16F output-sized residual plus a 4 KiB readback
(about 33.7 MiB plus allocation alignment at 2804x1577). This is shared code for
RTX 30/40/50, not evidence of validation on every GPU generation.

Validation: existing WARP shader regressions plus static rejection/no-pulse,
midpoint displacement, invalid/offscreen vectors and accepted-FG identity cases;
`tests/nr_residual_fallback_dx12_smoke.cpp` executes the embedded production DXIL
on the local RTX 3080. These tests do not load NGX or the game and do not establish
the cause of the user's flicker. Full game A/B validation is still required.

Limitations remain: the one-frame downstream post-processing mismatch and
approximate camera transforms described above; possible stale NR at disocclusions
or unreported scene cuts with apparently valid motion (no depth-history rejection
in this fallback). If NVIDIA reports no suppression, this candidate is not a fix
for the remaining interpolation/composition mismatch. FF16 and RTX 40 hardware
remain unvalidated until equivalent game captures are available.

## Local cadence candidate a2 (14 September 2026)

The new Requiem capture (user-tested SM86 0.2.4) exposes a second failure, independent
of the suppression flag. At 01:02:49-01:02:57 there are **313 NR anchors, zero skips**.
All 312 consecutive logged render-epoch deltas equal 2. The earlier a1 capture has
the same failure at 00:40:29-00:40:49 (771 anchors, zero skips) and 00:40:58-00:41:06
(288 anchors, zero skips). Other intervals alternate correctly; the earlier
quality/cadence logs alone missed the zero-skip intervals.

Cause: the native seam clock imported jumps from the swapchain Present counter.
Two Presents per upscale therefore advanced the render epoch by two.
`PrepareHalfRate` interpreted each jump as a missing rendered frame, reset FG
history and selected another NR anchor, so it never reached a skip. This wastes
the intended NR saving and repeatedly primes FG. It does not prove the entire
reported pacing or menu-flicker problem has this one cause.

A2 advances native temporal identity exactly once per observed Before-upscale
call. After retains that identity even if Present advances on another thread.
Bridge submission epochs, including duplicates and actual gaps, are unchanged.
The raw counter is still passed separately to the existing creation guards;
GPU-completion markers and resource retirement are unchanged. Explicit game
Reset, cancelled/failed pairs, invalid inputs and generation changes still
invalidate histories. Raw Present jumps alone no longer imply scene cuts.
Unreported cuts or a renderer bypassing the seam without Reset remain a limitation.

The INFO log now reports anchors, skips and render-epoch discontinuities every
240 successful half-rate compositions, even when no frames were skipped.
It does not depend on suppression readback becoming available. With INFO logging,
the user need not produce a per-frame TRACE capture to check the cadence.

Regression: the updated seam-clock test fails against a1 and passes with a2.
The 313-evaluate/2-Presents sequence now schedules 157 anchors and 156 skips in
the headless scheduling model. Tests also cover 1/3/4/6 Presents per render,
varying rates, stalled/reset Present counts, explicit history Reset and preserved
bridge gaps. These synthetic rates do not claim MFG hardware/runtime support.
The scheduling test does not execute NGX or measure actual displayed pacing.

No shader, camera, NR strength, resolution, WorkingScale, presentation hook,
SM86 runtime or user configuration is changed from a1. The code is common to
native D3D12 RTX 30/40/50; only Requiem/RTX 3080 provided the failing game log.
FF16 and RTX 40 still require live validation. A1's one-frame post-processing
mismatch, approximate camera guides and reprojection limitations remain.

## Current-raster candidate a3 (14 September 2026)

The A2 Requiem log proves the cadence fix is active: five half-rate runs alternate
anchors and skips 1:1, every logged render-epoch delta is one, and all cadence
reports show zero render-epoch gaps. The same capture reports zero NVIDIA
suppression across more than 700 observed interpolations. A1's rejection fallback
therefore never runs, and A2's remaining flicker is not a scheduling or suppression
failure.

A3 removes the remaining delayed-raster/NVIDIA-interpolation path. Anchor frames
compose their new residual with the current clean SR output. Skipped frames compose
the previous anchor's residual with the current clean SR output after one-frame
motion reprojection. Invalid, nonfinite, or offscreen motion preserves the current
clean pixel. The private residual SR still receives the composed two-frame motion
field on anchors, so it does not pretend that consecutive NR samples are one frame
apart.

The setting retains its old `ResidualFG` INI key for compatibility, but A3 does not
create or evaluate a residual-only NVIDIA FG feature and does not require the
approximate-camera option. It also removes two clean-history textures, the
interpolated residual, suppression resources/readback, and the FG feature from the
half-rate allocation. Ordinary full-game FG remains independent.

This is still an experimental temporal approximation. Disocclusions and areas
without valid motion fall back to the clean raster for the skipped frame, and the
NR edit can still change at half-rate. Only live Requiem/FF16 A/B testing can show
whether that trade is preferable to A2's severe flicker.
