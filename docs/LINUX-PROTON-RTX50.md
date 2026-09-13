# Linux / Proton / RTX 50 validation notes

Findings from an extended test session: 007 First Light (Streamline-based DX12 title), Ubuntu,
RTX 5070 (Blackwell), driver 615.71.09 (Linux R615 branch; equivalent to Windows 616.92 - the two
platforms number the same branch differently, so check the branch, not the raw number, against
this fork's stated 616.56 minimum). Proton 11.0 and Proton Experimental both tested.

## OptiScaler vs. a from-scratch NGX-forwarder addon: same runtime, different outcome

A separate NR host - a ReShade add-on driving `nvngx_dlssnr.dll` 310.8 through a hand-written NGX
forwarder - reliably reproduced Xid 31 (`FAULT_PDE`, `ACCESS_TYPE_VIRT_READ`) on this exact
GPU/driver/game combination: 17 confirmed occurrences, the fault address never moved by a single
bit across four independent fix attempts (NGX FrameGen-handle tracking, descriptor-heap shadowing,
a from-source ReShade patch, a custom vkd3d-proton build). A clean negative control - the addon
removed, everything else unchanged - ran a full session with no fault, isolating the addon's own
evaluation path as the trigger.

Swapping to this fork as the sole NR host, same `nvngx_dlssnr.dll` 310.8, same driver, same GPU:
zero Xid 31 across every subsequent test. This is comparative evidence the original fault lived in
that addon's resource/evaluation contract, not in the NR runtime itself - useful if anyone else
lands here from the same symptom.

## The game's own native Streamline FG plugin crashes on its own

Independent of any of the above: the game's own `sl.dlss_g.dll` (its native Streamline Frame
Generation plugin) produced Xid 32 (invalid/corrupted push-buffer stream), then on a later run
Xid 13 + Xid 32 together, within seconds of launch - **before DLSS-NR was ever enabled and before
any Frame Generation toggle was touched**. Both crashes coincided with the plugin's own repeated
`cloneFakeBuffers`/fullscreen-state churn during Streamline's plugin-manager init - the game's
own FG negotiation, not anything OptiScaler does. `[FrameGen] Enabled=false` in `OptiScaler.ini`
does not prevent this: that key only gates OptiScaler's *own* FG implementation, not the game's
native Streamline FG negotiation, which happens regardless.

**Removing `sl.dlss_g.dll` from the game's own directory** (leaving the rest of its native
Streamline set - `sl.common.dll`, `sl.dlss.dll`, `sl.dlss_d.dll`, `sl.interposer.dll`, `sl.nis.dll`,
`sl.pcl.dll`, `sl.reflex.dll` - untouched) fully resolved it: zero further Xid faults across every
subsequent NR-enabled session. If a game hangs or Xid-faults on launch before you've touched any
NR or FG setting, check whether its own Streamline FG plugin is present and try removing it before
suspecting OptiScaler.

Upgrading the remaining native Streamline plugins from the game's stock 2.13.0 to the official
2.14.1 SDK release worked without incident for SR + NR.

## Native NVIDIA Frame Generation does not work under DXVK-NVAPI here

Tested two paths: the game's own native Streamline FG (after temporarily restoring
`sl.dlss_g.dll` in isolation to confirm this specifically, then removing it again) and OptiScaler's
own `[FrameGen] FGOutput=dlssg` (using a private Streamline 2.14.1 copy under
`OptiScaler/streamline/`, keeping the game's own copy absent). Both produce a dark/black output,
not a crash - no Xid fault in either case.

Root cause, from the log: `NvAPI_D3D_SetReflexSync(...)` returns error `-3` (not-implemented
class), and Streamline retries in a tight loop, logging `RSYNC: setDynamicMFGParams failed with
status 1` and `RSYNC: setReflexTiming failed with status 1` dozens of times per second. DXVK-NVAPI
does not implement the Reflex-sync surface that Streamline's Dynamic Multi-Frame-Generation
negotiation depends on for pacing, on this stack. This is a platform-layer gap below either
injection path - it reproduces identically whether the `sl.dlss_g.dll` plugin loads from the game's
own folder or from OptiScaler's private `OptiScaler/streamline/` copy, since both go through the
same DXVK-NVAPI shim. Choosing a different plugin location does not route around it.

**Update: FSR-FG through OptiScaler also fails on this game, via a different root cause.** Tested
`FGInput=dlssg` + `FGOutput=fsrfg` (with `amd_fidelityfx_loader_dx12.dll` and
`amd_fidelityfx_framegeneration_dx12.dll` present) specifically to avoid the Reflex-Sync path
above - it doesn't depend on the same NVIDIA API surface, and other users in this project's
discussions reported it working where DLSS-FG didn't. Here it produced Xid 13 (Graphics
Exception) and a hung process ~9 seconds after launch. The log shows the actual mechanism:
repeated `Streamline supports only one DXGISwapChain (...). Skipping some Present() hooks for
DXGISwapChain (...)` warnings, immediately followed by `vkQueueSubmit error: FFFFFFFC`
(`VK_ERROR_DEVICE_LOST`) specifically inside OptiScaler's own `MenuOverlayVk` - a dual-swapchain /
Vulkan-interop conflict between the FFX FG bridge and OptiScaler's own overlay renderer, unrelated
to Streamline or Reflex-Sync entirely. Confirmed against NVIDIA's own published Vulkan Reflex SDK
(`NvLowLatencyVk.h`) that no Reflex-Sync/Dynamic-MFG function is published in the public API at
all - not merely unimplemented, there's no documented hook to build one against. Filed as
[vkd3d-proton#3292](https://github.com/HansKristian-Work/vkd3d-proton/issues/3292) rather than
ship a speculative patch against undocumented driver-internal behaviour.

Not attempted: XeFG (`FGOutput=xefg`) - a third, separate FG technology this fork supports that
also doesn't depend on Reflex-Sync. Given FSR-FG's failure was an overlay/swapchain conflict
rather than anything FSR-specific, XeFG may hit the same conflict, or may not - worth testing
before concluding no FG technology works here.

**Update: official Streamline docs confirm the requirement, not just the symptom.** Now that
`research/nvidia/streamline` (v2.14.1, `tools/research/fetch_sources.py`) is available locally,
its `docs/ProgrammingGuideDLSS_G.md` section 8.0 states plainly: "**It is required** for sl.reflex
to be integrated in the host application. **Any existing Reflex SDK integration that does not use
Streamline cannot be used with DLSS-G.**" This is not an implementation detail DXVK-NVAPI happens
to be missing - Streamline's own documentation makes full Reflex integration a hard prerequisite
for DLSS-G, with no documented fallback. The same section's troubleshooting note - "If you see a
warning... `common constants cannot be found for frame N`... the sl.reflex markers
`eReflexMarkerPresentStart`/`eReflexMarkerPresentEnd` are out of sync with the frame being
presented" - names exactly the frame-index-desync failure class this investigation observed
(`RSYNC: setDynamicMFGParams failed`/`setReflexTiming failed`). Strengthens, rather than changes,
the conclusion already filed as
[vkd3d-proton#3292](https://github.com/HansKristian-Work/vkd3d-proton/issues/3292): this is a
genuine gap in what a Vulkan/DXVK-NVAPI Reflex shim can implement without NVIDIA's undocumented
driver-internal Dynamic-MFG negotiation surface, not a workaround-able configuration issue.

## Forcing DLSS-RR on a game with no in-menu RR option: fails safely, not a usable path

This game's engine never calls `NVSDK_NGX_D3D12_CreateFeature` with
`NVSDK_NGX_Feature_RayReconstruction` - only `SuperSampling` - so its graphics menu has no Ray
Reconstruction toggle. `TryCreateOptiFeature` (`inputs/NVNGX_DLSS_Dx12.cpp`) picks the upscaler
backend for a `SuperSampling` call from `[Upscalers] Dx12Upscaler` regardless of which NGX feature
the game actually requested, so setting `Dx12Upscaler=dlssd` does substitute a real DLSSD (RR)
feature for the game's SR request, bypassing the missing menu option.

Tested it. `DLSSD::Init` fails immediately and OptiScaler falls back to FSR 2.1.2 as the base
upscaler (`Feature 'DLSSD' initialization failed falling back to FSR 2.1.2`) - no crash, no Xid,
the fallback path itself works correctly. Root cause: DLSS-RR's NGX contract needs G-buffer
signals SR does not (`DLSS.Input.DiffuseAlbedo`, `SpecularAlbedo`, `GBuffer.Normals`,
`GBuffer.Roughness`, `MotionVectorsReflection`, plus specular/diffuse hit-distance buffers) - a
game whose engine never intends to call the RR feature has no reason to ever populate those NGX
parameters, so `DLSSD::Init` fails against its own contract. The resulting FSR 2.1.2 + NR
combination is a quality *downgrade* from DLSS + NR, not an upgrade - any FPS or stability
improvement observed here is explained entirely by FSR 2.1.2 being cheaper than DLSS, not by
Ray Reconstruction doing anything.

**Update: implemented and validated.** `TryCreateOptiFeature` now checks, before substituting
DLSSD for a `SuperSampling` call, whether the game's own NGX parameter block carries
`GBuffer.Normals` and `GBuffer.Roughness` (`GameSuppliesRRInputs()`) - the two most fundamental
inputs any ray/path-traced renderer would expose. Absent either, it refuses the substitution and
uses the auto-detected backend instead. Built and live-tested against this exact game: the log now
shows a clear refusal followed by a successful plain-DLSS creation, with no `BAD00005` and no
FSR 2.1.2 fallback. DX12 only for now - the Vulkan backend-selection path is structurally
different and untouched.

**ControlMask experiment, run to a conclusion.** Two live-tested rounds (motion-vectors as probe,
then colour buffer as probe - deliberately wrong format/content both times, reusing existing
resources rather than building new synthetic-pattern generation blind): `EvaluateFeature` returned
`0x1` (success) both times, identical to the null baseline, zero Xid, no visible difference
observed (round 1 only, human-confirmed; round 2 launched unattended). Inconclusive-leaning-negative,
not a clean negative - two different probes producing the same non-effect is real signal, but a
controlled synthetic pattern (all-max, checkerboard) would be a stronger test than this session
built. Full write-up: `workflow/decisions/ADR-013`.

**Known conservatism, from the official RR programming guide**: Streamline's own DLSS-RR guide
(`research/nvidia/streamline/docs/ProgrammingGuideDLSS_RR.md`, section 4.1) documents that a real
integration may pack roughness into the normals texture's alpha channel
(`kBufferTypeNormalRoughness` with `normalRoughnessMode=ePacked`) instead of tagging a separate
roughness buffer - at the NGX layer this could plausibly mean `GBuffer.Roughness` is legitimately
never set even by a genuinely RR-capable renderer. `GameSuppliesRRInputs()` would then refuse a
substitution that might have actually worked. Left as-is deliberately: the substitution this gate
guards is already speculative (ini-forced against a feature the game never requested), so a false
negative here just means "does not force RR" - the same safe outcome as before this gate existed -
never a false positive that would risk the `BAD00005`/FSR21 regression this was built to prevent.

## Multipass (`Passes=2`) reintroduces grain and costs ~30% FPS at default tuning

Upstream PR #43 (interpass raw-output clamp fix) was believed to be the blocker for testing
`Passes>1` safely. It is not: this fork's current tree already implements the equivalent fix
independently, for both DX12 and Vulkan (`DlssNr_Dx12_Run.cpp`'s `passClamp`/
`DlssNrMode_ClampProxy`, `DlssNrFeature_Vk.cpp`'s `state.passClamp`), landed as part of the
refactor that split the old monolithic `DlssNr_Dx12.cpp` into the current per-concern files. PR
#43's diff no longer applies (its base predates that split by ~18k lines in this directory alone)
and is not worth porting - the fix it proposes is already present under a different, more complete
implementation.

Tested `Passes=2` anyway, since the presumed blocker was gone. Measured via MangoHud, not felt:
median FPS dropped from 88-89 (`Passes=1`) to 61.8 (p1_low 49.9) - about -30%, matching this
project's own ini documentation ("2 and 3... cost almost exactly 2x and 3x the model time").
Grain also visibly returned to the image.

Root cause: `OptiScaler/dlssnr/PassProfiles.h` defaults pass 2 and pass 3 to
`intensity=1.0`/`structure=1.0`, inherited from pass 1's own config, unless a per-pass override
(`DlssNrPass2Intensity`, etc.) is set. The same full-strength detail-injection operator that
already ran once therefore reapplies at full gain to its own already-processed output - not new
information, the same operator twice. This matches the general finding in iterative
super-resolution/refinement work (e.g. SR3, "Image Super-Resolution via Iterative Refinement",
arXiv:2104.07636): iteration count and per-iteration strength need to be balanced against noise
amplification, or repeated refinement compounds high-frequency artifacts instead of improving the
image. This project's own ini already calls multipass "deliberately over-processed" - the grain is
that tradeoff surfacing, not a new bug.

Not fully closed: damping pass 2 via `DlssNrPass2Intensity`/`DlssNrPass2LocalStructure` below 1.0
(a light reinforcement rather than a full reapplication) is a plausible follow-up this fork
already exposes and this session did not test. `Passes=1` is the deployed config until that's
tried.

## External research: sharpening/grain ordering, and a future-R&D direction

NVIDIA's own NVIDIA Image Scaling SDK (`NIS`, MIT-licensed, not currently integrated into
OptiScaler - checked, no `NIS`/`NVScaler`/`NVSharpen` references anywhere in this tree) documents
in its README exactly the mechanism behind the multipass grain finding above, independently of
this project: "sharpening algorithms can enhance noisy or grainy regions... certain effects such
as film grain should occur after NVScaler or NVSharpen." That is NVIDIA's own stated ordering rule
for their spatial sharpening/scaling pass, and it corroborates ADR-009's mechanism (detail
injection amplifies whatever noise-like signal is already present) from an independent,
official source rather than only the general iterative-refinement literature cited there.

`arXiv:2605.23902` ("PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion" - Ren,
Fidler, et al.) reformulates latent-to-pixel decoding as a few-step, sigma-aware (noise-level
aware) conditional diffusion process, distilled to 4 inference steps, decoding 512x512 to
2048x2048 in under 1s on an RTX 5090 (210ms on a GB200 datacenter GPU). Classified per this
project's own applicability tiers: **interesting but impractical today** - even its fastest
reported figure (210ms, on hardware far above a 5070) is roughly 15x this game's entire 13.3ms/
frame budget at 75Hz, so it cannot run per-frame in a real-time NR pipeline as published. The
architecturally relevant idea for **future research** is the "sigma-aware adapter": the model
knows how corrupted/uncertain its own input is and conditions its output strength on that,
rather than applying a fixed-strength operator regardless of input confidence. That is precisely
what `PassProfiles.h`'s multipass tuning lacks today (ADR-009) - a per-pass sense of how much
new information is actually left to extract, rather than a fixed `intensity=1.0` inherited
unconditionally from pass 1.

## DLSS-NR's fixed-blend history has no confidence signal (NRD comparison)

Compared this fork's own temporal accumulation against NVIDIA's real-time denoisers (NRD v4.18.0,
`research/nvidia/nrd`) - the closest official analogue to what the DLSS-NR pass does, since both
accumulate a per-pixel signal across frames using reprojection.

`dlssnr_residual.hlsl`'s accumulate mode is `history_t = lerp(reproject(history_{t-1}), edited -
original, gResidualBlend)`, where `gResidualBlend` is one scalar for the whole frame
(`DlssNrResidualAcrossRrBlend`, default 0.08). Reprojection validity is binary: invalid (off-screen
/ bad motion vector) zeroes history at that pixel, valid gets the same fixed blend rate regardless
of how much actually changed there. `DlssNr_Dx12_Run.cpp`'s only other reset path is a full,
whole-frame reset on the game's own signal - also binary.

NRD's own README states this directly: "An application should not rely solely on the anti-lag
provided by REBLUR/RELAX" - a fixed/binary scheme is documented by NVIDIA as an insufficient
baseline, not a legitimate design choice to converge on. Their fix is a continuous `[0,1]`
per-pixel confidence signal (`IN_DIFF_CONFIDENCE`/`IN_SPEC_CONFIDENCE`), computed from a gradient
between stored and re-evaluated radiance, that scales the effective accumulated history length
every frame - at NRD's own stated cost of under 5% of frame time. This retroactively explains why
`ReversibleMode`/`TransferStrength` tuning (this doc, above) was needed to fight ghosting: one
global blend rate is a compromise between stable pixels (want slow blend) and just-disoccluded/
fast-changing pixels (want fast blend) - tuning the single constant only moves where that
compromise sits, it can't remove it.

**Update: implemented and built.** `gResidualBlend` is now the stable floor of a per-pixel gate:
`disagreement = saturate(length(delta - history) / gResidualConfidenceSensitivity)`, blend rate
`a = lerp(gResidualBlend, 1.0, disagreement)`. New `[DlssNr] ResidualConfidenceSensitivity` ini key
(default `0.25`). Built clean on the guest VM (both DX12 `.cso` and Vulkan `.spv` blobs
recompiled), deployed, and smoke-tested live: zero Xid, all baseline signals normal. **What this
confirms and what it doesn't**: the whole code path is gated on Ray Reconstruction being active
(`g.rayReconstruction`), which this game doesn't support (ADR-008) - so this run only confirms the
change is a correct no-op with RR off, not that the gate improves anything. Real validation needs
a genuinely RR-capable game. Full rationale: `workflow/decisions/ADR-011` (proposal) and
`ADR-012` (implementation).

**Bonus finding, same source**: NRD's "Interaction with Frame Generation" section states that FG's
boosted display rate does *not* speed up the underlying denoising pass rate, so
`GetMaxAccumulatedFrameNum`-equivalent history-length math must be driven by the *real* render FPS,
not the generated one, or temporal lag increases by the FG multiplier. Not currently relevant here
(FG is confirmed non-viable, ADR-003) but worth checking first if that ever changes upstream -
DLSS-NR's own history-length/accumulation logic would need the same real-vs-generated FPS
distinction NRD documents.

## NR evaluation-cadence decoupling: real, measured GPU-cost reduction

Implemented the `neural-upstream` community technique (see the NRD-comparison section above)
generalized for this fork's main NR path: reproject last frame's real model answer through the
current frame's motion vectors instead of paying for a fresh NGX evaluate every frame. New
`[DlssNr] EvaluationCadence` key (default `1` = every frame, the existing behaviour, unreachable
otherwise - the new code path is provably a no-op unless raised). Reuses the already-separate
`dlssnr_residual.hlsl` blob (ADR-011/012) with a new `ReprojectOnly` mode rather than new shader
infrastructure.

Live-tested against 007 First Light at `EvaluationCadence=2`, measured via the new rolling-vitals
summary (below), across three consecutive 256-frame windows: model GPU time dropped from ~5.08ms
mean (cadence=1 baseline) to ~2.57-2.61ms mean - almost exactly half, matching the theoretical
expectation of alternating one real evaluate with one cheap reprojection. p99 stayed close to the
full-evaluate cost in every window, as expected (the real-evaluate frames dominate the tail). Zero
Xid, zero dispatch failures, stable across ~1800 frames. **Not yet validated: whether the image
looks right doing this** - reprojection artifacts (ghosting on missed disocclusion, drift on fast
motion) need a human watching, which a log/crash check cannot substitute for. Reverted to
`EvaluationCadence=1` pending that. Full write-up: `workflow/decisions/ADR-014`.

## Rolling vitals: mean/p99 GPU cost over a real window, not one sample

The existing periodic split-log (every 600 frames) reported whatever single frame happened to land
on the 600th tick - it could miss every real spike in between. Added a 256-slot rolling window
(`OptiScaler/gpu_time/Vitals.h`, project-wide - not DLSS-NR-specific, placed alongside
`GpuTime_Dx12/Dx11`) behind it, adapting a lock-free ring-buffer/percentile pattern from an
independent GreenBoost Vulkan layer (`~/Dev/greenboost_all/greenboost_gaming`) - std::sort instead
of a hand-rolled insertion sort (STL is available here), and reporting the metric this project
already uses (milliseconds) rather than converting to a framerate. Emits both a human-readable line
and a pipe-delimited `DLSS-NR-VITALS|...` one, greppable by hand or any external tool without
needing MangoHud or a separate post-processing script for NR's own GPU cost specifically.

This is what actually caught the cadence-decoupling result above - the model-mean drop was visible
in the log without needing MangoHud, a Python script, or a human watching a frame counter. Full
write-up: `workflow/decisions/ADR-015`.

## Signal-quality audit: a latent motion-vector-scale gap, found and fixed

Audited the 4 real explicit inputs (color/depth/motion/exposure) against NVIDIA's own reference
DLSS SDK helper code (`research/nvidia/dlss/include/nvsdk_ngx_helpers_d3d.h`) rather than assuming
this fork's existing handling was correct. Found one real, if latent, gap: `PreExposure` already
gets a "treat a degenerate value as the safe default" guard
(`DlssNr_Pipeline_Dx12.cpp`, `if (frame.PreExposure <= 1e-6f) frame.PreExposure = 1.0f;`), but the
immediately-adjacent `MvScaleX`/`MvScaleY` reads had no equivalent - despite NVIDIA's own reference
helper explicitly doing exactly this for MV.Scale (`InMVScaleX == 0.0f ? 1.0f : InMVScaleX`). Not
triggered on 007 First Light today (its logged scale is a clearly nonzero `-853 x 480`), but a game
that ever reports an explicit `0.0` (a transient not-yet-computed value, an init-order race) would
have made every motion vector this pass reads collapse to zero - breaking MV-based reprojection in
both the RR residual accumulator (ADR-011/012) and the new evaluation-cadence carry-forward
(ADR-014) exactly the way both are designed to prevent. Fixed with the same guard NVIDIA's own code
uses. Depth subrect handling and exposure tracking showed no equivalent gap in the same pass.

## Cadence=2 capture attempt: technically clean, landed on a static screen

Used this fork's own built-in frame-capture mechanism (`dlssnr-capture.trigger`, writes real
before/after `.raw` frame pairs - decoded locally with a small script, R11G11B10_FLOAT, Reinhard
tonemap) to get a real visual read on cadence=2 without needing a human watching live. Result:
technically clean (no crash, no corruption, no visible artifact in the NR edit itself), but the
8 captured frames showed essentially zero motion (max pixel delta 2/255 across the whole run) -
the game was sitting on a static, non-gameplay screen (a menu or completed loading screen, most
likely, matching a ~70% DLSS history-reset rate observed in the same session - games commonly
force continuous resets while paused/at a menu). This doesn't exercise reprojection under real
motion, so it doesn't answer the real question. Genuine validation still needs a live play session.

## ReversibleMode was undiscoverable

`DlssNrReversibleMode` has existed in `Config.h` since the hybrid-proxy commits (7ffcf8ee,
76fce1b4) and the shader already implements all five modes, but the ini key was never added to the
shipped template - it silently sat at its default (0, soft-knee) with no way to discover modes 3/4
short of reading the source. Mode 3 (hybrid + composed) measurably improved highlight detail
recovery without the midtone cost of mode 1, in this testing. Documented in the ini template as
part of this PR.
