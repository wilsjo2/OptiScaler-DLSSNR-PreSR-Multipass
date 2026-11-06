# Display Filter validation — 20 September 2026

Hardware: RTX 5090, NVIDIA driver 616.64, Windows 11. Original NVIDIA-signed NR 310.8 runtime. Ordinary driver feature creation and evaluation succeeded using this application's own project identity. No compatibility loader, DLL patching, game injection or game process-memory access was used.

## Feature/control parity update

The capture-compatible NR controls now include up to ten separately configured passes, style/preset/auto mask, 25�200% working resolution with the eight in-game spatial filters, highlight guard, hold/hide edit, full comparison controls/labels and debug views. Controls are grouped into tabs. See `PARITY.md` for scope and the small UI differences.

Headless GPU checks with the D3D12 debug layer passed the new settings round trip (temporary INI), independent ten-pass transitions and pass-10 edits, inactive profile isolation, held-frame identity while source/model size changes, hidden edit/live resume, all spatial filters, 200% scaling, supersampled HDR/three-pass motion and hold-release reset, highlight guard, debug views, exact wipe regions, side-by-side zoom and label sizes. Existing two-pass private-DLSS/motion/appearance checks passed, including exact SDR/HDR bypass and finite HDR highlights. The actual OptiScaler wrapper and standalone retained bit-identical model/compose results on the existing matched-input fixture, including 12 retained-history frames.

Logs: `out-parity/features.log`, `out-parity/private-regression.log`, `out-parity/default-parity.log`. These are correctness checks while the PC is in use, not controlled performance measurements. The new UI was built and statically inspected; no desktop/game was operated or launched. Live visual and game verification is left to the user. This feature update does not claim to fix the previously reported live image-quality gap.

## Monitor mode update

Added screen selection, monitor geometry, capture exclusion for the monitor overlay, and pause/resume while using Display Filter's controls. A live monitor run processed 4K frames, and the monitor fixture confirmed changed NR pixels and exact zero-strength output. The full monitor check remains inconclusive: an initial screen/window comparison exceeded its tolerance, and the latest run stopped when foreground focus differed from the expected fixture window. These results do not establish feedback prevention or a complete focus-switching pass. Further live testing is left to the user as requested.

## Focus crash and estimated motion update

The reported Windows crash record identified an execute access violation in `_nvngx.dll_unloaded`. The dispatcher is now pinned until process exit, including failed initialization paths, while model/GPU resources still shut down normally. This addresses the recorded unloaded-code failure; the original exception was not reproduced live under the debugger.

- Five consecutive NGX initialize/shutdown cycles passed, with the dispatcher remaining resident.
- Window capture of an already-running Raid: Shadow Legends session ran for several minutes in zero-motion mode, including focus loss/resume and normal stop/close, without another observed crash. No game was launched. This establishes operation during that session, not comprehensive image-quality verification.
- New optional GPU motion estimation passed a known (+12, -8) pixel translation fixture: current-to-previous vectors had 0.000-pixel median error, with 97.1% of sampled vectors within 2.5 pixels. Finite-vector, static-frame, first-frame, explicit-reset and scene-cut checks passed.
- The motion-enabled NR path at 1440p produced changed pixels, exact SDR/HDR zero-strength identity, finite HDR output and retained highlights. It measured 6.04 ms mean / 6.69 ms p95 with 906 MiB application VRAM on a static image over 30 frames.

Estimated motion defaults off. It uses GPU luminance pyramids and patch matching, retains NR history during continuous captures, and resets on discontinuities. Depth is still constant; motion quality and performance in live gameplay remain unverified.

## Motion enable follow-up

The reported persistent disappearance of the effect was not reproduced in independent fixtures. Source tracing and measurement did establish a first-enable stall: runtime compilation of the three optical-flow shaders took 3,211.7 ms in the initialization check. Compiling those shaders during the build reduced the same check to 8.1 ms, retaining the same 97.1% translation accuracy. The application also now replaces stale running statistics with explicit pause/resume messages when its output is hidden; pause time is excluded from resumed FPS statistics.

Before this fix, the independent moving-window check already passed 90 frames at 1916×1146, including changed pixels after the first frame and exact comparison of the presented output. The headless model check now also uses a shared capture-like texture, moves the entire image, switches between zero and estimated motion, and requires changed pixels throughout the sequence and consecutive HDR evaluations. This distinguishes successful model processing from the user's still-unconfirmed live presentation symptom. No further desktop interaction or game testing was performed after the user requested headless work only.

The final 4K headless run passed those checks, including exact SDR/HDR zero-strength identity. Motion-enabled processing measured 10.83 ms mean / 11.56 ms p95 over 10 timed frames, with 1,573 MiB application VRAM. This is processing time on the fixture, not live application throughput or end-to-end latency.

## Verified

### FPS cap and NR loader update

Removed the application's Authenticode precheck for `nvngx_dlssnr.dll` and its corresponding packaging gate. The ordinary NGX driver loader remains responsible for initialization and compatibility errors. The build records the actually loaded NR module, rather than identifying only the requested file.

Added a persisted 0–360 processing FPS limit (0 uncapped), an interruptible high-resolution pacing timer, and `Present(0)` in place of refresh-synchronized `Present(1)`. While capped, captures are drained and their GPU-copy waits completed, retaining only the latest input for processing. Stop/capture events interrupt pacing, and visibility/settings are rechecked at most 20 ms later. The source/GPU and compositor refresh still constrain observed frame rates. Private DLSS receives measured capture intervals.

Headless timer checks passed caps of 30, 120 and 240 FPS (29.7, 116.7 and 219.9 measured scheduling FPS on this run), plus uncapped behavior, capture-event wakeup without bypassing the cap, and stop interruption. These are timer-only measurements, not live WGC or presentation throughput. No desktop interaction or game launch was performed.

The matched-settings static history check processed 60 frames, with exactly one reset. Mean edit was 0.0151868 for reset mode versus 0.0155371 with retained history, and their output difference was 0.0031046. This does not explain the user's reported settings-parity quality/strength gap. Logs: `out/parity-history.log`. The settings/default changes from the previous audit must not be presented as a demonstrated fix for that gap.

### Model application alignment audit

See `ALIGNMENT.md` for the source comparison and its limits. Corrected first-pass Local tone default (1), later-pass Local tone (0), separate NR parameter maps, model control ranges (0–2), separate final Detail strength (0–2), all five HDR mapping choices, and finished-SDR input/output conversion. The shared in-game shader and game builds remain unchanged. Existing preferences are preserved except model values over 2 are clamped on load; the running app's settings file was not edited during development.

GPU readback confirmed that the SDR model input matches original sRGB within FP16 precision. SDR-on-HDR white normalization differed by at most 0.000000045 between equivalent 80/320-nit fixtures. Exact disabled/zero-strength output, alpha, full skin/environment protection, one/two passes, colour endpoints, private DLSS switching, moving frames and both HDR codecs passed. Tests used saved files and synthetic colour patches, without GUI interaction or game launch. The new content selector, live strength and reported eye artefacts still need user verification.

Replaying the old defaults reproduced mean RGB edit 0.007578 on the 720p wallpaper fixture; aligned defaults produced 0.012894. At fixed Local tone 1, removing the extra proxy curve did not increase mean edit magnitude; see the audit for that distinction. Logs: `out/alignment-old-defaults.log`, `out/alignment-sdr.log`, `out/alignment-dlss-multipass.log`. The 1440p/50%/two-pass/private-DLSS/motion run measured 7.70 ms mean / 8.21 ms p95 and 1,164 MiB application VRAM over eight timed frames. Measurements are processing-only, not live FPS/latency guarantees.

### Private DLSS residual enlargement

The standalone runtime now owns a separate DLSS Super Resolution feature and parameter map under the existing NGX initialization. The unchanged shared shader encodes the difference between the original low-resolution proxy and the final NR answer into its signed carrier (mode 9), then decodes the enlarged carrier during matched-residual composition (transfer 2). Unit exposure, constant depth, model-resolution zero/estimated motion and zero jitter are supplied. Colour and skin controls are still applied once, after enlargement. All allocation, mode switching and release wait for the app's GPU fence.

Headless checks passed using NVIDIA-signed DLSS SR 310.1.0.0, SHA-256 `AD3E9C07EE864E9702032459A59C6825166766C2CB75BD0318D5626595693BDB`. They covered actual finite changed pixels, distinct DLSS versus Bilinear output, DLSS → Bilinear → DLSS switching, native-size bypass, one/two NR passes, estimated motion across moving frames, exact zero-strength SDR/HDR output, highlight preservation and the appearance controls. The tests included 1280×720 at 50%, 3840×2160 at 50% with two passes/motion, and 2561×1441 at 25% (640×360 model). D3D12 debug validation was requested; no reported errors failed these checks.

Identical-input DLSS evaluations with Reset asserted showed approximately 0.00017–0.00020 mean linear RGB variation on the fixtures. Recreating features to compare the same feature age restored exact normalized brightness invariance and exact appearance-control endpoints; the test tolerance was not loosened. The live path does not recreate a feature every frame. This is a temporal-quality limitation to inspect visually, not evidence of an HDR normalization regression. Synthetic depth/motion and absence of source jitter are not equivalent to engine-supplied DLSS guides.

The 4K/two-pass/50%/motion run measured 9.97 ms mean / 10.38 ms p95 over 12 timed frames, using 1,835 MiB application VRAM. These are short processing-only measurements on a shared GPU, not a live FPS or latency guarantee. Logs: `out/private-dlss-colour.log`, `out/private-dlss-4k.log`, `out/private-dlss-minimum.log`. No GUI interaction, live capture, game launch, game installation change or upstream source modification was performed.

### Separate colour and skin controls

Added the shared shader's independent model-colour strength (0–4), the runtime's skin-structure control (-1–2), and optional skin/environment detail and colour attenuation (0–1), including the colour-based mask preview. Existing defaults are preserved. Final composition runs once after either one or two model passes; these controls reuse the unchanged OptiScaler shader and model parameter helper.

Headless appearance checks passed at native 1280×720 with one pass and at 2560×1440 with a 50% model scale and two passes. Colour strengths 0, 1 and 4 produced different finite results. Enabling separation with all strengths at 1 preserved the previous output exactly. All strengths at 0 preserved the original pixels exactly, including HDR tests with both tone curves. Synthetic warm/blue patches verified mask selection and exact independent protection of skin-coloured versus environment regions. The runtime accepted skin structure 2; these fixtures do not establish the visual quality of that parameter on faces.

The reduced-resolution run also passed resolution/pass switching, white-level invariance, moving frames with estimated motion and HDR highlight preservation. Checks requested the D3D12 debug layer. Logs are `out/appearance-native.log` and `out/appearance-reduced.log`. No GUI interaction, live capture, game launch or existing installation changes were performed. The expanded controls remain for user verification.

### White slider, model resolution and matched residuals

The standalone controls now include an adjustable reference-white slider (0 means desktop; exact numeric entry remains), a 25–100% model-resolution slider, classic/matched-residual selection and the current bilinear enlargement option. The shared in-game colour shader remains unchanged. The native-size path retains its original arithmetic; below native size, both the unchanged model input and final answer are sampled with the same linear sampler before the shared matched-residual composition.

Headless checks passed with 1281×721 capture at 37% (474×267 model), two passes and motion enabled, including a run requesting the D3D12 debug layer. They verified finite pixels and preserved alpha, exact disabled/zero-strength SDR/HDR output, distinct classic/matched output at reduced resolution, identical transfer-mode output at native size, and exact reset-frame output after reduced → native → reduced switching. Motion remained active across a moving sequence; both HDR codecs retained highlights. White-level invariance and explicit reference overrides still passed.

A 4K capture at 50% (1920×1080 model), two passes and estimated motion passed the same checks. Over 12 timed frames, processing measured 15.08 ms mean / 19.72 ms p95 and 1,281 MiB application VRAM. These are short headless measurements on a shared GPU, not a guaranteed live 60 FPS result or end-to-end latency. Both model passes use the selected working size; estimated motion is calculated in that same pixel grid.

No GUI interaction, desktop capture or game launch was used for this update. The new controls and live visual quality still require user testing. Checks used the existing Windows wallpaper file, not a screenshot of the user's desktop.

### Desktop white reference and two-pass update

Read-only display metadata reported 140-nit SDR white on the primary HDR display. Previously, the standalone processor always assumed 203 nits for HDR displays. For SDR desktop images this divided the model's linear input by 203/140 (about 1.45), making it approximately 31% darker than the correctly normalized input before the proxy curve. The application now reads Windows' SDR white level for the selected display, with an explicit reference override for native HDR content. Windows settings are not changed.

The Hybrid menu item also incorrectly passed mode 0 (soft knee) to the shared shader. It now selects mode 3 (Hybrid composed); Neutwo continues to use mode 1. No in-game shaders or source files were modified.

The headless white-level regression uses the same image at 80 and 320 nits. Correct normalization produced zero mean RGB error after dividing out the physical brightness difference for both codecs. Holding the old 203-nit assumption instead produced measurable differences (0.0038590 Hybrid / 0.0036923 Neutwo at 720p). Explicit reference-white overrides also matched the corresponding automatic reference exactly.

Two-pass processing creates independent NR feature histories under one NGX runtime. The first result is clamped to the existing proxy range before the second evaluation; the final edit is resolved once against the original frame. A 2 -> 1 -> 2 pass switch passed, and the second pass produced a measurable additional change. At 1440p with estimated motion, processing measured 9.71 ms mean / 10.29 ms p95 and 1,432 MiB application VRAM. Zero-strength SDR/HDR identity, moving frames, finite HDR output and retained highlights passed. These are headless processing measurements, not live capture or game FPS. No desktop interaction or game launch was used for this update.

The final 4K two-pass run with estimated motion passed the same checks. Processing measured 19.98 ms mean / 21.07 ms p95 over 12 timed frames, with 2,569 MiB application VRAM. This workload exceeded the 16.67 ms processing budget for 60 FPS; the extra pass is optional and the default remains one pass.

### Earlier checks

- Release x64 builds both the application and independent check executable.
- Actual finite, changed output pixels from NR, including an existing recorded KCD2 frame. This used a saved file; KCD2 was not launched.
- Exact FP16 zero-strength identity for SDR and HDR, including highlights above SDR white.
- Hybrid and Neutwo HDR output remains finite and retains highlights above 1.0 scRGB.
- Windows Graphics Capture → shared GPU texture → NR → DirectComposition works in the check executable's moving/text test window.
- A second WGC session captures the presented output and compares it against the NR output texture. The tested 720p, 1440p and 4K frames matched exactly, ruling out a visible-but-empty output window.
- The underlying test window keeps focus, and `WindowFromPoint` resolves through the output to that window. No input synthesis or forwarding is used.
- Decorated-window client cropping, resize to 640×360, capture stop/restart, minimize/restore, and focus-loss hiding pass.
- Packaged control-window startup, automatic local-runtime selection, appearance-setting persistence, stop-message handling and clean close pass. This UI check did not start capture.
- Historical prototype signature-gate checks no longer apply: the selected NR DLL signature gate was subsequently removed. Driver loading/compatibility errors remain explicit.

## Measurements

These are short standalone measurements, not application benchmarks. The selected application will compete with NR for GPU resources.

| Workload | Result |
| --- | --- |
| 1440p model + colour processing, serial GPU completion | 4.33 ms mean; 4.65 ms p95; 883 MiB application VRAM |
| 4K model + colour processing, serial GPU completion | 8.45 ms mean; 9.75 ms p95; 1,522 MiB application VRAM |
| 4K full capture/display, 180 frames | 59.9 steady processed FPS; 1,852 MiB application VRAM |
| 1440p full capture/display, 180 frames | 60.0 steady processed FPS; 1,014 MiB application VRAM |
| 4K acquisition → GPU completion | 8.0 ms p95 |
| 4K acquisition → `Present` return | 0.9 ms mean; 1.3 ms p95 |

The small `Present` return measurement is CPU submission timing: it does **not** include completion of the GPU work, all capture delay, DWM display scheduling, or physical scanout. It must not be advertised as end-to-end added latency. Full-pipeline throughput was approximately the test desktop's 60 Hz cadence.

A subsequent 4K processing run measured 7.96 ms mean / 8.39 ms p95, with the same 1,522 MiB footprint. The table retains the earlier measurement rather than selecting the faster run.

## Remaining verification

No games were launched. Beyond the limited already-running Raid session described above, compatibility with individual applications, text/control readability, motion quality and input feel need verification. Game anti-cheat acceptance is unverified. Constant depth with either zero or estimated motion is an experimental capture-based route, not equivalent to native temporal guides.

Actual HDR content on a physical HDR display, mixed-DPI/multi-monitor moves, hour-long operation, device-removal recovery, and physical input-to-display latency still need validation. Numeric HDR tests and successful scRGB presentation do not establish those results.

The normal OptiScaler release DLL remains unchanged at SHA-256 `36E6162226092A04402597CF5E7114C1DBCAADA7798926C0CDDE43F836554924`. No existing tracked source files, game installations, or upstream pull requests were modified.

Screen-border update: monitor sessions request Windows borderless-capture access before starting. Window sessions are unchanged. Build verified; no live capture or permission dialog was exercised during this update because the desktop is in use.

FPS counter: shader and application compiled; no live desktop/UI verification performed. Counter uses fresh processed captures, resets on pause, and is composited after NR at 50% opacity. Headless model checks omit it by default.

GPU fence-wait fix: replaced the stop-event-only 2 ms polling delay with a completion-fence event plus stop/message wakeups. Headless D3D12-debug-layer check passed: a controlled 3 ms completion delay woke in 3.31 ms on average across 20 iterations; stop cancellation, subsequent queue submissions, and Drain passed. FPS-cap tests also passed. Actual game/capture throughput remains for user verification.

## Capture cadence investigation
The previous application never set WGC MinUpdateInterval. On Windows versions exposing that property, capture now requests 1 ms before StartCapture; the application's selected FPS cap is unchanged. The previous and accepted intervals are logged. This addresses a reported WGC low-cadence behaviour for default/sub-millisecond intervals, but live confirmation is still required.

A headless `--performance-check 1` mode measures display queries and processing both with and without FPS labels. On this machine, the 1440p run measured display queries around 0.004 ms and white-point queries around 0.005 ms. With motion enabled, a subsequent timestamped test measured ~13.5 ms from CPU recording through completion and ~10.7 ms GPU timestamp span. These tests ran alongside the user's existing desktop workload; timings are not isolated benchmarks or proof of the live bottleneck. They do not include WGC or presentation. Motion on/off made no difference to the user's observed ~40 FPS ceiling.

Every existing two-second status report now has a separate Pipeline log entry: acquired pool-frame rate and average CPU elapsed time spent acquiring, waiting for captures, FPS pacing, waiting for GPU completion, recording/preparing, and Present. Record/prepare includes the scene-cut readback wait when motion estimation is enabled. Acquired frames include superseded frames dequeued from the capture pool, not unseen frames dropped by Windows.

Release application/checker build passed. No live capture, foreground changes or games were launched. Keep the previous fence-fix package for comparison.

## Independent capture producer and selectable DXGI backend

Release builds the application and checker. Headless `--capture-queue-check 1` passed with the D3D12 debug layer: FP16 HDR highlights above SDR white, SDR-to-linear conversion, all four display rotations, window-style crop, concurrent producer/consumer operation, protection of the pinned processing surface, newest-only selection, changing dimensions, and backend-setting persistence. Tests use synthetic application-owned textures; they do not start WGC/DXGI desktop capture or create windows. Frame-pacing and cancellation checks also pass (3.29 ms average for a controlled 3 ms completion signal).

The queue's producer waits on its explicit shared GPU copy fence before releasing a source frame and publishing the owned texture. The consumer only changes its pinned slot after processing GPU completion. At most three output surfaces and one reusable conversion input are retained. A capture-copy timeout retains unresolved ownership instead of releasing potentially live GPU resources. DXGI access loss ends capture and reveals the original display; restart is explicit.

Live 1440p/4K throughput, WGC-versus-DXGI comparison, self-capture exclusion, cursor handling, window focus and visible HDR output were NOT exercised: the user is using the PC and requested headless work. Desktop screenshots and games were not captured or launched. The new backend is for user verification, not a claim of 120 FPS.

## External DLSS Frame Generation

See FRAMEGEN.md for the new app-local focus adapter, headless interpolation results, driver-override limitation, and remaining live validation. This optional mode adds a separate HWND SDR/HDR10 presentation path; historical scRGB checks above describe FG-off operation.


## Capture/processing throughput audit — 21 September 2026

The live log at 09:30:33 showed DXGI acquiring/publishing 116.5 FPS while processing 45.3 FPS. Waiting for capture was 0.00 ms per processed frame; preparation took 19.67 ms, final GPU waiting 2.22 ms and Present 0.10 ms. That approximately 22 ms processing cycle accounts for 45 FPS without a 60 FPS sampler cap. Other intervals did deliver fewer captures and sometimes added idle time; this does not establish that capture cadence is always sufficient. Capture already runs on an independent producer thread, so its frame interval is not simply added to processing time for every frame.

The same session selected target 100 with 2x FG, deliberately limiting real processing to 50 FPS. Target 120 with 2x caps real processing at 60. The app does not change these saved settings automatically.

Changes:

- At native NR model resolution, FG reuses the full-resolution current-to-previous scene-motion guide already computed for NR, with the same scene-cut reset. This removes a second full-resolution estimator and its synchronous readback/submit. NR pixels and settings are unchanged. FG interpolation now uses the scene guide rather than re-estimating motion from the NR-modified picture. Lower/supersampled model resolutions, disabled NR motion and bypass modes retain the independent FG estimator; guide dimensions are checked, and switching resets the fallback estimator's history.
- The motion shader caches the current-image patch per thread group, rejects candidates whose accumulated nonnegative cost cannot beat the current best, and skips searches for perfect zero-displacement matches. Search radius, refinement, confidence and scene-cut thresholds are unchanged. All five existing motion fixtures produced identical vector fingerprints before/after, including a translated texture with zero median error and 97.1% of checked pixels within 2.5 pixels.
- Removed empty GPU submissions when draining already-completed captures during FPS limiting/pauses. The bounded capture queue already completes its own GPU copy before publication; selecting a frame queues no GPU wait. Actual processing/FG completion fences remain intact.
- Added separate preparation elapsed-time logs for Reflex, NR recording/motion and FG guides/encoding/copy. These CPU intervals explicitly include internal GPU waits, rather than being described as model-only times.

In a short 4K/two-pass/native-model headless comparison on the shared GPU, independent FG preparation measured 39.105 ms GPU span / 41.326 ms through completion; shared guides measured 35.656 / 39.517 ms. These fixtures include NR and preparation but exclude actual FG interpolation, capture and presentation. They are not a live FPS guarantee, and the currently running desktop/game workload affects absolute timings.

Passed: expanded NR + private SR + FG lifecycle check with native shared guides and 60/80-percent fallback guides, pause/resume/style/resolution switching, explicit assertion that shared guides add no submission/wait, D3D12 validation, capture-queue pinning/rotation/colour checks, pacing and fence wakeups. No game was launched and no live desktop was captured. Visible FG quality and live throughput remain for user verification.

## Private-DLSS edge correction and HDR output controls - 21 September 2026

The private upscaler reconstructs a bounded, signed NR residual. Small excursions outside its current source neighborhood can be amplified by residual decoding and, especially, by Hybrid/Neutwo replace's inverse curves near white. The standalone path now bounds DLSS excursions to the current 2x2 spatial footprint and uses the spatial reference where DLSS alone would cross the HDR inverse's white limit. The shared in-game shader, model settings and ordinary bilinear path are unchanged. One FP16 output-sized texture and one compute dispatch are added while private DLSS is active (about 63.3 MiB at 3840x2160).

--residual-check 1 passed with D3D12 validation. Deliberately injected ringing reproduced bright spikes: a Hybrid inverse-limit case rose from a source value of 5 to 176.25 linear units; correction retained 5. A separate eight-frame sequence through the real DLSS runtime produced 2,760,195 channel samples outside the fixture's carrier envelope, with peak carrier 0.901367; all corrected samples stayed within the source bounds. The extreme white-spike example is an injected regression fixture, not a reproduction of the user's captured scene. Neutral carriers and valid in-range samples were checked for exact identity.

The release application and checker built. --appearance-check 1 passed at 1280x720, 70% model resolution, two NR passes, estimated motion, private DLSS and Hybrid composed: exact bypass, DLSS/bilinear switching, colour and skin controls, reference-white invariance, moving frames and HDR highlight preservation. This also confirmed the edge correction does not make private DLSS identical to bilinear.

The new HDR output controls run in the existing label/output pass, after model composition and before FG conversion. Both default to zero. --feature-check 1 passed using synthetic HDR ramps/colours across all five mapping modes: 1000-nit peak ceiling, exact preservation below the shoulder, shadow lift/darken, unchanged reference white/highlights under black tuning, and an exact return to neutral pixels. Original comparison halves, swapped sides, bars/divider, disabled/zero-strength/hidden-edit bypass, debug views and SDR-display output remained unchanged. Settings persistence and the combined two-pass / estimated-motion / 70%-private-DLSS / Hybrid-replace path passed too.

The hidden FG swapchain check passed again: 119/178/237 reported presents from 60 real submissions at 2x/3x/4x, including an initial non-FG frame per mode. The NR/private-SR integration exercised shared and fallback motion guides, model resize, style rebuild, pause/resume and a bilinear switch. SDR/PQ numerical colour checks and D3D12 validation passed.

All checks were headless, using application-owned textures; no game was launched, live desktop captured or visible test window opened. Physical HDR brightness, perceived black levels, the user's white-pixel artifact and live performance remain for user verification.

## Stage timings and HWiNFO investigation - 21 September 2026

The old Throughput executable (SHA256 27F7846DDF68526D3653665039D99A0450A9FC1316B18F13B0A401FA9CFFD9B4) was observed running during the 10:47 slowdown. At 10:47:28 it processed 56.7 FPS from 121.4 acquired FPS; at 10:47:47 it processed 32.5 FPS from 32.5 acquired FPS. The corresponding processing/preparation plus completion-wait intervals were approximately 14.6 and 12.7 ms. Other samples on the HDR/edge build remained above 100 captured FPS but took around 25-30 ms in processing. These are two different limiting conditions; the logs do not establish one root cause for both.

HWiNFO had no running process, but its HWiNFO64 ETW session remained active with DXGI, DxgKrnl, DWM and other graphics providers enabled. The HWiNFO_215 driver also remained loaded. Windows recorded an HWiNFO nvml.dll access-violation crash at 08:45:37. The crash is a possible explanation for leftover instrumentation, not proof of when this session became orphaned or that it caused the user's current regression. The attempt to stop only HWiNFO64 tracing was rejected by Windows with Access denied; user administrator action is required. No GPU reset, game operation, driver unload or monitoring-configuration edit was performed.

The rebuilt app samples per-stage GPU timestamps every 32 frames, resolving/readback only across existing completion fences. The headless performance checker can also load an isolated or existing INI with --settings <path> (later command-line arguments override it). At 4K, 65% model resolution, two passes, private DLSS and estimated motion, the static synthetic fixture measured about 0.077 ms for edge correction and 0.067 ms for HDR/labels in the independent-FG preparation run. Its overall processing-through-completion mean was 11.865 ms. This excludes live capture, actual FG interpolation and physical presentation and does not reproduce moving gameplay. The orphaned trace was still active, so these are not before/after proof of its effect.

Release build and the 2x/3x/4x hidden FG integration check passed with per-stage timing enabled and D3D12 validation. The check exercised private DLSS, two NR passes, shared/fallback motion, resizing and pause/resume while requiring complete, finite timing readbacks. The timing change does not alter shader pixels, model parameters, capture pacing or FG synchronization. Live improvement is not claimed until the named trace is stopped and the same game scene is compared.

## Bounded production logging - 21 September 2026

Detailed CPU pipeline logging and GPU stage sampling are now opt-in through NR_CAPTURE_TIMINGS=1. With that environment variable absent, the application creates no GPU timing query/readback resources and writes routine performance summaries no more often than every 30 seconds. UI status and FPS updates remain unchanged. Startup, state transitions, errors and runtime messages still reach the bounded logger.

capture.log is capped at 2 MiB with one capture.log.1 backup capped at 2 MiB. Rotation retains the newest 2 MiB from an oversized legacy log. Oversized individual messages are bounded, and a failed backup write does not permit unlimited growth or raise an exception through the renderer.

The release build passed. The isolated --logging-check 1 passed exact boundary handling, one-backup replacement, oversized legacy logs/messages, newest-byte retention and failed-rotation behavior. Headless HDR/control/private-DLSS checks, the synthetic residual-edge check and the hidden 2x/3x/4x FG integration check passed with D3D12 validation. No game or visible test window was launched. Pixel shaders, HDR controls and the private-DLSS edge correction are unchanged in this logging update.

## Capture producer pacing - 2026-09-21

The live 3840x2160 log showed about 120 acquired/published frames per second for approximately 57 processed frames at a 116 FPS / 2x FG target. The capture producer previously ignored this processing limit and issued a full-resolution copy (plus SDR conversion when needed) for every acquired image. This is confirmed redundant work, not proof that it explains all source-FPS-dependent slowdown. The same logs also contain approximately 57 processed FPS with 120 FPS acquisition, so source acquisition rate alone is not a sufficient explanation for every slow interval.

Capture now uses the same fractional target/multiplier rate as NR. Its producer remains independent of processing. Dynamic target changes and stop wake its timer; zero remains uncapped. DXGI retains its previous frame during the wait and releases immediately before reacquiring, allowing Windows to accumulate changes instead of repeatedly copying intermediate desktop updates. WGC drains to the newest frame available at each acquisition; Windows' own WGC capture rate remains unchanged. No processing pixels, shader settings, GPU priorities, source-app settings or completion fences were changed.

Headless Release validation on the RTX 5090, with the D3D12 debug layer:
- Existing capture queue checks passed: exact HDR/SDR conversion, rotations, cropping, resizing, concurrent publication, newest-only selection and protection of the processing frame.
- The actual capture pacing helper plus GPU publication, driven by synthetic 120/240 FPS source timestamps, measured 56.9/57.0 copies per second at the fractional 116/2x cap. These are owned-texture scheduling checks, not live desktop throughput measurements.
- Capture target changes, uncapped mode and cancellation during a one-FPS wait passed.
- Existing 30/120/240 FPS timer checks, capture-event wake, stop cancellation and GPU completion wake passed. No 60 FPS ceiling was introduced.

Microsoft's DXGI frame-ownership explanation: https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-releaseframe

WoW was not launched, controlled or captured for these checks. Live A/B testing at fixed filter settings with 60 versus 120 source FPS remains required. Even with the redundant copies removed, the source game, desktop compositor, NR and FG share GPU resources; reducing the game's internal rendering resolution does not reduce a 4K capture/output surface.
- Hidden FG regression check also passed: 180 real submissions, 534 SDK-reported presents across 2x/3x/4x, two NR passes with private DLSS, shared/fallback motion, resizing, pause/resume and SDR/PQ encoding. No test window was shown or activated.

## Classic composition with private DLSS - 2026-09-21

Private DLSS now supports both transfer modes below native resolution. Classic passes the bounded NR colour answer through SR and the existing Classic composition shader. Matched residuals retains its signed-edit carrier and inverse. The same SR feature, textures and local overshoot guard are reused; the Classic guard treats its input as colour and protects DLSS-only crossings of the HDR inverse's white limit. There are no new per-frame allocations or changes to the shared in-game shader. This adds a comparison route; it does not establish that all reported hair/skin halos are fixed.

The transfer selector remains available with DLSS, and loading settings no longer changes Classic to Matched. Switching the two DLSS inputs resets temporal history. The status identifies DLSS SR colour versus DLSS SR edit. The capture pacing performance fix and bounded, opt-in detailed logging remain intact.

Headless Release checks passed with the D3D12 debug layer:
- Classic+DLSS survives an isolated settings round trip.
- Real DLSS produces changed pixels, distinct from both Classic+bilinear and matched-residual DLSS; transfer/backend round trips restore the same reset-frame Classic output.
- Zero strength, hidden edit and native-size upscaler selection preserve their expected identity.
- Switching Classic/Matched/Classic resets history once per switch and then resumes temporal reuse.
- Two NR passes, estimated motion and all five HDR mappings produce finite output.
- Classic colour overshoot and HDR white-pole fixtures pass. Existing matched-residual fixtures still pass, including the real-DLSS edge sequence (2,760,195 raw out-of-envelope channel samples, all bounded after correction).
- Existing feature and HDR output-control checks pass.

No game was launched or controlled, and no desktop capture was used by these tests. Live appearance, temporal edge stability and gameplay performance remain for user verification.
- Hidden FG lifecycle check also passed with Classic DLSS frames in each 2x/3x/4x sequence: 180 real submissions and 534 SDK-reported presents, including motion, resizing, pause/resume and bilinear switching. No test window was shown or activated.

## Matched-residual halo correction - 2026-09-21

Added a spatial correction to the standalone private-DLSS matched-residual pass. Classic colour SR and bilinear transfer retain their previous paths. The existing NR input is bound as a colour guide in the unused t3 slot, with its extent in the correction pass's guide dimensions. The shared in-game shader, motion estimator, model settings, synchronization and capture pacing were not changed. No new render target, history buffer, DLSS evaluation or dispatch was added.

The correction measures full-resolution contrast within one low-resolution pixel, then compares the decoded DLSS edit with a 3x3 reference weighted by spatial distance and input-colour similarity. A local mean plus two standard deviations, a small absolute allowance and 20% contrast allowance define supported brightening. Only excess positive luminance near strong edges with a reliable matching guide is blended toward the reference. Returning the original carrier outside the mask preserves those values exactly. The existing range and HDR inverse-pole guards remain in place.

The initial 10% contrast allowance slightly reduced a legitimate bright-strand fixture with modest reconstruction contrast. Raising the allowance to 20% preserved that fixture while retaining strong suppression of cross-edge leakage.

Release/headless validation on RTX 5090:
- D3D12 debug-layer edge fixtures passed for SDR and all five HDR mappings. A misplaced +0.06994 edit on a dark pixel beside a light surface was reduced to +0.01400 (supported local edit +0.01), while the bright side remained bit-identical. This is injected, controlled leakage; it is not a measured WoW artifact.
- Flat-area detail, supported bright strands, negative edits, a thin feature without a reliable guide, and Classic colour SR preserved their tested pixels.
- Existing residual range, white-pole, neutral identity and real-DLSS edge-sequence checks passed. The latter still bounded all 2,760,195 raw out-of-envelope channel samples in its flat-guide fixture.
- Full feature checks passed, including Classic/matched switching, native and bypass identity, two-pass temporal processing, supersampling, comparison and HDR output controls.
- At 3840x2160, 65% model scale, two passes, estimated motion and private DLSS, the static synthetic performance check measured correction means of 0.164 / 0.222 / 0.167 / 0.167 ms across its four modes. Before this change they were 0.082 / 0.182 / 0.082 / 0.114 ms. Scheduling outliers occurred; these are short headless measurements, not gameplay FPS or physical presentation latency. Whole-frame timings varied with system load and must not be used to claim a speedup.

Build completed with existing compiler warnings. An offline shader compile with the new correction disabled also reproduced the FXC BoundPrivateUpscale and glyph analysis warnings; GPU pixel and debug-layer checks passed.

The game was not launched or controlled, and live desktop capture was not used. The user must verify hair/skin halos, fine bright detail and temporal stability on actual content. This first conservative correction deliberately leaves ambiguous cases alone.
- Hidden FG lifecycle check passed on the final build: 2x/3x/4x, 180 real submissions and 534 SDK-reported presents, with two NR passes, private DLSS, Classic/matched switching, shared/fallback motion, resize and pause/resume. The fixture window was never shown or activated.

## Matched lighting correction for both upscalers - 2026-09-21

The user reported that the preceding private-DLSS residual heuristic did not remove the hair/skin halos, and that the issue also occurs with bilinear matched transfer. That heuristic and its extra guide binding have been removed. The basic private-DLSS range and HDR inverse-pole guards remain.

The standalone output pass now checks final lighting gain at skin-coloured edges after composition and the skin/environment controls, before HDR tuning and labels. Both subnative matched upscalers use this pass. It compares processed/original luminance with similar original colours farther into the bright side. This also detects insufficient darkening, which the previous positive-residual-only test excluded. The correction preserves hue, uses no temporal state and does not reduce the skin slider globally. Classic, native resolution, bypass, debug views and comparison originals are excluded. No shared game shader, model parameters, motion estimator, frame pacing or frame-generation lifecycle code changed.

A simple additive transfer can produce this failure without DLSS: for full-resolution F, low-resolution L, model M=gL and upsampling U, F+U(M-L) differs from gF by (1-g)*(F-U(L)). A negative edit can therefore leave the bright side of a sharp edge relatively too bright. This is a plausible mechanism established by the transfer arithmetic, not a measured identification of the user's WoW pixels. The final-lighting check also covers HDR replace output without assuming its transfer arithmetic is identical.

Release validation:
- Controlled final-image fixtures passed with the D3D12 debug layer at 99%, 65% and 25% scales. For an entirely negative edit, a skin-side red value of 0.4951 was reduced to 0.3958 versus a supported interior value of 0.3850. For a positive edit, 0.7700 was reduced to 0.6787 versus 0.6600. These are synthetic rims, not game measurements.
- Uniform darkening, neutral lighting, uniform brightening, the tested genuine original highlight, non-skin colours, disabled correction, original comparison regions, dividers and letterboxing remained unchanged. In these tests changes were confined to the injected rim; results stayed finite and did not brighten pixels.
- Existing residual/Classic overshoot, inverse-pole and real-DLSS sequence checks passed. Full feature/HDR checks passed, including both transfer modes and backend switching, bypass/native identity, two-pass motion and history reset/resume.
- Hidden FG lifecycle check passed: 2x/3x/4x, 180 real submissions and 534 SDK-reported presents, resize, pause/resume and bilinear switching. The test window was never shown or activated.
- The first 4K synthetic benchmark (65% model, two passes, motion and private DLSS) completed. The output stage including the new guard measured about 0.069 ms mean in that fixture; it does not exercise a screen full of skin edges and is not a worst-case cost or gameplay measurement.
- A subsequent benchmark-fixture creation script failed before producing its intended skin/edge content. During that attempt Windows recorded an NVIDIA TDR at 14:46:41-43. The new test process failed while creating its NR root signature with DXGI_ERROR_DEVICE_REMOVED (0x887A0005), before processing an image or dispatching the correction. The running older Display Filter also reported shutdown around this reset. Its cause has not been isolated; further GPU tests were stopped. The skin-heavy benchmark and reset-free concurrent-runtime operation are NOT validated.

No live desktop/game image was captured and no game was launched or controlled. The new build was not launched. The user still needs to check the actual hair/skin boundary. Colour selection is approximate: uncertain edges can retain halos and real model-generated edge lighting can be reduced. This replaces the failed heuristic; it is not a claim that the live artifact has been proven fixed.

## Correct subnative lighting reconstruction - 2026-09-21

The user confirmed that the 99% raw model output is clean while final composition has halos, and that 100% and 101% do not have the artifact. The input downsampler and NR evaluation have therefore been left unchanged. The unsuccessful post-composition skin-edge heuristic and its bindings were removed.

Below native resolution the standalone pipeline now prepares a bounded relative-lighting/chromaticity field from the model output and its matching input. Lighting is (modelY-inputY)/inputY; chromaticity is R/Y and B/Y. Classic carries model chromaticity; Matched carries its difference from the input, preserving native colour detail. Bilinear decodes the carrier before interpolation. Private DLSS uses the same field and existing local carrier bounds. The enlarged field reconstructs a native-size model image; the shared resolve then sees model and reference at the same resolution and applies the existing tone mapping, strength, colour and skin controls once. Lighting is kept separate from chroma/gamut fitting, preventing a colour boundary from changing the requested luminance gain.

Near-black input samples infer gain from paired neighbours rather than interpolating an undefined gain into a visible edge. Completely black native pixels remain black. This is a deliberate limitation of multiplicative lighting transfer: model-generated light in a truly zero-valued source pixel is suppressed below native resolution. New colour/gamut limits may also change subnative colour near extreme gamut boundaries. Actual content still needs visual assessment.

The code is isolated to standalone/NrCapture. ModelResize.hlsli contains the reconstruction. The shared OptiScaler shader, model parameters, motion, capture pacing, FG implementation, game installations and upstream PRs were not changed. The 100% path and supersampling above 100% retain their prior processing. The new textures are allocated on resize and reused: compared with the preceding build, private DLSS adds one full-size RGBA16F intermediate; bilinear adds that and one model-size carrier (about 63 MiB and 90 MiB respectively at 4K/65%). Existing fences cover their use/release.

Release checks on the RTX 5090 with the D3D12 debug layer:
- 324 compiled-shader fixtures passed: 99/65/25%, Classic/Matched, SDR/five HDR modes, spatial/full-size carrier paths, uniform darkening/brightening, neutral identity, cool-dark/warm-skin boundaries, zero colour channels and black boundaries. Full-size carrier fixtures inject reconstruction overshoot; they do not claim to emulate every DLSS behavior.
- The old 99% soft-knee composition produced a bright-edge error of 0.066895 against a roughly 0.385 native reference (about 17%). The corrected maximum relative RGB error over the applicable native-reference comparisons was 0.3446%. Classic intentionally retains resized chroma at coloured boundaries, where its luminance was checked instead of requiring identical RGB. Neutral Matched reconstruction preserved the encoded full-size proxy exactly.
- Both HDR inverse-pole fallback fixtures passed.
- Real DLSS edge sequence passed: all 2,760,195 raw out-of-envelope channel samples were bounded after correction.
- Full feature/HDR checks passed, including held frame, independent passes, supersampling filters, debug/comparison views, exact bypass/native identity, transfer/backend switching and two-pass temporal processing. Real NR/DLSS checks at exactly 99/100/101% passed; DLSS selection was bit-identical to spatial selection at/above native, where SR is bypassed.
- Hidden FG lifecycle passed: 2x/3x/4x, 180 real submissions and 534 SDK-reported presents, two NR passes, resize, style changes, motion, pause/resume and bilinear switching. No fixture window was shown or activated.
- A static 4K/65%/two-pass/motion/private-DLSS/soft-knee benchmark completed. NR-only processing through GPU completion averaged 9.704 ms; with the FPS label and shared/fallback FG preparation it was about 10 ms. These exclude capture, interpolation and physical presentation and are not gameplay FPS. The total GPU span includes the new reconstruction dispatches, which do not yet have separate stage counters.

An attempted software-only run failed in Windows d3d10warp.dll while executing the pre-existing encode shader, before reconstruction. The temporary WARP adapter option was removed. Final shader checks run without loading NGX; runtime/FG checks were then run sequentially after the user closed Display Filter. No new NVIDIA driver-reset event was found during these final checks.

No live desktop/game image was captured, no game was launched or controlled, and the packaged app was not launched. The remaining check is the user's actual hair/skin boundary and temporal image quality.

## 2026-10-28: current shared code, edge-aware enlargement and halo controls

Merged main 6125a75b into the standalone source branch. Added spatial edge-aware enlargement below native model resolution, the shared dark-rim filters and radius controls, and Automatic/Original/Optimized model labels. Legacy preset 3 remains available for saved profiles. NR runtime 310.8 is unchanged; this package does not supply the newer optimized network. In-game RR history rejection is not part of the external-capture pipeline.

A fresh x64 Release build passed. Tests used the RTX 5090, NVIDIA NR 310.8, the existing DLSS SR runtime, and the D3D12 debug layer:
- Feature checks passed: settings round trips, 1/10 passes, inactive profiles, hold/resume, all eight supersampling filters, motion, comparison/debug views, private DLSS, 99/100/101% transitions and HDR controls.
- New filters changed actual pixels. Tests covered both halo strengths and radius endpoints, 50/100/125% model resolution, two passes, SDR and all five HDR mappings. Disabling the filters restored identical pixels; hidden model edits preserved the original. Edge-aware bypassed private DLSS and had no effect at or above native size.
- All 324 reconstruction shader fixtures passed, including neutral identity, colour boundaries, black pixels, carrier overshoot and HDR inverse-pole fallbacks.
- The packaged runtime loaded successfully. Standalone/OptiScaler model, resolve and retained-history comparisons reported zero mean RGB difference on the tested input.

The old HDR peak fixture also failed on the pre-update executable: Hybrid replace produced a 658.8-nit peak, below its fixed 1000-nit test target. The fixture now selects a target below the measured peak (up to 1000 nits), and still requires highlight reduction, the ceiling, unchanged lower tones and exact neutral restoration. No production HDR adjustment was changed.

The rebuilt controls window opened and displayed the new tabs and model selector. Live capture/gameplay and visual image quality were not tested in this update. Full logs are in the source standalone/NrCapture folder as *-20261028.log. The previous executable is retained in out-before-20261028.
## 2026-11-05: nonconstant depth sensitivity probe

Added `--depth-check 1` to the headless checker. The Release checker built successfully, and the probe completed on the RTX 5090 / driver 621.01 under the D3D12 debug layer without error/corruption messages. The experiment uses the existing saved Cyberpunk image (`out-flicker/capture-preview.png`), resized to 640x360. The source is panned, two textured/tinted foreground shapes move independently, and brightness varies over 20 frames. No live capture is involved.

There are 11 depth maps: flat far, flat mid, horizontal/vertical/radial gradients, moving foreground shapes, reversed shapes, checkerboard, moving step, deterministic noise and image luminance. A twelfth trial repeats flat far. All trials see identical colour sequences. Three suites use reset on every frame, retained history with zero motion, or retained history with known synthetic motion. Every trial creates a fresh feature, and each uploaded colour/depth/motion texture is verified by GPU readback. Measurements use raw FP16 NR output, before composition, private SR, FG or PNG encoding.

Runtime: the packaged NVIDIA NR 310.8.0.0, verified as the actually loaded DLL, SHA-256 `E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E`. Model preset/style 0/0; intensity/structure/tone 1/1/1; skin structure -1; automatic masking on. Depth is R32_FLOAT, normal (`DepthInverted=0`), at native fixture resolution. The source PNG SHA-256 is `24BC69DFE8C2FCF3436C35BE30D432B0FE7B4CC0FD9EE369CE3A802901B41192`.

Results from 36 trials / 720 probe evaluations:

| Comparison | Maximum absolute RGB difference |
| --- | ---: |
| Every tested depth map versus flat far, within each suite | 0 |
| Repeated flat far versus its first run | 0 |
| Retained history / zero motion versus reset every frame | 0.260864258 |
| Retained history / known motion versus reset every frame | 0.623046875 |
| Known motion versus zero motion, both retaining history | 0.606445312 |

NR was active: the per-frame mean absolute edit from the altered input ranged from 0.026350674 to 0.037267467. Outputs responded to the changing image. All supplied depth patterns reached GPU memory as intended, and every output remained finite. **No raw RGB sensitivity to depth was observed in this runtime/configuration**, while history and motion had measurable effects. This supports deferring a depth-estimation model until a runtime/configuration demonstrates a depth response. It does not prove that all NR models ignore depth, and it does not evaluate private SR or frame generation's depth use.

From the repository root, reproduce with:

```powershell
$env:NR_CAPTURE_DEBUG = '1'
./standalone/NrCapture/out-depth/Release/NrCaptureCheck.exe --runtime 'C:/NR/nvngx_dlssnr.dll' --depth-check 1 --width 640 --height 360 --frames 20 --image ./standalone/NrCapture/out-flicker/capture-preview.png --depth-output ./standalone/NrCapture/out-depth/results-640
```

`out-depth` is a separate CMake build of the `NrCaptureCheck` target. Reports and PNG previews are in `out-depth/results-640/` (`frames.csv`, `summary.csv`, `metadata.txt`); the run log is `depth-check-640.log`. The packaged overlay executable was not rebuilt or replaced. For another machine, supply its NR runtime and an SDR test image; the README shows the standard build command.

## Independent pass resolution and foveation (November 6, 2026)

Both standalone and addon Release/x64 builds succeeded. Standalone validation used the saved Cyberpunk street/vendor fixture at 640x360 with NR 310.8 and `NR_CAPTURE_DEBUG=1`.

`NrCaptureCheck --pass-resolution-check 1` passed:

- Independent 50/125/67% passes evaluated at 320x180, 800x450 and 429x241, with distinct private-DLSS, bilinear and edge-aware results.
- Shared-profile foveation evaluated both passes at 448x270. Independent layouts evaluated pass 1 at 448x270 and pass 2 at 480x316, against ordinary sizes of 640x360 and 800x450.
- Per-pass centre changes, selected/inactive-pass grid previews, exact restoration after hiding the grid, settings persistence and inactive-profile isolation.
- Temporal accumulation resumed after local and upstream layout resets; the test directly inspected the downstream pass's reset flag.
- Frozen-source preservation across source changes and pass resizing, original alpha, finite output, SDR/HDR processing and exact disabled/zero-strength/hidden-edit/skin bypasses.
- No D3D12 error/corruption diagnostics during these checks.

The existing `--feature-check 1` suite passed, including 1/10-pass transitions, supersampling filters, private DLSS, motion, appearance/comparison and all five HDR mappings. The existing `--quality-check 1 --passes 2 --motion 1` check also passed: raw, composed and 12-frame temporal comparisons with the actual OptiScaler wrapper all had zero mean RGB difference on the unchanged shared-profile path.

Evidence: `out-per-pass/pass-resolution.log`, `features.log`, `quality.log`, and `results/`. Both selected-pass wireframe PNGs were visually inspected: spacing becomes denser in each pass's fixed central region and wider toward its compressed periphery. The new controls were reviewed in source; a live GUI/capture session was not restarted because the user's existing Display Filter was running. These checks do not establish in-game performance or visual parity for independent addon passes. Addon profile and raw-reconstruction WARP checks passed; no game was launched or addon installed into a running game.

## Performance update (November 6, 2026)

The investigation used the actual current standalone per-pass build as the baseline. A separate instrumented baseline executable was preserved in `out-perf/baseline/` before pipeline changes. Original settings and the running user application were not replaced.

Confirmed code costs and changes:

- Independent passes estimated the same captured source repeatedly, with a blocking scene-cut readback per estimate. Passes with identical ordinary dimensions and input filter now share the exact estimate and independently transform it for foveation. Local/upstream resets retain their previous semantics.
- The independent chain encoded the source again during final composition. The initial encoded proxy and original carrier are now reused. Intermediate raw stages use a small equivalent resolve after reconstruction instead of running the general appearance shader.
- The compose-only host allocated unused proxy, NR answer, depth and zero-motion textures. Removing them saves 24 bytes per capture pixel: 189.8 MiB at 3840x2160. The unused SDR input carrier is also avoided when applicable.
- Motion search now caches the zero-offset previous-image patches, skips previously scored or provably losing candidates, and avoids search dispatches/readback waits on already-known reset frames. Normal-frame scene cuts are still detected immediately.
- The old GPU stage timer overwrote prior passes with the final pass. It now accumulates distinct timestamp intervals. The benchmark also measures per-frame Resize (only a few microseconds in these runs), uses a deterministic moving fixture and hashes final RGBA output outside timing.

### Repeated 4K comparison

RTX 5090, driver 621.01, NR 310.8; saved Cyberpunk fixture translated by 7/3 pixels per frame. HDR/Hybrid composed, 213-nit reference, Original model, Cinematic style. Each run measured 60 frames after five warmup frames. Each profile ran old/new/new/old in separate processes, with two runs averaged for each build. Debug validation was disabled for timing. Game/desktop activity continued sharing the GPU; these are headless processing measurements, not isolated timings or live game FPS. No capture, presentation or FG interpolation is included in this table.

| Pass profile | Baseline GPU span | Updated GPU span | Interpretation |
| --- | ---: | ---: | --- |
| One native pass, motion off (control) | 14.94 ms | 14.29 ms | Unchanged rendering path; variation illustrates contention/noise |
| 100% then 25%, bilinear, motion off | 20.63 ms | 20.45 ms | No meaningful measured throughput improvement |
| 100% then 25%, private DLSS, motion off | 21.21 ms | 21.26 ms | No meaningful measured throughput improvement |
| 25% private DLSS then 100%, motion off | 21.58 ms | 21.26 ms | No meaningful measured throughput improvement |
| Two independent 50% passes, motion on | 20.19 ms | 17.15 ms | About 15.1% less GPU processing time |

For the final row, the accumulated input-motion interval fell from 8.35 to 4.63 ms (about 44.5%). The earlier single sweep measured 23.59 to 17.21 ms for that case; the repeated comparison above is the reported result. Full-resolution NR remains the dominant cost in the motion-off cases. The severe reported live slowdown with a quarter-resolution extra pass was not established as fixed by these measurements.

A 25% pass has 6.25% of native NR pixels, but still reconstructs to the capture size. Capture, optional private SR, external FG guides/interpolation and presentation retain output-resolution costs. Addon percentages can refer to the engine render subrect before upscaling, and it receives engine guides. Therefore identical percentages/pass counts alone do not establish matched standalone/addon workloads. The matched-input wrapper correctness comparison still passes; no live addon performance comparison is claimed.

### Correctness and evidence

Release build, D3D12-debug-layer per-pass and motion checks, full feature regression checks, and the two-pass temporal quality/reference check passed. The new shared-motion check compares a moving six-frame sequence against independent estimates, including distinct foveation layouts and a local reset, with exact output equality. All seven saved source/processed/wireframe PNGs matched the earlier per-pass version. All six initial benchmark output hashes and all repeated old/new hashes matched for their respective fixture/profile/frame counts. No model scale, appearance control, preset or reconstruction setting was reduced to obtain these results.

Evidence: `out-perf/motion.log`, `pass-resolution.log`, `features.log`, `quality.log`, `paired-measurements.csv`, `paired-summary.csv`, and the `paired-old-*` / `paired-new-*` logs. Profiles and reproduction scripts are in `out-perf/profiles`, `bench.ps1` and `paired.ps1`.

The original performance check incorrectly reset independent FG preparation every frame when NR motion was off. This has been corrected to match the live app's reset condition. The initial `out-perf/baseline-fg/` results are explicitly marked invalid and are not used for improvement claims. Benchmark FG preparation excludes actual interpolation and presentation.

A supplementary retained-motion run with two independent 50% passes plus independent FG guide preparation measured 24.66 -> 20.46 ms GPU span; its full-resolution FG motion interval fell from 5.02 -> 3.07 ms. Final NR-output hashes matched. This single old/new pair is supporting evidence for the motion-search changes, not a measured interpolation or live-game FPS result. Logs: out-perf/fg-motion-old and out-perf/fg-motion-new.

## NR model map and full-body recurrence (2026-11-06)

The earlier NVFP4 implementation was recovered first: d2b65cda / v0.7.2-hybrid replaced eight ViT feed-forward expansion/contraction pairs (blocks 31-38); other operators remained NVIDIA FP8. Later db065ec2 and 67fa0e5a supplied kernel profiling and bottleneck-reuse hooks. Recovered source/provenance: out-model-map/nvfp4-history/recovered-files.json. The new native probe reuses that NVAPI boundary.

### Local model and measured floor

Audited runtime 310.8.0.0 SHA256: e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e. WEIGHTS_HT has 153 records, 71 blocks, 152 executable layers and 641 mapped tensor regions. Resource SHA256: 836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4. The encoder/decoder uses 32/64/128/256/512-channel stages around eight 1024-channel ViT blocks. Record offsets and layouts: out-model-map/weights.

Native tracing confirms 156 kernel calls per ordinary evaluation at 320x180, 960x540 and 3840x2160. The 320x180 input uses a 320x320 working image. An ascending/descending six-resolution sweep measured roughly 2.2-2.8 ms median at small sizes, 3.3-3.5 ms at 1080p and 7.3-8.8 ms at 4K. The user's live GPU workload remained active; these do not establish an intrinsic minimum or uncontended benchmark.

A 320x180 instrumented evaluation took 2.24 ms: split-Swin approximately 0.70 ms, ViT 0.63 ms and 256-channel Swin 0.49 ms. Pre-block, post-block and final copy together were approximately 0.038 ms. At 960x540 those three deep groups accounted for about 2.04 ms of 2.60 ms. Per-launch timestamps alter scheduling, so these are approximate interval attributions. The evidence does not support treating the entire 2.5 ms as removable image-entry/exit overhead. CPU launch time is separate from GPU elapsed time.

NrModelProbe feeds model-space RGBA16F directly, excluding capture, scaling and colour composition. Frames 5-24 supply timings; frame 25 is the excluded trace/control sample. Release build and D3D12 debug checks passed. Off/census/timed modes produced bit-identical frame-25 output at 384x216, SHA256 4fd957bfc0fddd185a1f04f8fe1d987282756a110f63ee5e71098b8d608d9ac4.

### Full-model controls before pruning

tests/nr_model_loop.py uses locally extracted weights and the separate mauri870/dlssnr-pytorch reference at commit 3beef7935a8ef976b47ab78b906e75600c52d2de. It compares:

- Complete model cycles: all 152 layers each cycle, linked by internal RGB tensors with crop/clamp/mirror-padding and optional FP16 rounding. No uint8 conversion or intermediate image readback.
- Complete feature-body cycles: pre-block 0 once, all 150 interior layers in blocks 1-69 each cycle, post-block 70 once. Interior encoder skips are recomputed; the original pre-block skip and residual RGB reference survive. Two cycles execute 302 learned layers versus 304 for two complete passes.

One-cycle schedulers match exactly. Both boundary blocks contain learned attention/projections and nonlinear processing. Later tone conditioning enters through the pre-block, so feature-only repetition cannot reproduce independent later tone settings. CPU checks cover original, translated and contrast-altered 384x216 images, including style 2 with tone 1 then 0. The reconstruction is single-frame only; no temporal-quality claim is made.

### Native full-body feedback

The native experiment copies final 32-channel features into the original body-input buffer, resets completion counters and replays all 152 body kernels (150 learned layers plus two layout repacks). The final post-block executes once. No interior stage is pruned. Each extra body also adds one counter-clear kernel and one GPU copy dispatch. At 384x216 the feedback tensor is 983,040 FP8 bytes.

The probe requires the exact audited DLL hash and kernel names/parameter layouts. Pool allocation, UAV state and GetGPUVirtualAddress provenance were checked in the runtime. Alias, dimension, counter-span and preserved-skip guards run before submission. Conservative global UAV barriers separate replayed kernels. This code is linked only into NrModelProbe, not display_filter.

Native two-body tests completed 26 reset evaluations on each of three images with D3D12 debugging enabled, finite output and no validation errors. Repeating the base test produced identical first and final outputs. Compared with two complete native passes (style 0, tone/structure/intensity 1 on both):

| Fixture | One normal pass PSNR | Full body repeated twice PSNR |
| --- | ---: | ---: |
| Original | 27.89 dB | 12.40 dB |
| Translated | 28.08 dB | 12.51 dB |
| Contrast altered | 26.34 dB | 12.51 dB |

The native feature loop produces severe colour noise, qualitatively matching the reconstruction. Matching tensor dimensions does not make decoder features suitable encoder inputs. This direct loop fails the two-pass-quality objective. Its conservative implementation measured about 5.0-5.3 ms median at 384x216; these are correctness-test timings, not an optimized result or a speedup claim.

Native evidence: out-model-map/native/full-body-native-results.json and .csv, full-body-*-2.log, full-body-base-2-repeat.log, and native-full-loop-comparison.png. Kernel traces and resolution measurements are beside them. CPU controls remain under out-model-map/ablation/full-body-loops. Reproduce from the standalone directory:

```powershell
./out-perf/Release/NrModelProbe.exe <absolute-runtime.dll> <model-space-input.rgba16f> 384 216 <output.rgba16f> 1 1 1 1 off 2
```

The previously delivered performance standalone remains the runnable application. Feature feedback was not promoted into its rendering path.

## Narrow NR bookend bridge experiment (2026-11-06)

The narrow boundaries are after block70's full-resolution Swin output and before block0's Swin input. The learned RGB projection is3x32; the reset-mode colour lift is32x3. Their composition has rank3, so direct32-channel identity is invalid. CPU exact fusion preserves composition clamps, FP16 rounding, noise/conditioning, mirror padding and the output residual anchor; both lifted features and final two-pass output match the complete reconstruction exactly on three384x216 fixtures.

The native probe now extracts the audited DLL's PTX and JITs only its bookends. Original JIT control, export-only, pre-lift self-import and raw-half residual-anchor control each passed26 D3D12-debug evaluations with bit-identical base output. The final two-cycle path preserves both pre/post Swin bodies and every interior layer; it removes the first post RGB tail, imports the next lifted features, and carries a small residual anchor. Its factored shader is approximate because the original lift is already FP16-rounded and it changes projection accumulation.

Against actual two-full-pass DLL output, native PSNR was42.40/41.67/41.67dB and SSIM0.99729/0.99733/0.99631 for original/translated/contrast fixtures. All26 evaluations per fixture were finite, debug validation passed, and first/final images matched exactly. Removing the first RGB tail was bit-exact against the earlier export-only bridge on all three images. Final files: out-model-map/bookends/native/latent-*.rgba16f; metrics: native-bookend-results.json; CPU controls: bookends/reference/fused.

This is a correctness prototype only. Large FP16 feature buffers and conservative per-kernel barriers remain. Existing low-resolution traces put complete pre/post/copy work at about0.038ms; a rank3 bridge cannot remove the repeated backbone's cost. No speed improvement or live-application integration is claimed. Work moved to compute-queue overlap, native negative ViT repetition and second-pass block ablation at the user's direction.

## Queue overlap, native negative ViT repeat and pass-2 bypasses (2026-11-06)

These experiments follow the user's redirection away from bookend fusion. All native work used the same audited NR 310.8 DLL on RTX 5090 / driver 621.01. The game remained running; one snapshot showed 85% GPU utilization before the benchmarks. Timings are workload-contended observations, not isolated hardware minima. No live capture session or game settings were changed.

### Independent features on compute queues

ModelOverlap runs two distinct NR features on the same device, with explicit tone 1 on both. A produces frame N+1 into a different intermediate from the one B reads for frame N. Producer/consumer and slot-reuse fences, COMMON-state handoffs, and a common submission gate are explicit. Each feature has at most one evaluation in flight. Inputs alternate between the base image and a (7,3) translation. Every A/B output matched the corresponding DIRECT reference exactly, including all 480 measured waves and warmups. D3D12 debug validation passed. Initial kernel censuses each contain 156 calls and show distinct writable counter/feature arenas.

Each mode ran twice at each resolution, with 40 measured waves after six warmups per run. Order was single, parallel, forced-serial, parallel, single, forced-serial. Values below average the two run means, in milliseconds:

| Input | One compute queue | Two queues, forced serial | Two queues, parallel | Parallel reduction vs one queue |
| --- | ---: | ---: | ---: | ---: |
| 384x216 | 9.122 | 11.984 | 5.502 | 39.7% |
| 960x540 | 8.172 | 11.557 | 5.934 | 27.4% |

Parallel timestamp intervals overlapped by 5.250 / 5.927 ms on average. Forced-serial overlap was zero; single-queue overlap was zero / 0.009 ms (calibration noise). Timestamps use each actual queue's frequency and clock calibration. Interval overlap alone does not prove simultaneous kernel execution, but the repeated pair-makespan reduction establishes useful overlap under these conditions. Each wave drains before rerecording and performs readback outside NR timing; this is not a measurement of steady-state capture FPS or end-to-end latency. Two-queue forced serialization adds scheduling cost, so the conservative comparison is against the single queue.

Evidence: out-model-map/overlap/paired-summary.csv and .json; timing-*-*.csv/.json/.log, per-feature A/B trace files and saved reference/output images. ModelOverlap.cpp is linked only into NrModelProbe.

### Actual DLL ViT repetition at alpha -1

The new mode snapshots the first ViT result, replays the original repack and all eight ViT blocks (42 native launches including the final repack), mixes 2z minus the repeated result, rounds through FP16 to FP8, and decodes once. Only the ViT completion-counter span is cleared. Strict DLL, ABI, arena, counter, dimension and external-alias checks run before submission. Native repacking preserves padded token lanes. Conservative barriers remain between replayed kernels.

Alpha 0 restored the ordinary one-pass result bit-for-bit over 26 reset evaluations. Alpha -1 completed 26 reset evaluations on each of the three altered fixtures with finite output, no D3D12 errors and identical first/final images. Exhaustive CPU checks of the mixing shader's conversion rules covered all 65,536 half encodings and 64,516 finite FP8 pairs without mismatch.

| 384x216 fixture | One pass vs actual two-pass target | Native ViT repeat vs target | Repeat SSIM |
| --- | ---: | ---: | ---: |
| Original | 27.89 dB | 31.93 dB | 0.97974 |
| Translated | 28.08 dB | 31.36 dB | 0.97765 |
| Contrast altered | 26.34 dB | 30.91 dB | 0.96305 |

The repeat captured 68.4–82.2% of the second-pass change's directional amplitude, with cosine 0.747–0.825. This is improved agreement, not equivalent two-pass output or an objective image-quality judgment. At 960x540 the base fixture improved from 26.67 to 29.40 dB; directional amplitude was 43.5% and cosine 0.685. No retained-history or game-motion stability claim is made.

Cost was measured in one process, interleaving one pass, the ViT repeat and two complete passes on the same DIRECT queue. Both full passes use tone 1, reset true and RGBA16F handoff. Nine warmup evaluations precede 20 samples per mode; triplet order reverses, although ViT remains in the middle. Each size ran twice; averages of run means:

| Input | One full pass | One pass + ViT repeat | Two full passes |
| --- | ---: | ---: | ---: |
| 384x216 | 4.318 ms | 5.337 ms | 8.809 ms |
| 960x540 | 4.204 ms | 5.040 ms | 8.204 ms |

The repeat added about 1.019 / 0.836 ms to one pass and cost about 39% less than two passes in these runs. The 384x216 repeat run means varied from 4.371 to 6.303 ms, so the absolute cost is not stable under this GPU load. Timing includes snapshot, repacks, mixing and conservative barriers; readback is excluded. All interleaved first/final outputs matched their independent native controls and repeated-run counterparts exactly. CPU time includes evaluation/final validation and excludes BeginVitLoop setup.

Evidence: out-model-map/vit-repeat/quality-summary.csv/json, native-vit-comparison.png, cost-summary.csv/json, cost-*.csv/json/log and alpha0/alpha-minus1 outputs. The exact audited graph/counter contracts are in vit-repeat-audit.

### Second-pass block bypass quality/cost catalog

The catalog maps all 71 blocks and identifies 49 conservative identity candidates: 28 plain Swin blocks, 13 complete split-Swin blocks, and eight complete ViT blocks. Ten layout-boundary blocks require separate repacking work; twelve geometry/bookend blocks are excluded from simple whole-block identity. All candidate destinations are distinct from their inputs and feed subsequent completion counters, so omitting launches alone is invalid: a native replacement must forward/repack data and publish completion correctly.

The CPU sweep feeds each candidate the ACTUAL native first-pass output, then bypasses only selected blocks in pass 2. The complete second-pass reconstruction is the ablation control, and actual native two-pass output is a separate fidelity target. Baseline replay remained tensor-exact after every sweep. Coverage: 49 individual blocks on the base image, six selected blocks on translated/contrast inputs, and two combinations across all three images (67 ablated evaluations).

| Block | Family | Worst PSNR vs complete reference, three fixtures | Largest additional native-reference PSNR loss | Instrumented 4K interval proxy |
| --- | --- | ---: | ---: | ---: |
| 42 | Decoder split-Swin | 46.46 dB | 0.788 dB | 0.0574 ms |
| 43 | Decoder split-Swin | 45.99 dB | 1.245 dB | 0.0588 ms |
| 44 | Decoder split-Swin | 45.74 dB | 0.261 dB | 0.0548 ms |
| 41 | Decoder split-Swin | 44.68 dB | 0.495 dB | 0.0595 ms |
| 46 | Decoder split-Swin | 41.90 dB | 1.117 dB | 0.0594 ms |
| 52 | Decoder Swin-256 | 43.58 dB | 2.066 dB | 0.0689 ms |

Blocks 42–44 are the least disruptive small group in this sample. Bypassing all three together scored 42.62–43.52 dB against the full reference and lost 1.03–1.82 dB of fidelity to the native target. Bypassing 41+42+43+44+46+52 fell to 35.23–36.83 dB and lost 5.06–5.96 dB. Individual ViT block omissions were much more disruptive (28.04–36.37 dB versus complete reference on the base image).

These bypasses were evaluated in the CPU reconstruction, not injected into the DLL. Native full-reference fidelity of roughly 40.7–42.9 dB bounds interpretation. Cost columns come from earlier per-launch traces, which perturb scheduling; replacement copy/repack/counter work is not included. They are not measured savings. The three-block interval sum is only about 0.171 ms at 4K, and quality costs do not combine independently. Inputs cover three versions of one scene and fixed controls, not a broad temporal corpus.

Evidence: out-model-map/pass2-ablation/catalog (71-block contracts, all 49 ranked candidates, native ranges and quality/cost tables); quality/results.csv, quality/combined/results.csv. Reproducer: tests/nr_second_pass_ablation.py. No body bypass was promoted to the standalone or addon rendering path.

## Mixed-resolution overlap validation (2026-11-06, load-on phase)

Work was committed before further builds: `b16dc70b` contains the independent pass profiles, foveation and product performance work; `9aa9e01b` contains the pre-existing research probes. Bookend-bridge and block-bypass development is stopped. These follow-up changes affect NrModelProbe only; no application pipeline option or new display_filter build was made.

The live WoW and Display Filter client extents were both DPI-aware 3840x2160. The requested 100%/25% test therefore uses independent 3840x2160 and 960x540 NR features (percentage of width and height, not pixel count). Both have preset 1, intensity/structure/tone 1 and auto-mask enabled. B's queue clamps and area-downsamples A's previous-frame answer with original-frame alpha. Handoff and NR are separately timed; the complete pair includes handoff. Final reconstruction, composition and SR are outside this model scheduling experiment.

The probe now supports separate pass extents, progressive synthetic pans with known backward motion, and captured frame directories. Retained-history references execute chronologically on DIRECT with one reset at frame 0. Both features are then released/recreated before COMPUTE playback. A0 is primed once; each wave evaluates A(N+1) and B(N), with no repeated B0 in sequence mode. Per-feature guides and ping-pong source/answer slots stay immutable until their owning fences complete. All outputs, including alpha, are compared byte for byte. Error paths terminate only the disposable probe if outstanding GPU work cannot be drained safely.

Validation: Release probe build passed (only existing Detours missing-PDB linker warnings). D3D12-debug runs at the actual 4K/540p sizes passed reset, synthetic nonreset and real-capture nonreset modes. No debug errors were reported. The real sequence contains 96 distinct WGC frames over 3966.708 ms from WoW's character-selection animation, with measured timestamp spacing 33.31..66.70 ms (mean 41.75 ms). The first 47 frames drive each measured sequence run. This is real character/environment animation with a stationary camera, not broad gameplay/camera-motion coverage. Raw scRGB and HDR/white metadata are retained; CPU encoding uses the shared default soft-knee proxy with observed SDR-white 1.75 scRGB. Captured depth and motion are unavailable and explicitly zero, so these results do not validate the motion-estimation path.

World of Warcraft PID 8436 and the user's existing Display Filter PID 18232 remained running throughout. Two runs per mode, 40 measured waves each, six warmup waves per run, order single/parallel/serial/parallel/single/serial. Values below pool 80 samples per row; p10 uses linear interpolation at p*(N-1). All 552 paired waves including warmups matched the fresh sequential references exactly.

| Input/history | Queue scheduling | Minimum ms | P10 ms | Mean ms |
|---|---|---:|---:|---:|
| Alternating input, reset each evaluation | One COMPUTE queue | 8.9958 | 9.9675 | 14.4851 |
| Alternating input, reset each evaluation | Two queues, forced serial | 9.2555 | 10.7223 | 15.7462 |
| Alternating input, reset each evaluation | Two queues, parallel | 8.6630 | 9.3336 | 14.1133 |
| Captured animation, reset only frame 0 | One COMPUTE queue | 9.6228 | 11.4396 | 15.4156 |
| Captured animation, reset only frame 0 | Two queues, forced serial | 9.6777 | 12.2340 | 16.5301 |
| Captured animation, reset only frame 0 | Two queues, parallel | 9.4024 | 11.6713 | 15.4618 |

The appropriate performance baseline is one queue, not the forced two-queue serial control. Parallel reset means improved only 2.57%; retained-history means worsened 0.30%, with p10 also worse. Per-run reset parallel means varied 12.9483..15.2784 ms, larger than the pooled benefit. There is no demonstrated reliable speedup for the loaded 4K/540p retained-history case. Timestamp intervals overlap, but this alone does not establish useful hardware concurrency: in sequence mode B's apparent NR interval expands from 3.7147 ms on one queue to 15.3523 ms in parallel, while full pair time stays almost unchanged. Sequence handoff means are 0.1223 ms single and 0.1077 ms parallel.

Artifacts: `out-model-map/validation-20261106/overlap-summary.csv` and `.json`; per-run CSV/JSON/logs and load snapshots under `game-on-reset/` and `game-on-sequence/`; native traces and exact output samples beside each run. Full raw/model capture and contact sheet are under `wow-animation/`. Runner and summarizer scripts are retained with these local artifacts. Reset-run probe SHA256: 253862d9605815a380c6a923f4d6caf046a2dcd9eef73ae8ebcf8e8f9c92307a. Sequence-run/final probe SHA256: 921a0514214c9c6192b3a0ef1baaefa734396aa21bc7b66ff95dfea4641e5ef3; the intervening rebuild only added the reviewed failure-path guard. DLL SHA256 remains e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e.

The game-closed control is ready and the user has been asked to close WoW and stop live capture. It has not yet run. ViT alpha=-1 validation at these resolutions is deliberately pending that control, following the requested order; no new ViT quality or flicker conclusion is claimed. Existing genuine 4K WoW, Cyberpunk, Kingdom Come II and Factorio captures have been identified for its four-scene set.

## Headless Cyberpunk at 1080p, 1440p and 4K (2026-11-06)

The user redirected validation to Cyberpunk and authorized control of running programs. WoW and Display Filter were closed through their normal UI before this matrix; neither supplied any new input. This is headless NrModelProbe replay of saved Cyberpunk captures, not a live Cyberpunk benchmark. The game-closed control is now complete. There is still desktop GPU activity, and clock drift limits small performance conclusions. Bookend-bridge and block-bypass development remains stopped. No overlap or ViT option was added to the application, and no new application executable was built.

Product changes were committed separately as `b16dc70b`, pre-existing research as `9aa9e01b`, overlap/capture validation as `2641d88d`, and the reviewed ViT temporal harness as `f2acc516` before building it. Only NrModelProbe was rebuilt. Release build passed with existing Detours missing-PDB linker warnings; 1080p retained-history alpha-zero and 4K alpha-minus-one static D3D12-debug runs passed.

### Input provenance and protocol

The old `out-model-map/native/input-3840x2160.rgba16f` was an upscaled 960x540 preview. It is excluded here: previous 4K inference dimensions did not establish genuine 4K source detail. The replacement corpus contains four distinct genuine 3840x2160 Cyberpunk sources: nightclub, car and portrait photomode PNGs plus a vendor model-proxy capture. The three PNGs use RGB/255 and alpha 1. `before_00..07.raw` are already encoded model proxies from `nr.colorCopy`, not untouched scene-linear captures; values are preserved without double encoding. Historical internal render/upscaling and proxy-encoding settings are unavailable. 1080p/1440p fixtures use area filtering directly from 4K; the 4K raw inputs are byte-identical to their originals. Source/output hashes and exact paths are in the artifact `manifest.json`.

The real clip has eight distinct chronological frames with animation/traffic and a stationary camera. It has no recorded timestamps or geometric guides; no FPS/duration is inferred. Its motion/depth guides are explicitly zero. Longer tests use a controlled wrapped 8/4-pixel pan of the genuine vendor capture with known backward motion and zero depth. This is synthetic motion of real captured content, not a longer recorded game sequence. The real eight frames are never looped to inflate sample count.

Hardware/runtime: RTX 5090, driver 621.01, NR 310.8.0.0. DLL SHA256 `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`. Overlap probe SHA256 `921a0514214c9c6192b3a0ef1baaefa734396aa21bc7b66ff95dfea4641e5ef3`; new ViT probe SHA256 `5b708324c423d5204fc81eb500cd1a7be94104c086ab648e88b509e16a96b6b7`. All percentages below scale each dimension: 25% width/height is 6.25% of the pixels.

### Overlap with retained history

All features use preset 1, intensity/structure/tone 1 and auto-mask. For each size pair, two runs per mode measured 40 waves after six warmups, in order single/parallel/serial/parallel/single/serial. Statistics pool 80 samples per mode; p10 is linear interpolation at 0.1*(N-1). Both histories reset only on frame zero. Every output is checked against fresh chronological DIRECT references. The primary performance baseline is one COMPUTE queue, not forced two-queue serialization.

Full-pair GPU time includes the second feature's mixed-resolution clamp/area-downsample handoff. Equal-size overlap retains the earlier raw-output handoff without that clamp; comparisons within each row use identical preparation. Capture, upload/readback, validation, SR, final reconstruction/composition and presentation are excluded. Each paired wave drains before the next wave: these are scheduling/makespan measurements, not steady-state app FPS or capture-to-display latency.

| A -> B | One queue min / p10 / mean ms | Parallel min / p10 / mean ms | Parallel mean change |
|---|---:|---:|---:|
| 1920x1080 -> 1920x1080 | 5.5468 / 6.6792 / 7.5854 | 6.2096 / 6.2407 / 7.1005 | -6.39% |
| 1920x1080 -> 480x270 | 4.7766 / 4.8034 / 6.0171 | 3.4399 / 3.4608 / 4.2101 | -30.03% |
| 2560x1440 -> 2560x1440 | 8.0538 / 8.6771 / 9.7285 | 7.4355 / 8.2701 / 9.5059 | -2.29% |
| 2560x1440 -> 640x360 | 6.7898 / 6.9719 / 7.8264 | 5.4157 / 5.4351 / 6.3350 | -19.06% |
| 3840x2160 -> 3840x2160 | 15.4847 / 18.6261 / 19.2861 | 15.6030 / 18.9227 / 19.6006 | +1.63% |
| 3840x2160 -> 960x540 | 9.2667 / 10.2502 / 12.3378 | 11.3989 / 11.6158 / 12.5482 | +1.71% |

Mixed-size overlap helps at 1080p and 1440p in this matrix; neither 4K configuration improves. At 4K/25%, p10 worsens 13.32%, and the two single-queue run means span 11.85..12.82 ms versus parallel 12.44..12.65 ms. Full/full changes are small and 1440p/4K repeat-mean ranges overlap. Overlapping timestamp intervals establish concurrent scheduling, not free physical execution; at 4K the intervals lengthen while total pair time fails to improve.

All 42 runs passed: 36 performance runs plus six parallel real-eight-frame parity runs. Including warmups there are 1,698 paired waves, 3,396 paired feature-output comparisons and 42 priming A0 comparisons: 3,438 byte-exact outputs. Main performance has 1,440 measured waves. Each real-eight-frame run evaluates A0..7 and B0..6, so it has seven pairs and only one post-warmup timing sample; it is parity evidence, not a performance distribution. An independent CPU audit recalculated all CSV statistics, checked chronology/reset/dimensions, and rehashed 27 unique inputs against Cyberpunk provenance.

### ViT repeat at alpha -1: static cost and agreement

The repeat uses the real DLL, repeats only the eight ViT blocks, mixes `2z - ViT(z)`, and decodes once. Every evaluation validates the audited DLL/ABI, 156 original launches, 42 replayed kernels and counter/alias contracts. Snapshot, repacks, mix and conservative replay barriers are timed. This remains a DLL-specific probe experiment with no on-disk DLL modification.

At each resolution, four scenes each run 20 samples per mode in interleaved one/ViT/two triplets, reversing order every triplet, after nine warmup evaluations. Both full passes use tone 1 and reset true. The two-pass reference clamps intermediate RGB and restores original-frame alpha; its preparation is timed. The table pools 80 samples per mode/resolution. This DIRECT/reset experiment differs from the COMPUTE/retained-history overlap experiment; compare modes within a table, not absolute timings between tables. Feature creation, uploads and readback are excluded. Reported CPU Evaluate time excludes BeginVitLoop setup and is not total CPU recording time.

| Resolution | One pass min / p10 / mean ms | ViT repeat min / p10 / mean ms | Two passes min / p10 / mean ms |
|---|---:|---:|---:|
| 1920x1080 | 2.7643 / 2.7733 / 3.2039 | 3.4971 / 3.5048 / 4.0893 | 5.4863 / 5.5042 / 6.5057 |
| 2560x1440 | 3.6369 / 3.6515 / 4.2128 | 4.4221 / 4.4444 / 5.1569 | 7.2333 / 7.9840 / 8.5668 |
| 3840x2160 | 6.7187 / 6.9314 / 7.5767 | 8.4833 / 8.9719 / 9.4431 | 14.1925 / 14.6035 / 15.5056 |

The repeat costs 37.14%, 39.80% and 39.10% less than two full passes on mean GPU time, adding 0.8854 / 0.9440 / 1.8664 ms over one pass. One-pass minimum rises from 2.7643 ms at 1080p to 6.7187 ms at 4K: the earlier tiny-input floor does not describe cost across these target sizes.

All 12 static cases move closer to two complete passes in raw RGB error: PSNR gains range +2.81..+6.30 dB, with 44..70% projection onto the second-pass change. This is agreement with a chosen reference, not perceptual ground truth or equivalent two-pass output. SSIM is slightly lower than one pass for all three car cases (about -0.0011) and essentially tied/slightly lower for the 4K portrait (-0.000061). Native crops show tone moving toward two passes without reproducing every local structural change. All 36 modes have finite, byte-identical first/last reset outputs.

The new chronological harness gives each mode fresh features, resets only frame zero, saves every output, and uses the same clamped/original-alpha two-pass handoff. Six alpha-zero controls (12-frame pan plus eight real frames at each size) matched one-pass RGBA bytes on all 60 corresponding frames, including warmups. All six alpha-minus-one runs (48-frame pan and eight real frames at each size) completed with finite outputs: 504 mode outputs. Separate whole-mode sequence timings are secondary because clocks can drift; the interleaved table above is the primary cost comparison.

### ViT retained-history stability

The CPU analyser reads raw FP16 RGB without image resizing/quantization, checks full RGBA finiteness and alpha-zero identity, and uses 11x11 Gaussian SSIM (sigma 1.5). Its stripe SSIM, motion-alignment sign and byte checks were independently checked. The synthetic clip discards six warmups, leaving 42 frames and 41 transitions; the real clip discards frame zero, leaving seven frames and six transitions. Pan metrics use a common source-space crop after exact 8/4-pixel unrolling, excluding wrapped edges plus a 32-pixel margin. The input within that crop is analytically unchanged; global attention can still see moving/wrapped context outside it.

Adjacent RMS is the root mean square of consecutive-frame RGB changes. Change-error is RMS((candidate[t]-candidate[t-1]) - (two[t]-two[t-1])). A change-error ratio above 1 means ViT follows two-pass temporal changes less closely than one pass does. For real clips, input animation remains in the adjacent RMS; it must not all be labelled flicker.

| Sequence | PSNR vs two: one -> ViT dB | Adjacent RMS: one / ViT / two | Change-error ratio ViT / one |
|---|---:|---:|---:|
| 1080p-pan | 27.50 -> 30.82 | 0.002254 / 0.002805 / 0.002481 | 1.406 |
| 1080p-real8 | 27.99 -> 31.10 | 0.017473 / 0.017671 / 0.017955 | 1.221 |
| 1440p-pan | 27.34 -> 31.56 | 0.001962 / 0.002679 / 0.002178 | 1.435 |
| 1440p-real8 | 27.31 -> 30.49 | 0.017222 / 0.017539 / 0.017456 | 1.279 |
| 2160p-pan | 28.48 -> 31.87 | 0.001854 / 0.002171 / 0.001981 | 1.327 |
| 2160p-real8 | 28.92 -> 32.30 | 0.017852 / 0.017668 / 0.017349 | 1.175 |

All three controlled pans improve per-frame agreement while increasing aligned variation. ViT adjacent RMS is approximately 13%, 23% and 10% above two passes at 1080p, 1440p and 4K. Its temporal change-error versus two passes is worse than the one-pass baseline on all six clips. The eight-frame real clip has unavailable motion guides and initial history settling, so it cannot establish long-duration gameplay stability. Native crop inspection and per-transition plots complement the raw numerical results; there is no claim of equivalent two-pass detail or flicker-free operation.

Decision: retain the guarded ViT probe for research, but do not carry its DLL-specific replay into the application on this evidence. Its 37-40% static model-time reduction versus two full-size passes is substantial, but the temporal penalty is consistent and structural agreement is incomplete. These full-size ViT results are not a measured replacement for the mixed 100%/25% pipeline. Queue overlap remains a conditional 1080p/1440p mixed-size result, with no demonstrated 4K gain. No further block or bookend experiments were performed.

All local artifacts are under `out-model-map/validation-cyberpunk-20261106/`: `manifest.json` (source provenance), `overlap-summary.csv/json`, `overlap-audit.txt/json`, `vit-cost-summary.csv/json`, `vit-static-cost-audit.txt/json`, `quality-temporal-report.txt`, `analysis-summary.csv/json`, `temporal-pan-analysis.png`, and per-frame raw outputs/timing CSVs. Native run commands and executable/runtime hashes are in the `vit-*-manifest.json` files and overlap run manifests. Reproduction runners and independent audits remain alongside the ignored artifacts. The reviewed reusable CPU analyser is retained in `../../tests/nr_analyse_vit_outputs.py`, with commands in README.md. Large captures, DLLs and build outputs are not committed.
