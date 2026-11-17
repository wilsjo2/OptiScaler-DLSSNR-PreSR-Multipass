# External DLSS Frame Generation — test build

The optional Streamline path uses Reflex markers, full-output estimated motion, constant planar depth and synthetic camera constants. The output swapchain is sRGB/8-bit SDR or PQ/10-bit HDR10; FG-off retains the existing FP16/scRGB composition path. Capture and NR retain floating-point textures. HDR conversion preserves absolute brightness numerically, subject to 10-bit quantization; live HDR appearance still needs verification.

Select 2x/3x/4x before starting capture. Unsupported multipliers report an error. Target FPS is divided by the requested multiplier to cap real processing: 120/2 = 60, 120/3 = 40, 120/4 = 30. Fractional processing rates are preserved; 0 is uncapped. This limits Display Filter, not the underlying application. The on-screen counter uses runtime-reported presents with FG enabled; the status also reports real processing FPS and observed multiplier.

## Process isolation and focus

NVIDIA initially suppressed interpolation with `DLSS-G disabled: window not focused`. Static tracing identified Streamline's shared `IKeyboard::hasFocus` helper, which calls GetForegroundWindow and compares its owning process. NVIDIA's source documents that implementation in [extra.cpp](https://github.com/NVIDIA-RTX/Streamline/blob/main/source/core/sl.extra/extra.cpp).

FrameGenerationFocus redirects only the loaded common plugin's GetForegroundWindow import in this application's private address space. While FG is enabled it returns our own output HWND; otherwise it calls the real API. The HWND's ownership is checked. The actual common module is resolved through its feature API, including NVIDIA's driver-supplied plugin versions. No hardcoded binary offsets or code patches are used. Unknown/already-redirected imports fail explicitly. Teardown disables and restores the import.

This changes no outside process, DLL file, driver profile, Windows focus, source input or game configuration. Our capture/focus checks continue using the real Windows API. No injection, global hooks, focus stealing or input forwarding is used. This is not an anti-cheat acceptance guarantee.

## Verified headlessly on the RTX 5090

- Limiter division, fractional pacing, uncapped mode and setting persistence.
- Streamline/Reflex initialization and NR feature creation in the same process.
- SDR encoding, 320-nit HDR10/PQ conversion and guide reset.
- A hidden application-owned swapchain exercised tagging, Present, disable and destruction. The window was never shown or activated, and no desktop/game capture was used.
- Final run: **120 real submissions, 354 runtime-reported presents**. Requested 2x produced 176 presents from 60 real frames; requested 3x produced 178 from 60.
- D3D12 debug messages were checked after the FG workload; no error/corruption messages failed the check.
- Existing capture-queue, pacing and GPU-fence-wakeup checks passed.

The installed NVIDIA App profile forces Dynamic FG and overrides the requested multiplier; the runtime reports a maximum of 3x here. The app supplies the selected output target to Dynamic FG and reports the observed ratio, but does not alter driver profiles. For exact fixed factors, select application-controlled FG in NVIDIA App. The real processing cap always uses the factor selected in Display Filter.

Runtime-reported extra presents establish that background interpolation executes. They do **not** verify physical scanout, visible image quality, live capture exclusion with the new HWND swapchain, click-through operation during FG, actual HDR display, latency or sustained 1440p/4K throughput. Those remain user tests. Estimated motion and flat depth are experimental; expect limitations around HUD/text, occlusion and fast movement. FG adds processing cost and does not repair a slow capture cadence.

The bundled six FG DLLs came from NVIDIA's official streamline-sdk-v2.12.0.zip, SHA-256 F5C0A3D870707DDDC3570FB4BCD3655CF48A8A68C3A9D342910CFA21B77DCF48, verified against its release digest. NVIDIA can still load its driver-provided overrides. DLSS SR is included; supply NR separately. No game builds/installations or upstream PRs were changed.

## Retest after disabling the NVIDIA App FG override

The fresh hidden-window test passed using the packaged runtime. Each selected multiplier was honored exactly: 60 real submissions produced 120 presents at 2x, 180 at 3x and 240 at 4x (180 real / 540 reported total). The runtime now reports support up to 6x; the application currently exposes up to 4x. NR coexistence, SDR/HDR conversion and D3D12 validation also passed. This supersedes the active-override limitation observed above for the current driver settings. No code or executable change was required. Visible quality and physical scanout remain unverified.

## WoW crash investigation and presentation lifecycle fix (21 September 2026)

Located the actual WoW Classic beta reports under `D:/Battle.net/World of Warcraft Forever/World of Warcraft/_classic_beta_/Errors`. Read the live Display Filter log through the localhost file share to avoid the desktop host's redirected AppData view. The failures at 08:45 and 08:51 coincide with NVIDIA TDR resets. Display Filter reports DXGI_ERROR_DEVICE_REMOVED, followed by WoW rendering assertions/access violations; HWiNFO also faulted in nvml.dll. Both sequences followed a controls-focus pause and resume with private SR active. This establishes a shared GPU failure, not a hook into WoW; it does not identify the failing driver instruction.

Static review found a concrete DLSS-G integration violation. Presentation::Hide changed window visibility before disabling FG, while Presentation::Present called SetWindowPos on every FG frame. NVIDIA's [DLSS-G guide, sections 12 and 15](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md) requires FG off before window manipulation to avoid deadlock/instability, and states that option changes take effect at the next Present. Merely setting Off after hiding, or immediately overwriting Off with On on resume, did not quiesce asynchronous presentation.

The corrected path:

- Sets FG Off and consumes it with a synchronous Present before hiding, moving, resizing or destroying the output window.
- Retains FG allocations across temporary pauses, without keeping interpolation active during window changes.
- Positions the window while FG is off, presents one real processed frame, shows it without activation, then enables interpolation on the next frame.
- Removes the per-frame SetWindowPos call from the FG path.
- Applies staged FG settings once per frame and chains NVIDIA's explicit input-consumption fence into the application's resource-completion fence.

Expanded hidden-window checks passed with NR x2 + private DLSS SR + FG, estimated motion, 60-to-80 percent model resizing, style changes, switching to bilinear, pause/resume and repositioning a never-shown window. Each multiplier begins with one ordinary frame: 60 real submissions produced 119/178/237 reported presents at 2x/3x/4x respectively (534 total). D3D12 validation passed; the repeated-options warning no longer appeared. No game or desktop capture was launched, and no game files, outside processes or driver profiles were modified.

This fixes the documented lifecycle violation matching the failures. Live WoW stability and fullscreen/compositor transitions still require user verification; the headless checks do not prove that every driver-reset cause is resolved.

## Motion reuse and throughput update

With NR motion estimation enabled, FG reuses NR's scene-motion vectors and reset flag at any model resolution: at 100% they are handed over as they are, otherwise FG's encode pass resamples them to the output size and rescales their pixel units (no extra pass). Disabled NR motion and bypass modes keep the independent FG estimator. The status identifies which route is active. Completion fencing and the window-lifecycle crash fixes above remain in place. See VALIDATION.md for capture-rate evidence, shader parity checks and the measured headless performance improvement; no fixed live FPS is promised.
