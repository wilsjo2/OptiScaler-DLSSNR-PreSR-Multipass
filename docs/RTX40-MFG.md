# v0.8.0 with RTX 40 MFG

This branch retains the built-in Ada unlock as an optional build feature. Both builds include the Starfield tracking fix. The unlock is compiled into OptiScaler: no extra helper, ASI loader or external-FG mode is needed.

## Build

The default build excludes the unlock implementation, hooks, capability overrides, menu and configuration field. Ordinary DLSS FG/MFG remains available. Legacy `AdaMfgUnlock` settings are ignored and removed on save.

```powershell
# Without the unlock (default)
MSBuild OptiScaler.sln /p:Configuration=Release /p:Platform=x64 /p:OptiScalerRtx40Mfg=false
./package_release.ps1 -Version nr-standard

# Compile the optional unlock
MSBuild OptiScaler.sln /p:Configuration=Release /p:Platform=x64 /p:OptiScalerRtx40Mfg=true
./package_release.ps1 -Version nr-rtx40-mfg -EnableRtx40Mfg
```

Enabled builds use `x64/Release-RTX40-MFG`; standard builds use `x64/Release`. Separate intermediate folders prevent mixing objects/PCH files. Packaging checks the DLL flavour even with `-SkipBuild` and omits the unlock INI default for standard packages.

`OptiScalerRtx40Mfg=true` defines `OPTISCALER_RTX40_MFG`. The runtime toggle still defaults off. Keeping the restoration/build-flag commits separate lets upstream reviews omit their source changes too: a disabled build flag alone does not remove them from a PR diff.

## Enable at runtime

1. Install the complete **unlock-enabled** package, preserving your INI and separately supplied NR runtime.
2. Under frame-generation settings, enable **RTX 40 MFG unlock (restart)**, save and restart the game. Alternatively set `[DLSSG] AdaMfgUnlock=true` before launch.
3. Enable the game's DLSS FG or configure OptiScaler's normal DLSSG output. Start at 3x and check motion as well as the FPS counter.

The option defaults off and only patches RTX 40/Ada. It requires a supported NVIDIA DLSSG runtime; game multiplier overrides need Streamline 2.7.1+. Keep the game's working runtime. This package does not include NVIDIA FG/NR DLLs or another MFG unlocker.

The patch retargets compatible Blackwell interpolation kernels for Ada and changes two frame-count gates in memory. It exposes up to five generated frames (6x including the real frame) only when both gates and a kernel group match. Unknown/ambiguous signatures remain unchanged. Disabling also requires a restart; it does not undo a live patch.

RTX 20/30 unlocks, external-FG ownership, residual frame interpolation and NVFP4 remain absent. This does not add a missing FG integration to a game. Dynamic MFG and real RTX 40 motion quality remain unverified here; the available test GPU is RTX 5090.

## Validation and source

`tests/mfg_unlock/run.ps1` compiles the production patcher/scanner against controlled PE images. Cases cover both gate layouts, kernel retargeting, repeat calls, unsupported GPUs, missing/ambiguous gates, malformed kernels and restart semantics. `-Runtime <nvngx_dlssg.dll>` additionally patches an image mapped without DLL initialization, under simulated Ada identity; it checks that the disk file is unchanged. Neither test proves real RTX 40 interpolation works.

The branch changes are the patcher, DLSSG/Streamline/load hooks, the toggle and INI handling, project registrations, tests and package documentation. The [NR upstream inventory](NR-UPSTREAM-DIFF-INVENTORY.md) describes the v0.8.0 base.

Adapted from [y4my4my4m's work](https://github.com/y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG/commit/7b7220bb) and the earlier fork's Ada kernel retargeting, under GPL-3.0. This is the built-in implementation, not Dashdogy's separate unlocker.

## Unlock options

Everything below is applied at load, so Save Settings and restart after changing it. It exists only in the unlock-enabled build (`OptiScalerRtx40Mfg=true`). The options sit under **RTX 40 (Ada) MFG Unlock Options** in the overlay, which is drawn under the unlock checkbox on an RTX 40 while the unlock is on. The overlay reports the result directly under each control.

The unlock patches the DLSS-G module and the Streamline plugin in memory only, and no file on disk is changed.

- **Provider discovery.** The DLSS-G module is found by the game's `nvngx_dlssg.dll`, by the driver's OTA store (`models\dlssg\...\<hash>.bin`), and, for a renamed snippet, by a rate-limited walk of the loaded modules. Finding a module never patches it: the signatures decide.
- **Plugin frame ceiling.** Streamline's `sl.dlss_g` lowers its compiled maximum to a device value its wrapper cached. A wrapper that cached 1 then rejects 3X and 4X with `sl::Result` 38. Once the snippet unlock has landed, the one-byte clamp is neutralised. The compiled maximum stays as a hard bound. The source fork applies this only together with its flip-metering option; here it applies whenever the unlock has landed.
- **Frame timing fix** (`AdaTemporalFix = auto | retarget | ptx`). Above 2X every generated frame can land at the midpoint between two real frames. `auto` and `retarget` reuse the Blackwell interpolation kernel the module carries. `ptx` rewrites the Ada kernel's PTX so each frame is blended at its own time, and only works on the DLSS-G builds it has an exact profile for.
- **Software frame pacing** (`AdaFlipMeteringPatch`, default false). Only for a freeze at 3X or more. It edits NVIDIA's plugin in memory, when it loads, so that it takes its own software-pacing fallback, and refuses unless the plugin's code is of the shape it recognises. Try `[NvApi] DisableFlipMetering=true` first, which is milder and ini-only.

A line under **Override DLSSG Ratio** shows what the game asked for, what was sent, and what Streamline says it presented. That is the check to use: confirm the presented count follows the ratio, not just that an FPS counter rose.

### What was checked, and what was not

Real modules were mapped (never run) and searched: `nvngx_dlssg.dll` 310.9 and 310.8, and `sl.dlss_g.dll` 2.13.0.0. Both frame-count gates match exactly once in both snippet builds. The PTX rewrite finds and rebuilds its kernel in both (8 descriptors each). The plugin's frame-count clamp is found once, with a compiled maximum of 5. The flip-metering state is derived from the plugin: context +0x44F0, value 1, one register store to rewrite. `tests/mfg_real_module_check.cpp` reports the same for any build you give it, without modifying the file. Run it on a new DLSS-G build to see whether the patches still find their targets.

**Not run in a game and not on RTX 40 hardware.** Nobody here has confirmed that intermediate frames advance, that the plugin ceiling patch removes the `sl::Result` 38 rejection, or that software pacing ends a freeze. Test moving scenes at 2X then 3X, read the presented count in the overlay, and report the GPU, driver, game, the DLSS-G version shown in the overlay status line, and `OptiScaler.log`.