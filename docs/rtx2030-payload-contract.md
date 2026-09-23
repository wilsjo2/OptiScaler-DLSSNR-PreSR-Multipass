# RTX 20/30 (SM75/SM86) payload contract - dlssg_for_sm86 v0.3.5

Load-bearing facts for the sidecar loader in `OptiScaler/framegen/dlssg/AmpereMfgLoader.{cpp,h}`.
Every statement below is taken from the pinned revision
`sdli1995/dlssg_for_sm86@9621db573e07ed54f50c15bbb585ed9a7bdfac28` (release tag `0.3.5`) - the tag itself
carries **no release assets**, so the payload is taken from the commit tree. Hashes and sizes are in
`vendor/dlssg_sm86/PIN.json`; `tools/check_payload_pin.py` re-verifies them.

Staging: `vendor/dlssg_sm86/{dlssg_sm86.dll,dlssg_sm86.ini,THIRD_PARTY_NOTICES.txt}`. The binary is staged
RENAMED (`version.dll` -> `dlssg_sm86.dll`, bytes unchanged). The ship location decided by todo 2 is
`OptiScaler/dlssg_sm86/`.

## Required file names

| File | Required? | Note |
| --- | --- | --- |
| `dlssg_sm86.dll` | yes | The module. Ships upstream as `version.dll`; upstream `alternatives/` carries the same build under other proxy names. Renamed here; `bundled_name` decided by todo 2. |
| `dlssg_sm86.ini` | yes | **Fixed name, not configurable.** It is read from the *module's own directory*; the two files must always be deployed together in the same folder. |
| `THIRD_PARTY_NOTICES.txt` | yes (distribution obligation) | NVIDIA runtime/models/`nvngx_dlssg.dll` + Coldwood1026's SM75 kernel family are not GPL; upstream asserts GPLv3 in prose but the repo has no `LICENSE` file (residual risk is recorded in the plan). |

## Bundled name (decided by todo 2)

- `bundled_name` = `dlssg_sm86.dll` (renamed, bytes unchanged); `fallback` = `version.dll`.
- Decided by todo 2's real-payload probe: both variants loaded cleanly (`handle=nonnull`,
  `DlssgProxy_Role=1` = active), so the candidate won. Receipts:
  `.omo/evidence/rtx2030-mfg-integration/02/`. The loader takes the name as a parameter (upstream
  `wilsjo2/main` hardcodes `dlssg_sm86.dll`).

## Explicit-load support statement

**Not documented upstream; measured working in todo 2's probe.** Upstream's documented entry path is
proxy-by-name, not `LoadLibrary` from an arbitrary path:

- "A proxy DLL gets into the process by *the game loading a system DLL with the same name*"
  (`alternatives/README.md`).
- "The first proxy loaded in the process (the one whose `DllMain` runs first) takes a process-wide named
  marker and becomes **active**: it installs the `LoadLibrary` hook, reads the INI and loads the backend";
  every later proxy of the same family becomes **standby** - it forwards its exports but installs no hooks,
  reads no INI and writes no log (`docs/INSTALL.en.md`, "Several proxies at once").
- "Every proxy forwards all of its own exports to the real DLL of the same name in `System32`, and
  intercepts only the loading of `nvngx_dlssg.dll`" (`docs/INSTALL.en.md`, uninstall notes).

So, for a sidecar loader that loads the module itself:

1. The module may still become **active** if it is the first of this proxy family in the process - that
   role is decided by the process-wide marker, not by the file name. Loaded after another copy of the
   family, it goes standby and silently does nothing (only forwarding), which the loader must treat as a
   **named failure, never as "installed"** (C5).
2. **Correction (todo 2 measurement): forwarding does NOT follow the file name.** This document originally
   claimed that "the module resolves its forward target from its own file name
   (`%SystemRoot%\System32\<own name>.dll`)". Measured in todo 2 (`scratch-role-encoding.txt`, probe3): the renamed
   copy's forwarded `GetFileVersionInfoSizeW` resolves exactly like the `version.dll` copy (`=1844`,
   `GetLastError=0`), and `DlssgProxy_Name` returns the wide string `version.dll` **in both variants**. So the
   module forwards to the **canonical build name** (`version.dll`), not to the name it was loaded under; the
   rename is therefore not load-relevant for forwarding. The identity export cannot be used to validate the
   rename either, for the same reason. (Related measurement, todo 1: the export set is exactly `version.dll`'s 17
   exports plus two mod-specific exports, `DlssgProxy_Name` and `DlssgProxy_Role`, 19 total, so a loader can
   query which name/role the module actually took. The system `version.dll` has only the 17.)
3. An explicit load of the renamed module does arm the module: todo 2's real-payload probe loaded
   `dlssg_sm86.dll` by explicit absolute path in its own child process and got `status=PayloadLoaded`,
   `handle=nonnull`, `role=1` (active) with the INI applied (`arm:ini-write` -> `arm:load-call` ->
   `arm:result status=PayloadLoaded` -> `init:boundary`, load before the simulated Streamline boundary).
   The `version.dll` branch was equally clean, so the renamed candidate is what ships. Load success is
   load evidence only; no backend signal (`backend_install.status=0`, `install.active=true`) is produced
   by an out-of-game load (C5).

## Arming order (constrains the loader, not the payload)

The payload's architecture rewrite is armed at game start (`dllmain` / `nvapi_load` / `dependency_load`);
`docs/INSTALL.en.md` (`SpoofArchToGame`) states outright that **"anything armed 'once the runtime is
loaded' is too late"** - Streamline 2.x decides "this platform does not support DLSS-G" inside `slInit`,
before a D3D device exists. An "after GPU init" arming point does not satisfy C2; todo 6 owns the seam.

## INI schema keys (v0.3.5)

Sections are added by the writer when absent; a wrong value costs only that key
(`configuration_warning{section,key,value,default,reason}`), the mod stays enabled. The loader-relevant keys:

| Key | Section | Values | Factory | Meaning |
| --- | --- | --- | --- | --- |
| `Optimized` | `[FrameGeneration]` | `0`-`3` | `1` | Consistency tier: `0` stock kernels, `1` bit-identical to the official runtime (default), `2`/`3` lossy, faster. `2`/`3` degrade to `1` on the 310.1 build (`kernel_selection_unsupported`). Out-of-range falls back to **`1`**, not `0`. |
| `MaxGeneratedFrames` | `[FrameGeneration]` | `0`-`5` | `3` | Ceiling on extra generated frames: `5` = 6X, `3` = 4X, `2` = 3X, `1` = 2X, `0` = runtime's own value. 310.1 clamps `5` back to `3` and logs `limit_clamped`. The *actual* count is requested by the game. |
| `Router` | `[Compatibility]` | `Auto`/`SM86`/`SM75` | `Auto` | Kernel family, resolved from the physical GPU; SM75 (Turing) is present in both builds. |
| `Mode` | `[Runtime]` | `Bundled`/`Auto`/`Pinned` | `Bundled` | `Bundled` = the runtime embedded in the proxy (what we ship). `Auto`/`Pinned` need a matching backend; `Pinned` uses `[Runtime] Path`. A missing/mismatched `Path` falls back to `Bundled`. |

Also present in the factory INI: `[General] Enabled=1`, `[Compatibility] Preset=Auto`,
`[Logging] Level=1`, `[Logging] Directory=dlssg_sm86\logs`, `[Runtime] CacheDirectory=`.
Other supported-but-omitted keys (diagnostics/diagnostics-only, e.g. `SM75Family`,
`SpoofArchToGame`, `SpoofArchValue`, `SkipRepeatedRealCopy`, `ImagePatches`) take safe
defaults and are not written by our loader. `KernelImage` is written only for a PTX/Cubin override.
`HardwareBilinear` is forced off by the pinned 310.9 payload (upstream `docs/INSTALL.en.md`,
"Advanced / diagnostic keys"); OptiScaler therefore does not expose or save that control.

## Machine signals (acceptance-relevant, C5)

Logs: `<game dir>\dlssg_sm86\logs\loader_<PID>.jsonl` and `backend_<PID>.jsonl` (`[Logging] Level`; `1` =
errors only, use `2` for the records below).

- `loader: backend_install` - **`status=0`** means the backend installed successfully.
- `backend: install` - **`install.active=true`** means the route is enabled this run; also carries
  `actual_sm`, `image`, `target_sm`/`router`, `optimized_tier`/`optimized_tier_name`, `optimized_knobs`.
- Upstream's own instruction: "To confirm the route installed, check both `install.active=true` and the
  matching `backend_install.status=0`. `image=ptx_sm86` on its own does not mean a kernel was created or
  executed" - the later `kernel_create` / `evaluate` records are the proof. `LoadLibrary` returning a
  handle is never evidence of working MFG.

## Self-sign identity

The release proxy DLLs are code-signed with a **self-signed** certificate:
`CN=DLSSG for SM86 (self-signed)`, SHA-1 thumbprint `85BA66762F851E49148D706915D09026281418E6`.
Re-measured on the staged binary in this todo (`Get-AuthenticodeSignature`): subject and thumbprint match
the README exactly; `Status = UnknownError` because a self-signed chain is untrusted on a stock box - that
is a trust/reputation state (SmartScreen "unknown publisher"), not a tamper signal and not an antivirus
detection. The certificate proves signer identity + file integrity only.

## The 0.3.0 native-to-proxy change

From the 0.3.0 release notes: *"Reverted from the native build to the proxy build. The native approach (a
self-built NGX host) had game-compatibility problems that were hard to fix; this build uses a proxy DLL
around the unmodified factory runtime, leaving the game's calls to NGX unchanged."*

Consequences the loader must respect at 0.3.5: the module is a proxy of a **same-name system DLL**, the
companion INI sits beside it and is always named `dlssg_sm86.ini`, and the 0.2.x native surface
(`[Runtime]` cache/`Mode` semantics of the old spec, `native_<PID>.jsonl` logs, the native NGX host path)
no longer applies.

## Alternatives and the 310.1 runtime variant (recorded, NOT downloaded)

Recorded for the loader's failure matrix. **None of these were fetched** (each is ~28-30 MB):

- `alternatives/` proxy-named builds of the same 0.3.5 module: `d3d12.dll` (30,021,920), `dbghelp.dll`
  (30,039,840), `dinput8.dll` (30,021,408), `dxgi.dll` (30,021,920), `winmm.dll` (30,033,184) bytes.
  Purpose upstream: if the game does not load `version.dll`, use the proxy name the game actually loads;
  `dxgi.dll`/`d3d12.dll` are the higher-risk rendering-path proxies (at most one of the two at a time).
  Note our staged binary is 30,021,920 bytes - identical in size to `alternatives/d3d12.dll`/`dxgi.dll`,
  i.e. the root `version.dll` bytes only.
- `310.1/` runtime variant: the same layout one level down - `310.1/version.dll` (27,995,424) plus
  `310.1/alternatives/{d3d12,dinput8,dxgi}.dll` (27,995,424 each), `dbghelp.dll` (28,013,344),
  `winmm.dll` (28,006,688). The root build embeds the newer **310.9.1** runtime (6X capable); 310.1 is the
  older 4X runtime, where `MaxGeneratedFrames=5` is clamped to `3` and tiers `2`/`3` degrade to `1`.
  We ship the root (310.9) build only.

## R1 verdict

**R1 (upstream `wilsjo2/main:OptiScaler/framegen/dlssg/AmpereMfgLoader.h`): CHANGED - the "Native 0.2.3"
loader assumptions do NOT hold at v0.3.5.** The status-enum shape (`Enabled/DllFound/IniWritten/DllLoaded/
ErrorMessage` -> `Status{...}`) and the "generate an INI with strict clamping" intent survive, but the
native-mode INI spec, the try-setup-after-GPU-init timing, and the hardcoded renamed module name do not.

Adaptation list for todo 3 (vs the upstream main loader):

1. **Module name is a parameter**, sourced from `PIN.json` `bundled_name` - never hardcoded
   `dlssg_sm86.dll` (todo 2 decides between it and `version.dll`).
2. **INI section layout is the 0.3.5 one**: `[General] Enabled`, `[FrameGeneration] Optimized` +
   `MaxGeneratedFrames`, `[Runtime] Mode=Bundled`; `Router` lives in `[Compatibility]`. The 0.2.4 native
   layout (`[Compatibility] Router/KernelImage/HardwareBilinear` + `[FrameGeneration] MaxGeneratedFrames`
   only, no `Optimized`) must not be emitted.
3. **Clamp ranges per the 0.3.5 schema**: `MaxGeneratedFrames` `0..5` (310.1 ceiling 3, `limit_clamped`),
   `Optimized` `0..3` (out-of-range -> `1`; 310.1 tiers 2/3 -> 1), `Router` `Auto|SM86|SM75`,
   `Mode` `Bundled|Auto|Pinned`.
4. **Arming completes before `slInit`** (C2) - a post-GPU-init hook is too late for the architecture gate;
   the loader must expose one explicit `Arm()` and the hook boundary owns the ordering.
5. **Status/logging mirror v0.8.7 `MfgUnlock`** and record the payload's own signals, not load success:
   `backend_install.status=0` + `install.active=true` (+ `kernel_create`/`evaluate`), i.e.
   `Disabled/Ineligible/Conflict -> PayloadValidated -> Loaded -> BackendInstalled -> FeatureCreated ->
   Evaluating -> Presenting`.
6. **Deploy the module and its INI together** (fixed INI name, read from the module's own directory), and
   treat "loaded but standby" (another proxy of the family already active in the process) as a named
   refusal rather than a success.
