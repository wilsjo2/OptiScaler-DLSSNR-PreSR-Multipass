# Neural Rendering implementation

NR calls NVIDIA feature 18 through the installed driver's NGX core. Users supply `nvngx_dlssnr.dll`; there is no NR helper DLL. See [installation](../../INSTALL-DLSSNR.md) and [credits](../../docs/CREDITS.md).

## Ownership and dispatch

`IFeature_Dx12` and `IFeature_Vk` own model contexts, scratch/history, captures and timers. Resources are lazy; NR defaults off. Vulkan needs NR enabled before device/swapchain creation to prepare extensions and transfer usage.

Adapters pass explicit colour, depth, motion, output and frame metadata into shared pipeline ordering. D3D11/Vulkan bridges reuse D3D12 NR; native Vulkan owns its implementation. Hooks select live owners; menus exchange values/requests without holding GPU resources.

`RunBeforeSR` redirects owned colour before SR/RR+SR, then restores game bindings and resource states. Invalid pre-colour rectangles fall back after SR where valid. Colour must start at origin zero; depth/motion have independent regions.

| Route | Implementation notes |
| --- | --- |
| Ordinary pre/post NR | [Placement and independent pass histories](design/pre-sr-multipass.md) |
| Early generation, late application | [Private edit upscaling](../../docs/DEFERRED-NR-DLSS.md); source RR adds [motion accumulation](../../docs/RESIDUAL-ACROSS-RR.md) |
| Finished picture | Guides captured at the upscaler seam; [presentation bridges](../../docs/NR-FINISHED-BRIDGES.md) |
| Reduced post/finished NR | [Matched residual + DLSS](../../docs/NR-DLSS-ENLARGEMENT.md), always private SR |

## Source map

Paths are relative to `OptiScaler/dlssnr` unless prefixed with `shaders/`, which starts at `OptiScaler`.

| Files | Purpose |
| --- | --- |
| `DlssNr_Pipeline_Dx12.*`, `DlssNrPipeline_Vk.h` | Upscaler adapters |
| `DlssNrFeature_Vk*`, `DlssNrFinished_Vk*` | Native Vulkan model/resources/presentation |
| `DlssNr_Menu.cpp`, `DlssNr_MenuControls.cpp`, `DlssNr_MenuOverlay.*`, `DlssNr_PipelineUi.h` | Menu routing/status, controls, comparison tags and chart |
| `shaders/dlssnr/DlssNr_Dx12_Run.cpp` | Encoding, model execution and composition |
| `shaders/dlssnr/DlssNr_Dx12_{Models,Resources,State,ModelState}*` | Model/resource ownership |
| `shaders/dlssnr/DlssNr_Dx12_{DeferredSr,Enlarge}*`, `shaders/dlssnr/DlssNr_Upscaler_Dx12*` | Private upscaling |
| `shaders/dlssnr/DlssNr_Dx12_{Late,FinishedQueue,FinishedCompose}*` | Capture, submission and presentation |
| `shaders/dlssnr/DlssNr_Dx12_Hold*` | Held inputs |
| `shaders/dlssnr/DlssNr_Common.h`, `shaders/dlssnr/precompile/*.{hlsl,hlsli}` | Shared codec contract and shaders |

D3D12 completion markers protect retirement; CPU frame counts only pair logical frames. Replaced owners stay registered until recordings and GPU work finish. Unresolved teardown work remains alive for process exit. Vulkan drains before resource replacement and separates creation/evaluation with events. See [GPU lifetime](../../docs/NR-GPU-RETIREMENT.md).


The model menu's **Edit pass** selector exposes all 30 profiles, independently of the active pass count.
Inactive profiles can be prepared in advance; later passes inherit pass 1 except for Local tone, which defaults to zero.

## Independent resolution and upscaling per pass

Choose **Edit pass**, open **Resolution and upscaling**, and enable **Independent resolution and upscaling** for
that pass. Each pass can use its own model scale (25–200%), supersample downsampler, enlargement method,
private upscaler backend, and peripheral compression layout. Model tuning and resolution inheritance are
separate. Disabling the resolution override restores inheritance without discarding model tuning.

Pass 1 uses the existing global controls. INI overrides use `Pass2WorkingScale`, `Pass2ScalingDownscaler`,
`Pass2Transfer`, `Pass2PrivateUpscaler`, and `Pass2SpatialCompression`/`Pass2SpatialCenterX` etc.; replace `2`
with any pass number through `30`. Every absent key inherits pass 1. Overrides on inactive passes have no
effect. For example, `Pass2WorkingScale=1.25` and `Pass2Transfer=3` let pass 2 supersample independently of a
50% first pass. See the comments in the packaged `OptiScaler.ini` for all keys.

With no active resolution overrides, the existing shared-resolution pipeline is unchanged. With an
override active, the scene is encoded once. Each pass resizes or warps that encoded picture, runs its own
NR history, and reconstructs a native-size encoded answer for the next pass. Inter-pass answers are
clamped to the model's input range. HDR decoding, strength, colour, skin protection, comparison and halo
suppression apply once to the cumulative result. Halo radius is measured in native pixels in this mode;
the compressed-input preview shows the final pass's actual model input.

DX12 and its existing DX11/Vulkan bridges support the independent chain. The private backend selection
also controls temporal enlargement for each pass. Native Vulkan supports spatial reconstruction and
reports a pass-numbered error for unsupported temporal enlargement. Model creation or a failed stage
leaves the game output unchanged; skipped histories reset before the next complete chain. Spatial
fallbacks remain visible in the per-pass status.

The new configuration/profile checks and software WARP reconstruction checks pass, including gradients,
shapes, signed edits and native-detail preservation. The changed DX12/Vulkan units compile. Independent
chains have not been exercised in a live game; GPU memory use, frame-time cost and game-specific image
quality remain unverified.

## Validation

From an x64 Visual Studio developer PowerShell:

```powershell
msbuild OptiScaler.sln /m /p:Configuration=Release /p:Platform=x64 /p:PostBuildEventUseInBuild=false
./tests/dlssnr_proxy/run.ps1
./tests/run_nr_gpu_lifetime.ps1
./tests/run_nr_pipeline_capture.ps1
./tests/run_nr_private_upscaler_smoke.ps1
./tests/run_nr_pass_profiles_smoke.ps1
./tests/run_nr_raw_residual_smoke.ps1
```

Shader edits must regenerate DX12 and Vulkan binaries/headers with matching constants and stable operation numbers. C++ units include `pch.h` first per [CONTRIBUTING.md](../../CONTRIBUTING.md). See [runtime test boundaries](../../tests/nr_private_upscaler_smoke.md) and [game results](../../docs/NR-UPSTREAM-REVIEW.md).
