# Native Vulkan NR

Incorporates [y4my4my4m's commit 7b7220bb](https://github.com/y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG/commit/7b7220bb) (GPL-3.0).

`IFeature_Vk` owns NR like RCAS/output scaling, using reverse setup and forward dispatch around SR or RR+SR. Early processing uses owned colour scratch; game pointers/layouts are restored. Invalid active rectangles fall back after SR where supported. [Depth/motion regions](NR-MOTION-METADATA.md) remain independent.

Each pass owns its model and temporal history. Passes alternate output images and compose once. For NVIDIA features, a GPU event separates creation uploads from evaluation. Vulkan-to-D3D12 bridges use the D3D12 runtime.

Native Vulkan supports [finished-picture NR](NR-FINISHED-BRIDGES.md). Private-upscaler composition requires D3D12.

## Independent runtimes

Keep the packaged `OptiScaler/nvfp4` folder beside the proxy DLL. Set `Enabled=true` under `[DlssNr]` before starting the game so OptiScaler can request buffer device addresses, binary import and 64-bit image handles when Vulkan creates the device. These models run directly on the game's Vulkan command buffer using the packaged kernels and weights. They do not require `nvngx_dlssnr.dll`.

| Preset | Model | Target GPUs |
| --- | --- | --- |
| 25 | INT8 Original | RTX 30, 40, 50 |
| 26 | INT8 2xPass-in-one | RTX 30, 40, 50 |
| 4 | NVFP4 Original | RTX 50 |
| 21 | NVFP4 2xPass-in-one | RTX 50 |
| 22 | NVFP4 3xPass-in-one | RTX 50 |

For INT8 Original before super resolution, keep your native Vulkan upscaler selection and set the following in `OptiScaler.ini`.

```ini
[DlssNr]
Enabled=true
Preset=25
RunBeforeSR=true
DeferredDLSS=false
Passes=1
WorkingScale=1.0
```

The NR status and log identify a model that cannot run on the current device. The custom FP8 pass-in-one presets require the signed-kernel D3D12 path.

On Windows with an RTX 5090, all five independent presets produced byte-identical RGBA16F output to the D3D12 host on three 321×193 test frames per preset, covering reset, temporal history and Cinematic grading. Gameplay, Proton and RTX 30/40 hardware remain unverified.
