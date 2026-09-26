# v0.8.92 (fork) - matched guides and motion scale for the NR model

This is a fork release by [jlrouzies-fr](https://github.com/jlrouzies-fr/OptiScaler-DLSSNR-PreSR-Multipass), built from v0.8.91 plus two fixes to what the neural model is given. It is offered to upstream as a pull request.

## Fix 1: guides at the model's working size

### The problem

Every Model resolution below 100% flickered and kept "settling" for several frames after the camera stopped, while 100% was steady. The same model at 100% of a frame the game had already shrunk (DLSS5-Feeder's own work resolution: same pixel count, guides made at that size) was steady too, which put the difference inside this fork's reduced path.

Below 100% the model was given a colour at the working size and depth and motion vectors at the frame's size, with only the vector magnitudes rescaled (`DLSSNR.Width` = working size, `DLSSNR.DepthSubrectWidth` / `MVecSubrectWidth` = frame size). Nothing documents that the model resamples a guide larger than its colour, and the picture says it does not: each pixel's depth and motion belonged to a different place in the frame, the history never lined up, and the model re-decided every frame.

### The fix

Depth and motion are resampled to the working size with the point resample the DLSS-enlargement path already builds for its private upscaler (`DlssNrMode_ResizePrivateGuides`), and handed to the model as a full zero-origin region. The vectors keep the game's units; the working-size scale still applies. Two work-size scratch textures, parked with the other per-size resources. Peripheral spatial compression is untouched: it packs its own guides already.

`[DlssNr] MatchGuides=true` (default) switches it; `false` restores the previous contract for an A/B on the same build. `OptiScaler.log` prints once which guides the model got:

```
DLSS-NR guides matched to the working size: depth and motion 2150x1210 for a 2150x1210 model (the frame's guides are 3072x1728)
```

## Fix 2: motion scale measured against the right size

### The problem

The model path turned the game's motion-vector scale into model pixels with `working width / frame width`. But the model reads the scale in pixels of the motion texture it is handed, and the game's scale turns its vectors into pixels of the size they are measured in, which for low-resolution vectors is the render size. Below 100% in a game upscaling with low-resolution vectors, that conversion shrank every vector. Onimusha: Way of the Sword at DLSS Performance, 4K: the game's scale is 1920 for a 3840-wide frame, and at 70% the model got 1344 where the motion, in the matched 2688-wide guides, was 2688 -- every moving pixel reprojected halfway, which is flicker in motion and a settle after the camera stops. Fix 1 alone did not help there. At 100% the model kept the game's 1920-wide vectors with the game's own scale of 1920, which is why 100% had always been steady.

Games fed at their output size (DLAA, or DLSS5-Feeder, whose frame and vectors are the same size) were never affected, which is why Fable only needed fix 1.

### The fix

The model's scale is `game scale x motion texture the model reads / size the vectors are measured in`: the texture is the matched working-size resample below 100% and the game's own region otherwise; the reference is the render size for low-resolution vectors and the output size otherwise -- the DLSS-enlargement path's formula (`DlssNr_Dx12_Enlarge.cpp`), whose texture is always its own resample. `[DlssNr] RenderMotionScale=true` (default); `false` restores the old conversion. `OptiScaler.log` prints the scale once per change:

```
DLSS-NR model motion scale 2688.0 x 1512.0: game scale 1920 x 1080 measured against 1920x1080 (render size, low-resolution vectors), motion texture 2688x1512 (matched to the working size), model 2688x1512
```

The first v0.8.92 build measured against the working size in both cases. That is the same number with matched guides, but at 100%, where the guides are not resampled, the model got a scale for a frame-size texture while reading the game's render-size one -- 3840 where 1920 was right in the Onimusha case, twice too large -- and 100% flickered where it had been steady. Fixed in the build after v0.8.92; `MatchGuides=false` builds got the same wrong scale at every resolution.

## Verified

- Fable Anniversary (DirectX 9 through dgVoodoo2, 4K, DLSS5-Feeder 32-bit helper, RTX 5090, driver 617.14): 70% Model resolution is as steady as 100% with `MatchGuides=true`, and flickers with `false`, on the same build.
- Onimusha: Way of the Sword (native D3D12 DLSS, Performance, 4K, 70% Model resolution, 2 passes, Finished Picture, Transfer=2): very flickery with fix 1 alone; steady with both.
- The helper's 300-evaluate self-test at 640x360 with `WorkingScale=0.7`: 300/300 both ways, no measurable cost.

Not run: RTX40 MFG. The native Vulkan path (`DlssNrFeature_Vk.cpp`) had the same shape as the D3D12 path before either fix -- full-size guides for a smaller colour, and a `working / frame` scale -- and both fixes are ported to it in the build after v0.8.92 with the same two switches and the same shader mode (two work-size images, RGBA32F for motion as the packed guides already are). That port compiles and is otherwise untested: there is no Vulkan NR rig here. A Vulkan game below 100% is the test; `OptiScaler.log` prints `DLSS-NR Vulkan model motion scale ...` with the texture the scale is for.

## Package

Same layout as v0.8.91 with `OptiScaler.dll` replaced and this file added. NVIDIA model/FG runtime DLLs are not bundled. Source: branch `fix/reduced-scale-guides-0.8.91` on the fork.
