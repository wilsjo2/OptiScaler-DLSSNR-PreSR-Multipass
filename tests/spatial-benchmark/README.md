# Offscreen spatial NR benchmark

From the repository root on a supported NVIDIA host, supply local NGX driver and NR runtime paths:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/spatial-benchmark/run.ps1 -Driver 'C:/path/to/nvngx.dll' -RuntimeDirectory 'C:/path/to/NR-runtime' -Frames 20
```

`-VcVars` can select a Visual Studio toolchain; `-OutputDirectory` defaults to ignored `x64/nr-spatial-benchmark`. Add `-ReverseOrder` to run the four cases in reverse. The runner builds against the current production spatial DX12 shader blobs and `DlssNr_CompatibilityRuntime`, then calls NVIDIA NGX feature 18. It does not start a game or modify an install. Python with NumPy and Pillow analyzes the full RGBA16F output images. Output includes source/runtime hashes, per-frame GPU timestamps, full raw images, PNGs, error heatmaps, and `analysis.json`.

| Case | Global scale | Centre/work per axis | Model extent | Timed passes |
| --- | ---: | --- | --- | --- |
| native100 | 100% | disabled | 3840×2160 | NR |
| spatial100 | 100% | 80/90, offsets 0 | 3456×1944 | colour/guide pack, NR, unpack |
| uniform85 | 85% | linear map, 100% work | 3264×1836 | uniform resample, NR |
| spatial85 | 85% | 80/90, offsets 0 | 2938×1654 | colour/guide pack, NR, unpack |

The 100% pair is compared at 3840×2160; the 85% pair is compared at its common ordinary grid of 3264×1836. Median and p95 GPU times exclude the first two evaluations for each newly created feature. A moving bright block crosses the image centre over four synthetic frames, with uniform depth and zero motion. Separate typed DX12 guide assertions check offset depth/motion subrects, nonzero motion scaling, and curve extrapolation at boundaries.

On an RTX 5090 with driver 616.64, two runs of 20 frames per case (18 timed frames after warmup) gave these median GPU totals:

| Pair | Forward | Reverse | Difference in total time |
| --- | --- | --- | --- |
| native100 → spatial100 | 7.293 → 6.521 ms | 7.317 → 6.420 ms | 10.6–12.3% lower |
| uniform85 → spatial85 | 5.719 → 4.639 ms | 5.647 → 4.644 ms | 17.8–18.9% lower |

Paired output differences on the synthetic final frame were 0.00174 mean absolute RGB / 48.85 dB PSNR at 100%, and 0.00210 / 48.31 dB at 85%. All four full-frame readbacks were finite and nonblack. Exact local DLL and shader hashes live in each run's `manifest.json`.

This is a controlled shader plus real-NR measurement. It excludes the host codec/encode, final transfer/composition, and presentation, so its timing savings and image metrics do not claim whole-game frame gains or moving-scene quality. The baseline `uniform85` resamples through the spatial colour shader's exact linear-map branch as a stand-in for the ordinary pipeline's downsample pass. No game capture is compared here.
