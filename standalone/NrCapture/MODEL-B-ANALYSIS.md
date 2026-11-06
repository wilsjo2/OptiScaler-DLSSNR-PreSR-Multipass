# NR 310.9.16 Model B compared with 310.8 Model A

Measured November 6, 2026, on the local RTX 5090 / driver 621.01. This is an inspection and headless forward-pass comparison. No application runtime was replaced, no app executable was built, and no replay/bypass patch was applied to the new DLL.

## Identification and acquisition

The second model is Model B (`WEIGHTS_CC_SMALL_FP8`) inside `nvngx_dlssnr.dll` 310.9.16.0. [NVIDIA's October announcement](https://www.nvidia.com/en-gb/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/) describes a smaller model and lower GPU cost. The measurements below are independent local observations, not a restatement of those performance claims.

The DLL was downloaded from the [runtime link published by RTX Encore](https://github.com/SilyNoMeta/rtx-encore), `https://encore-dl.nometa.fr/nvngx_dlssnr-310.9.16.zip`. The GitHub connector cannot download binary assets, so the archive was downloaded directly into the ignored research directory. Only its single DLL member was extracted. Windows Authenticode reported **Signature verified**, signer **NVIDIA Corporation**. Its size and SHA256 also match the pinned input documented by [OpenNR](https://github.com/clshortfuse/openNR).

| File | Bytes | SHA256 |
|---|---:|---|
| Existing 310.8.0.0 DLL | 165,840,496 | `e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e` |
| Downloaded 310.9.16.0 DLL | 266,424,944 | `2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f` |
| Download archive | — | `e59e782ac9236353859df3c708ba48e58ee636fbe68827e1a0fe14bd14d399a9` |

The larger new DLL contains both models. Its original `WEIGHTS_HT` resource is **byte-identical** to the entire resource in 310.8, including all 153 records. The added B resource is separate; all 77 of its payload records differ from every old-model record.

Local artifact root: `out-model-map/model-b-20261106/`. The downloaded file is `runtime-310.9.16/nvngx_dlssnr.dll`; acquisition and signature records are `acquisition.json` and `signature.json`. The application's existing `runtime/nvngx_dlssnr.dll` still has the original hash above.

## Graph and weight changes

The PE resource envelopes were parsed independently. Tensor layouts and graph reconstruction use the already pinned [dlssnr-pytorch reference at 3beef793](https://github.com/mauri870/dlssnr-pytorch/tree/3beef7935a8ef976b47ab78b906e75600c52d2de), checked against every record's size, embedded kernel names and decoded FP8 matrices. The native launch census then confirms the executable layer-family counts. Reference comments sometimes say "77 layers" for B; the actual count is **76 learned layers plus one blend scalar**, comprising 77 serialized records.

| Stage | Model A | Model B |
|---|---|---|
| RGB adapter | Block 0 | Block 0 |
| Fused Swin encoder, widths 32/64/128/256 | Blocks 1–22; depths **4/4/6/8** | Blocks 1–8; depths **2/2/2/2** |
| Split-Swin encoder, width 512 | Blocks 23–30; **8 blocks** | Blocks 9–10; **2 blocks** |
| ViT, width 1024, FFN width 4096 | Blocks 31–38; **8 blocks** | Blocks 11–18; **8 blocks** |
| Decoder bridge, 1024 → 512 | Block 39 | Block 19 |
| Split-Swin decoder, width 512 | Blocks 40–47; **8 blocks** | Blocks 20–21; **2 blocks** |
| Fused Swin decoder, widths 256/128/64/32 | Blocks 48–69; depths **8/6/4/4** | Blocks 22–29; depths **2/2/2/2** |
| RGB output | Block 70 | Block 30 |

| Quantity | Model A | Model B |
|---|---:|---:|
| Logical blocks | 71 | 31 |
| Learned layers / native model kernels | 152 | 76 |
| All native launches, including clear/repacks/copy | 156 | 80 |
| Serialized weight resource | 147,695,410 bytes | 113,893,343 bytes |
| Weight payload excluding record envelopes | 147,683,778 bytes | 113,887,490 bytes |
| Mapped tensor regions, excluding blend scalar | 641 | 285 |
| FP8 matrix regions | 342 | 146 |
| Decoded FP8 matrix elements | 143,831,040 | 112,799,744 |

The packed payload is **22.88% smaller**. These serialized sizes and matrix-element counts are not a count of all training parameters. The cuts remove 28 fused-Swin kernels and 48 split-Swin kernels per forward pass. The 40 learned ViT kernels and two ViT repacks remain, along with the same channel widths and stage transitions. At the three tested sizes, A and B have matching pre-block and ViT grid extents; this speedup is not obtained by lowering the requested input resolution.

Both models use **FP8 E4M3 matrices**, with the existing mixed scalar/head formats. This is not the earlier experimental NVFP4 hybrid. All 146 mapped B matrix regions decode without invalid FP8 NaN codes, none is entirely zero, and none has the same decoded hash as any A matrix. The eight corresponding ViT blocks also have different weights: same-role coefficient cosine values range from -0.000357 to +0.000647. This establishes different learned coefficients, not their training lineage; it does not prove how NVIDIA trained or distilled B.

The ViT weight payload stays 100,697,232 bytes, now **88.42% of B's payload**, versus 68.18% of A's. Weight share is not a GPU-time share. The output blend scalar changes from 0.73974609375 to 0.76708984375. All remaining B weight payloads differ too: simply bypassing the omitted blocks in A would not reproduce B.

Full record/tensor maps, absolute offsets, shapes, hashes and family totals are under `weights-a/` and `weights-b/`; `weight-comparison.json` records the comparisons. The new model was also unpacked into 302 named exact-byte tensors in the local `weights-b.safetensors` file. The extra tensor count comes from splitting interleaved Q/K/V matrices and including the blend scalar.

## GPU code and routing

Both DLLs still expose 15 GPU fatbins, each with SM120 PTX and cubin entries. The newer package removes **111 non-FP8 kernel variants**, retaining 120 named kernels. Of those 120, 71 cubin function-code sections are byte-identical and 49 differ. Kernel names alone therefore do not establish code identity. PTX diffs also contain register, label and symbol renumbering, so textual changes are not counted as semantic optimizations.

The new DLL compresses its main cubins with Zstandard; those cubins were uncompressed in 310.8. Combined stored GPU-module payload drops from 17,163,072 to 3,654,424 bytes through variant removal and compression. This is a packaging change, not evidence of lower steady-state kernel time. Extracted PTX/cubins, per-function code hashes and comparisons are under `fatbins/`, `ptx-diffs/`, `fatbin-comparison.json` and `ptx-comparison.json`.

Native routing checks observed:

- Preset 1: A, 156 launches, on both DLL versions.
- Preset 2: B, 80 launches, on 310.9.16.
- Preset 0: automatic selected B at 1080p and produced byte-identical output to explicit preset 2 in this process/driver configuration.

The old ViT feedback hook remains restricted to the old DLL hash and 156-launch contract. It was not run on the new DLL. B retains the eight-block ViT structurally, but its weights, launch indices and compiled code need a fresh audit before a repeat experiment could be ported.

## Native performance and correctness

Headless NrModelProbe replay used the same genuine 4K Cyberpunk vendor proxy and direct area-filtered 1080p/1440p versions from the previous validation. Depth/motion are zero; reset is true; intensity/structure/tone are 1, style 0 and automatic mask enabled. No game or live filter was running. Feature creation, input upload, readback and presentation are excluded.

At each resolution, order was old-A/new-A/new-B/new-B/new-A/old-A. Each process records 20 timed evaluations after five warmups. The final extra evaluation is reserved for census/parity and excluded. Rows pool 40 samples, with linearly interpolated p10. These are model costs, not game FPS. Two repeats and variable desktop clocks limit small differences, particularly comparisons between the two A runtimes.

| Resolution | 310.8 A min / p10 / mean ms | 310.9.16 A min / p10 / mean ms | 310.9.16 B min / p10 / mean ms |
|---|---:|---:|---:|
| 1920×1080 | 2.7836 / 3.2660 / 3.3498 | 2.8867 / 2.8968 / 3.2867 | 1.9106 / 1.9248 / 2.0294 |
| 2560×1440 | 3.6556 / 4.1552 / 4.4452 | 4.2562 / 4.2884 / 4.4722 | 2.1516 / 2.1550 / 2.4809 |
| 3840×2160 | 6.8221 / 7.0740 / 7.7744 | 6.4896 / 6.6133 / 7.1810 | 3.7625 / 3.7984 / 4.6974 |

B reduces mean time **39.42% / 44.19% / 39.58%** versus our old A DLL, or **38.25% / 44.53% / 34.59%** versus A in the same new DLL. Thus the model choice, not merely replacing the DLL while retaining preset A, provides the substantial gain in this sample.

All 18 primary runs completed. First/final outputs and repeat-run outputs are byte-identical within each model/version/size. A's entire RGBA output is also byte-identical across DLL versions at all three sizes, corroborating the identical weight resource despite code changes. All outputs are finite. D3D12-debug B runs passed at 1080p and 4K. The probe build passed with existing Detours missing-PDB warnings.

Source was checkpointed as `298f46dd` before rebuilding only NrModelProbe. Probe SHA256: `5a60b634efe98c1497efd94eb7ee594265e7a4a7672a5e5c409f97e32b5ab9a7`. Raw samples and launch parameters are under `native/`; `native-manifest.json`, `native-summary.csv/json` and `native-quality.csv/json` record commands, hashes and results. The ordinary probe's p10/p95 reporting was corrected to linear interpolation and its raw samples are now retained.

## Reconstruction and image differences

Every one of B's 76 layers was executed in the CPU reference from the newly extracted weights, with no skipped layers or feedback. The exact FP16 1080p input was promoted to FP32 without PNG conversion. The reference used a 1920×1152 working extent and produced finite RGB output at **49.24 dB PSNR**, mean absolute error **0.00230**, against the native B output. This supports the decoded topology/layout interpretation but is approximate, not bit-exact. The 15.61-second CPU run is not a measure of native inference performance. Details: `reconstruction-b.json`, `reconstruct_b.py`, `cpu-b-1080.npy`.

Four different genuine 4K Cyberpunk captures were compared at identical settings. The native-pixel contact sheet is `model-a-b-crops.png`. It shows different lighting/detail choices, with B generally applying a milder edit in this small corpus:

| Capture | A/B RGB PSNR | B edit RMS / A edit RMS |
|---|---:|---:|
| Nightclub | 37.19 dB | 0.752 |
| Car | 38.14 dB | 0.854 |
| Portrait | 39.32 dB | 0.845 |
| Vendor | 34.90 dB | 0.865 |

Here edit RMS is distance from the input, not an image-quality score. B's changes have 13.5–24.8% less RMS magnitude on these captures, but they are not just a scalar reduction of A. Similarity to A is not ground truth, and this turn did not establish B's moving-sequence stability or two-pass quality. All additional static first/final outputs matched and were finite.

## Reproduction

From this repository root, with a separate reviewed `dlssnr-pytorch` checkout:

```powershell
python tests/nr_inventory_model.py C:/NR/nvngx_dlssnr.dll --reference C:/research/dlssnr-pytorch --variant small --output C:/results/b-weights
standalone/NrCapture/out-perf/Release/NrModelProbe.exe --preset 2 C:/NR/nvngx_dlssnr.dll C:/fixtures/vendor-3840x2160.rgba16f 3840 2160 C:/results/b.rgba16f 1 1 1 1 census
```

Use `--variant ht` and `--preset 1` for A. Use an empty inventory output directory. The research directory retains acquisition, full native commands/manifests, CPU reconstruction and comparison scripts. DLLs, weights, captures, PTX/cubins and large outputs remain local ignored artifacts; only probe/inventory source and this report are committed.
