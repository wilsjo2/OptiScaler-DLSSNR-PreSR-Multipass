"""Compare complete NR model passes with repetition of the complete latent body.

CPU research harness; does not load or modify NVIDIA's DLL. Requires Python3.11+,
CPU PyTorch, numpy, Pillow, safetensors, and a separate checkout of
https://github.com/mauri870/dlssnr-pytorch (tested commit
3beef7935a8ef976b47ab78b906e75600c52d2de). Supply weights extracted from your own
runtime. This repository does not vendor that implementation or its weights.

Full-model tensor loop: blocks0..70, all152 layers, execute each cycle. An RGB
tensor bridge crops, clamps and mirror-pads the result for the next cycle; no
uint8 conversion or host image readback occurs between cycles.

Feature-body loop: block0 executes once, every interior layer in blocks1..69
executes N times, then block70 executes once. The first pre-block's full-resolution
skip and original RGB residual reference survive every cycle. This is NOT N full
model passes. Later tone/style conditioning cannot be applied without block0.

Example (run using the Python environment containing the dependencies):
  python tests/nr_model_loop.py --reference <checkout> --weights <weights.safetensors>
    --input <frame.png> --output <directory> --loops 2 --style 2 --tone 1 --later-tone 0
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

EXPECTED_COMMIT = "3beef7935a8ef976b47ab78b906e75600c52d2de"


def main_value(value):
    return value[0] if isinstance(value, tuple) else value


def tensor_bridge(net, result, fp16=False):
    """RGB stays a tensor; its valid rectangle determines the next padding."""
    import torch
    cropped = result[:, :, :net.height, :net.width].clamp(0, 1)
    if fp16:
        cropped = cropped.to(torch.float16).to(torch.float32)
    rows = torch.arange(net.padding.height, device=result.device)
    cols = torch.arange(net.padding.width, device=result.device)
    rows = torch.where(rows < net.height, rows, 2*net.height-rows-2).clamp(0, net.height-1)
    cols = torch.where(cols < net.width, cols, 2*net.width-cols-2).clamp(0, net.width-1)
    return cropped[:, :, rows][:, :, :, cols].contiguous()


def feature_body_loop(net, image, profiles):
    """Reuse the previously validated range scheduler, for the whole body only."""
    from dlssnr import int4, int8, quant
    outputs = {}
    first, end = 1, len(net.plan)-1
    int4.reset()

    def execute(begin, stop, controls):
        for i in range(begin, stop):
            spec, layer = net.plan[i], net.layers[i]
            if hasattr(layer, "controls"):
                layer.controls = controls
            int8.current = quant.current_block = spec.block
            source = net.skip_source[i]
            skip = None if source is None else outputs[source]
            previous = outputs[i-1] if i else None
            if spec.kind == "PreBlock":
                outputs[i] = layer(image)
            elif spec.kind == "PostBlock":
                outputs[i] = layer(main_value(previous), skip[1], image)
            elif spec.kind == "SplitFfwdProj":
                outputs[i] = layer(previous)
            else:
                if spec.kind == "DecInputUpsample":
                    skip = main_value(skip)
                outputs[i] = layer(main_value(previous), skip)

    execute(0, end, profiles[0])
    initial_pre = outputs[0]
    for controls in profiles[1:]:
        state = main_value(outputs[end-1])
        if state.shape != main_value(initial_pre).shape:
            raise ValueError("The full latent body does not return to its input shape")
        # Keep the original full-resolution skip; recompute every interior skip.
        outputs[0] = (state, *initial_pre[1:])
        execute(first, end, controls)
    execute(end, len(net.plan), profiles[-1])
    return main_value(outputs[end])


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loops", type=int, choices=(1,2,3), default=2)
    parser.add_argument("--style", type=int, choices=(0,1,2), default=0)
    parser.add_argument("--tone", type=float, default=1)
    parser.add_argument("--later-tone", type=float, default=None)
    parser.add_argument("--structure", type=float, default=1)
    parser.add_argument("--input-fp16", action="store_true", help="Round initial PNG/255 values to the native texture's FP16 precision")
    parser.add_argument("--bridge-fp16", action="store_true", help="Round RGB between complete passes to FP16")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    sys.path.insert(0, str(args.reference.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from dlssnr.io import Controls, frame_to_tensor, tensor_to_frame
    from dlssnr.metrics import ssim
    from dlssnr.model import NRNet
    from dlssnr.weights import Weights

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    frame = np.asarray(Image.open(args.input).convert("RGB")).copy()
    height, width = frame.shape[:2]
    weights = Weights.load(str(args.weights))
    net = NRNet(weights, width, height).to("cpu")
    if len(net.plan) != 152 or net.plan[0].kind != "PreBlock" or net.plan[-1].kind != "PostBlock":
        raise ValueError("This harness requires the original 152-layer HT model")
    if any(not 1 <= spec.block <= 69 for spec in net.plan[1:-1]):
        raise ValueError("Unexpected full-body layer boundaries")
    later_tone = args.tone if args.later_tone is None else args.later_tone
    profiles = [Controls(style=args.style, intensity=1, local_tone=args.tone if index==0 else later_tone,
                         local_structure=args.structure, skin_structure=-1, auto_mask=True, seed=0)
                for index in range(args.loops)]
    image = frame_to_tensor(frame, "cpu", "ht")
    if args.input_fp16:
        image = image.to(torch.float16).to(torch.float32)
    try:
        observed_commit = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        observed_commit = None
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        started = time.perf_counter()
        full = image
        first_result = None
        full_times = []
        for index, controls in enumerate(profiles):
            if index:
                full = tensor_bridge(net, full, args.bridge_fp16)
            cycle_start = time.perf_counter()
            full = main_value(net.run(full, controls)[len(net.plan)-1])
            full_times.append(time.perf_counter()-cycle_start)
            if index==0:
                first_result = full
        full_seconds = time.perf_counter()-started
        # The latent scheduler must be exact when it traverses the graph once.
        once = feature_body_loop(net, image, profiles[:1])
        if not torch.equal(once, first_result):
            raise RuntimeError("One-cycle latent scheduler differs from the complete reference graph")
        started = time.perf_counter()
        feature = feature_body_loop(net, image, profiles)
        feature_seconds = time.perf_counter()-started

    def crop(value):
        return value[0, :, :height, :width].permute(1,2,0).cpu().numpy().copy()
    floats = {"first-full-pass":crop(first_result), "full-model-tensor-loop":crop(full),
              "feature-body-loop":crop(feature)}
    byte_images = {}
    for name, value in (("first-full-pass",first_result),("full-model-tensor-loop",full),("feature-body-loop",feature)):
        byte_images[name] = tensor_to_frame(value,width,height)
        Image.fromarray(byte_images[name]).save(args.output / f"{name}.png")
        np.save(args.output / f"{name}.npy",floats[name])
    target = floats["full-model-tensor-loop"].astype(np.float64)
    first = floats["first-full-pass"].astype(np.float64)
    wanted = target-first
    wanted_norm = float(np.sum(wanted*wanted))
    rows=[]
    for name in ("first-full-pass","feature-body-loop"):
        actual = floats[name].astype(np.float64)
        error = actual-target
        mse = float(np.mean(error*error))
        change = actual-first
        divisor = np.linalg.norm(change)*np.linalg.norm(wanted)
        rows.append({"candidate":name,"target":"full-model-tensor-loop",
                     "float_psnr_db":None if mse==0 else -10*math.log10(mse),
                     "identical":bool(np.array_equal(actual,target)),
                     "mean_abs_rgb_error":float(np.abs(error).mean()),
                     "ssim_8bit":ssim(byte_images[name],byte_images["full-model-tensor-loop"]),
                     "target_delta_projection":float(np.sum(change*wanted)/wanted_norm) if wanted_norm else None,
                     "target_delta_cosine":float(np.sum(change*wanted)/divisor) if divisor else None,
                     "finite":bool(np.isfinite(actual).all())})
    report={"reference_commit":observed_commit,"tested_reference_commit":EXPECTED_COMMIT,
            "weights_sha256":hash_file(args.weights),"weights_metadata":weights.metadata,
            "input_sha256":hash_file(args.input),"width":width,"height":height,
            "working_padding":net.padding._asdict(),"torch":torch.__version__,"device":"cpu",
            "loops":args.loops,"profiles":list(map(dataclasses.asdict,profiles)),
            "input_fp16":args.input_fp16,"rgb_bridge_fp16":args.bridge_fp16,
            "one_cycle_scheduler_exact":True,
            "full_model_tensor_loop":{"input_stage":"block0 each cycle, fresh seed0 and that cycle's conditioning",
                "body":"all blocks1..69,150 interior layers, each cycle",
                "output_stage":"block70 each cycle; RGB crop/clamp/mirror-pad tensor bridge between cycles",
                "executed_layers":152*args.loops,"cpu_seconds":full_seconds,"cycle_cpu_seconds":full_times},
            "feature_body_loop":{"input_stage":"block0 once, first-cycle conditioning and original full-resolution skip",
                "body":"all blocks1..69,150 interior layers, each cycle; all interior skip tensors recomputed",
                "output_stage":"block70 once, original RGB residual reference and original pre-block skip",
                "executed_layers":2+150*args.loops,"cpu_seconds":feature_seconds,
                "later_conditioning_applied":False,"final_post_intensity":profiles[-1].intensity},
            "limits":["CPU reconstruction, not a modified NVIDIA runtime or GPU speed measurement.",
                "Feature-body repetition does not execute intermediate block70/block0 and cannot reproduce later tone conditioning.",
                "Full-model control executes every learned layer every cycle; it still crosses the internal RGB representation.",
                "No temporal motion/history implementation; seed0 is used on every complete pass."],"comparisons":rows}
    (args.output / "results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"loops":args.loops,"scheduler_exact":True,"comparisons":rows,
                      "full_model_layers":152*args.loops,"feature_body_layers":2+150*args.loops},indent=2))


if __name__ == "__main__":
    main()
