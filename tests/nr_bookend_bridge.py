"""Isolate NR's RGB projection/lift bookends while preserving every Swin layer.

CPU research harness. Dependencies: CPU PyTorch, numpy, Pillow, safetensors and a
separate checkout of https://github.com/mauri870/dlssnr-pytorch, tested at commit
3beef7935a8ef976b47ab78b906e75600c52d2de. Supply locally extracted model weights;
neither the reference implementation nor weights are vendored here.

Every candidate retains post-block skip mixing and Swin, pre-block Swin and
pooling, and all 150 interior layers. The only alternate connection is from the
post-block's full-resolution 32-channel Swin output to the next pre-block's
32-channel lifted input, before its Swin operation.

Candidates: ordinary RGB projection/composition/FP16/crop/padding/lift control;
an exact functional fused bridge returning lifted32 plus a float32 residual
carrier; direct 32-channel identity (diagnostic); composed learned residual update
A_next = narrow(A_original_with_next_conditioning + (C @ P) @ z / 32).
Here C sums the pre-lift's two colour triplets and P is the post RGB projection.
The latter is approximate: intermediate clamps and colour rounding are omitted.

The approximate candidates use the exact first-pass RGB as the second post-block's
residual anchor, isolating bridge-input error. The fused candidate instead rebuilds
that anchor from original RGB plus its float32 carry, with an exact round-trip
guard. It still computes three colour values. CPU tensor temporaries do not prove
that a native kernel avoids a materialized texture or reduces memory traffic.
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


def mirror_pad(net, tensor, fp16=False):
    """Crop any channel count to the valid frame and reproduce ordinary padding."""
    import torch
    cropped = tensor[:, :, :net.height, :net.width]
    if fp16:
        cropped = cropped.to(torch.float16).to(torch.float32)
    rows = torch.arange(net.padding.height, device=tensor.device)
    cols = torch.arange(net.padding.width, device=tensor.device)
    rows = torch.where(rows < net.height, rows, 2*net.height-rows-2).clamp(0, net.height-1)
    cols = torch.where(cols < net.width, cols, 2*net.width-cols-2).clamp(0, net.width-1)
    return cropped[:, :, rows][:, :, :, cols].contiguous()


def split_pass(net, image, controls, lifted_override=None, residual_carry=None):
    """Expose only the pointwise bookends; all learned attention is untouched."""
    import torch
    from dlssnr import int4, int8, quant
    from dlssnr.layers.pre_post import compose
    from dlssnr.layers.swin import narrow, pool2x2, round8, swin_core
    int4.reset()
    pre, post = net.layers[0], net.layers[-1]
    pre.controls = post.controls = controls
    int8.current = quant.current_block = 0
    accumulator = None
    if lifted_override is None:
        accumulator = torch.einsum("ck,bkhw->bchw", pre.lift, pre.features(image))
        lifted = narrow(accumulator)
    else:
        lifted = lifted_override
    expected_shape = (image.shape[0], pre.lift.shape[0], image.shape[2], image.shape[3])
    if lifted.shape != expected_shape:
        raise ValueError("Bridge must preserve the full-resolution32-channel pre-Swin shape")
    wide, before_rounding = swin_core(pre.params, lifted, pre.spec.shift_x, pre.spec.shift_y, accumulator=True)
    outputs = {0:(pool2x2(before_rounding), round8(wide))}
    for index in range(1, len(net.plan)-1):
        spec, layer = net.plan[index], net.layers[index]
        int8.current = quant.current_block = spec.block
        source = net.skip_source[index]
        skip = None if source is None else outputs[source]
        previous = outputs[index-1]
        if spec.kind == "SplitFfwdProj":
            outputs[index] = layer(previous)
        else:
            if spec.kind == "DecInputUpsample":
                skip = main_value(skip)
            outputs[index] = layer(main_value(previous), skip)
    int8.current = quant.current_block = 70
    post_wide = swin_core(post.params, post.blend(main_value(outputs[len(net.plan)-2]), outputs[0][1]),
                          post.spec.shift_x, post.spec.shift_y)
    residual = torch.einsum("rc,bchw->brhw", post.project, post_wide)
    anchor = image if residual_carry is None else (image+residual_carry).to(torch.float16).to(torch.float32)
    rgb = compose(anchor, residual, controls.intensity)
    return {"rgb":rgb,"post_wide":post_wide,"lifted":lifted,"lift_accumulator":accumulator,
            "unclamped_rgb":anchor+residual*0.25}


def fused_bookend_bridge(net, original_image, post_wide, first_controls, next_controls):
    """Exact functional bridge: return next lifted32 and an RGB residual carrier.

    Projection, clipping, FP16 conversion, centering and the original16-column
    lift order survive. Unlike the approximate dense matrix, this is the factored
    P -> colour operations -> L path. Three colour tensor temporaries exist in
    Python; native fusion would need to keep them local to a kernel.
    """
    import torch
    from dlssnr.layers.pre_post import CONSTANT_FEATURE, _centred_colour, compose
    from dlssnr.layers.swin import narrow
    pre, post = net.layers[0], net.layers[-1]
    pre.controls = next_controls
    residual = torch.einsum("rc,bchw->brhw", post.project, post_wide)
    colour_registers = compose(original_image, residual, first_controls.intensity).to(torch.float16).to(torch.float32)
    # Keep carry in float32. Half originals/results in[0,1] lie on a2^-24 grid,
    # so their subtraction and round-trip addition are representable exactly.
    carry = mirror_pad(net, colour_registers-original_image)
    reconstructed = (original_image+carry).to(torch.float16).to(torch.float32)
    expected = mirror_pad(net, colour_registers)
    if not torch.equal(reconstructed, expected):
        raise RuntimeError("Float32 residual carry failed exact FP16 anchor reconstruction")
    del colour_registers, expected
    colour = _centred_colour(reconstructed)
    _, _, height, width = original_image.shape
    device = original_image.device
    conditioning = torch.tensor(next_controls.conditioning(), device=device).to(torch.float16).float()
    def spread(values):
        return values.view(1,-1,1,1).expand(1,-1,height,width)
    features = torch.cat([pre.noise(height,width,device),
        spread(torch.tensor([CONSTANT_FEATURE],device=device)), colour, colour,
        spread(conditioning), spread(torch.zeros(1,device=device))],dim=1)
    lifted = narrow(torch.einsum("ck,bkhw->bchw",pre.lift,features))
    return lifted, carry


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda:handle.read(1024*1024),b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--native-one",type=Path)
    parser.add_argument("--native-two",type=Path)
    parser.add_argument("--style",type=int,choices=(0,1,2),default=0)
    parser.add_argument("--tone",type=float,default=1)
    parser.add_argument("--later-tone",type=float,default=1)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:
        parser.error("--threads must be positive")
    sys.path.insert(0,str(args.reference.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from dlssnr.io import Controls,frame_to_tensor,tensor_to_frame
    from dlssnr.layers.swin import narrow
    from dlssnr.metrics import ssim
    from dlssnr.model import NRNet
    from dlssnr.weights import Weights
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    frame=np.asarray(Image.open(args.input).convert("RGB")).copy()
    height,width=frame.shape[:2]
    weights=Weights.load(str(args.weights))
    net=NRNet(weights,width,height).to("cpu")
    if len(net.plan)!=152 or net.plan[0].kind!="PreBlock" or net.plan[-1].kind!="PostBlock":
        raise ValueError("The bookend experiment requires the original152-layer HT model")
    first_controls=Controls(style=args.style,local_tone=args.tone,local_structure=1,intensity=1,
                            skin_structure=-1,auto_mask=True,seed=0)
    next_controls=Controls(style=args.style,local_tone=args.later_tone,local_structure=1,intensity=1,
                           skin_structure=-1,auto_mask=True,seed=0)
    image=frame_to_tensor(frame,"cpu").to(torch.float16).to(torch.float32)
    args.output.mkdir(parents=True,exist_ok=True)

    def cropped(tensor):
        return tensor[0,:,:height,:width].permute(1,2,0).cpu().numpy().copy()
    def tensor_stats(value):
        array=value.detach().cpu().numpy().astype(np.float64)
        return {"shape":list(array.shape),"rms":float(np.sqrt(np.mean(array*array))),
                "max_abs":float(np.max(np.abs(array))),"finite":bool(np.isfinite(array).all())}
    def image_metrics(candidate,target):
        a,b=candidate.astype(np.float64),target.astype(np.float64)
        difference=a-b
        mse=float(np.mean(difference*difference))
        a8=np.floor(np.clip(a,0,1)*255+0.5).astype(np.uint8)
        b8=np.floor(np.clip(b,0,1)*255+0.5).astype(np.uint8)
        return {"identical":bool(np.array_equal(a,b)),"float_psnr_db":None if mse==0 else -10*math.log10(mse),
                "mean_abs_rgb_error":float(np.abs(difference).mean()),"max_abs_rgb_error":float(np.max(np.abs(difference))),
                "ssim_8bit":ssim(a8,b8),"finite":bool(np.isfinite(a).all())}
    def load_native(path):
        return None if path is None else np.fromfile(path,dtype=np.float16).reshape(height,width,4)[:,:,:3].astype(np.float32)
    native_one,native_two=load_native(args.native_one),load_native(args.native_two)

    with torch.no_grad():
        ordinary_first=main_value(net.run(image,first_controls)[151])
        first=split_pass(net,image,first_controls)
        if not torch.equal(first["rgb"],ordinary_first):
            raise RuntimeError("Splitting the first pass's bookends changed the reference output")
        next_image=mirror_pad(net,first["rgb"].clamp(0,1),fp16=True)
        ordinary_second=main_value(net.run(next_image,next_controls)[151])
        started=time.perf_counter()
        exact=split_pass(net,next_image,next_controls)
        if not torch.equal(exact["rgb"],ordinary_second):
            raise RuntimeError("Ordinary second-pass bookends failed exact decomposition guard")
        exact_seconds=time.perf_counter()-started
        pre,post=net.layers[0],net.layers[-1]
        pre.controls=next_controls
        # Recompute the UNROUNDED original-colour lift with the next pass's
        # conditioning and seed. Adding to the already-narrowed lift loses bits.
        original_accumulator=torch.einsum("ck,bkhw->bchw",pre.lift,pre.features(image))
        colour_lift=pre.lift[:,4:7]+pre.lift[:,7:10]
        matrix=(colour_lift @ post.project) / 32.0
        post_features=mirror_pad(net,first["post_wide"])
        candidates={"ordinary_bookends":exact,
                    "identity32":narrow(post_features),
                    "composed_learned_update":narrow(original_accumulator+torch.einsum("ck,bkhw->bchw",matrix,post_features))}
        timings={"ordinary_bookends":exact_seconds}
        for name in ("identity32","composed_learned_update"):
            started=time.perf_counter()
            candidates[name]=split_pass(net,next_image,next_controls,candidates[name])
            timings[name]=time.perf_counter()-started
        started=time.perf_counter()
        fused_lift, carry=fused_bookend_bridge(net,image,first["post_wide"],first_controls,next_controls)
        if not torch.equal(fused_lift,exact["lifted"]):
            raise RuntimeError("Exact fused bridge changed the second pre-block's lifted features")
        fused=split_pass(net,image,next_controls,fused_lift,residual_carry=carry)
        if not torch.equal(fused["rgb"],exact["rgb"]):
            raise RuntimeError("Exact fused bridge changed the complete second-pass output")
        candidates["exact_fused_residual_carry"]=fused
        timings["exact_fused_residual_carry"]=time.perf_counter()-started

    first_array,target=cropped(first["rgb"]),cropped(exact["rgb"])
    Image.fromarray(tensor_to_frame(first["rgb"],width,height)).save(args.output / "first-pass.png")
    np.save(args.output / "first-pass.npy",first_array)
    rows=[]
    desired=target.astype(np.float64)-first_array
    for name,result in candidates.items():
        output=cropped(result["rgb"])
        Image.fromarray(tensor_to_frame(result["rgb"],width,height)).save(args.output / f"{name}.png")
        np.save(args.output / f"{name}.npy",output)
        lift_difference=(result["lifted"]-exact["lifted"]).cpu().numpy().astype(np.float64)
        change=output.astype(np.float64)-first_array
        desired_norm=float(np.sum(desired*desired))
        divisor=np.linalg.norm(change)*np.linalg.norm(desired)
        row={"candidate":name,"vs_exact_two":image_metrics(output,target),
             "vs_native_two":None if native_two is None else image_metrics(output,native_two),
             "lift_statistics":tensor_stats(result["lifted"]),
             "lift_error_mae":float(np.mean(np.abs(lift_difference))),"lift_error_max":float(np.max(np.abs(lift_difference))),
             "second_delta_projection":float(np.sum(change*desired)/desired_norm) if desired_norm else None,
             "second_delta_cosine":float(np.sum(change*desired)/divisor) if divisor else None,
             "cpu_diagnostic_seconds":timings[name]}
        rows.append(row)
        print(json.dumps(row),flush=True)
    unclamped=cropped(first["unclamped_rgb"])
    try:
        commit=subprocess.check_output(["git","-C",str(args.reference),"rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit=None
    report={"reference_commit":commit,"tested_reference_commit":EXPECTED_COMMIT,
            "weights_sha256":sha256(args.weights),"weights_metadata":weights.metadata,
            "input_sha256":sha256(args.input),"width":width,"height":height,"working_padding":net.padding._asdict(),
            "device":"cpu","torch":torch.__version__,"first_controls":dataclasses.asdict(first_controls),
            "second_controls":dataclasses.asdict(next_controls),"input_precision":"FP16","rgb_boundary_precision":"FP16",
            "first_split_exact":True,"second_split_exact":True,
            "fused_lift_exact":True,"fused_output_exact":True,"fused_carry_anchor_roundtrip_exact":True,
            "fused_bridge_output":{"lifted":"full-resolution32-channel FP16 values held as float32",
                "carry":"three-channel float32 y_half-original_half, mirror-padded; never narrowed to half",
                "final_anchor":"FP16(original_half+carry32), checked against ordinary intermediate output"},
            "retained_operations":["Both pre-block Swin bodies and pooling", "Both post-block skip mixing and Swin bodies", "All150 interior layers on both passes", "Fresh second-pass skip tensors", "Correct first-pass RGB residual anchor for the second output"],
            "approximate_bridge_formula":"narrow(L*features(original_RGB,next_controls)+(C*P)*mirror_pad(post_Swin32)/32); C=L[:,4:7]+L[:,7:10]",
            "identity_bridge_scope":"Diagnostic32-channel input only; no pre/post attention block is omitted",
            "first_post_feature_statistics":tensor_stats(first["post_wide"]),
            "first_pass_unclamped_rgb_fraction_outside_0_1":float(np.mean((unclamped<0)|(unclamped>1))),
            "first_vs_native":None if native_one is None else image_metrics(first_array,native_one),
            "single_vs_exact_two":image_metrics(first_array,target),
            "limits":["Modified paths execute only in a CPU reconstruction; native outputs are external comparison targets.",
                      "Approximate candidates use exact first-pass RGB as output residual anchor. The exact fused candidate uses original RGB plus float32 carry and proves the reconstructed anchor identical.",
                      "The fused bridge still computes three RGB values and CPU temporary tensors. This functional test does not demonstrate native texture elimination, memory savings or GPU speed.",
                      "Composed learned update omits first composition clipping and intermediate colour rounding and changes floating-point summation order.",
                      "No retained history/motion implementation and no native GPU performance claim."],"comparisons":rows}
    (args.output / "results.json").write_text(json.dumps(report,indent=2))
    print("PASS: ordinary decomposition, fused lifted features, carry anchor and final output are tensor-exact",flush=True)


if __name__=="__main__":
    main()
