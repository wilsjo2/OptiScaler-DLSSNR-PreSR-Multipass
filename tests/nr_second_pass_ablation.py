"""Rank whole-block identity bypasses only in the second NR pass, on CPU.

Requires CPU PyTorch, numpy, Pillow, safetensors and a separate checkout of
https://github.com/mauri870/dlssnr-pytorch (tested commit
3beef7935a8ef976b47ab78b906e75600c52d2de). Supply locally extracted HT weights.
No NVIDIA DLL is loaded or modified, and no GPU work is submitted.

Each candidate consumes the saved ACTUAL native first-pass output. The complete
reference second pass on that same input is the ablation baseline; saved native
second-pass output is an independent fidelity target. This isolates pass2 from
first-pass reconstruction error. Only plain same-shape Swin, complete split-Swin
blocks and complete ViT blocks are eligible. Pooling, up/downsampling, pre/post
blocks and layout-boundary variants remain intact.

Base fixture: all 49 candidates, individually. The best --followup-count candidates
by distance from the complete reference second pass are repeated on translated
and contrast-altered versions of the same scene. CPU times are diagnostic only.
Native dispatch feasibility and measured speed savings require separate tests.
Use --combined-only to test just two explicit unions (42+43+44 and
41+42+43+44+46+52), across all three fixtures, without repeating the broad sweep.
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main_value(value):
    return value[0] if isinstance(value,tuple) else value


def eligible_blocks(plan):
    groups=collections.defaultdict(list)
    for spec in plan:
        groups[spec.block].append(spec)
    candidates={}
    split=set(range(24,30))|set(range(40,47))
    vit=set(range(31,39))
    for block,specs in groups.items():
        plain_swin=len(specs)==1 and specs[0].kind.startswith("Swin") and specs[0].variant in (None,"","-")
        if plain_swin or block in split or block in vit:
            candidates[block]={"block":block,"family":"swin" if plain_swin else "split" if block in split else "vit",
                "layer_indices":[s.index for s in specs],"channels":specs[0].channels,"tokens":specs[0].tokens}
    if len(candidates)!=49:
        raise ValueError(f"Unexpected HT candidate set: {len(candidates)} blocks, expected49")
    return candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--fixture-directory",type=Path,required=True)
    parser.add_argument("--native-directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--followup-count",type=int,default=6)
    parser.add_argument("--combined-only",action="store_true")
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if not 0<=args.followup_count<=49 or args.threads<1:
        parser.error("followup count must be0..49 and threads positive")
    sys.path.insert(0,str(args.reference.resolve()))
    import numpy as np
    from PIL import Image
    import torch
    from dlssnr.io import Controls,tensor_to_frame
    from dlssnr.metrics import ssim
    from dlssnr.model import NRNet
    from dlssnr.weights import Weights
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    controls=Controls(style=0,intensity=1,local_tone=1,local_structure=1,skin_structure=-1,auto_mask=True,seed=0)
    dimensions=Image.open(args.fixture_directory / "input-base.png").size
    width,height=dimensions
    weights=Weights.load(str(args.weights))
    net=NRNet(weights,width,height).to("cpu")
    if len(net.plan)!=152:
        raise ValueError("The second-pass experiment requires the original152-layer HT graph")
    candidates=eligible_blocks(net.plan)
    combinations={"42+43+44":(42,43,44),"41+42+43+44+46+52":(41,42,43,44,46,52)}
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        commit=subprocess.check_output(["git","-C",str(args.reference),"rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit=None
    metadata={"reference_commit":commit,"weights_sha256":sha256(args.weights),"weights_metadata":weights.metadata,
        "width":width,"height":height,"working_padding":net.padding._asdict(),"device":"cpu","torch":torch.__version__,
        "controls":dataclasses.asdict(controls),"input":"Actual native first-pass RGBA16F output, RGB clamped0..1 and mirror-padded",
        "candidate_policy":"Whole-block identity only in pass2; no pool/up/downsample, pre/post or layout-boundary bypasses",
        "candidates":list(candidates.values()),"followup_count":args.followup_count,
        "combined_only":args.combined_only,"combined_candidates":combinations if args.combined_only else None,
        "limits":["Modified inference is CPU reconstruction only; no native bypass implementation or GPU speed measurement.",
            "Native output is an independent reference, not an image-quality ground truth.",
            "Rank by distance from complete reference second pass; native fidelity differences can include reconstruction error.",
            "Three altered versions of one scene, one resolution, fixed controls, fresh seed0 and no retained history.",
            "Individual sweep bypasses one block at a time; combined-only mode tests exactly the two named unions."]}
    (args.output / "metadata.json").write_text(json.dumps(metadata,indent=2))
    rows=[]
    baselines=[]
    selected=[]

    def image8(array):
        return np.floor(np.clip(array,0,1)*255+0.5).astype(np.uint8)
    def raw(value):
        return value[0,:,:height,:width].permute(1,2,0).cpu().numpy().copy()
    def distance(candidate,target):
        a,b=candidate.astype(np.float64),target.astype(np.float64)
        difference=a-b
        mse=float(np.mean(difference*difference))
        return {"psnr_db":None if mse==0 else -10*math.log10(mse),"mae":float(np.abs(difference).mean()),
                "max_abs":float(np.max(np.abs(difference))),"mse":mse,
                "ssim":ssim(image8(candidate),image8(target)),"finite":bool(np.isfinite(candidate).all())}
    def write_progress():
        (args.output / "results.json").write_text(json.dumps(rows,indent=2))
        (args.output / "baselines.json").write_text(json.dumps(baselines,indent=2))
        if rows:
            with (args.output / "results.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)

    for fixture in ("base","translated","contrast"):
        if fixture!="base" and not selected and not args.combined_only:
            continue
        output=args.output / fixture
        output.mkdir(exist_ok=True)
        one_path=args.native_directory / f"parity-{fixture}-pass1.rgba16f"
        two_path=args.native_directory / f"parity-{fixture}-pass2.rgba16f"
        if one_path.stat().st_size!=width*height*8 or two_path.stat().st_size!=width*height*8:
            raise ValueError("Native output dimensions do not match fixture")
        one=np.fromfile(one_path,dtype=np.float16).reshape(height,width,4)[:,:,:3].astype(np.float32)
        native_two=np.fromfile(two_path,dtype=np.float16).reshape(height,width,4)[:,:,:3].astype(np.float32)
        valid=torch.from_numpy(one.copy()).permute(2,0,1).unsqueeze(0).clamp(0,1)
        yy=torch.arange(net.padding.height);xx=torch.arange(net.padding.width)
        yy=torch.where(yy<height,yy,2*height-yy-2).clamp(0,height-1)
        xx=torch.where(xx<width,xx,2*width-xx-2).clamp(0,width-1)
        image=valid[:,:,yy][:,:,:,xx].contiguous()
        with torch.no_grad():
            net.skip=set()
            capture=net.run(image,controls,capture=fixture=="base")
            full_tensor=main_value(capture[151])
            if fixture=="base":
                for block,candidate in candidates.items():
                    indices=candidate["layer_indices"]
                    before=main_value(capture[indices[0]-1]);after=main_value(capture[indices[-1]])
                    if before.shape!=after.shape:
                        raise RuntimeError(f"Block{block} is not shape-preserving")
                    candidate["verified_tensor_shape"]=list(before.shape)
            full=raw(full_tensor)
            del capture
            baseline_metrics=distance(full,native_two)
            baselines.append({"fixture":fixture,"native_one_sha256":sha256(one_path),"native_two_sha256":sha256(two_path),
                "full_reference_vs_native_two":baseline_metrics,"native_one_vs_native_two":distance(one,native_two),
                "baseline_replay_exact":False})
            Image.fromarray(tensor_to_frame(full_tensor,width,height)).save(output / "complete-second-pass.png")
            np.save(output / "complete-second-pass.npy",full)
            print(f"{fixture}: full reference second pass versus native {baseline_metrics['psnr_db']:.3f}dB",flush=True)
            blocks=list(combinations) if args.combined_only else list(candidates) if fixture=="base" else selected
            for block in blocks:
                if args.combined_only:
                    members=combinations[block]
                    candidate={"family":"combined","channels":None,
                        "layer_indices":[i for member in members for i in candidates[member]["layer_indices"]]}
                    block_label=block
                else:
                    candidate=candidates[block]
                    block_label=f"{block:02}"
                net.skip=set(candidate["layer_indices"])
                started=time.perf_counter()
                result=main_value(net.run(image,controls)[151])
                elapsed=time.perf_counter()-started
                values=raw(result)
                vs_full,vs_native=distance(values,full),distance(values,native_two)
                target_delta=native_two.astype(np.float64)-one
                actual_delta=values.astype(np.float64)-one
                denominator=float(np.sum(target_delta*target_delta))
                cosine_denominator=np.linalg.norm(target_delta)*np.linalg.norm(actual_delta)
                row={"fixture":fixture,"block":block,"family":candidate["family"],"channels":candidate["channels"],
                    "skipped_reference_layers":len(candidate["layer_indices"]),
                    "psnr_vs_complete_reference":vs_full["psnr_db"],"ssim_vs_complete_reference":vs_full["ssim"],
                    "mae_vs_complete_reference":vs_full["mae"],"mse_vs_complete_reference":vs_full["mse"],
                    "psnr_vs_native_two":vs_native["psnr_db"],"ssim_vs_native_two":vs_native["ssim"],
                    "mae_vs_native_two":vs_native["mae"],
                    "native_psnr_loss_db":baseline_metrics["psnr_db"]-vs_native["psnr_db"],
                    "native_second_delta_projection":float(np.sum(actual_delta*target_delta)/denominator),
                    "native_second_delta_cosine":float(np.sum(actual_delta*target_delta)/cosine_denominator),
                    "finite":vs_full["finite"],"cpu_seconds_diagnostic":elapsed}
                rows.append(row)
                Image.fromarray(tensor_to_frame(result,width,height)).save(output / f"skip-block{block_label}.png")
                write_progress()
                full_score="identical" if vs_full["psnr_db"] is None else f"{vs_full['psnr_db']:.3f}dB"
                print(f"{fixture} block {block_label} {candidate['family']:5}: {full_score} vs full, {vs_native['psnr_db']:.3f}dB vs native, native loss {row['native_psnr_loss_db']:+.3f}dB",flush=True)
            net.skip=set()
            replay=main_value(net.run(image,controls)[151])
            if not torch.equal(replay,full_tensor):
                raise RuntimeError("Complete second pass changed after block-ablation sweep")
            baselines[-1]["baseline_replay_exact"]=True
        if fixture=="base" and not args.combined_only:
            ranking=sorted((r for r in rows if r["fixture"]=="base"),key=lambda r:r["mse_vs_complete_reference"])
            selected=[r["block"] for r in ranking[:args.followup_count]]
            (args.output / "base-ranking.json").write_text(json.dumps(ranking,indent=2))
            metadata["followup_blocks"]=selected
            metadata["verified_candidates"]=list(candidates.values())
            (args.output / "metadata.json").write_text(json.dumps(metadata,indent=2))
            print(f"Selected follow-up blocks {selected}",flush=True)
        write_progress()
    print("DONE: complete-pass replay remained tensor-exact",flush=True)


if __name__=="__main__":
    main()
