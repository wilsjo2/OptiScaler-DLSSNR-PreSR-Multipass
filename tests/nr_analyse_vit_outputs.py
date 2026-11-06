"""Streaming CPU analysis of native one/ViT/two outputs; never runs NR or GPU work.

Dependencies: numpy, scipy, Pillow; matplotlib only with --contacts.
  python tests/nr_analyse_vit_outputs.py static --prefix <vit-cost prefix> --output <dir>
  python tests/nr_analyse_vit_outputs.py sequence --prefix <vit-sequence prefix> --output <dir>

Float metrics use decoded native RGBA16F RGB, peak1. SSIM uses an11x11 Gaussian
window (sigma1.5), reflect-without-edge-repeat padding, and raw floating RGB in
range1. No image resizing or8-bit quantization is used for numerical metrics.
Files are memory-mapped and processed in row stripes. Sequence transitions open
only the current and previous frame triplets, not the whole sequence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter

MODES=("one","vit","two")
RADIUS=5


def mapped(stack,path,width,height):
    path=Path(path)
    if path.stat().st_size!=width*height*8:
        raise ValueError(f"Unexpected RGBA16F size: {path}")
    array=np.memmap(path,dtype="<f2",mode="r",shape=(height,width,4))
    stack.callback(array._mmap.close)
    return array


def integrity(path,width,height,rows):
    count=0;minimum=math.inf;maximum=-math.inf
    with ExitStack() as stack:
        array=mapped(stack,path,width,height)
        for y in range(0,height,rows):
            stripe=array[y:min(y+rows,height)]
            finite=np.isfinite(stripe)
            count+=int(stripe.size-np.count_nonzero(finite))
            if finite[:,:,:3].all():
                minimum=min(minimum,float(stripe[:,:,:3].min()))
                maximum=max(maximum,float(stripe[:,:,:3].max()))
    return {"finite":count==0,"nonfinite_rgba_values":count,
            "rgb_min":None if minimum==math.inf else minimum,"rgb_max":None if maximum==-math.inf else maximum}


def bitwise_compare(a_path,b_path,width,height,rows):
    different=0
    with ExitStack() as stack:
        a=mapped(stack,a_path,width,height);b=mapped(stack,b_path,width,height)
        for y in range(0,height,rows):
            different+=int(np.count_nonzero(a[y:y+rows].view(np.uint16)!=b[y:y+rows].view(np.uint16)))
    return {"rgba16_bit_exact":different==0,"different_half_words":different}


def view(array,roi,offset=(0,0)):
    x0,y0,x1,y1=roi;dx,dy=offset
    if x0+dx<0 or y0+dy<0 or x1+dx>array.shape[1] or y1+dy>array.shape[0]:
        raise ValueError("Aligned crop would read a wrapped border")
    return array[y0+dy:y1+dy,x0+dx:x1+dx,:3]


def blur(array):
    return gaussian_filter(array,sigma=(1.5,1.5,0),radius=(5,5,0),mode="mirror")


def stripe_ssim(a,b,rows):
    """Exact stripe accumulation of full-image11x11 SSIM, including halo rows."""
    height=a.shape[0];total=0.0;count=0
    for y in range(0,height,rows):
        end=min(y+rows,height);lo=max(0,y-RADIUS);hi=min(height,end+RADIUS)
        x=np.asarray(a[lo:hi],dtype=np.float64);z=np.asarray(b[lo:hi],dtype=np.float64)
        mx=blur(x);mz=blur(z)
        sx=blur(x*x)-mx*mx;sz=blur(z*z)-mz*mz;cross=blur(x*z)-mx*mz
        score=((2*mx*mz+0.01**2)*(2*cross+0.03**2))/((mx*mx+mz*mz+0.01**2)*(sx+sz+0.03**2))
        valid=score[y-lo:end-lo]
        total+=float(valid.sum(dtype=np.float64));count+=valid.size
    return total/count


def psnr(mse):
    return None if mse==0 else -10*math.log10(mse)


def triplet_quality(paths,width,height,roi,offset,rows):
    sums={"one_squared":0.0,"vit_squared":0.0,"one_abs":0.0,"vit_abs":0.0,
          "dot":0.0,"wanted_squared":0.0,"candidate_squared":0.0}
    count=0
    with ExitStack() as stack:
        arrays={name:view(mapped(stack,path,width,height),roi,offset) for name,path in paths.items()}
        h=roi[3]-roi[1]
        for y in range(0,h,rows):
            one=np.asarray(arrays["one"][y:y+rows],dtype=np.float64)
            vit=np.asarray(arrays["vit"][y:y+rows],dtype=np.float64)
            two=np.asarray(arrays["two"][y:y+rows],dtype=np.float64)
            d_one=one-two;d_vit=vit-two;wanted=two-one;candidate=vit-one
            sums["one_squared"]+=float(np.sum(d_one*d_one));sums["vit_squared"]+=float(np.sum(d_vit*d_vit))
            sums["one_abs"]+=float(np.abs(d_one).sum());sums["vit_abs"]+=float(np.abs(d_vit).sum())
            sums["dot"]+=float(np.sum(wanted*candidate));sums["wanted_squared"]+=float(np.sum(wanted*wanted))
            sums["candidate_squared"]+=float(np.sum(candidate*candidate));count+=one.size
        one_ssim=stripe_ssim(arrays["one"],arrays["two"],rows)
        vit_ssim=stripe_ssim(arrays["vit"],arrays["two"],rows)
    one_mse=sums["one_squared"]/count;vit_mse=sums["vit_squared"]/count
    one_psnr,vit_psnr=psnr(one_mse),psnr(vit_mse)
    cosine_denominator=math.sqrt(sums["wanted_squared"]*sums["candidate_squared"])
    return {"one_psnr_vs_two_db":one_psnr,"vit_psnr_vs_two_db":vit_psnr,
        "one_identical_two":one_mse==0,"vit_identical_two":vit_mse==0,
        "vit_psnr_gain_db":None if one_psnr is None or vit_psnr is None else vit_psnr-one_psnr,
        "one_mse_vs_two":one_mse,"vit_mse_vs_two":vit_mse,
        "one_mae_vs_two":sums["one_abs"]/count,"vit_mae_vs_two":sums["vit_abs"]/count,
        "one_ssim_vs_two":one_ssim,"vit_ssim_vs_two":vit_ssim,
        "second_change_projection":None if sums["wanted_squared"]==0 else sums["dot"]/sums["wanted_squared"],
        "second_change_cosine":None if cosine_denominator==0 else sums["dot"]/cosine_denominator,
        "second_change_rms":math.sqrt(sums["wanted_squared"]/count),
        "vit_change_from_one_rms":math.sqrt(sums["candidate_squared"]/count),"rgb_values":count}


def adjacent_changes(previous,current,width,height,roi,previous_offset,current_offset,rows):
    accum={name:{"sq":0.0,"abs":0.0,"signed":0.0} for name in (*MODES,"vit_minus_two","one_minus_two")}
    count=0
    with ExitStack() as stack:
        before={name:view(mapped(stack,previous[name],width,height),roi,previous_offset) for name in MODES}
        after={name:view(mapped(stack,current[name],width,height),roi,current_offset) for name in MODES}
        for y in range(0,roi[3]-roi[1],rows):
            delta={name:np.asarray(after[name][y:y+rows],dtype=np.float64)-np.asarray(before[name][y:y+rows],dtype=np.float64) for name in MODES}
            delta["vit_minus_two"]=delta["vit"]-delta["two"]
            delta["one_minus_two"]=delta["one"]-delta["two"]
            for name,value in delta.items():
                accum[name]["sq"]+=float(np.sum(value*value));accum[name]["abs"]+=float(np.abs(value).sum())
                accum[name]["signed"]+=float(value.sum())
            count+=delta["one"].size
    result={"rgb_values":count}
    for name,sums in accum.items():
        result[name+"_change_rms"]=math.sqrt(sums["sq"]/count)
        result[name+"_change_mean_abs"]=sums["abs"]/count
        result[name+"_change_signed_mean"]=sums["signed"]/count
    return result


def input_pair_change(previous,current,width,height,roi,rows):
    squared=absolute=signed=0.0;count=0
    with ExitStack() as stack:
        before=view(mapped(stack,previous,width,height),roi)
        after=view(mapped(stack,current,width,height),roi)
        for y in range(0,roi[3]-roi[1],rows):
            delta=np.asarray(after[y:y+rows],dtype=np.float64)-np.asarray(before[y:y+rows],dtype=np.float64)
            squared+=float(np.sum(delta*delta));absolute+=float(np.abs(delta).sum());signed+=float(delta.sum());count+=delta.size
    return {"input_change_rms":math.sqrt(squared/count),"input_change_mean_abs":absolute/count,
            "input_change_signed_mean":signed/count,"input_change_measured":True}


def paths_for(prefix,frame=None,last=False):
    if frame is not None:
        return {mode:Path(str(prefix)+f".{mode}.frame-{frame:04}.rgba16f") for mode in MODES}
    suffix=".last.rgba16f" if last else ".rgba16f"
    return {mode:Path(str(prefix)+"."+mode+suffix) for mode in MODES}


def write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(rows)


def aggregate_quality(rows):
    result={"measured_frames":len(rows)}
    for mode in ("one","vit"):
        mse=float(np.mean([r[mode+"_mse_vs_two"] for r in rows]))
        result[mode+"_psnr_from_mean_mse_db"]=psnr(mse)
        for metric in ("mae_vs_two","ssim_vs_two"):
            result[mode+"_mean_"+metric]=float(np.mean([r[mode+"_"+metric] for r in rows]))
    for metric in ("second_change_projection","second_change_cosine"):
        values=[r[metric] for r in rows if r[metric] is not None]
        result[metric+"_mean"]=None if not values else float(np.mean(values))
        result[metric+"_min"]=None if not values else min(values)
        result[metric+"_max"]=None if not values else max(values)
    return result


def contact_plot(prefix,output,width,height,roi,frames,step,source=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    crop_width=min(420,roi[2]-roi[0]);crop_height=min(300,roi[3]-roi[1])
    # Fixed centers chosen from the untouched source contact sheet. Each mode
    # and chronological frame uses the same source-space crop.
    centers={"nightclub":(0.55,0.40),"car":(0.55,0.58),"portrait":(0.42,0.40),"vendor":(0.38,0.50)}
    label=next((name for name in centers if name in (prefix.name+" "+source).lower()),None)
    center=centers.get(label,(0.5,0.5))
    x=max(roi[0],min(roi[2]-crop_width,round(width*center[0]-crop_width/2)))
    y=max(roi[1],min(roi[3]-crop_height,round(height*center[1]-crop_height/2)))
    crop=(x,y,x+crop_width,y+crop_height)
    fig,axes=plt.subplots(len(frames),3,figsize=(13,3.3*len(frames)),squeeze=False,layout="constrained")
    for row,frame in enumerate(frames):
        paths=paths_for(prefix,frame)
        offset=(0,0) if frame is None else (step[0]*frame,step[1]*frame)
        with ExitStack() as stack:
            for column,mode in enumerate(MODES):
                pixels=np.asarray(view(mapped(stack,paths[mode],width,height),crop,offset),dtype=np.float32)
                axes[row,column].imshow(np.clip(pixels,0,1),interpolation="nearest")
                axes[row,column].set_title(f"{mode} | "+("static" if frame is None else f"frame {frame}"))
                axes[row,column].axis("off")
    fig.suptitle(f"Native output crops: {prefix.name} | original pixel sampling",fontsize=12)
    fig.savefig(output / "native-crops.png",dpi=140);plt.close(fig)
    return {"scene":label,"source_crop_xyxy":crop,"sampling":"native pixels, same source-space crop for every mode/frame"}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind",choices=("static","sequence"))
    parser.add_argument("--prefix",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rows",type=int,default=128)
    parser.add_argument("--crop-margin",type=int,default=32,help="Additional source-space border excluded from synthetic pans")
    parser.add_argument("--contacts",action="store_true")
    args=parser.parse_args()
    if args.rows<16 or args.crop_margin<0:
        parser.error("rows must be at least16 and crop margin nonnegative")
    started=time.perf_counter()
    metadata=json.loads(Path(str(args.prefix)+".json").read_text())
    if args.kind=="sequence" and metadata.get("status")!="complete":
        raise ValueError("Sequence metadata must mark the inference run complete")
    width,height=int(metadata["width"]),int(metadata["height"])
    if (width,height) not in ((1920,1080),(2560,1440),(3840,2160)):
        raise ValueError("This validation only accepts1080p,1440p or4K native outputs")
    args.output.mkdir(parents=True,exist_ok=True)
    alpha_zero=float(metadata.get("alpha",-1))==0
    summary={"schema":1,"kind":args.kind,"prefix":str(args.prefix.resolve()),"width":width,"height":height,
        "model_metadata":metadata,"metric_encoding":"Raw FP16 RGB converted to float64; peak1, no quantization/resizing",
        "ssim":"11x11 Gaussian sigma1.5, mirror padding without edge repetition; raw float RGB constants0.01²/0.03²",
        "null_psnr":"Exact equality, infinite PSNR, is represented by null plus an identity flag",
        "cpu_only":True,"alpha_zero_control":alpha_zero}
    quality=[];changes=[];checks=[];alpha_checks=[]
    if args.kind=="static":
        roi=(0,0,width,height)
        paths=paths_for(args.prefix)
        check={name:integrity(path,width,height,args.rows) for name,path in paths.items()}
        checks.append({"frame":0,"modes":check})
        if not all(value["finite"] for value in check.values()):
            raise RuntimeError("Nonfinite native static output")
        quality.append({"frame":0,**triplet_quality(paths,width,height,roi,(0,0),args.rows)})
        stability={}
        last=paths_for(args.prefix,last=True)
        for name in MODES:
            stability[name]={"first_last":bitwise_compare(paths[name],last[name],width,height,args.rows),
                             "last_integrity":integrity(last[name],width,height,args.rows)}
        checks.append({"frame":"last","modes":{name:stability[name]["last_integrity"] for name in MODES}})
        if not all(stability[name]["last_integrity"]["finite"] for name in MODES):
            raise RuntimeError("Nonfinite native static last output")
        summary["reset_first_last_stability"]=stability
        if alpha_zero:
            alpha_checks.append({"frame":"first",**bitwise_compare(paths["one"],paths["vit"],width,height,args.rows)})
            alpha_checks.append({"frame":"last",**bitwise_compare(last["one"],last["vit"],width,height,args.rows)})
        if args.contacts:
            summary["contact_crop"]=contact_plot(args.prefix,args.output,width,height,roi,[None],(0,0),metadata.get("source_path",""))
    else:
        frames=int(metadata["frames_per_mode"]);warmups=int(metadata["warmup_frames_per_mode"])
        if not 0<=warmups<frames:
            raise ValueError("Invalid sequence warmup range")
        pan=metadata["source"]=="synthetic-progressive-pan"
        step=(int(metadata.get("pan_step_x",8)),int(metadata.get("pan_step_y",4))) if pan else (0,0)
        margin=args.crop_margin if pan else 0
        dxs=(0,step[0]*(frames-1));dys=(0,step[1]*(frames-1))
        roi=(max(0,-min(dxs))+margin,max(0,-min(dys))+margin,
             min(width,width-max(dxs))-margin,min(height,height-max(dys))-margin)
        if roi[2]<=roi[0]+10 or roi[3]<=roi[1]+10:
            raise ValueError("Sequence pan leaves no valid common crop")
        summary.update({"source":metadata["source"],"warmup_frames_discarded":warmups,"frames_per_mode":frames,
            "aligned_roi_xyxy":roi,"pan_step_xy":step,"alignment":"Equivalent to roll(-dx*t,-dy*t), then crop; direct slices avoid wrapped reads",
            "temporal_interpretation":"Synthetic pan: aligned output variation under known backward flow. Global attention can still see wrapped context outside the crop." if pan else
                "Unmodified real clip with zero motion guides. Animation/content changes remain; report adjacent changes and their error relative to full two-pass, not all changes as flicker.",
            "short_real_clip_warning":None if pan else "Short clip with initial history settling; not steady-state performance or broad motion coverage."})
        source_path=Path(metadata.get("source_path",""))
        input_files=[source_path/f"frame-{frame:04}.rgba16f" for frame in range(frames)] if not pan and source_path.is_dir() else []
        if input_files and any(not path.is_file() for path in input_files):
            raise ValueError("Source directory is missing an exact chronological frame filename")
        summary["input_unroll_check"]=(
            "Known roll(+8t,+4t), or metadata step, analytically unrolls to the same original input crop. Expected input change is zero; generated GPU input images were not separately dumped." if pan else
            "Real input-frame adjacent changes measured from untouched source files." if input_files else
            "Input files unavailable; output change error is measured relative to full two-pass.")
        previous=None;previous_frame=None
        for frame in range(frames):
            paths=paths_for(args.prefix,frame)
            check={name:integrity(path,width,height,args.rows) for name,path in paths.items()}
            checks.append({"frame":frame,"warmup":frame<warmups,"modes":check})
            if not all(value["finite"] for value in check.values()):
                (args.output / "integrity.json").write_text(json.dumps(checks,indent=2))
                raise RuntimeError(f"Nonfinite native sequence output on frame{frame}")
            if alpha_zero:
                alpha_checks.append({"frame":frame,**bitwise_compare(paths["one"],paths["vit"],width,height,args.rows)})
            if frame<warmups:
                continue
            offset=(step[0]*frame,step[1]*frame)
            row={"frame":frame,**triplet_quality(paths,width,height,roi,offset,args.rows)}
            quality.append(row)
            if previous is not None:
                change={"from_frame":previous_frame,"to_frame":frame,
                    **adjacent_changes(previous,paths,width,height,roi,(step[0]*previous_frame,step[1]*previous_frame),offset,args.rows)}
                if pan:
                    change.update({"input_change_rms":0.0,"input_change_mean_abs":0.0,"input_change_signed_mean":0.0,"input_change_measured":False})
                elif input_files:
                    change.update(input_pair_change(input_files[previous_frame],input_files[frame],width,height,roi,args.rows))
                changes.append(change)
            previous,previous_frame=paths,frame
            write_csv(args.output / "quality.csv",quality);write_csv(args.output / "temporal.csv",changes)
            print(f"{args.prefix.name}: frame{frame+1}/{frames}, ViT/two PSNR{row['vit_psnr_vs_two_db']}, transitions{len(changes)}",flush=True)
        summary["measured_adjacent_pairs"]=len(changes)
        summary["temporal_metrics"]={}
        if changes:
            for name in (*MODES,"vit_minus_two","one_minus_two"):
                summary["temporal_metrics"][name]={
                    "rms_over_adjacent_samples":math.sqrt(float(np.mean([r[name+"_change_rms"]**2 for r in changes]))),
                    "mean_abs_change":float(np.mean([r[name+"_change_mean_abs"] for r in changes])),
                    "signed_mean_change":float(np.mean([r[name+"_change_signed_mean"] for r in changes]))}
        if args.contacts:
            summary["contact_crop"]=contact_plot(args.prefix,args.output,width,height,roi,[warmups,frames-1],step,str(source_path))
    summary["quality"]=aggregate_quality(quality)
    summary["all_outputs_finite"]=all(v["finite"] for r in checks for v in r["modes"].values())
    summary["alpha_zero_checks"]=alpha_checks
    summary["alpha_zero_all_bit_exact"]=None if not alpha_checks else all(c["rgba16_bit_exact"] for c in alpha_checks)
    summary["analysis_cpu_seconds"]=time.perf_counter()-started
    write_csv(args.output / "quality.csv",quality);write_csv(args.output / "temporal.csv",changes)
    (args.output / "quality.json").write_text(json.dumps(quality,indent=2))
    (args.output / "temporal.json").write_text(json.dumps(changes,indent=2))
    (args.output / "integrity.json").write_text(json.dumps(checks,indent=2))
    (args.output / "summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"output":str(args.output),"quality":summary["quality"],
        "adjacent_pairs":len(changes),"alpha_zero_exact":summary["alpha_zero_all_bit_exact"],
        "analysis_cpu_seconds":summary["analysis_cpu_seconds"]},indent=2),flush=True)


if __name__=="__main__":
    main()
