"""Encode saved WGC scRGB with shared NR transfer formulas; keep source untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("capture", type=Path)
p.add_argument("--curve", choices=["auto", "sdr", "soft-knee"], default="auto")
args = p.parse_args()
meta = json.loads((args.capture / "capture.json").read_text())
w, h = meta["width"], meta["height"]
hdr = meta["display_start"]["hdr_enabled"]
white = (meta["display_start"]["desktop_sdr_white_scrgb"] or 203/80) if hdr else 1.0
curve = args.curve
if curve == "auto":
    curve = "soft-knee" if meta["display_start"]["hdr_enabled"] else "sdr"
out = args.capture / "model"
out.mkdir(exist_ok=True)
report = {"encoding": curve, "white_scrgb": white, "width": w, "height": h,
          "source": "raw WGC RGBA16F scRGB; no synthetic image motion",
          "note": "CPU float32 transfer formulas from dlssnr.hlsl; FP16 output. HDR display status selects default soft-knee proxy, not a claim about current app settings. SDR uses CaptureColour mode21. No depth or motion was captured.",
          "frames": []}
selected = set(np.linspace(0, len(meta["frames"])-1, min(8, len(meta["frames"])), dtype=int).tolist())
thumbs = []
previous = None
for info in meta["frames"]:
    source = args.capture / info["file"]
    rgba = np.fromfile(source, dtype="<f2").reshape(h, w, 4).astype(np.float32)
    if not np.isfinite(rgba).all():
        raise ValueError(f"Nonfinite captured input: {source}")
    rgb = np.maximum(rgba[..., :3] / np.float32(white), np.float32(0))
    if curve == "soft-knee":
        y = rgb[..., 0]*np.float32(.2126) + rgb[..., 1]*np.float32(.7152) + rgb[..., 2]*np.float32(.0722)
        rolled = np.float32(.75) + np.float32(.25)*(1-np.exp(-(y-np.float32(.75))/np.float32(.25)))
        rgb *= np.where(y > .75, rolled / np.maximum(y, 1e-8), 1)[..., None]
        rgb /= np.maximum(np.max(rgb, axis=2), 1)[..., None]
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb < np.float32(.0031308), rgb*np.float32(12.92),
                   np.float32(1.055)*np.power(np.maximum(rgb, np.float32(1e-8)), np.float32(1/2.4))-np.float32(.055))
    rgba[..., :3] = rgb
    half = rgba.astype("<f2")
    target = out / f"frame-{info['index']:04d}.rgba16f"
    half.tofile(target)
    delta = float(np.mean(np.abs(rgb-previous))) if previous is not None else 0
    report["frames"].append({"index": info["index"], "relative_ms": info["relative_ms"],
                             "rgb_delta_previous": delta, "sha256": hashlib.sha256(half.tobytes()).hexdigest()})
    previous = rgb.copy()
    if info["index"] in selected:
        thumb = Image.fromarray((np.clip(rgb,0,1)*255+.5).astype(np.uint8)).resize((640,360), Image.Resampling.LANCZOS)
        ImageDraw.Draw(thumb).text((8,8), f"Frame {info['index']} / {info['relative_ms']:.0f} ms", fill="white", stroke_width=1, stroke_fill="black")
        thumbs.append(thumb)
    print(f"Encoded {info['index']}: delta {delta:.6f}", flush=True)
(args.capture / "model.json").write_text(json.dumps(report, indent=2))
preview = Image.new("RGB", (640*min(4,len(thumbs)),360*((len(thumbs)+3)//4)))
for i, thumb in enumerate(thumbs): preview.paste(thumb, ((i%4)*640,(i//4)*360))
preview.save(args.capture / "contact-sheet.png")
