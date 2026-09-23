"""Summarize fenced D3D12 timestamp and full RGBA16F readback results."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_image(path: Path, width: int, height: int) -> np.ndarray:
    pixels = np.fromfile(path, dtype="<f2")
    expected = width * height * 4
    if pixels.size != expected:
        raise ValueError(f"{path}: {pixels.size} samples; expected {expected}")
    image = pixels.reshape(height, width, 4).astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: non-finite pixels")
    if image[..., :3].mean() < 0.01:
        raise ValueError(f"{path}: near-black output")
    return image


def png_rgb(image: np.ndarray, path: Path) -> None:
    encoded = (np.clip(image[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(encoded, "RGB").save(path)


def compare(a: np.ndarray, b: np.ndarray, path: Path) -> dict:
    diff = np.abs(a[..., :3] - b[..., :3])
    h, w = diff.shape[:2]
    central = diff[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    periphery = diff.copy()
    periphery[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)] = 0
    periphery_count = diff.size - central.size
    mse = float(np.mean((a[..., :3] - b[..., :3]) ** 2))
    scale = max(float(np.percentile(diff, 99)), 1e-5)
    heat = np.clip(diff.mean(axis=2) / scale, 0, 1)
    hot = np.empty((h, w, 3), dtype=np.uint8)
    hot[..., 0] = (heat * 255).astype(np.uint8)
    hot[..., 1] = (np.sqrt(heat) * 90).astype(np.uint8)
    hot[..., 2] = ((1 - heat) * 35).astype(np.uint8)
    Image.fromarray(hot, "RGB").save(path)
    return {
        "mae_rgb": float(diff.mean()),
        "p95_absolute_rgb": float(np.percentile(diff, 95)),
        "max_absolute_rgb": float(diff.max()),
        "psnr_db": float(10 * np.log10(1 / mse)) if mse else None,
        "center_mae_rgb": float(central.mean()),
        "periphery_mae_rgb": float(periphery.sum() / periphery_count),
        "heatmap_scale_p99": scale,
    }


def main(out: Path) -> None:
    rows = list(csv.DictReader((out / "timings.csv").open(newline="")))
    if not rows:
        raise ValueError("No GPU timestamp rows")
    timings = {}
    images = {}
    for name in ("native100", "spatial100", "uniform85", "spatial85"):
        selected = [r for r in rows if r["case"] == name]
        if len(selected) < 5:
            raise ValueError(f"{name}: expected at least five frames")
        # First two evaluations warm the feature and its temporal history.
        settled = selected[2:]
        timings[name] = {
            field: {"median_ms": float(np.median([float(r[field]) for r in settled])),
                    "p95_ms": float(np.percentile([float(r[field]) for r in settled], 95))}
            for field in ("pack_ms", "nr_ms", "unpack_ms", "total_ms")
        }
        w, h = int(selected[0]["ordinary_w"]), int(selected[0]["ordinary_h"])
        images[name] = read_image(out / f"{name}.rgba16f", w, h)
        png_rgb(images[name], out / f"{name}.png")
    metrics = {
        "spatial100_vs_native100": compare(images["native100"], images["spatial100"], out / "difference100.png"),
        "spatial85_vs_uniform85": compare(images["uniform85"], images["spatial85"], out / "difference85.png"),
    }
    speed = {
        "spatial100_vs_native100_percent": 100 * (1 - timings["spatial100"]["total_ms"]["median_ms"] / timings["native100"]["total_ms"]["median_ms"]),
        "spatial85_vs_uniform85_percent": 100 * (1 - timings["spatial85"]["total_ms"]["median_ms"] / timings["uniform85"]["total_ms"]["median_ms"]),
    }
    report = {"gpu_timings_exclude_first_two_frames": timings, "median_total_savings": speed,
              "paired_image_difference": metrics,
              "fixture": "4K synthetic checker/wave scene, four moving bright-block positions, uniform depth and zero motion",
              "timed_work": "GPU timestamp interval encloses actual NVIDIA NR feature 18 evaluate; spatial cases also include production color/guide pack and unpack, uniform85 includes a linear resample via the color pack shader"}
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
