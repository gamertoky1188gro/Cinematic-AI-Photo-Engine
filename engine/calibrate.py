import json
import time
from pathlib import Path
from PIL import Image

from .pipeline import CinematicPipeline
from .evaluate import compute_metrics, measure


def calibrate(
    image_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    report_path: str | Path | None = None,
) -> dict:
    """Run full calibration across a directory of images."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted([f for f in image_dir.rglob("*") if f.suffix.lower() in extensions])

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    pipeline = CinematicPipeline(seed=seed)
    results = []

    print(f"Calibrating on {len(images)} images from {image_dir}")
    print(f"Seed: {seed}")
    print(f"Output: {output_dir}\n")

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name} ... ", end="", flush=True)

        original = Image.open(img_path).convert("RGB")

        out_lut = output_dir / f"{img_path.stem}_lut{img_path.suffix}"
        out_full = output_dir / f"{img_path.stem}_full{img_path.suffix}"

        lut_result, lut_ms, lut_mem = measure(
            pipeline.adaptive_lut.process,
            original,
            pipeline.analyzer.analyze(original),
        )
        lut_result.save(out_lut, quality=95)

        full_result, full_ms, full_mem = measure(
            pipeline.run,
            str(img_path),
            str(out_full),
        )

        lut_metrics = compute_metrics(original, lut_result, lut_ms, lut_mem)
        full_metrics = compute_metrics(original, Image.open(out_full).convert("RGB"), full_ms, full_mem)

        scene = pipeline.analyzer.analyze(original)
        plan = pipeline.intelligence.generate(scene)

        entry = {
            "file": img_path.name,
            "resolution": f"{original.width}x{original.height}",
            "scene": scene,
            "plan": {k: round(v, 3) for k, v in plan.__dict__.items()},
            "lut": {
                "psnr": round(lut_metrics.psnr, 2),
                "ssim": round(lut_metrics.ssim, 4),
                "max_delta": round(lut_metrics.max_delta, 1),
                "mean_delta": round(lut_metrics.mean_delta, 2),
                "clipping_pct": round(lut_metrics.clipping_pct, 3),
                "brightness_change": round(lut_metrics.brightness_change, 2),
                "saturation_change": round(lut_metrics.saturation_change, 2),
                "time_ms": round(lut_metrics.processing_ms, 1),
            },
            "full": {
                "psnr": round(full_metrics.psnr, 2),
                "ssim": round(full_metrics.ssim, 4),
                "max_delta": round(full_metrics.max_delta, 1),
                "mean_delta": round(full_metrics.mean_delta, 2),
                "clipping_pct": round(full_metrics.clipping_pct, 3),
                "brightness_change": round(full_metrics.brightness_change, 2),
                "saturation_change": round(full_metrics.saturation_change, 2),
                "time_ms": round(full_metrics.processing_ms, 1),
            },
        }
        results.append(entry)

        print(f"LUT PSNR={lut_metrics.psnr:.1f} SSIM={lut_metrics.ssim:.3f} | Full PSNR={full_metrics.psnr:.1f} SSIM={full_metrics.ssim:.3f}")

    summary = _summarize(results)

    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"results": results, "summary": summary}, f, indent=2)
        print(f"\nReport saved: {report_path}")

    _print_summary(summary)
    return {"results": results, "summary": summary}


def _summarize(results: list) -> dict:
    import numpy as np

    lut_psnrs = [r["lut"]["psnr"] for r in results]
    full_psnrs = [r["full"]["psnr"] for r in results]
    lut_ssims = [r["lut"]["ssim"] for r in results]
    full_ssims = [r["full"]["ssim"] for r in results]
    full_clips = [r["full"]["clipping_pct"] for r in results]
    full_times = [r["full"]["time_ms"] for r in results]

    return {
        "count": len(results),
        "lut_psnr_mean": round(float(np.mean(lut_psnrs)), 2),
        "lut_psnr_min": round(float(np.min(lut_psnrs)), 2),
        "full_psnr_mean": round(float(np.mean(full_psnrs)), 2),
        "full_psnr_min": round(float(np.min(full_psnrs)), 2),
        "lut_ssim_mean": round(float(np.mean(lut_ssims)), 4),
        "full_ssim_mean": round(float(np.mean(full_ssims)), 4),
        "full_clipping_mean": round(float(np.mean(full_clips)), 3),
        "full_clipping_max": round(float(np.max(full_clips)), 3),
        "full_time_mean": round(float(np.mean(full_times)), 1),
        "full_time_total": round(float(np.sum(full_times)), 1),
    }


def _print_summary(summary: dict):
    print(f"\n{'='*50}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*50}")
    print(f"Images:        {summary['count']}")
    print(f"LUT PSNR:      {summary['lut_psnr_mean']:.2f} (min: {summary['lut_psnr_min']:.2f})")
    print(f"Full PSNR:     {summary['full_psnr_mean']:.2f} (min: {summary['full_psnr_min']:.2f})")
    print(f"LUT SSIM:      {summary['lut_ssim_mean']:.4f}")
    print(f"Full SSIM:     {summary['full_ssim_mean']:.4f}")
    print(f"Clipping:      mean={summary['full_clipping_mean']:.3f}% max={summary['full_clipping_max']:.3f}%")
    print(f"Time (per img):{summary['full_time_mean']:.1f} ms")
    print(f"Time (total):  {summary['full_time_total']:.1f} ms")
    print(f"{'='*50}")
