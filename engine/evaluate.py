import time
import tracemalloc
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class Metrics:
    resolution: tuple
    processing_ms: float
    memory_peak_kb: float
    max_delta: float
    mean_delta: float
    psnr: float
    ssim: float
    clipping_pct: float
    saturation_change: float
    brightness_change: float


def _psnr(original: np.ndarray, processed: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _ssim(original: np.ndarray, processed: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    o = original.astype(np.float64)
    p = processed.astype(np.float64)

    mu_o = np.mean(o)
    mu_p = np.mean(p)
    sig_o_sq = np.var(o)
    sig_p_sq = np.var(p)
    sig_op = np.mean((o - mu_o) * (p - mu_p))

    num = (2 * mu_o * mu_p + c1) * (2 * sig_op + c2)
    den = (mu_o ** 2 + mu_p ** 2 + c1) * (sig_o_sq + sig_p_sq + c2)

    return float(num / den)


def measure(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ms, peak / 1024


def compute_metrics(original: Image.Image, processed: Image.Image, elapsed_ms: float = 0, memory_kb: float = 0) -> Metrics:
    a = np.array(original)
    b = np.array(processed)

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))

    h, w = a.shape[:2]

    o_luma = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
    p_luma = 0.2126 * b[:, :, 0] + 0.7152 * b[:, :, 1] + 0.0722 * b[:, :, 2]

    o_sat = np.max(a, axis=2).astype(float) - np.min(a, axis=2).astype(float)
    p_sat = np.max(b, axis=2).astype(float) - np.min(b, axis=2).astype(float)

    clipping = np.any(b == 0, axis=2) | np.any(b == 255, axis=2)

    return Metrics(
        resolution=(w, h),
        processing_ms=elapsed_ms,
        memory_peak_kb=memory_kb,
        max_delta=float(diff.max()),
        mean_delta=float(diff.mean()),
        psnr=_psnr(a, b),
        ssim=_ssim(a, b),
        clipping_pct=float(clipping.sum() / (h * w) * 100),
        saturation_change=float(np.mean(p_sat) - np.mean(o_sat)),
        brightness_change=float(np.mean(p_luma) - np.mean(o_luma)),
    )


def regression_report(input_path: str, pipeline, overrides: dict | None = None) -> dict:
    """Run full pipeline and per-stage regression, return structured report."""
    original = Image.open(input_path).convert("RGB")
    from .analyzer import SceneAnalyzer
    from .stages.film_transform import FilmTransformStage
    from .stages.optical_renderer import OpticalRenderer
    from .stages.geometry_renderer import GeometryRenderer

    analyzer = SceneAnalyzer()
    scene = analyzer.analyze(original)

    plan = pipeline.intelligence.generate(original, scene)
    if overrides:
        plan = pipeline.heuristic.merge(plan, overrides)

    stages = {}

    img, ms, mem = measure(pipeline.adaptive_lut.process, original, scene)
    stages["lut"] = compute_metrics(original, img, ms, mem)

    film = FilmTransformStage(params=plan.to_film_params())
    img2, ms2, mem2 = measure(film.process, img, scene)
    stages["film"] = compute_metrics(original, img2, ms2, mem2)

    optical = OpticalRenderer(params=plan.to_optical_params())
    img3, ms3, mem3 = measure(optical.process, img2, scene)
    stages["optical"] = compute_metrics(original, img3, ms3, mem3)

    geometry = GeometryRenderer(params=plan.to_geometry_params())
    img4, ms4, mem4 = measure(geometry.process, img3, scene)
    stages["geometry"] = compute_metrics(original, img4, ms4, mem4)

    total_ms = sum(s.processing_ms for s in stages.values())
    total_mem = max(s.memory_peak_kb for s in stages.values())

    return {
        "resolution": f"{original.width}x{original.height}",
        "scene": scene,
        "plan": {k: round(v, 3) for k, v in plan.__dict__.items()},
        "stages": {k: {
            "resolution": f"{v.resolution[0]}x{v.resolution[1]}",
            "time_ms": round(v.processing_ms, 1),
            "max_delta": round(v.max_delta, 1),
            "mean_delta": round(v.mean_delta, 2),
            "psnr": round(v.psnr, 2),
            "ssim": round(v.ssim, 4),
            "clipping_pct": round(v.clipping_pct, 3),
            "saturation_change": round(v.saturation_change, 2),
            "brightness_change": round(v.brightness_change, 2),
        } for k, v in stages.items()},
        "total_ms": round(total_ms, 1),
        "total_memory_kb": round(total_mem, 0),
    }
