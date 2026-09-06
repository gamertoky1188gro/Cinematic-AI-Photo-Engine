import json
from pathlib import Path
from PIL import Image

from .pipeline import CinematicPipeline
from .intelligence import IntelligenceController
from .quality_evaluator import evaluate_quality, QualityScores
from .evaluate import compute_metrics, measure


def ab_compare(
    input_path: str | Path,
    output_dir: str | Path,
    predictor_checkpoint: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """Run A (heuristic), B (learned), C (hybrid) comparison."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(input_path).convert("RGB")
    stem = input_path.stem

    pipeline_heuristic = CinematicPipeline(seed=seed)
    pipeline_hybrid = CinematicPipeline(
        predictor_checkpoint=predictor_checkpoint,
        seed=seed,
        hybrid_confidence=0.5,
    )

    results = {}

    out_a = output_dir / f"{stem}_A_heuristic.jpg"
    a_result, a_ms, a_mem = measure(pipeline_heuristic.run, str(input_path), str(out_a))
    a_img = Image.open(out_a).convert("RGB")
    a_quality = evaluate_quality(original, a_img)
    a_metrics = compute_metrics(original, a_img, a_ms, a_mem)
    results["A_heuristic"] = {
        "output": str(out_a),
        "plan": a_result["plan"],
        "quality": a_quality.__dict__,
        "metrics": {
            "psnr": round(a_metrics.psnr, 2),
            "ssim": round(a_metrics.ssim, 4),
            "max_delta": round(a_metrics.max_delta, 1),
            "mean_delta": round(a_metrics.mean_delta, 2),
        },
        "time_ms": round(a_ms, 1),
    }

    if predictor_checkpoint is not None:
        out_b = output_dir / f"{stem}_B_learned.jpg"
        pipeline_learned = CinematicPipeline(
            predictor_checkpoint=predictor_checkpoint,
            seed=seed,
        )
        pipeline_learned.intelligence.base_confidence = 1.0
        b_result, b_ms, b_mem = measure(pipeline_learned.run, str(input_path), str(out_b))
        b_img = Image.open(out_b).convert("RGB")
        b_quality = evaluate_quality(original, b_img)
        b_metrics = compute_metrics(original, b_img, b_ms, b_mem)
        results["B_learned"] = {
            "output": str(out_b),
            "plan": b_result["plan"],
            "quality": b_quality.__dict__,
            "metrics": {
                "psnr": round(b_metrics.psnr, 2),
                "ssim": round(b_metrics.ssim, 4),
                "max_delta": round(b_metrics.max_delta, 1),
                "mean_delta": round(b_metrics.mean_delta, 2),
            },
            "time_ms": round(b_ms, 1),
        }

        out_c = output_dir / f"{stem}_C_hybrid.jpg"
        c_result, c_ms, c_mem = measure(pipeline_hybrid.run, str(input_path), str(out_c))
        c_img = Image.open(out_c).convert("RGB")
        c_quality = evaluate_quality(original, c_img)
        c_metrics = compute_metrics(original, c_img, c_ms, c_mem)
        results["C_hybrid"] = {
            "output": str(out_c),
            "plan": c_result["plan"],
            "quality": c_quality.__dict__,
            "metrics": {
                "psnr": round(c_metrics.psnr, 2),
                "ssim": round(c_metrics.ssim, 4),
                "max_delta": round(c_metrics.max_delta, 1),
                "mean_delta": round(c_metrics.mean_delta, 2),
            },
            "time_ms": round(c_ms, 1),
        }

    _print_comparison(results)
    return results


def _print_comparison(results: dict):
    print(f"\n{'='*70}")
    print("A/B/C COMPARISON")
    print(f"{'='*70}\n")

    for name, data in results.items():
        q = data["quality"]
        m = data["metrics"]
        print(f"  {name}")
        print(f"    PSNR={m['psnr']:.1f}  SSIM={m['ssim']:.4f}  Time={data['time_ms']:.0f}ms")
        print(f"    Quality: natural={q['naturalness']:.3f}  cinematic={q['cinematicity']:.3f}  "
              f"color={q['color_quality']:.3f}  highlight={q['highlight_quality']:.3f}")
        print(f"             shadow={q['shadow_quality']:.3f}  detail={q['detail_preservation']:.3f}  "
              f"optical={q['optical_realism']:.3f}  OVERALL={q['overall']:.3f}")
        print()

    if len(results) > 1:
        best = max(results.items(), key=lambda x: x[1]["quality"]["overall"])
        print(f"  Best overall: {best[0]} (score={best[1]['quality']['overall']:.3f})")
    print(f"{'='*70}")
