import argparse
import json
from pathlib import Path

from engine.pipeline import CinematicPipeline
from engine.evaluate import regression_report
from engine.ab_compare import ab_compare


def main():
    parser = argparse.ArgumentParser(description="Cinematic AI Photo Engine")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic grain")
    parser.add_argument("--report", action="store_true", help="Generate full evaluation report")
    parser.add_argument("--report-json", type=str, default=None, help="Save report as JSON")
    parser.add_argument("--compare", action="store_true", help="Run A/B/C heuristic vs learned vs hybrid")
    parser.add_argument("--predictor", type=str, default=None, help="Path to predictor checkpoint")
    parser.add_argument("--film-strength", type=float, default=None)
    parser.add_argument("--highlight-rolloff", type=float, default=None)
    parser.add_argument("--shadow-tint", type=float, default=None)
    parser.add_argument("--color-separation", type=float, default=None)
    parser.add_argument("--halation", type=float, default=None)
    parser.add_argument("--bloom", type=float, default=None)
    parser.add_argument("--grain", type=float, default=None)
    parser.add_argument("--vignette", type=float, default=None)
    parser.add_argument("--lens-distortion", type=float, default=None)
    parser.add_argument("--chromatic-aberration", type=float, default=None)
    parser.add_argument("--edge-softness", type=float, default=None)
    parser.add_argument("--lut-only", action="store_true", help="Skip all effects, LUT output only")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.compare:
        output_dir = Path(args.output) if args.output else Path("output") / "comparison"
        results = ab_compare(
            input_path=str(input_path),
            output_dir=str(output_dir),
            predictor_checkpoint=args.predictor,
            seed=args.seed or 42,
        )
        comparison_path = output_dir / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nComparison saved: {comparison_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("output") / f"{input_path.stem}_cinematic{input_path.suffix}"

    overrides = {}
    if args.lut_only:
        overrides = {
            "film_strength": 0.0, "highlight_rolloff": 0.0,
            "shadow_tint": 0.0, "color_separation": 0.0,
            "halation": 0.0, "bloom": 0.0, "grain": 0.0, "vignette": 0.0,
            "lens_distortion": 0.0, "chromatic_aberration": 0.0, "edge_softness": 0.0,
        }
    else:
        for key in [
            "film_strength", "highlight_rolloff", "shadow_tint", "color_separation",
            "halation", "bloom", "grain", "vignette",
            "lens_distortion", "chromatic_aberration", "edge_softness",
        ]:
            val = getattr(args, key, None)
            if val is not None:
                overrides[key] = val

    pipeline = CinematicPipeline(
        seed=args.seed,
        predictor_checkpoint=args.predictor,
    )
    result = pipeline.run(input_path, output_path, overrides=overrides or None)

    print(f"\nInput : {result['input']}")
    print(f"Output: {result['output']}")
    print(f"\nScene analysis:")
    for k, v in result["scene"].items():
        print(f"  {k}: {v}")
    print(f"\nRender plan:")
    for k, v in result["plan"].items():
        print(f"  {k}: {v:.4f}")

    if args.report or args.report_json:
        print(f"\n{'='*50}")
        print("CINEMATIC AI PHOTO ENGINE — EVALUATION REPORT")
        print(f"{'='*50}\n")

        report = regression_report(str(input_path), pipeline, overrides=overrides or None)

        print(f"Input:  {report['resolution']}")
        print()

        print("Scene:")
        for k, v in report["scene"].items():
            print(f"  {k}: {v}")
        print()

        print("RenderPlan:")
        for k, v in report["plan"].items():
            print(f"  {k}: {v}")
        print()

        print(f"{'Stage':<12} {'Time(ms)':>9} {'MaxD':>7} {'MeanD':>7} {'PSNR':>8} {'SSIM':>8} {'Clip%':>8} {'DBright':>8} {'DSat':>8}")
        print("-" * 90)
        for name, s in report["stages"].items():
            print(f"{name:<12} {s['time_ms']:>9.1f} {s['max_delta']:>7.1f} {s['mean_delta']:>7.2f} {s['psnr']:>8.2f} {s['ssim']:>8.4f} {s['clipping_pct']:>8.3f} {s['brightness_change']:>8.2f} {s['saturation_change']:>8.2f}")
        print("-" * 90)
        print(f"{'TOTAL':<12} {report['total_ms']:>9.1f}")
        print(f"\nPeak memory: {report['total_memory_kb']:.0f} KB")

        if args.report_json:
            json_path = Path(args.report_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
