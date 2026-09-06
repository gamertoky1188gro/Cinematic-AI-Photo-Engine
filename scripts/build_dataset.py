"""Generate reference plans for Unsplash images in the dataset."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.dataset import ReferenceDataset
from engine.pipeline import CinematicPipeline


def main():
    dataset_root = project_root / "dataset"
    unsplash_dir = dataset_root / "unsplash_images"

    print("Initializing dataset...")
    dataset = ReferenceDataset(dataset_root)

    print("Ingesting Unsplash images...")
    ingested = dataset.ingest_unsplash(unsplash_dir)
    print(f"  {len(ingested)} images ingested")

    print("\nGenerating render plans (this may take a while)...")
    # Use heuristic intelligence for plan generation (no predictor needed)
    from engine.intelligence import IntelligenceController
    heuristic = IntelligenceController()
    plans = {}
    images_dir = dataset.images_dir
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
            continue
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        scene = dataset._analyzer.analyze(image) if hasattr(dataset, '_analyzer') else None
        if scene is None:
            from engine.analyzer import SceneAnalyzer
            analyzer = SceneAnalyzer()
            scene = analyzer.analyze(image)
        plan = heuristic.generate(scene)
        plan_data = {k: round(v, 4) for k, v in plan.__dict__.items()}
        plan_file = dataset.plans_dir / f"{img_path.stem}.json"
        with open(plan_file, "w") as f:
            json.dump(plan_data, f, indent=2)
        plans[img_path.name] = {"scene": scene, "plan": plan_data}
        if len(plans) % 50 == 0:
            print(f"  {len(plans)} plans generated...")
    print(f"  {len(plans)} plans generated")

    print("\nGenerating reference outputs...")
    pipeline = CinematicPipeline(seed=42)
    references = []
    for img_path in sorted(dataset.images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
            continue
        ref_path = dataset.references_dir / f"{img_path.stem}_ref{img_path.suffix}"
        try:
            import time
            import tracemalloc
            from PIL import Image
            from engine.evaluate import compute_metrics

            original = Image.open(img_path).convert("RGB")
            tracemalloc.start()
            t0 = time.perf_counter()
            result = pipeline.run(str(img_path), str(ref_path))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_kb = peak / 1024

            ref_img = Image.open(ref_path).convert("RGB")
            metrics = compute_metrics(original, ref_img, elapsed_ms, memory_kb)
            references.append({
                "file": img_path.name,
                "reference": ref_path.name,
                "metrics": {
                    "psnr": round(metrics.psnr, 2),
                    "ssim": round(metrics.ssim, 4),
                    "max_delta": round(metrics.max_delta, 1),
                    "mean_delta": round(metrics.mean_delta, 2),
                    "clipping_pct": round(metrics.clipping_pct, 3),
                    "brightness_change": round(metrics.brightness_change, 2),
                    "saturation_change": round(metrics.saturation_change, 2),
                },
            })
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
        if len(references) % 50 == 0:
            print(f"  {len(references)} references generated...")
    print(f"  {len(references)} references generated")

    print("\nBuilding metadata...")
    metadata = dataset.build_metadata(plans, references)
    print(f"  metadata.json written with {len(metadata['images'])} entries")

    print("\nDataset build complete!")
    summary = dataset.summary()
    print(f"  Total images: {summary['count']}")
    print(f"  Mean PSNR: {summary['psnr_mean']}")
    print(f"  Mean SSIM: {summary['ssim_mean']}")


if __name__ == "__main__":
    main()
