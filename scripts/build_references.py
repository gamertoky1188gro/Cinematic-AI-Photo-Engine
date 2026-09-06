"""Generate reference outputs for Unsplash images in the dataset."""

import sys
import json
import time
import tracemalloc
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.dataset import ReferenceDataset
from engine.pipeline import CinematicPipeline
from engine.evaluate import compute_metrics


def main():
    dataset_root = project_root / "dataset"

    print("Initializing dataset...")
    dataset = ReferenceDataset(dataset_root)

    print("Checking existing images...")
    images = sorted([f for f in dataset.images_dir.iterdir() 
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}])
    print(f"  {len(images)} images found")

    print("\nGenerating reference outputs...")
    pipeline = CinematicPipeline(seed=42)
    references = []
    
    for i, img_path in enumerate(images):
        ref_path = dataset.references_dir / f"{img_path.stem}_ref{img_path.suffix}"
        
        # Skip if reference already exists
        if ref_path.exists():
            continue
            
        try:
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
        
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} processed...")
    
    print(f"  {len(references)} references generated")

    print("\nBuilding metadata...")
    # Load existing plans
    plans = {}
    for plan_file in dataset.plans_dir.glob("*.json"):
        with open(plan_file) as f:
            plan_data = json.load(f)
        # Find corresponding image
        img_name = plan_file.stem + ".jpg"  # Default extension
        for img in images:
            if img.stem == plan_file.stem:
                img_name = img.name
                break
        plans[img_name] = {
            "scene": {},  # Will be filled from plan if needed
            "plan": plan_data,
        }

    metadata = dataset.build_metadata(plans, references)
    print(f"  metadata.json written with {len(metadata['images'])} entries")

    print("\nDataset build complete!")
    summary = dataset.summary()
    print(f"  Total images: {summary['count']}")
    print(f"  Mean PSNR: {summary['psnr_mean']}")
    print(f"  Mean SSIM: {summary['ssim_mean']}")


if __name__ == "__main__":
    main()
