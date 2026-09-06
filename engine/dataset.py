import json
import shutil
from pathlib import Path
from PIL import Image

from .pipeline import CinematicPipeline
from .evaluate import compute_metrics, measure


class ReferenceDataset:
    """Builds and manages reference datasets for learned parameter prediction."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.references_dir = self.root / "references"
        self.plans_dir = self.root / "plans"
        self.metadata_path = self.root / "metadata.json"

        for d in [self.images_dir, self.references_dir, self.plans_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def ingest(self, source_dir: str | Path, copy: bool = True) -> list:
        """Copy/link images from source into dataset structure."""
        source_dir = Path(source_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = sorted([f for f in source_dir.rglob("*") if f.suffix.lower() in extensions])

        ingested = []
        for img_path in images:
            dest = self.images_dir / img_path.name
            if copy and not dest.exists():
                shutil.copy2(img_path, dest)
            ingested.append(img_path.name)

        return ingested

    def ingest_unsplash(self, unsplash_dir: str | Path, copy: bool = True) -> list:
        """Ingest images from Unsplash download directory."""
        unsplash_dir = Path(unsplash_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = sorted([f for f in unsplash_dir.iterdir() if f.suffix.lower() in extensions])

        ingested = []
        for img_path in images:
            dest = self.images_dir / img_path.name
            if copy and not dest.exists():
                shutil.copy2(img_path, dest)
            ingested.append(img_path.name)

        return ingested

    def generate_plans(
        self,
        pipeline: CinematicPipeline | None = None,
        seed: int = 42,
    ) -> dict:
        """Generate render plans for all images using intelligence layer."""
        if pipeline is None:
            pipeline = CinematicPipeline(seed=seed)

        plans = {}
        images = sorted(self.images_dir.glob("*"))

        for img_path in images:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                continue

            image = Image.open(img_path).convert("RGB")
            scene = pipeline.analyzer.analyze(image)
            plan = pipeline.intelligence.generate(scene)

            plan_data = {k: round(v, 4) for k, v in plan.__dict__.items()}

            plan_file = self.plans_dir / f"{img_path.stem}.json"
            with open(plan_file, "w") as f:
                json.dump(plan_data, f, indent=2)

            plans[img_path.name] = {
                "scene": scene,
                "plan": plan_data,
            }

        return plans

    def generate_references(
        self,
        pipeline: CinematicPipeline | None = None,
        seed: int = 42,
    ) -> list:
        """Generate cinematic reference outputs for all images."""
        if pipeline is None:
            pipeline = CinematicPipeline(seed=seed)

        results = []
        images = sorted(self.images_dir.glob("*"))

        for img_path in images:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                continue

            ref_path = self.references_dir / f"{img_path.stem}_ref{img_path.suffix}"
            original = Image.open(img_path).convert("RGB")

            result, ms, mem = measure(
                pipeline.run,
                str(img_path),
                str(ref_path),
            )

            ref_img = Image.open(ref_path).convert("RGB")
            metrics = compute_metrics(original, ref_img, ms, mem)

            results.append({
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

        return results

    def build_metadata(self, plans: dict, references: list):
        """Combine plans and references into metadata.json."""
        metadata = {
            "version": "0.8",
            "images": [],
        }

        ref_map = {r["file"]: r for r in references}

        for filename, data in plans.items():
            entry = {
                "image": filename,
                "reference": ref_map.get(filename, {}).get("reference"),
                "scene": data["scene"],
                "render_plan": data["plan"],
                "metrics": ref_map.get(filename, {}).get("metrics"),
            }
            metadata["images"].append(entry)

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def full_build(
        self,
        source_dir: str | Path,
        seed: int = 42,
    ) -> dict:
        """Complete pipeline: ingest → plans → references → metadata."""
        print(f"Building reference dataset at {self.root}\n")

        print("Step 1: Ingesting images...")
        ingested = self.ingest(source_dir)
        print(f"  {len(ingested)} images ingested\n")

        print("Step 2: Generating render plans...")
        pipeline = CinematicPipeline(seed=seed)
        plans = self.generate_plans(pipeline, seed)
        print(f"  {len(plans)} plans generated\n")

        print("Step 3: Generating reference outputs...")
        references = self.generate_references(pipeline, seed)
        print(f"  {len(references)} references generated\n")

        print("Step 4: Building metadata...")
        metadata = self.build_metadata(plans, references)
        print(f"  metadata.json written\n")

        print(f"Dataset ready: {self.root}")
        return metadata

    def summary(self) -> dict:
        """Load and summarize metadata."""
        with open(self.metadata_path) as f:
            metadata = json.load(f)

        import numpy as np

        psnrs = [e["metrics"]["psnr"] for e in metadata["images"] if e.get("metrics")]
        ssims = [e["metrics"]["ssim"] for e in metadata["images"] if e.get("metrics")]

        return {
            "count": len(metadata["images"]),
            "psnr_mean": round(float(np.mean(psnrs)), 2) if psnrs else 0,
            "ssim_mean": round(float(np.mean(ssims)), 4) if ssims else 0,
            "images": [e["image"] for e in metadata["images"]],
        }
