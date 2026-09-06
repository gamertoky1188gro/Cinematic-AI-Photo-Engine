from pathlib import Path

import torch
from PIL import Image

from .analyzer import SceneAnalyzer
from .intelligence import IntelligenceController
from .hybrid_intelligence import HybridIntelligence
from .render_plan import RenderPlan
from .stages.adaptive_lut import AdaptiveLUTStage
from .stages.film_transform import FilmTransformStage
from .stages.optical_renderer import OpticalRenderer
from .stages.geometry_renderer import GeometryRenderer

RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}


def load_image(path: Path) -> Image.Image:
    """Load image, supporting RAW formats via rawpy."""
    if path.suffix.lower() in RAW_EXTENSIONS:
        try:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            return Image.fromarray(rgb)
        except Exception:
            pass
    return Image.open(path).convert("RGB")


class CinematicPipeline:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
        predictor_checkpoint: str | Path | None = None,
        hybrid_confidence: float = 0.3,
    ):
        self.analyzer = SceneAnalyzer()
        self.heuristic = IntelligenceController()
        self.adaptive_lut = AdaptiveLUTStage(model_dir=model_dir, device=device)
        self.seed = seed

        if predictor_checkpoint is not None:
            self.intelligence = HybridIntelligence(
                checkpoint_path=predictor_checkpoint,
                device=device,
                base_confidence=hybrid_confidence,
            )
        else:
            self.intelligence = None

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        overrides: dict | None = None,
        on_progress: callable = None,
        max_size: int | None = None,
    ) -> dict:
        def emit(stage, pct, detail=""):
            if on_progress:
                on_progress({"stage": stage, "percent": pct, "detail": detail})

        input_path = Path(input_path)
        output_path = Path(output_path)

        emit("analyzing", 5, "Loading image")
        image = load_image(input_path)

        if max_size:
            w, h = image.size
            scale = max_size / max(w, h)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)

        emit("analyzing", 15, "Scene analysis")
        scene = self.analyzer.analyze(image)

        emit("planning", 25, "Generating render plan")
        if self.intelligence is not None:
            plan = self.intelligence.generate(image, scene)
        else:
            plan = self.heuristic.generate(scene)

        if overrides:
            plan = self.heuristic.merge(plan, overrides)

        emit("lut", 35, "Adaptive 3D LUT")
        film = FilmTransformStage(params=plan.to_film_params())
        optical = OpticalRenderer(params=plan.to_optical_params(), seed=self.seed)
        geometry = GeometryRenderer(params=plan.to_geometry_params())

        result = self.adaptive_lut.process(image, scene)

        emit("film", 50, "Film transform")
        result = film.process(result, scene)

        emit("optical", 65, "Optical effects")
        result = optical.process(result, scene)

        emit("geometry", 80, "Geometry rendering")
        result = geometry.process(result, scene)

        emit("saving", 95, "Saving output")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, quality=95)

        emit("done", 100, "Complete")

        return {
            "input": str(input_path),
            "output": str(output_path),
            "scene": scene,
            "plan": {k: round(getattr(plan, k), 4) for k in [
                "lut_strength", "film_strength", "highlight_rolloff",
                "shadow_tint", "color_separation", "halation", "bloom",
                "grain", "vignette", "lens_distortion",
                "chromatic_aberration", "edge_softness",
            ]},
        }
