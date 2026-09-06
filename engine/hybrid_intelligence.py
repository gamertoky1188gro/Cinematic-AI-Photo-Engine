import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

_models_dir = Path(__file__).resolve().parents[1] / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from parameter_predictor import MultiHeadPredictor, RENDER_PLAN_KEYS
from .render_plan import RenderPlan
from .intelligence import IntelligenceController
from .analyzer import SceneAnalyzer


class HybridIntelligence:
    """Blends heuristic and learned predictions with confidence weighting."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | None = None,
        base_confidence: float = 0.3,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.base_confidence = base_confidence

        self.heuristic = IntelligenceController()
        self.analyzer = SceneAnalyzer()

        self.model = MultiHeadPredictor(pretrained=False)
        self.has_learned = False

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state.get("model_state_dict", state))
            self.has_learned = True

        self.model.to(self.device).eval()

    def generate(self, image: Image.Image, scene: dict | None = None) -> RenderPlan:
        if scene is None:
            scene = self.analyzer.analyze(image)

        heuristic_plan = self.heuristic.generate(scene)

        if not self.has_learned:
            return heuristic_plan

        img = resize_and_to_tensor(image).unsqueeze(0).to(self.device)
        result = self.model.predict(img)

        learned_plan = RenderPlan(**result["parameters"])
        confidence = result["confidence"]

        return self._fuse(heuristic_plan, learned_plan, confidence)

    def _fuse(
        self,
        heuristic: RenderPlan,
        learned: RenderPlan,
        confidence: dict,
    ) -> RenderPlan:
        fused = {}
        for key in RENDER_PLAN_KEYS:
            h = getattr(heuristic, key)
            l = getattr(learned, key)
            c = confidence.get(key, 0.5) * self.base_confidence
            c = max(0.0, min(1.0, c))
            fused[key] = h * (1.0 - c) + l * c

        return RenderPlan(**fused)


def resize_and_to_tensor(image: Image.Image) -> torch.Tensor:
    from torchvision.transforms.functional import resize
    image = resize(image, (224, 224))
    return to_tensor(image)
