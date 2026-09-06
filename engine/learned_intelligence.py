import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

import sys
from pathlib import Path

_models_dir = Path(__file__).resolve().parents[1] / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from parameter_predictor import ParameterPredictor, RENDER_PLAN_KEYS
from .render_plan import RenderPlan
from .intelligence import IntelligenceController


class LearnedIntelligence:
    """Uses a trained ParameterPredictor to generate RenderPlans."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = ParameterPredictor(pretrained=False)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state.get("model_state_dict", state))
        self.model.to(self.device).eval()

        self.heuristic = IntelligenceController()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> RenderPlan:
        img = to_tensor(image).unsqueeze(0).to(self.device)
        params = self.model.predict(img)

        return RenderPlan(**params)

    def generate(self, image: Image.Image, scene: dict | None = None) -> RenderPlan:
        return self.predict(image)
