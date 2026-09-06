import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

_backend = Path(__file__).resolve().parents[3] / "Image-Adaptive-3DLUT"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from models_modern import (
    Classifier,
    Generator3DLUT_identity,
    Generator3DLUT_zero,
    apply_3d_lut,
)


class AdaptiveLUTStage:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: torch.device | None = None,
    ):
        if model_dir is None:
            model_dir = Path(__file__).resolve().parents[3] / "Image-Adaptive-3DLUT" / "pretrained_models" / "sRGB"
        else:
            model_dir = Path(model_dir)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model_dir = model_dir
        self._load()

    def _load(self):
        identity_lut = _backend / "IdentityLUT33.txt"
        self.LUT0 = Generator3DLUT_identity(dim=33, lut_file=str(identity_lut))
        self.LUT1 = Generator3DLUT_zero(dim=33)
        self.LUT2 = Generator3DLUT_zero(dim=33)
        self.classifier = Classifier()

        lut_file = self.model_dir / "LUTs.pth"
        cls_file = self.model_dir / "classifier.pth"

        LUTs = torch.load(lut_file, map_location="cpu")
        cls_state = torch.load(cls_file, map_location="cpu")

        self.LUT0.load_state_dict(LUTs["0"])
        self.LUT1.load_state_dict(LUTs["1"])
        self.LUT2.load_state_dict(LUTs["2"])
        self.classifier.load_state_dict(cls_state)

        self.LUT0.to(self.device).eval()
        self.LUT1.to(self.device).eval()
        self.LUT2.to(self.device).eval()
        self.classifier.to(self.device).eval()

    @torch.no_grad()
    def process(self, image: Image.Image, scene: dict | None = None) -> Image.Image:
        img = to_tensor(image).unsqueeze(0).to(self.device)

        pred = self.classifier(img).flatten(1)
        weights = pred[0, :3].cpu().tolist()

        adaptive_lut = (
            weights[0] * self.LUT0.LUT
            + weights[1] * self.LUT1.LUT
            + weights[2] * self.LUT2.LUT
        )

        result = apply_3d_lut(adaptive_lut, img)

        result = (
            result.squeeze(0)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        return Image.fromarray(result, mode="RGB")
