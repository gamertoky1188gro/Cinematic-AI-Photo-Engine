import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


COLOR_KEYS = [
    "lut_strength", "film_strength", "highlight_rolloff",
    "shadow_tint", "color_separation",
]

OPTICAL_KEYS = [
    "halation", "bloom", "grain", "vignette",
]

GEOMETRY_KEYS = [
    "perspective", "lens_distortion", "chromatic_aberration", "edge_softness",
]

RENDER_PLAN_KEYS = COLOR_KEYS + OPTICAL_KEYS + GEOMETRY_KEYS
NUM_PARAMS = len(RENDER_PLAN_KEYS)

HEAD_SIZES = {
    "color": len(COLOR_KEYS),
    "optical": len(OPTICAL_KEYS),
    "geometry": len(GEOMETRY_KEYS),
}


class MultiHeadPredictor(nn.Module):
    """Multi-head vision model with uncertainty estimation."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        backbone_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Linear(backbone_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.color_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, HEAD_SIZES["color"]),
            nn.Sigmoid(),
        )

        self.optical_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, HEAD_SIZES["optical"]),
            nn.Sigmoid(),
        )

        self.geometry_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, HEAD_SIZES["geometry"]),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, NUM_PARAMS),
            nn.Sigmoid(),
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        shared = self.shared(features)

        color = self.color_head(shared)
        optical = self.optical_head(shared)
        geometry = self.geometry_head(shared)
        uncertainty = self.uncertainty_head(shared)

        params = torch.cat([color, optical, geometry], dim=1)

        return params, uncertainty

    def predict(self, x: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            params, uncertainty = self.forward(x)
        return {
            "parameters": {
                k: float(params[0, i])
                for i, k in enumerate(RENDER_PLAN_KEYS)
            },
            "confidence": {
                k: float(1.0 - uncertainty[0, i])
                for i, k in enumerate(RENDER_PLAN_KEYS)
            },
        }


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
