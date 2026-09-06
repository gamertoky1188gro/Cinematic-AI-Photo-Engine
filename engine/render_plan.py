from dataclasses import dataclass, field

SCHEMA_VERSION = 1
PARAMETER_KEYS = [
    "lut_strength", "film_strength", "highlight_rolloff", "shadow_tint",
    "color_separation", "halation", "bloom", "grain", "vignette",
    "perspective", "lens_distortion", "chromatic_aberration", "edge_softness",
]


@dataclass
class RenderPlan:
    """Normalized render plan — 0.0=disabled, 1.0=max cinematic strength."""

    lut_strength: float = 1.0

    film_strength: float = 1.0
    highlight_rolloff: float = 1.0
    shadow_tint: float = 1.0
    color_separation: float = 1.0

    halation: float = 0.0
    bloom: float = 0.0
    grain: float = 0.0
    vignette: float = 0.0

    perspective: float = 0.0
    lens_distortion: float = 0.0
    chromatic_aberration: float = 0.0
    edge_softness: float = 0.0

    def to_film_params(self) -> dict:
        return {
            "response": {"toe": 0.05, "shoulder": 0.92, "mid_gamma": 1.05, "strength": self.film_strength},
            "rolloff": {"threshold": 0.78, "softness": 0.25, "strength": self.highlight_rolloff},
            "shadow_tint": {"tint_color": (0.0, 0.04, 0.08), "shadow_threshold": 0.35, "strength": self.shadow_tint},
            "color_sep": {"red_shift": 0.02, "green_shift": 0.0, "blue_shift": -0.01,
                          "red_gamma": 0.98, "green_gamma": 1.0, "blue_gamma": 1.02, "strength": self.color_separation},
        }

    def to_optical_params(self) -> dict:
        return {
            "halation": {"threshold": 0.85, "blur_radius": 12.0, "tint": (0.9, 0.5, 0.2), "strength": self.halation},
            "bloom": {"threshold": 0.75, "blur_radius": 18.0, "intensity": 0.35, "strength": self.bloom},
            "grain": {"amount": 0.03, "size": 1.0, "strength": self.grain},
            "vignette": {"radius": 1.2, "softness": 0.6, "strength": self.vignette},
        }

    def to_geometry_params(self) -> dict:
        return {
            "perspective": {"horizontal": 0.0, "vertical": 0.0, "scale": 1.0, "strength": self.perspective},
            "distortion": {"k1": -0.001, "k2": 0.0, "strength": self.lens_distortion},
            "chromatic": {"shift": 0.1, "strength": self.chromatic_aberration},
            "edge_softness": {"radius": 0.8, "blur_sigma": 1.0, "strength": self.edge_softness},
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "parameters": {k: getattr(self, k) for k in PARAMETER_KEYS},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RenderPlan":
        params = d.get("parameters", d)
        return cls(**{k: params[k] for k in PARAMETER_KEYS if k in params})
