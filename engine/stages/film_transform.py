import numpy as np
from PIL import Image


def film_response(
    image: Image.Image,
    toe: float = 0.05,
    shoulder: float = 0.92,
    mid_gamma: float = 1.05,
    strength: float = 1.0,
) -> Image.Image:
    """S-curve mimicking film's tonal response (toe → mid → shoulder)."""
    arr = np.array(image, dtype=np.float64) / 255.0

    mid_point = (toe + shoulder) / 2.0

    def curve(x):
        m = (x - toe) / (shoulder - toe)
        m = np.clip(m, 0.0, 1.0)
        m = np.power(m, mid_gamma)
        m = toe + m * (shoulder - toe)
        return m

    toe_region = arr < toe
    shoulder_region = arr > shoulder
    mid_region = ~toe_region & ~shoulder_region

    result = arr.copy()
    result[mid_region] = curve(arr[mid_region])

    result[toe_region] = arr[toe_region] * (toe / 0.15)
    result[shoulder_region] = shoulder + (arr[shoulder_region] - shoulder) * 0.3

    result = np.clip(result, 0.0, 1.0)
    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def highlight_rolloff(
    image: Image.Image,
    threshold: float = 0.78,
    softness: float = 0.25,
    strength: float = 1.0,
) -> Image.Image:
    """Smoothly compress highlights above threshold — prevents hard clipping."""
    arr = np.array(image, dtype=np.float64) / 255.0

    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    t = np.clip((luma - threshold) / softness, 0.0, 1.0)

    roll = np.sin(t * np.pi / 2.0)
    scale = 1.0 - t * (1.0 - roll)

    for c in range(3):
        arr[:, :, c] *= scale

    arr = np.clip(arr, 0.0, 1.0)

    result = np.where(
        luma[:, :, np.newaxis] > threshold,
        arr + strength * (arr - arr),
        arr,
    )
    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def shadow_tint(
    image: Image.Image,
    tint_color: tuple = (0.0, 0.04, 0.08),
    shadow_threshold: float = 0.35,
    strength: float = 1.0,
) -> Image.Image:
    """Add color tint to shadow regions — classic film stock behavior."""
    arr = np.array(image, dtype=np.float64) / 255.0

    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    mask = np.clip((shadow_threshold - luma) / shadow_threshold, 0.0, 1.0)
    mask = mask[:, :, np.newaxis]

    tint = np.array(tint_color, dtype=np.float64)
    tint = tint[np.newaxis, np.newaxis, :]

    result = arr + mask * tint * strength
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def color_separation(
    image: Image.Image,
    red_shift: float = 0.02,
    green_shift: float = 0.0,
    blue_shift: float = -0.01,
    red_gamma: float = 0.98,
    green_gamma: float = 1.0,
    blue_gamma: float = 1.02,
    strength: float = 1.0,
) -> Image.Image:
    """Per-channel contrast and offset — simulates film stock color response."""
    arr = np.array(image, dtype=np.float64) / 255.0

    shifts = np.array([red_shift, green_shift, blue_shift], dtype=np.float64)
    gammas = np.array([red_gamma, green_gamma, blue_gamma], dtype=np.float64)

    result = arr.copy()
    for c in range(3):
        result[:, :, c] = np.power(arr[:, :, c], gammas[c]) + shifts[c]

    result = np.clip(result, 0.0, 1.0)
    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


class FilmTransformStage:
    """Configurable film response transform — four independent operations."""

    def __init__(self, params: dict | None = None):
        self.params = params or self.defaults()

    @staticmethod
    def defaults() -> dict:
        return {
            "response": {
                "toe": 0.05,
                "shoulder": 0.92,
                "mid_gamma": 1.05,
                "strength": 1.0,
            },
            "rolloff": {
                "threshold": 0.78,
                "softness": 0.25,
                "strength": 1.0,
            },
            "shadow_tint": {
                "tint_color": (0.0, 0.04, 0.08),
                "shadow_threshold": 0.35,
                "strength": 0.6,
            },
            "color_sep": {
                "red_shift": 0.02,
                "green_shift": 0.0,
                "blue_shift": -0.01,
                "red_gamma": 0.98,
                "green_gamma": 1.0,
                "blue_gamma": 1.02,
                "strength": 0.7,
            },
        }

    def process(self, image: Image.Image, scene: dict | None = None) -> Image.Image:
        p = self.params

        result = film_response(image, **p["response"])
        result = highlight_rolloff(result, **p["rolloff"])
        result = shadow_tint(result, **p["shadow_tint"])
        result = color_separation(result, **p["color_sep"])

        return result
