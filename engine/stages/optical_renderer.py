import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def _luma(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def halation(
    image: Image.Image,
    threshold: float = 0.85,
    blur_radius: float = 12.0,
    tint: tuple = (0.9, 0.5, 0.2),
    strength: float = 1.0,
) -> Image.Image:
    """Warm red glow around very bright regions — light bouncing off film base."""
    arr = np.array(image, dtype=np.float64) / 255.0
    luma = _luma(arr)

    mask = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0)
    mask = gaussian_filter(mask, sigma=blur_radius)

    tint_arr = np.array(tint, dtype=np.float64)
    glow = mask[:, :, np.newaxis] * tint_arr[np.newaxis, np.newaxis, :]

    result = arr + glow * strength
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def bloom(
    image: Image.Image,
    threshold: float = 0.75,
    blur_radius: float = 18.0,
    intensity: float = 0.35,
    strength: float = 1.0,
) -> Image.Image:
    """Soft light bleed from bright regions — optical lens glow."""
    arr = np.array(image, dtype=np.float64) / 255.0
    luma = _luma(arr)

    mask = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0)
    blurred = np.stack([
        gaussian_filter(arr[:, :, c], sigma=blur_radius)
        for c in range(3)
    ], axis=2)

    glow = mask[:, :, np.newaxis] * blurred * intensity

    result = arr + glow * strength
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def film_grain(
    image: Image.Image,
    amount: float = 0.03,
    size: float = 1.0,
    strength: float = 1.0,
    seed: int | None = None,
) -> Image.Image:
    """Luminance-oriented stochastic grain — real film grain structure."""
    arr = np.array(image, dtype=np.float64) / 255.0
    h, w = arr.shape[:2]

    gh = max(1, int(h / size))
    gw = max(1, int(w / size))

    rng = np.random.RandomState(seed)
    noise_small = rng.randn(gh, gw).astype(np.float64)

    from scipy.ndimage import zoom
    noise = zoom(noise_small, (h / gh, w / gw), order=1)
    noise = noise[:h, :w]

    luma = _luma(arr)

    grain_luma = noise * amount * (0.5 + luma)

    result = arr.copy()
    for c in range(3):
        result[:, :, c] += grain_luma * [1.0, 1.0, 1.0][c]

    result = np.clip(result, 0.0, 1.0)
    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


def vignette(
    image: Image.Image,
    radius: float = 0.65,
    softness: float = 0.35,
    strength: float = 1.0,
) -> Image.Image:
    """Gradual optical falloff from center — not a black border."""
    arr = np.array(image, dtype=np.float64) / 255.0
    h, w = arr.shape[:2]

    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0

    dist = np.sqrt(((x - cx) / (w / 2.0)) ** 2 + ((y - cy) / (h / 2.0)) ** 2)

    mask = np.clip((dist - radius) / softness, 0.0, 1.0)
    mask = mask ** 2

    result = arr * (1.0 - mask[:, :, np.newaxis])
    result = np.clip(result, 0.0, 1.0)

    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 1.0)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


class OpticalRenderer:
    """Four independently adjustable optical effects."""

    def __init__(self, params: dict | None = None, seed: int | None = None):
        self.params = params or self.defaults()
        self.seed = seed

    @staticmethod
    def defaults() -> dict:
        return {
            "halation": {
                "threshold": 0.85,
                "blur_radius": 12.0,
                "tint": (0.9, 0.5, 0.2),
                "strength": 0.4,
            },
            "bloom": {
                "threshold": 0.75,
                "blur_radius": 18.0,
                "intensity": 0.35,
                "strength": 0.3,
            },
            "grain": {
                "amount": 0.03,
                "size": 1.0,
                "strength": 0.5,
            },
            "vignette": {
                "radius": 1.2,
                "softness": 0.6,
                "strength": 0.35,
            },
        }

    def process(self, image: Image.Image, scene: dict | None = None) -> Image.Image:
        p = self.params

        highlights = halation(image, **p["halation"])
        result = bloom(highlights, **p["bloom"])
        grain_params = dict(p["grain"])
        if self.seed is not None:
            grain_params["seed"] = self.seed
        result = film_grain(result, **grain_params)
        result = vignette(result, **p["vignette"])

        return result
