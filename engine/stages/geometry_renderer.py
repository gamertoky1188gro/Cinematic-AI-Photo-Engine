import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter


def perspective_transform(
    image: Image.Image,
    horizontal: float = 0.0,
    vertical: float = 0.0,
    scale: float = 1.0,
    strength: float = 1.0,
) -> Image.Image:
    """Subtle viewpoint/keystone correction via homography."""
    arr = np.array(image)
    h, w = arr.shape[:2]

    dx = horizontal * w * 0.01
    dy = vertical * h * 0.01

    src = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])

    dst = np.float32([
        [dx, dy],
        [w - dx, -dy],
        [w + dx, h + dy],
        [-dx, h - dy],
    ])

    if scale != 1.0:
        sx = w * (1.0 - scale) / 2.0
        sy = h * (1.0 - scale) / 2.0
        dst += np.array([[sx, sy], [-sx, sy], [-sx, -sy], [sx, -sy]])

    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    result = arr.astype(np.float64) + strength * (result.astype(np.float64) - arr.astype(np.float64))
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(result, mode="RGB")


def lens_distortion(
    image: Image.Image,
    k1: float = -0.005,
    k2: float = 0.0,
    strength: float = 1.0,
) -> Image.Image:
    """Radial barrel/pincushion distortion: r' = r(1 + k1*r² + k2*r⁴)."""
    arr = np.array(image)
    h, w = arr.shape[:2]

    cx, cy = w / 2.0, h / 2.0

    j, i = np.mgrid[0:h, 0:w]
    x = (i - cx) / cx
    y = (j - cy) / cy
    r2 = x * x + y * y
    r4 = r2 * r2
    factor = 1.0 + k1 * r2 + k2 * r4

    map_x = (cx + x * factor * cx).astype(np.float32)
    map_y = (cy + y * factor * cy).astype(np.float32)

    distorted = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    result = arr.astype(np.float64) + strength * (distorted.astype(np.float64) - arr.astype(np.float64))
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(result, mode="RGB")


def chromatic_aberration(
    image: Image.Image,
    shift: float = 0.15,
    strength: float = 1.0,
) -> Image.Image:
    """Edge-only RGB channel separation — zero at center, increases radially."""
    arr = np.array(image, dtype=np.float64)
    h, w = arr.shape[:2]

    cx, cy = w / 2.0, h / 2.0
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    y, x = np.mgrid[0:h, 0:w]
    dx = (x - cx) / max_r
    dy = (y - cy) / max_r
    r = np.sqrt(dx ** 2 + dy ** 2)

    norm_r = r / r.max() if r.max() > 0 else r

    shift_r = shift * norm_r
    shift_b = -shift * norm_r

    map_x_r = (x + shift_r * dx * cx).astype(np.float32)
    map_y_r = (y + shift_r * dy * cy).astype(np.float32)
    map_x_b = (x + shift_b * dx * cx).astype(np.float32)
    map_y_b = (y + shift_b * dy * cy).astype(np.float32)

    r_ch = cv2.remap(arr[:, :, 0], map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g_ch = arr[:, :, 1]
    b_ch = cv2.remap(arr[:, :, 2], map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    result = np.stack([r_ch, g_ch, b_ch], axis=2)

    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(result, mode="RGB")


def edge_softness(
    image: Image.Image,
    radius: float = 0.75,
    blur_sigma: float = 1.2,
    strength: float = 1.0,
) -> Image.Image:
    """Gradual peripheral sharpness reduction — less digitally perfect."""
    arr = np.array(image, dtype=np.float64)
    h, w = arr.shape[:2]

    blurred = np.stack([
        gaussian_filter(arr[:, :, c], sigma=blur_sigma)
        for c in range(3)
    ], axis=2)

    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)

    mask = np.clip((dist - radius) / (1.0 - radius), 0.0, 1.0)
    mask = mask[:, :, np.newaxis]

    result = arr * (1.0 - mask) + blurred * mask

    result = arr + strength * (result - arr)
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(result, mode="RGB")


class GeometryRenderer:
    """Four independently adjustable geometry effects."""

    def __init__(self, params: dict | None = None):
        self.params = params or self.defaults()

    @staticmethod
    def defaults() -> dict:
        return {
            "perspective": {
                "horizontal": 0.0,
                "vertical": 0.0,
                "scale": 1.0,
                "strength": 0.0,
            },
            "distortion": {
                "k1": -0.001,
                "k2": 0.0,
                "strength": 0.08,
            },
            "chromatic": {
                "shift": 0.1,
                "strength": 0.05,
            },
            "edge_softness": {
                "radius": 0.8,
                "blur_sigma": 1.0,
                "strength": 0.1,
            },
        }

    def process(self, image: Image.Image, scene: dict | None = None) -> Image.Image:
        p = self.params

        result = perspective_transform(image, **p["perspective"])
        result = lens_distortion(result, **p["distortion"])
        result = chromatic_aberration(result, **p["chromatic"])
        result = edge_softness(result, **p["edge_softness"])

        return result
