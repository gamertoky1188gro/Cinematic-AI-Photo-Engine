import numpy as np
from PIL import Image


class SceneAnalyzer:
    """Analyzes input image to extract scene metadata for downstream stages."""

    def analyze(self, image: Image.Image) -> dict:
        arr = np.array(image, dtype=np.float32) / 255.0

        brightness = float(np.mean(arr))
        contrast = float(np.std(arr))

        r_mean = float(np.mean(arr[:, :, 0]))
        g_mean = float(np.mean(arr[:, :, 1]))
        b_mean = float(np.mean(arr[:, :, 2]))

        if r_mean > b_mean:
            warmth = "warm"
        elif b_mean > r_mean:
            warmth = "cool"
        else:
            warmth = "neutral"

        shadows = float(np.mean(arr[arr < 0.3])) if np.any(arr < 0.3) else 0.0
        highlights = float(np.mean(arr[arr > 0.7])) if np.any(arr > 0.7) else 1.0

        h, w = arr.shape[:2]
        center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        edge_mean = brightness
        center_mean = float(np.mean(center))
        vignette_hint = center_mean - edge_mean

        return {
            "brightness": brightness,
            "contrast": contrast,
            "rgb_means": (r_mean, g_mean, b_mean),
            "warmth": warmth,
            "shadows_mean": shadows,
            "highlights_mean": highlights,
            "vignette_hint": vignette_hint,
            "width": w,
            "height": h,
        }
