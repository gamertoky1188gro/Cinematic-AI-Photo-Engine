import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class QualityScores:
    naturalness: float
    cinematicity: float
    color_quality: float
    highlight_quality: float
    shadow_quality: float
    detail_preservation: float
    optical_realism: float
    overall: float


def _luma(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def _saturation(arr: np.ndarray) -> np.ndarray:
    return np.max(arr, axis=2).astype(float) - np.min(arr, axis=2).astype(float)


def evaluate_quality(original: Image.Image, processed: Image.Image) -> QualityScores:
    """Score perceptual quality of cinematic transformation."""
    orig = np.array(original, dtype=np.float64) / 255.0
    proc = np.array(processed, dtype=np.float64) / 255.0

    orig_luma = _luma(orig)
    proc_luma = _luma(proc)

    orig_sat = _saturation(orig)
    proc_sat = _saturation(proc)

    orig_contrast = float(np.std(orig_luma))
    proc_contrast = float(np.std(proc_luma))

    orig_clip = float(np.mean((orig_luma < 0.02) | (orig_luma > 0.98)))
    proc_clip = float(np.mean((proc_luma < 0.02) | (proc_luma > 0.98)))

    naturalness = _score_naturalness(proc, proc_luma, proc_sat, proc_contrast)
    cinematicity = _score_cinematicity(proc, proc_luma, proc_sat, proc_contrast)
    color_quality = _score_color_quality(orig, proc, orig_sat, proc_sat)
    highlight_quality = _score_highlights(orig_luma, proc_luma, orig_clip, proc_clip)
    shadow_quality = _score_shadows(orig_luma, proc_luma)
    detail_preservation = _score_detail(orig, proc)
    optical_realism = _score_optical_realism(proc, proc_luma)

    overall = (
        0.20 * naturalness +
        0.20 * cinematicity +
        0.15 * color_quality +
        0.15 * highlight_quality +
        0.10 * shadow_quality +
        0.10 * detail_preservation +
        0.10 * optical_realism
    )

    return QualityScores(
        naturalness=round(naturalness, 3),
        cinematicity=round(cinematicity, 3),
        color_quality=round(color_quality, 3),
        highlight_quality=round(highlight_quality, 3),
        shadow_quality=round(shadow_quality, 3),
        detail_preservation=round(detail_preservation, 3),
        optical_realism=round(optical_realism, 3),
        overall=round(overall, 3),
    )


def _score_naturalness(arr, luma, sat, contrast) -> float:
    score = 1.0

    mean_luma = float(np.mean(luma))
    if mean_luma < 0.1 or mean_luma > 0.9:
        score -= 0.2
    elif mean_luma < 0.2 or mean_luma > 0.8:
        score -= 0.1

    if contrast < 0.05 or contrast > 0.6:
        score -= 0.15

    mean_sat = float(np.mean(sat))
    if mean_sat > 0.5:
        score -= 0.2
    elif mean_sat > 0.4:
        score -= 0.1

    extreme_pixels = float(np.mean((luma < 0.01) | (luma > 0.99)))
    if extreme_pixels > 0.05:
        score -= 0.15

    return max(0.0, min(1.0, score))


def _score_cinematicity(arr, luma, sat, contrast) -> float:
    score = 0.5

    if 0.15 <= contrast <= 0.45:
        score += 0.15
    elif contrast < 0.1 or contrast > 0.55:
        score -= 0.1

    mean_sat = float(np.mean(sat))
    if 0.1 <= mean_sat <= 0.35:
        score += 0.15
    elif mean_sat > 0.45:
        score -= 0.1

    mean_luma = float(np.mean(luma))
    if 0.2 <= mean_luma <= 0.6:
        score += 0.1
    elif mean_luma < 0.15:
        score += 0.05

    shadows = float(np.mean(luma < 0.25))
    highlights = float(np.mean(luma > 0.75))
    if 0.1 <= shadows <= 0.5 and 0.05 <= highlights <= 0.4:
        score += 0.1

    return max(0.0, min(1.0, score))


def _score_color_quality(orig, proc, orig_sat, proc_sat) -> float:
    score = 0.8

    sat_change = float(np.mean(proc_sat) - np.mean(orig_sat))
    if abs(sat_change) > 0.3:
        score -= 0.3
    elif abs(sat_change) > 0.2:
        score -= 0.15

    proc_r = float(np.mean(proc[:, :, 0]))
    proc_g = float(np.mean(proc[:, :, 1]))
    proc_b = float(np.mean(proc[:, :, 2]))

    color_cast = max(proc_r, proc_g, proc_b) - min(proc_r, proc_g, proc_b)
    if color_cast > 0.15:
        score -= 0.2

    return max(0.0, min(1.0, score))


def _score_highlights(orig_luma, proc_luma, orig_clip, proc_clip) -> float:
    score = 0.8

    if proc_clip > orig_clip + 0.02:
        score -= 0.3
    elif proc_clip > orig_clip + 0.01:
        score -= 0.15

    orig_bright = float(np.mean(orig_luma > 0.9))
    proc_bright = float(np.mean(proc_luma > 0.9))
    if proc_bright > orig_bright * 1.5:
        score -= 0.2

    return max(0.0, min(1.0, score))


def _score_shadows(orig_luma, proc_luma) -> float:
    score = 0.8

    orig_dark = float(np.mean(orig_luma < 0.1))
    proc_dark = float(np.mean(proc_luma < 0.1))
    if proc_dark > orig_dark * 1.5:
        score -= 0.25

    orig_shadow_detail = float(np.std(orig_luma[orig_luma < 0.3]))
    proc_shadow_detail = float(np.std(proc_luma[proc_luma < 0.3]))
    if proc_shadow_detail < orig_shadow_detail * 0.3:
        score -= 0.2

    return max(0.0, min(1.0, score))


def _score_detail(orig, proc) -> float:
    orig_edges = _edge_energy(orig)
    proc_edges = _edge_energy(proc)

    if orig_edges > 0:
        ratio = proc_edges / orig_edges
        if 0.6 <= ratio <= 1.2:
            return 0.9
        elif 0.4 <= ratio <= 1.5:
            return 0.7
        else:
            return 0.4
    return 0.7


def _edge_energy(arr) -> float:
    luma = _luma(arr)
    gy, gx = np.gradient(luma)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def _score_optical_realism(arr, luma) -> float:
    score = 0.85

    extreme_bright = float(np.mean(luma > 0.95))
    if extreme_bright > 0.1:
        score -= 0.2

    mean_luma = float(np.mean(luma))
    if mean_luma < 0.1:
        score -= 0.15

    return max(0.0, min(1.0, score))
