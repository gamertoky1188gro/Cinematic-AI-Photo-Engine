from .render_plan import RenderPlan


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


class IntelligenceController:
    """Derives a RenderPlan from scene analysis — no pixel modification."""

    def generate(self, scene: dict) -> RenderPlan:
        brightness = scene.get("brightness", 0.5)
        contrast = scene.get("contrast", 0.2)
        warmth = scene.get("warmth", "neutral")
        shadows = scene.get("shadows_mean", 0.15)
        highlights = scene.get("highlights_mean", 0.85)

        plan = RenderPlan()

        self._adapt_film(plan, brightness, contrast, warmth, shadows, highlights)
        self._adapt_optical(plan, brightness, contrast, shadows, highlights)
        self._adapt_geometry(plan, brightness, contrast)

        return plan

    def _adapt_film(self, plan, brightness, contrast, warmth, shadows, highlights):
        if brightness < 0.25:
            plan.film_strength = _lerp(0.5, 0.8, (brightness - 0.05) / 0.2)
            plan.highlight_rolloff = _lerp(0.6, 0.9, contrast)
            plan.shadow_tint = _lerp(0.4, 0.7, shadows / 0.2)
            plan.color_separation = 0.4
        elif brightness > 0.65:
            plan.film_strength = _lerp(0.7, 0.5, (brightness - 0.65) / 0.3)
            plan.highlight_rolloff = _lerp(0.5, 0.8, highlights)
            plan.shadow_tint = 0.2
            plan.color_separation = _lerp(0.3, 0.2, brightness)
        else:
            plan.film_strength = 0.75
            plan.highlight_rolloff = _lerp(0.4, 0.7, contrast)
            plan.shadow_tint = 0.35
            plan.color_separation = _lerp(0.35, 0.45, contrast)

        if warmth == "warm":
            plan.shadow_tint *= 0.6
            plan.color_separation *= 0.7
        elif warmth == "cool":
            plan.shadow_tint = _lerp(plan.shadow_tint, plan.shadow_tint * 1.3, 0.5)

    def _adapt_optical(self, plan, brightness, contrast, shadows, highlights):
        if brightness < 0.25:
            plan.halation = _lerp(0.15, 0.3, contrast)
            plan.bloom = _lerp(0.1, 0.25, contrast)
            plan.grain = _lerp(0.2, 0.35, 1.0 - brightness)
            plan.vignette = _lerp(0.15, 0.25, contrast)
        elif brightness > 0.65:
            plan.halation = _lerp(0.05, 0.15, highlights)
            plan.bloom = _lerp(0.1, 0.2, highlights)
            plan.grain = 0.1
            plan.vignette = _lerp(0.1, 0.2, contrast)
        else:
            plan.halation = _lerp(0.08, 0.2, highlights)
            plan.bloom = _lerp(0.08, 0.18, contrast)
            plan.grain = _lerp(0.1, 0.2, contrast)
            plan.vignette = _lerp(0.1, 0.2, contrast)

        if shadows < 0.1:
            plan.grain *= 1.3
            plan.vignette *= 1.2

    def _adapt_geometry(self, plan, brightness, contrast):
        plan.lens_distortion = _lerp(0.03, 0.08, contrast)
        plan.chromatic_aberration = _lerp(0.02, 0.06, contrast)
        plan.edge_softness = _lerp(0.05, 0.12, 1.0 - contrast)
        plan.perspective = 0.0

    def merge(self, plan: RenderPlan, overrides: dict) -> RenderPlan:
        """Apply user CLI overrides on top of AI-generated plan."""
        for key, value in overrides.items():
            if value is not None and hasattr(plan, key):
                setattr(plan, key, value)
        return plan
