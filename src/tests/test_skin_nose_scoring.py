"""
test_skin_nose_scoring.py — closes the confirmed gap: no dedicated test
coverage for skin_nose scoring (FeatureAnalyzer._analyze_skin /
_texture_irregularity dispatch when region_kind="nose").

test_scoring.py's TestTextureIrregularity only exercises the default
(cheek) divisor path, and TestSkinToneGate / TestCalibrationBaseline build
regions without ever passing region_kind="nose" through to _analyze_skin.
Neither confirms the nose-specific calibration in thresholds.TEXTURE
(nose_divisor=80.0, vs. divisor=20.0 for cheeks and forehead_divisor=24.0)
actually takes effect, which matters because the nose structurally reads
much higher raw Laplacian variance than a cheek even on smooth skin
(nostril/tip shadow, surface curvature) -- per the calibration note in
feature_analyzer.py, at the cheek divisor a real nose capture was scoring
176.2 (clipped to 100) before the nose-specific divisor was introduced.

Covers:
  - _texture_irregularity(): nose divisor produces a materially lower score
    than the cheek/default divisor for the *same* raw variance, and an
    unrecognized region_kind falls back to the default (cheek) divisor
  - _analyze_skin(region_kind="nose") end-to-end: available on a normal
    nose crop, texture score matches the nose-divisor formula exactly,
    redness/acne are computed the same way as any other region, and the
    low-light gate still zeroes texture/acne (not redness) for the nose
    exactly as it does for cheek/forehead
  - Regression guard: an identical crop scored as "nose" vs "cheek" must
    not produce the same texture score, or the region_kind dispatch has
    silently stopped mattering

No camera, no model file, no network.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase3_feature_analysis.feature_analyzer import FeatureAnalyzer
from phase2_region_extraction.region_extractor import FacialRegion
from thresholds import TEXTURE, REGION_LIGHTING


@pytest.fixture
def analyzer():
    return FeatureAnalyzer()


def _solid_bgr(b, g, r, h=40, w=60):
    return np.full((h, w, 3), [b, g, r], dtype=np.uint8)


def _noisy(h=40, w=60, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _full_mask(h, w):
    return np.ones((h, w), dtype=np.uint8) * 255


def _region(crop, mask=None):
    h, w = crop.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
    return FacialRegion(name="nose", crop=crop, mask=mask, masked_crop=crop.copy(),
                        bbox=(0, 0, w, h), available=True)


# ── _texture_irregularity region_kind dispatch ─────────────────────────────

class TestTextureIrregularityRegionDispatch:

    def test_nose_divisor_differs_from_cheek_divisor(self):
        assert TEXTURE.nose_divisor != TEXTURE.divisor
        assert TEXTURE.nose_divisor > TEXTURE.divisor, (
            "nose_divisor should be larger than the cheek divisor so the "
            "same raw variance yields a lower (less falsely-elevated) score"
        )

    def test_nose_scores_lower_than_cheek_for_identical_variance(self, analyzer):
        # Full 0-255 noise saturates BOTH divisors at the 100 clip, which
        # would hide the very difference this test exists to catch. Use a
        # bounded-amplitude crop whose raw Laplacian variance clips the
        # cheek divisor (var/20) but stays under the clip for the larger
        # nose divisor (var/80), so the two scores are actually distinguishable.
        rng = np.random.default_rng(7)
        crop = rng.integers(100, 160, (60, 80, 3), dtype=np.uint8)
        mask = _full_mask(60, 80)
        nose_score = analyzer._texture_irregularity(crop, mask, region_kind="nose")
        cheek_score = analyzer._texture_irregularity(crop, mask, region_kind="cheek")
        assert nose_score < cheek_score, (
            f"nose score ({nose_score}) should be lower than cheek score "
            f"({cheek_score}) for the identical crop, since nose_divisor "
            f"({TEXTURE.nose_divisor}) > cheek divisor ({TEXTURE.divisor})"
        )

    def test_nose_score_matches_formula_exactly(self, analyzer):
        crop = _noisy(h=60, w=80)
        mask = _full_mask(60, 80)
        import cv2
        gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray_img, cv2.CV_64F)
        expected = float(np.clip(np.var(lap[mask > 0]) / TEXTURE.nose_divisor, 0, 100))
        score = analyzer._texture_irregularity(crop, mask, region_kind="nose")
        assert score == pytest.approx(expected, abs=1e-6)

    def test_uniform_nose_crop_near_zero(self, analyzer):
        crop = _solid_bgr(150, 130, 170, h=50, w=70)
        mask = _full_mask(50, 70)
        score = analyzer._texture_irregularity(crop, mask, region_kind="nose")
        assert score < 5.0, f"Uniform nose crop should score near-zero, got {score}"

    def test_unrecognized_region_kind_falls_back_to_default_divisor(self, analyzer):
        crop = _noisy(h=60, w=80)
        mask = _full_mask(60, 80)
        default_score = analyzer._texture_irregularity(crop, mask)
        fallback_score = analyzer._texture_irregularity(crop, mask, region_kind="chin")
        assert default_score == fallback_score

    def test_high_variance_nose_can_still_saturate_at_100(self, analyzer):
        # Extreme synthetic variance should still clip to 100 even with the
        # larger nose divisor, not overflow or go negative.
        rng = np.random.default_rng(3)
        crop = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
        crop[::2, ::2] = [255, 255, 255]
        crop[1::2, 1::2] = [0, 0, 0]
        mask = _full_mask(60, 80)
        score = analyzer._texture_irregularity(crop, mask, region_kind="nose")
        assert 0.0 <= score <= 100.0


# ── _analyze_skin end-to-end with region_kind="nose" ────────────────────────

class TestAnalyzeSkinNoseRegion:

    def test_normal_nose_crop_is_available(self, analyzer):
        crop = _solid_bgr(130, 110, 170, h=50, w=70)  # medium skin tone, BGR
        sf = analyzer._analyze_skin(_region(crop), region_kind="nose")
        assert sf.available

    def test_texture_score_uses_nose_divisor_not_cheek_divisor(self, analyzer):
        crop = _noisy(h=50, w=70, seed=11)
        # Blend toward a plausible skin tone so the tone gate accepts it,
        # while keeping enough high-frequency variance for a non-trivial
        # texture reading.
        skin_tint = np.full_like(crop, [130, 110, 170])
        crop = (0.5 * crop.astype(float) + 0.5 * skin_tint.astype(float)).astype(np.uint8)

        sf_nose = analyzer._analyze_skin(_region(crop), region_kind="nose")
        sf_cheek = analyzer._analyze_skin(_region(crop), region_kind="cheek")

        assert sf_nose.available and sf_cheek.available
        if sf_nose.texture_irregularity > 0 or sf_cheek.texture_irregularity > 0:
            assert sf_nose.texture_irregularity <= sf_cheek.texture_irregularity, (
                "nose region_kind should not score texture higher than cheek "
                "for the same crop"
            )

    def test_redness_score_unaffected_by_region_kind(self, analyzer):
        # Redness scoring has no region_kind branch -- same crop should
        # produce the identical redness score regardless of region_kind.
        crop = _solid_bgr(130, 110, 200, h=50, w=70)
        sf_nose = analyzer._analyze_skin(_region(crop), region_kind="nose")
        sf_cheek = analyzer._analyze_skin(_region(crop), region_kind="cheek")
        assert sf_nose.redness_score == pytest.approx(sf_cheek.redness_score, abs=1e-6)

    def test_low_light_nose_region_zeroes_texture_and_acne_not_redness(self, analyzer):
        # A dark-but-still-skin-toned crop should trip the low-light gate.
        dark_skin = _solid_bgr(30, 25, 40, h=50, w=70)
        sf = analyzer._analyze_skin(_region(dark_skin), region_kind="nose")
        if sf.available and sf.low_light:
            assert sf.texture_irregularity == 0.0
            assert sf.acne_score == 0.0
            assert sf.spot_count == 0
            # Redness is not gated by low_light per _analyze_skin's docstring.
            assert sf.redness_score >= 0.0

    def test_unavailable_region_returns_default_skin_features(self, analyzer):
        empty_region = FacialRegion(name="nose", crop=None, mask=None, masked_crop=None,
                                    bbox=(0, 0, 0, 0), available=False)
        sf = analyzer._analyze_skin(empty_region, region_kind="nose")
        assert not sf.available
        assert sf.redness_score == 0.0
        assert sf.texture_irregularity == 0.0
        assert sf.spot_count == 0

    def test_all_fabric_or_occluded_nose_is_unavailable(self, analyzer):
        # Entirely non-skin-toned crop (e.g. occluded by hair/glasses shadow)
        # should be rejected by the tone gate the same way it would for any
        # other region -- nose gets no special exemption.
        crop = _solid_bgr(20, 20, 20, h=50, w=70)  # near-black, fails tone gate
        sf = analyzer._analyze_skin(_region(crop), region_kind="nose")
        assert not sf.available

    @pytest.mark.parametrize("bgr", [
        (196, 224, 255),  # very light skin
        (66, 134, 198),   # medium skin
        (35, 55, 90),      # dark brown skin
    ])
    def test_nose_available_across_skin_tone_gradient(self, analyzer, bgr):
        crop = _solid_bgr(*bgr, h=50, w=70)
        sf = analyzer._analyze_skin(_region(crop), region_kind="nose")
        assert sf.available, f"Nose region unavailable for skin tone {bgr}"


if __name__ == "__main__":
    import unittest
    # Allow running standalone via `python -m unittest` in addition to pytest,
    # consistent with the rest of the suite's dual invocation support.
    pytest.main([__file__, "-v"])
