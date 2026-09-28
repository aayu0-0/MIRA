"""
A6 — test_scoring.py

Unit tests for Phase 3 (FeatureAnalyzer) scoring functions.
All inputs are synthetic numpy arrays — no camera, no model file.

Strategy: call the private scoring methods directly (bypass analyze()),
passing carefully constructed crops whose expected scores are
analytically computable from the thresholds constants.

Covered:
  - EAR normalization (closed → 0, open → 100, clamping)
  - Eye redness (red-heavy crop → high score, green-heavy → near 0)
  - Dark circles (dark lower-third → high score, bright → near 0)
  - Puffiness (flat/uniform crop → high, high-variance → low)
  - Skin redness (rel-R above baseline → positive score)
  - Texture irregularity (uniform → 0, noisy → high)
  - Acne spot count (0 blobs → 0 score, synthetic spots → non-zero)
  - Lip dryness (uniform → near 0, high-frequency noise → high)
  - Lip color consistency (uniform → 100, mixed → lower)
  - Lip pallor (desaturated → high, saturated → low)
  - Swelling score (ratio within typical → 0, above → positive)
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase3_feature_analysis.feature_analyzer import FeatureAnalyzer
from phase2_region_extraction.region_extractor import FacialRegion
from tests.fixtures import build_synthetic_landmarks, FACE_CX, FACE_CY, FACE_RX, FACE_RY
from thresholds import EAR, DARK_CIRCLE, PUFFINESS, SKIN_REDNESS, TEXTURE, SWELLING, SKIN_TONE_GATE, LIP_TONE_GATE, MOUTH_SHAPE


# ── test fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    return FeatureAnalyzer()


def _solid_bgr(b, g, r, h=40, w=60):
    """Solid-colour crop in BGR."""
    return np.full((h, w, 3), [b, g, r], dtype=np.uint8)


def _noisy(h=40, w=60, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _full_mask(h, w):
    return np.ones((h, w), dtype=np.uint8) * 255


# ── EAR ───────────────────────────────────────────────────────────────────────

class TestEAR:

    def _ear_from_value(self, analyzer, ear_value, side="left"):
        """
        Construct landmarks where the EAR formula yields exactly `ear_value`,
        then call _compute_ear.
        EAR = (v1 + v2) / (2h) so set h=100, v1=v2=ear_value*100.
        """
        from phase1_face_understanding.face_detector import LandmarkPoint
        from phase3_feature_analysis.feature_analyzer import LEFT_EYE_EAR_POINTS, RIGHT_EYE_EAR_POINTS

        pts_idx = LEFT_EYE_EAR_POINTS if side == "left" else RIGHT_EYE_EAR_POINTS
        # Enough landmark slots
        lm = [LandmarkPoint(x=0.5, y=0.5, px=300, py=300) for _ in range(max(pts_idx) + 1)]
        # p0 and p3: horizontal distance = 100 (h)
        lm[pts_idx[0]] = LandmarkPoint(x=0, y=0, px=0,   py=0)
        lm[pts_idx[3]] = LandmarkPoint(x=0, y=0, px=100, py=0)
        # p1 and p5: vertical distance = ear_value * 100 (v1)
        # Use round() not int() to avoid truncation error at the midpoint
        half_v = round(ear_value * 50)
        lm[pts_idx[1]] = LandmarkPoint(x=0, y=0, px=50, py=-half_v)
        lm[pts_idx[5]] = LandmarkPoint(x=0, y=0, px=50, py=+half_v)
        # p2 and p4: same (v2 = v1)
        lm[pts_idx[2]] = LandmarkPoint(x=0, y=0, px=50, py=-half_v)
        lm[pts_idx[4]] = LandmarkPoint(x=0, y=0, px=50, py=+half_v)
        return analyzer._compute_ear(lm, side)

    def test_closed_eye_score_zero(self, analyzer):
        score = self._ear_from_value(analyzer, EAR.closed)
        # Pixel coordinates are integer so exact EAR.closed may not be representable;
        # allow a small tolerance from rounding.
        assert score == pytest.approx(0.0, abs=5.0)

    def test_open_eye_score_100(self, analyzer):
        score = self._ear_from_value(analyzer, EAR.open)
        assert score == pytest.approx(100.0, abs=1.0)

    def test_below_closed_clamped_to_zero(self, analyzer):
        score = self._ear_from_value(analyzer, EAR.closed - 0.05)
        assert score == 0.0

    def test_above_open_clamped_to_100(self, analyzer):
        score = self._ear_from_value(analyzer, EAR.open + 0.05)
        assert score == 100.0

    def test_midpoint_is_50(self, analyzer):
        mid = (EAR.closed + EAR.open) / 2
        score = self._ear_from_value(analyzer, mid)
        # Tolerance accounts for pixel-coordinate rounding in the fixture
        assert score == pytest.approx(50.0, abs=5.0)


# ── Eye redness ───────────────────────────────────────────────────────────────

class TestEyeRedness:
    """
    C1/C2 — eye redness is now calibrated against the per-person forehead
    baseline (CalibrationBaseline), using the same relative-R formula as
    _skin_redness.  The old mean(R-G) formula produced large false-positive
    scores on neutral eye crops across every skin tone because skin naturally
    has R > G by varying amounts depending on tone.
    """

    def _baseline(self, analyzer, forehead_bgr):
        """Build a CalibrationBaseline from a solid-colour forehead crop."""
        from phase2_region_extraction.region_extractor import FacialRegion
        crop = np.full((60, 80, 3), forehead_bgr, dtype=np.uint8)
        h, w = crop.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        region = FacialRegion(name="forehead", crop=crop, mask=mask,
                              masked_crop=crop.copy(), bbox=(0, 0, w, h), available=True)
        return analyzer._compute_calibration_baseline(region)

    @pytest.mark.parametrize("bgr", [
        (196, 224, 255),  # very light skin
        (125, 194, 241),  # light skin
        (66,  134, 198),  # medium skin
        (36,   85, 141),  # tan skin
        (35,   55,  90),  # dark brown skin
    ])
    def test_neutral_eye_same_tone_as_forehead_scores_zero(self, analyzer, bgr):
        """
        C1 regression guard — the core bug this fix addresses.
        A neutral eye crop (same colour as the person's own forehead) must
        score ~0 redness regardless of skin tone.  With the old mean(R-G)
        formula, medium skin scored ~75 and dark skin scored ~41 here, both
        pure false-positive signal from skin-tone rel-R, not inflammation.
        """
        baseline = self._baseline(analyzer, bgr)
        crop = np.full((40, 60, 3), bgr, dtype=np.uint8)
        score = analyzer._eye_redness(crop, baseline)
        assert score == pytest.approx(0.0, abs=1.0), (
            f"Neutral eye crop at skin tone {bgr} should score ~0 "
            f"once calibrated, got {score}")

    def test_genuinely_red_eye_scores_high_across_tones(self, analyzer):
        """
        A clearly red-shifted eye crop (high rel-R vs any forehead baseline)
        must still register as redness after calibration.
        """
        red_crop = np.full((40, 60, 3), (50, 80, 220), dtype=np.uint8)  # vivid red, BGR
        for bgr in [(196, 224, 255), (66, 134, 198), (35, 55, 90)]:
            baseline = self._baseline(analyzer, bgr)
            score = analyzer._eye_redness(red_crop, baseline)
            assert score > 60.0, (
                f"Red eye crop should score high for skin tone {bgr}, got {score}")

    def test_no_baseline_falls_back_to_fixed_constant(self, analyzer):
        """Omitting baseline or passing None must not raise and must match each other."""
        crop = _solid_bgr(50, 80, 220)
        score_implicit = analyzer._eye_redness(crop)
        score_explicit_none = analyzer._eye_redness(crop, None)
        assert score_implicit == score_explicit_none

    def test_unavailable_baseline_falls_back(self, analyzer):
        from phase3_feature_analysis.feature_analyzer import CalibrationBaseline
        crop = _solid_bgr(50, 80, 220)
        unavailable = CalibrationBaseline(available=False)
        score_unavail = analyzer._eye_redness(crop, unavailable)
        score_none = analyzer._eye_redness(crop, None)
        assert score_unavail == score_none

    def test_score_in_range(self, analyzer):
        """Score must always be in [0, 100]."""
        for crop in [_solid_bgr(0, 0, 255), _noisy(), _solid_bgr(128, 128, 128)]:
            score = analyzer._eye_redness(crop)
            assert 0.0 <= score <= 100.0

    def test_redness_monotonic_with_red_shift(self, analyzer):
        """More red-shifted crops must produce higher or equal scores."""
        baseline_bgr = (66, 134, 198)
        baseline = self._baseline(analyzer, baseline_bgr)
        scores = []
        for extra_r in (0, 20, 50, 80):
            b, g, r = baseline_bgr
            shifted = (b, g, min(255, r + extra_r))
            crop = np.full((40, 60, 3), shifted, dtype=np.uint8)
            scores.append(analyzer._eye_redness(crop, baseline))
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Not monotonic: {scores}"


# ── Dark circles ──────────────────────────────────────────────────────────────

class TestDarkCircles:

    def test_very_dark_lower_third_high_score(self, analyzer):
        crop = _solid_bgr(5, 5, 5, h=30, w=60)   # very dark = low luminance
        score = analyzer._dark_circle_score(crop)
        assert score > 70.0, f"Dark crop should give high dark-circle score, got {score}"

    def test_very_bright_lower_third_near_zero(self, analyzer):
        crop = _solid_bgr(240, 240, 240, h=30, w=60)
        score = analyzer._dark_circle_score(crop)
        assert score < 15.0, f"Bright crop should give near-zero score, got {score}"

    def test_too_small_crop_returns_zero(self, analyzer):
        crop = np.zeros((4, 60, 3), dtype=np.uint8)
        score = analyzer._dark_circle_score(crop)
        assert score == 0.0

    def test_score_in_range(self, analyzer):
        for crop in [_solid_bgr(0, 0, 0), _solid_bgr(255, 255, 255), _noisy()]:
            assert 0.0 <= analyzer._dark_circle_score(crop) <= 100.0


# ── Puffiness ─────────────────────────────────────────────────────────────────

class TestPuffiness:

    def test_uniform_upper_half_high_score(self, analyzer):
        """Flat/uniform texture (variance → 0) should give max puffiness signal."""
        crop = _solid_bgr(180, 150, 140, h=40, w=60)
        score = analyzer._puffiness_score(crop)
        assert score > 80.0, f"Uniform crop should give high puffiness score, got {score}"

    def test_noisy_upper_half_low_score(self, analyzer):
        """High-variance texture means no puffiness."""
        rng = np.random.default_rng(0)
        crop = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
        score = analyzer._puffiness_score(crop)
        assert score < 30.0, f"Noisy crop should give low puffiness score, got {score}"

    def test_score_in_range(self, analyzer):
        for crop in [_solid_bgr(128, 128, 128), _noisy()]:
            assert 0.0 <= analyzer._puffiness_score(crop) <= 100.0


# ── Skin redness ──────────────────────────────────────────────────────────────

class TestSkinRedness:

    def test_high_rel_red_gives_positive_score(self, analyzer):
        """Pixels where R dominates should produce a positive redness score."""
        pixels = np.array([[0, 50, 200]], dtype=float)  # BGR: B=0, G=50, R=200
        score = analyzer._skin_redness(pixels)
        assert score > 0.0, f"Red-dominant pixels should score > 0, got {score}"

    def test_equal_channels_near_zero(self, analyzer):
        """Equal R/G/B → relative R = 1/3 ≈ baseline → score near 0."""
        pixels = np.array([[100, 100, 100]], dtype=float)
        score = analyzer._skin_redness(pixels)
        # rel_r = 100/300 = 0.333 ≈ SKIN_REDNESS.baseline_rel_r → score ≈ 0
        assert abs(score) < 5.0, f"Equal channels should be near 0, got {score}"

    def test_score_clamped(self, analyzer):
        pixels = np.array([[0, 0, 255]], dtype=float)
        score = analyzer._skin_redness(pixels)
        assert 0.0 <= score <= 100.0


# ── Texture irregularity ──────────────────────────────────────────────────────

class TestTextureIrregularity:

    def test_uniform_crop_near_zero(self, analyzer):
        crop = _solid_bgr(150, 130, 120, h=50, w=80)
        mask = _full_mask(50, 80)
        score = analyzer._texture_irregularity(crop, mask)
        assert score < 5.0, f"Uniform crop should have near-zero texture score, got {score}"

    def test_noisy_crop_higher_score(self, analyzer):
        crop = _noisy(h=50, w=80)
        mask = _full_mask(50, 80)
        score = analyzer._texture_irregularity(crop, mask)
        assert score > 5.0, f"Noisy crop should have higher texture score, got {score}"

    def test_empty_mask_returns_zero(self, analyzer):
        crop = _noisy(h=50, w=80)
        mask = np.zeros((50, 80), dtype=np.uint8)
        score = analyzer._texture_irregularity(crop, mask)
        assert score == 0.0


# ── Acne / spot detection ────────────────────────────────────────────────────

class TestAcneScore:

    def test_uniform_crop_no_spots(self, analyzer):
        crop = _solid_bgr(180, 160, 150, h=60, w=80)
        mask = _full_mask(60, 80)
        score, count = analyzer._acne_score(crop, mask)
        assert score == 0.0
        assert count == 0

    def test_too_small_returns_zero(self, analyzer):
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = _full_mask(8, 8)
        score, count = analyzer._acne_score(crop, mask)
        assert score == 0.0
        assert count == 0

    def test_synthetic_dark_spot_detected(self, analyzer):
        """
        Plant a small dark patch on a bright background.
        The acne detector uses local contrast (Gaussian blur - image),
        so a dark disk on a light crop should register as at least one spot.
        """
        crop = _solid_bgr(200, 180, 170, h=80, w=100)
        # 8×8 dark spot in the centre
        crop[35:43, 45:53] = [40, 30, 25]
        mask = _full_mask(80, 100)
        score, count = analyzer._acne_score(crop, mask)
        assert count >= 1, f"Expected ≥1 spot detected, got {count}"
        assert score > 0.0

    def test_score_proportional_to_count(self, analyzer):
        """More spots → higher score (up to the cap)."""
        def plant_spots(n):
            crop = _solid_bgr(200, 180, 170, h=120, w=160)
            for k in range(n):
                r, c = 20 + k * 15, 20 + k * 15
                if r + 8 < 120 and c + 8 < 160:
                    crop[r:r+8, c:c+8] = [30, 20, 20]
            mask = _full_mask(120, 160)
            score, count = analyzer._acne_score(crop, mask)
            return score, count

        score_1, cnt_1 = plant_spots(1)
        score_3, cnt_3 = plant_spots(3)
        if cnt_1 > 0 and cnt_3 > cnt_1:
            assert score_3 >= score_1


# ── Lip dryness ───────────────────────────────────────────────────────────────

class TestLipDryness:

    def test_smooth_lip_near_zero(self, analyzer):
        crop = _solid_bgr(80, 60, 150, h=30, w=60)  # solid pink/red
        mask = _full_mask(30, 60)
        score = analyzer._lip_dryness(crop, mask)
        assert score < 5.0, f"Smooth lip crop should score near 0, got {score}"

    def test_high_frequency_lip_higher_score(self, analyzer):
        crop = _noisy(h=30, w=60)
        mask = _full_mask(30, 60)
        score = analyzer._lip_dryness(crop, mask)
        assert score > 5.0, f"Noisy lip crop should have higher dryness score, got {score}"

    def test_empty_mask_returns_zero(self, analyzer):
        crop = _noisy(h=30, w=60)
        mask = np.zeros((30, 60), dtype=np.uint8)
        score = analyzer._lip_dryness(crop, mask)
        assert score == 0.0


# ── Lip color consistency ────────────────────────────────────────────────────

class TestLipColorConsistency:

    def test_uniform_pixels_score_100(self, analyzer):
        pixels = np.full((100, 3), [80, 60, 150], dtype=float)
        score = analyzer._lip_color_consistency(pixels)
        assert score == pytest.approx(100.0, abs=1.0)

    def test_varied_pixels_lower_score(self, analyzer):
        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, (100, 3)).astype(float)
        score = analyzer._lip_color_consistency(pixels)
        assert score < 80.0, f"Random pixels should give lower consistency, got {score}"

    def test_score_in_range(self, analyzer):
        for pixels in [
            np.full((50, 3), 128, dtype=float),
            np.random.default_rng(0).integers(0, 256, (50, 3)).astype(float),
        ]:
            score = analyzer._lip_color_consistency(pixels)
            assert 0.0 <= score <= 100.0


# ── Lip pallor ───────────────────────────────────────────────────────────────

class TestLipPallor:

    def test_desaturated_grey_high_pallor(self, analyzer):
        """Grey pixels (saturation≈0) → pallor score near 100."""
        pixels = np.full((50, 3), 150, dtype=float)  # equal B=G=R → grey
        score = analyzer._lip_pallor(pixels)
        assert score > 80.0, f"Grey pixels should give high pallor, got {score}"

    def test_vivid_red_low_pallor(self, analyzer):
        """Vivid red lips (high saturation in HSV) → pallor score near 0."""
        pixels = np.array([[0, 0, 200]] * 50, dtype=float)  # pure red in BGR
        score = analyzer._lip_pallor(pixels)
        assert score < 30.0, f"Vivid red should give low pallor, got {score}"

    def test_score_in_range(self, analyzer):
        for pixels in [np.full((30, 3), 100, dtype=float),
                       np.array([[0, 0, 200]] * 30, dtype=float)]:
            score = analyzer._lip_pallor(pixels)
            assert 0.0 <= score <= 100.0


# ── Swelling score ────────────────────────────────────────────────────────────

class TestSwellingScore:

    def _swelling(self, analyzer, cheek_w, eye_span):
        """
        Build minimal landmarks where cheek and eye points give the desired widths.
        Centre the geometry around FACE_CX, FACE_CY.
        """
        from phase1_face_understanding.face_detector import LandmarkPoint
        from phase1_face_understanding.face_detector import (
            LEFT_CHEEK_PT, RIGHT_CHEEK_PT,
            LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER,
        )

        max_idx = max(LEFT_CHEEK_PT, RIGHT_CHEEK_PT,
                      LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER)
        lm = [LandmarkPoint(x=0.5, y=0.5, px=300, py=300)
              for _ in range(max_idx + 1)]

        half_c = cheek_w // 2
        half_e = eye_span // 2
        cy = FACE_CY

        lm[LEFT_CHEEK_PT]          = LandmarkPoint(x=0, y=0, px=300 + half_c, py=cy)
        lm[RIGHT_CHEEK_PT]         = LandmarkPoint(x=0, y=0, px=300 - half_c, py=cy)
        lm[LEFT_EYE_OUTER_CORNER]  = LandmarkPoint(x=0, y=0, px=300 + half_e, py=cy)
        lm[RIGHT_EYE_OUTER_CORNER] = LandmarkPoint(x=0, y=0, px=300 - half_e, py=cy)

        return analyzer._swelling_score(lm)

    def test_typical_ratio_gives_zero(self, analyzer):
        """cheek/eye ≈ 1.45 (within typical range) → swelling score = 0."""
        score, label, ratio_out = self._swelling(analyzer, cheek_w=290, eye_span=200)
        ratio = 290 / 200  # 1.45
        assert ratio <= SWELLING.typical_high
        assert score == 0.0
        assert round(ratio_out, 2) == round(ratio, 2)
        # With score=0 < SWELLING.none (20), the label should be "No Swelling Indicators"
        assert label == "No Swelling Indicators"

    def test_high_ratio_gives_positive_score(self, analyzer):
        """cheek/eye >> typical_high → positive swelling score."""
        score, label, ratio_out = self._swelling(analyzer, cheek_w=400, eye_span=200)
        ratio = 400 / 200  # 2.0 — well above 1.60
        assert ratio > SWELLING.typical_high
        assert score > 0.0
        assert round(ratio_out, 2) == round(ratio, 2)

    def test_swelling_score_monotonic_with_ratio(self, analyzer):
        """Higher ratio → higher swelling score."""
        eye_span = 200
        scores = []
        for cheek_w in [280, 340, 400]:
            score, _, _ = self._swelling(analyzer, cheek_w=cheek_w, eye_span=eye_span)
            scores.append(score)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Not monotonic: {scores}"

    def test_zero_eye_span_returns_not_assessed(self, analyzer):
        from phase1_face_understanding.face_detector import LandmarkPoint
        from phase1_face_understanding.face_detector import (
            LEFT_CHEEK_PT, RIGHT_CHEEK_PT,
            LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER,
        )
        max_idx = max(LEFT_CHEEK_PT, RIGHT_CHEEK_PT,
                      LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER)
        lm = [LandmarkPoint(x=0.5, y=0.5, px=300, py=300)
              for _ in range(max_idx + 1)]
        # Both eye corners at the same point → eye_span = 0
        lm[LEFT_EYE_OUTER_CORNER]  = LandmarkPoint(x=0, y=0, px=300, py=300)
        lm[RIGHT_EYE_OUTER_CORNER] = LandmarkPoint(x=0, y=0, px=300, py=300)
        score, label, ratio_out = analyzer._swelling_score(lm)
        assert score == 0.0
        assert label == "Not Assessed"
        assert ratio_out == 0.0


# ── Smile / mouth geometry ─────────────────────────────────────────────────────
# _analyze_smile() computes three signals from 4 landmarks (MOUTH_LEFT,
# MOUTH_RIGHT, UPPER_LIP_CENTER, LOWER_LIP_CENTER): mouth-aspect-ratio (MAR),
# corner-asymmetry, and curvature. All three normalize by the true Euclidean
# corner-to-corner distance (math.hypot), not a fixed horizontal width — a
# vertical corner offset subtly lengthens that distance, so expected values
# below are computed the same way the production code computes them.

class TestSmileGeometry:

    def _smile_landmarks(self, l_px, l_py, r_px, r_py, u_px, u_py, lo_px, lo_py):
        from phase1_face_understanding.face_detector import (
            LandmarkPoint, MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP_CENTER, LOWER_LIP_CENTER,
        )
        max_idx = max(MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP_CENTER, LOWER_LIP_CENTER)
        lm = [LandmarkPoint(x=0.5, y=0.5, px=300, py=300) for _ in range(max_idx + 1)]
        lm[MOUTH_LEFT]       = LandmarkPoint(x=0, y=0, px=l_px, py=l_py)
        lm[MOUTH_RIGHT]      = LandmarkPoint(x=0, y=0, px=r_px, py=r_py)
        lm[UPPER_LIP_CENTER] = LandmarkPoint(x=0, y=0, px=u_px, py=u_py)
        lm[LOWER_LIP_CENTER] = LandmarkPoint(x=0, y=0, px=lo_px, py=lo_py)
        return lm

    # -- mouth aspect ratio (MAR) ---------------------------------------------
    # mar_raw = width/height; normalized over [mar_typical_low, mar_typical_high]
    # = [2.0, 6.0] into a 0-100 score, clamped at both ends.

    def test_mar_at_low_clamp_boundary_scores_zero(self, analyzer):
        # width=200, height=100 -> raw ratio 2.0 == mar_typical_low -> 0
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 250, 300, 350)
        sf = analyzer._analyze_smile(lm)
        assert sf.available
        assert sf.mouth_aspect_ratio == pytest.approx(0.0, abs=0.5)

    def test_mar_below_low_clamp_boundary_clamps_to_zero(self, analyzer):
        # width=100, height=100 -> raw ratio 1.0, below typical_low -> clamps to 0
        lm = self._smile_landmarks(350, 300, 250, 300, 300, 250, 300, 350)
        sf = analyzer._analyze_smile(lm)
        assert sf.mouth_aspect_ratio == 0.0

    def test_mar_at_high_clamp_boundary_scores_hundred(self, analyzer):
        # width=300, height=50 -> raw ratio 6.0 == mar_typical_high -> 100
        lm = self._smile_landmarks(450, 300, 150, 300, 300, 275, 300, 325)
        sf = analyzer._analyze_smile(lm)
        assert sf.mouth_aspect_ratio == pytest.approx(100.0, abs=0.5)

    def test_mar_above_high_clamp_boundary_clamps_to_hundred(self, analyzer):
        # width=500, height=50 -> raw ratio 10.0, above typical_high -> clamps to 100
        lm = self._smile_landmarks(550, 300, 50, 300, 300, 275, 300, 325)
        sf = analyzer._analyze_smile(lm)
        assert sf.mouth_aspect_ratio == 100.0

    def test_mar_at_midpoint_scores_fifty(self, analyzer):
        # width=200, height=50 -> raw ratio 4.0, midpoint of [2,6] -> 50
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 275, 300, 325)
        sf = analyzer._analyze_smile(lm)
        assert sf.mouth_aspect_ratio == pytest.approx(50.0, abs=0.5)

    # -- corner asymmetry ------------------------------------------------------
    # asym_ratio = vertical_diff / mouth_width (true hypot distance, not raw
    # horizontal width), normalized by MOUTH_SHAPE.asymmetry_divisor (0.15).

    def test_corner_asymmetry_zero_when_corners_level(self, analyzer):
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 270, 300, 330)
        sf = analyzer._analyze_smile(lm)
        assert sf.corner_asymmetry == 0.0

    def test_corner_asymmetry_uses_euclidean_corner_distance(self, analyzer):
        """Regression pin: mouth_width must be hypot(horiz, vert_offset), not
        the raw horizontal span — a vertical corner offset subtly lengthens
        the corner-to-corner distance used to normalize the asymmetry ratio."""
        horiz, offset = 200, 30
        lm = self._smile_landmarks(400, 300 + offset, 200, 300, 300, 270, 300, 330)
        sf = analyzer._analyze_smile(lm)

        true_width = math.hypot(horiz, offset)
        expected = min(100.0, (offset / true_width) / MOUTH_SHAPE.asymmetry_divisor * 100)
        assert sf.corner_asymmetry == pytest.approx(expected, abs=0.1)
        # Sanity check that using the naive (non-hypot) width would have
        # given a measurably different number -- i.e. the distinction matters.
        naive = min(100.0, (offset / horiz) / MOUTH_SHAPE.asymmetry_divisor * 100)
        assert expected != pytest.approx(naive, abs=0.1)

    def test_corner_asymmetry_scale_invariant(self, analyzer):
        """Doubling every coordinate (same width:offset ratio) must yield the
        same score -- the ratio is scale-invariant by construction."""
        lm_small = self._smile_landmarks(400, 330, 200, 300, 300, 270, 300, 330)
        lm_big   = self._smile_landmarks(600, 360, 200, 300, 300, 270, 300, 330)
        sf_small = analyzer._analyze_smile(lm_small)
        sf_big   = analyzer._analyze_smile(lm_big)
        assert sf_small.corner_asymmetry == pytest.approx(sf_big.corner_asymmetry, abs=0.1)

    def test_corner_asymmetry_monotonic_with_vertical_offset(self, analyzer):
        scores = []
        for offset in (10, 30, 60):
            lm = self._smile_landmarks(400, 300 + offset, 200, 300, 300, 270, 300, 330)
            sf = analyzer._analyze_smile(lm)
            scores.append(sf.corner_asymmetry)
        assert scores[0] < scores[1] < scores[2], f"Not monotonic: {scores}"

    def test_corner_asymmetry_clamps_at_hundred(self, analyzer):
        # Huge vertical offset relative to width -> ratio far exceeds 1.0
        lm = self._smile_landmarks(210, 1300, 200, 300, 300, 270, 300, 330)
        sf = analyzer._analyze_smile(lm)
        assert sf.corner_asymmetry == 100.0

    # -- curvature --------------------------------------------------------------
    # curvature_raw = (midline_y - corner_y) / mouth_width, normalized by
    # MOUTH_SHAPE.curvature_divisor (0.20), clamped to [-100, 100]. Corners
    # kept level (l_py == r_py) so mouth_width reduces cleanly to the
    # horizontal span for these tests.

    def test_curvature_positive_at_clamp_boundary(self, analyzer):
        # width=200, D=40 -> raw 0.2 == curvature_divisor -> +100
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 320, 300, 360)
        sf = analyzer._analyze_smile(lm)
        assert sf.curvature_score == pytest.approx(100.0, abs=0.5)

    def test_curvature_negative_at_clamp_boundary(self, analyzer):
        # width=200, D=-40 -> raw -0.2 -> -100
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 240, 300, 280)
        sf = analyzer._analyze_smile(lm)
        assert sf.curvature_score == pytest.approx(-100.0, abs=0.5)

    def test_curvature_midpoint_positive(self, analyzer):
        # width=200, D=20 -> raw 0.1 -> +50
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 300, 300, 340)
        sf = analyzer._analyze_smile(lm)
        assert sf.curvature_score == pytest.approx(50.0, abs=0.5)

    def test_curvature_midpoint_negative(self, analyzer):
        # width=200, D=-20 -> raw -0.1 -> -50
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 260, 300, 300)
        sf = analyzer._analyze_smile(lm)
        assert sf.curvature_score == pytest.approx(-50.0, abs=0.5)

    def test_curvature_zero_when_corners_at_midline(self, analyzer):
        # midline_y == corner_y -> D=0 -> score 0
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 280, 300, 320)
        sf = analyzer._analyze_smile(lm)
        assert sf.curvature_score == pytest.approx(0.0, abs=0.5)

    # -- guards -------------------------------------------------------------

    def test_zero_width_returns_unavailable(self, analyzer):
        # Both corners coincide -> mouth_width == 0 -> defaults, unavailable
        lm = self._smile_landmarks(300, 300, 300, 300, 300, 270, 300, 330)
        sf = analyzer._analyze_smile(lm)
        assert not sf.available
        assert sf.mouth_aspect_ratio == 0.0
        assert sf.corner_asymmetry == 0.0
        assert sf.curvature_score == 0.0

    def test_zero_height_mar_defaults_to_typical_high(self, analyzer):
        # Upper/lower lip-center coincide -> mouth_height == 0 -> mar_raw
        # falls back to MOUTH_SHAPE.mar_typical_high -> MAR score == 100,
        # while the region is still available since mouth_width > 0.
        lm = self._smile_landmarks(400, 300, 200, 300, 300, 300, 300, 300)
        sf = analyzer._analyze_smile(lm)
        assert sf.available
        assert sf.mouth_aspect_ratio == pytest.approx(100.0, abs=0.5)

    def test_missing_landmark_returns_unavailable(self, analyzer):
        # Landmark list too short to contain the required indices at all.
        from phase1_face_understanding.face_detector import LandmarkPoint
        short_lm = [LandmarkPoint(x=0, y=0, px=0, py=0) for _ in range(5)]
        sf = analyzer._analyze_smile(short_lm)
        assert not sf.available


# ── Skin/lip tone gating ──────────────────────────────────────────────────────
# Regression coverage for the bug found via real-photo testing: a turban edge
# intruding into the cheek-region polygon (oval-landmark boundary doesn't know
# about fabric/hair occlusion) silently contaminated the redness score, since
# the only prior filter was "exclude pure black". These tests exercise
# _analyze_skin/_analyze_lips end-to-end (not just the gate helpers in
# isolation) so a future change that bypasses gating would be caught here.

class TestSkinToneGate:

    def _region(self, crop, mask=None):
        h, w = crop.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        return FacialRegion(name="test", crop=crop, mask=mask, masked_crop=crop.copy(),
                            bbox=(0, 0, w, h), available=True)

    def test_pure_skin_region_is_available(self, analyzer):
        crop = np.full((60, 80, 3), [130, 110, 170], dtype=np.uint8)  # medium skin, BGR
        sf = analyzer._analyze_skin(self._region(crop))
        assert sf.available

    def test_fabric_contaminated_region_matches_pure_skin_score(self, analyzer):
        """
        The motivating bug: half the crop is navy fabric (turban), half is
        real skin. Redness should reflect only the skin half — i.e. match a
        pure-skin crop of the same color — not be diluted/skewed by fabric.
        """
        pure = np.full((60, 80, 3), [130, 110, 170], dtype=np.uint8)
        sf_pure = analyzer._analyze_skin(self._region(pure))

        mixed = pure.copy()
        mixed[:, :40] = [90, 55, 30]   # left half -> navy turban-like fabric
        sf_mixed = analyzer._analyze_skin(self._region(mixed))

        assert sf_mixed.available
        assert sf_mixed.redness_score == pytest.approx(sf_pure.redness_score, abs=1.0), (
            f"Fabric contamination changed redness from {sf_pure.redness_score} "
            f"to {sf_mixed.redness_score} — tone gate did not fully exclude it"
        )

    def test_all_fabric_region_is_unavailable(self, analyzer):
        """If literally nothing in the region passes the skin-tone gate
        (e.g. a region that's entirely occluded by fabric), it should be
        reported unavailable rather than silently scoring the fabric."""
        crop = np.full((60, 80, 3), [90, 55, 30], dtype=np.uint8)  # all navy fabric
        sf = analyzer._analyze_skin(self._region(crop))
        assert not sf.available
        assert sf.redness_score == 0.0

    def test_black_hair_contaminated_region_matches_pure_skin_score(self, analyzer):
        """Same contamination scenario but with hair (beard/eyebrow) instead
        of fabric — different color, same failure mode."""
        pure = np.full((60, 80, 3), [130, 110, 170], dtype=np.uint8)
        sf_pure = analyzer._analyze_skin(self._region(pure))

        mixed = pure.copy()
        mixed[:, :30] = [25, 22, 20]   # left third -> black hair
        sf_mixed = analyzer._analyze_skin(self._region(mixed))

        assert sf_mixed.available
        assert sf_mixed.redness_score == pytest.approx(sf_pure.redness_score, abs=1.0)

    @pytest.mark.parametrize("bgr", [
        (196, 224, 255),  # very light skin
        (125, 194, 241),  # light skin
        (66, 134, 198),   # medium skin
        (36, 85, 141),    # tan skin
        (35, 55, 90),      # dark brown skin
        (18, 25, 40),      # very deep dark skin
    ])
    def test_skin_gradient_all_kept(self, analyzer, bgr):
        """Every tone across the calibration gradient must be available —
        an over-tight gate would silently zero out real people's scores."""
        crop = np.full((40, 60, 3), bgr, dtype=np.uint8)
        sf = analyzer._analyze_skin(self._region(crop))
        assert sf.available, f"Skin tone {bgr} was rejected by the tone gate"


class TestSkinRegionLightingGate:
    """
    A1-followup — per-region lighting/noise reliability check. A real
    underlit capture produced near-100 'noise read as texture' scores from
    Laplacian variance, even though the overall frame passed AlignmentCheck's
    whole-face brightness gate. These tests cover the region-local floor
    that suppresses texture_irregularity/acne_score in that case (see
    thresholds.REGION_LIGHTING).
    """

    def _region(self, crop, mask=None):
        h, w = crop.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        return FacialRegion(name="test", crop=crop, mask=mask, masked_crop=crop.copy(),
                            bbox=(0, 0, w, h), available=True)

    def test_dark_skin_region_flags_low_light_and_suppresses_texture_and_acne(self, analyzer):
        # Same "very deep dark skin" BGR kept by the gradient test above —
        # confirmed to pass the tone gate (available) but with low gray
        # luminance (~29), well under REGION_LIGHTING.min_mean_luminance.
        crop = np.full((40, 60, 3), (18, 25, 40), dtype=np.uint8)
        sf = analyzer._analyze_skin(self._region(crop))

        assert sf.available, "Region should stay available — only texture/acne are suppressed"
        assert sf.low_light is True
        assert sf.texture_irregularity == 0.0
        assert sf.acne_score == 0.0
        assert sf.spot_count == 0
        # Redness is not gated by lighting — should still be a real reading.
        assert sf.redness_score >= 0.0

    def test_bright_skin_region_not_flagged(self, analyzer):
        crop = np.full((40, 60, 3), [130, 110, 170], dtype=np.uint8)  # medium skin, BGR
        sf = analyzer._analyze_skin(self._region(crop))
        assert sf.available
        assert sf.low_light is False

    def test_low_light_floor_is_a_hard_cutoff(self, analyzer):
        """A clearly-under-floor region flags low_light; a clearly-over-floor
        region of the same skin tone family does not — regression guard
        against an inverted or off-by-one comparison. Uses real skin-toned
        colors (not grey) so the tone gate doesn't interfere."""
        skin_below = np.full((40, 60, 3), (18, 25, 40), dtype=np.uint8)    # gray ~29
        skin_above = np.full((40, 60, 3), (66, 134, 198), dtype=np.uint8)  # gray well over 60

        sf_below = analyzer._analyze_skin(self._region(skin_below))
        sf_above = analyzer._analyze_skin(self._region(skin_above))

        assert sf_below.low_light is True
        assert sf_above.low_light is False


class TestLipToneGate:

    def _region(self, crop, mask=None):
        h, w = crop.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        return FacialRegion(name="test", crop=crop, mask=mask, masked_crop=crop.copy(),
                            bbox=(0, 0, w, h), available=True)

    def test_pure_lip_region_is_available(self, analyzer):
        crop = np.full((30, 60, 3), [100, 70, 180], dtype=np.uint8)
        lf = analyzer._analyze_lips(self._region(crop))
        assert lf.available

    def test_mustache_contaminated_lip_matches_pure_lip_score(self, analyzer):
        """A mustache/beard hair edge intruding into the lips polygon should
        not change pallor or color-consistency scores versus pure lip color."""
        pure = np.full((30, 60, 3), [100, 70, 180], dtype=np.uint8)
        lf_pure = analyzer._analyze_lips(self._region(pure))

        mixed = pure.copy()
        mixed[:, :20] = [25, 22, 20]   # left third -> black mustache hair
        lf_mixed = analyzer._analyze_lips(self._region(mixed))

        assert lf_mixed.available
        assert lf_mixed.pallor_score == pytest.approx(lf_pure.pallor_score, abs=1.0)
        assert lf_mixed.color_consistency == pytest.approx(lf_pure.color_consistency, abs=1.0)

    def test_all_hair_lip_region_is_unavailable(self, analyzer):
        crop = np.full((30, 60, 3), [25, 22, 20], dtype=np.uint8)  # all hair, no lip
        lf = analyzer._analyze_lips(self._region(crop))
        assert not lf.available

    @pytest.mark.parametrize("bgr", [
        (100, 70, 180),   # vivid pink lip
        (140, 130, 170),  # pale/dry lip
        (160, 150, 175),  # very pale lip
        (60, 40, 150),    # dark red lip
        (40, 20, 160),    # deep red lip
    ])
    def test_lip_color_range_all_kept(self, analyzer, bgr):
        """Pale/dry lips are exactly what this analysis exists to detect —
        an over-tight gate that rejects them would defeat the feature."""
        crop = np.full((30, 60, 3), bgr, dtype=np.uint8)
        lf = analyzer._analyze_lips(self._region(crop))
        assert lf.available, f"Lip tone {bgr} was rejected by the tone gate"


class TestLipRegionLightingGate:
    """Mirrors TestSkinRegionLightingGate for lips — see that class's
    docstring and thresholds.REGION_LIGHTING."""

    def _region(self, crop, mask=None):
        h, w = crop.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        return FacialRegion(name="test", crop=crop, mask=mask, masked_crop=crop.copy(),
                            bbox=(0, 0, w, h), available=True)

    def test_dark_lip_region_flags_low_light_and_suppresses_dryness(self, analyzer):
        # Same dark BGR used in the skin lighting-gate tests — also falls
        # inside LIP_TONE_GATE's band, with gray luminance ~29.
        crop = np.full((30, 60, 3), (18, 25, 40), dtype=np.uint8)
        lf = analyzer._analyze_lips(self._region(crop))

        assert lf.available, "Region should stay available — only dryness is suppressed"
        assert lf.low_light is True
        assert lf.dryness_score == 0.0
        # Color consistency / pallor are not gated by lighting.
        assert lf.color_consistency >= 0.0
        assert lf.pallor_score >= 0.0

    def test_bright_lip_region_not_flagged(self, analyzer):
        crop = np.full((30, 60, 3), [100, 70, 180], dtype=np.uint8)
        lf = analyzer._analyze_lips(self._region(crop))
        assert lf.available
        assert lf.low_light is False


class TestToneGateBandSanity:
    """Direct checks on the calibrated Cr/Cb bands themselves, independent
    of _analyze_skin/_analyze_lips, so a bad future edit to thresholds.py
    fails here with a clear message instead of a confusing downstream
    'region unavailable' somewhere else."""

    def test_navy_fabric_rejected_by_skin_gate(self, analyzer):
        crop = np.full((10, 10, 3), [90, 55, 30], dtype=np.uint8)
        mask = analyzer._tone_gate_mask_2d(crop, SKIN_TONE_GATE)
        assert mask.max() == 0, "Navy fabric should be fully rejected by the skin-tone gate"

    def test_black_hair_rejected_by_skin_gate(self, analyzer):
        crop = np.full((10, 10, 3), [25, 22, 20], dtype=np.uint8)
        mask = analyzer._tone_gate_mask_2d(crop, SKIN_TONE_GATE)
        assert mask.max() == 0, "Black hair should be fully rejected by the skin-tone gate"

    def test_black_hair_rejected_by_lip_gate(self, analyzer):
        crop = np.full((10, 10, 3), [25, 22, 20], dtype=np.uint8)
        mask = analyzer._tone_gate_mask_2d(crop, LIP_TONE_GATE)
        assert mask.max() == 0, "Black hair should be fully rejected by the lip-tone gate"


class TestCalibrationBaseline:
    """
    A2 — per-person skin-tone calibration. Before this, dark-circle scoring
    compared under-eye luminance against one fixed constant
    (DARK_CIRCLE.bright_luminance=200), and skin-redness scoring compared
    relative-red dominance against one fixed constant
    (SKIN_REDNESS.baseline_rel_r=0.33). Both luminance and relative-red
    dominance vary by skin tone independent of any actual dark-circle or
    redness condition, so a fixed global reference systematically
    over-scored darker/more-saturated skin tones. These tests guard the
    fix: scoring relative to the person's OWN forehead instead.
    """

    def _region(self, crop, mask=None):
        h, w = crop.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        return FacialRegion(name="test", crop=crop, mask=mask, masked_crop=crop.copy(),
                            bbox=(0, 0, w, h), available=True)

    # ── baseline computation itself ─────────────────────────────────────────

    def test_baseline_unavailable_when_forehead_is_none(self, analyzer):
        baseline = analyzer._compute_calibration_baseline(None)
        assert baseline.available is False

    def test_baseline_unavailable_when_forehead_region_unavailable(self, analyzer):
        region = FacialRegion(name="forehead", crop=None, mask=None, masked_crop=None,
                               bbox=(0, 0, 0, 0), available=False)
        baseline = analyzer._compute_calibration_baseline(region)
        assert baseline.available is False

    def test_baseline_unavailable_below_min_pixels(self, analyzer):
        tiny = _solid_bgr(130, 110, 170, h=3, w=3)  # 9px < CALIBRATION.min_pixels
        baseline = analyzer._compute_calibration_baseline(self._region(tiny))
        assert baseline.available is False

    def test_baseline_available_for_normal_forehead_crop(self, analyzer):
        crop = _solid_bgr(130, 110, 170)
        baseline = analyzer._compute_calibration_baseline(self._region(crop))
        assert baseline.available is True
        assert baseline.luminance > 0
        assert 0 < baseline.rel_r < 1

    @pytest.mark.parametrize("bgr", [
        (196, 224, 255),  # very light skin
        (125, 194, 241),  # light skin
        (66, 134, 198),   # medium skin
        (36, 85, 141),    # tan skin
        (35, 55, 90),      # dark brown skin
    ])
    def test_baseline_available_across_skin_tone_gradient(self, analyzer, bgr):
        """Same gradient as TestToneGate.test_skin_gradient_all_kept — the
        calibration baseline must be computable for every tone the tone
        gate itself accepts, or darker-skinned people would silently fall
        back to the uncalibrated (biased) fixed-constant path."""
        crop = _solid_bgr(*bgr)
        baseline = analyzer._compute_calibration_baseline(self._region(crop))
        assert baseline.available, f"Calibration baseline unavailable for skin tone {bgr}"

    # ── dark-circle scoring uses the baseline ───────────────────────────────

    def test_dark_circle_no_baseline_matches_old_fixed_constant_behavior(self, analyzer):
        """Backward compatibility: omitting baseline (or passing None) must
        reproduce the exact pre-calibration score, since every existing
        direct call site (and unit test) does this."""
        crop = _solid_bgr(40, 40, 40)  # dark eye crop
        score_implicit = analyzer._dark_circle_score(crop)
        score_explicit_none = analyzer._dark_circle_score(crop, None)
        assert score_implicit == score_explicit_none
        # And matches the formula against the fixed constant directly.
        gray_mean = 40  # solid BGR(40,40,40) -> grayscale 40
        expected = (DARK_CIRCLE.bright_luminance - gray_mean) / DARK_CIRCLE.divisor * 100
        assert score_implicit == pytest.approx(min(expected, 100.0), abs=1.0)

    def test_dark_circle_unavailable_baseline_falls_back_to_fixed_constant(self, analyzer):
        from phase3_feature_analysis.feature_analyzer import CalibrationBaseline
        crop = _solid_bgr(40, 40, 40)
        unavailable = CalibrationBaseline(available=False)
        score_with_unavailable = analyzer._dark_circle_score(crop, unavailable)
        score_with_none = analyzer._dark_circle_score(crop, None)
        assert score_with_unavailable == score_with_none

    def test_dark_circle_eye_matching_own_forehead_tone_scores_near_zero(self, analyzer):
        """Regression guard for the bug this feature exists to fix: a
        medium/dark-skinned person whose eye crop is the SAME tone as their
        own forehead (i.e. no actual dark-circle signal) must not be scored
        as having severe dark circles just because their skin is darker
        than the old fixed 200-luminance reference."""
        skin_bgr = (35, 55, 90)  # dark brown skin, BGR
        forehead_crop = _solid_bgr(*skin_bgr)
        eye_crop = _solid_bgr(*skin_bgr)

        baseline = analyzer._compute_calibration_baseline(self._region(forehead_crop))
        assert baseline.available

        score_uncalibrated = analyzer._dark_circle_score(eye_crop, None)
        score_calibrated = analyzer._dark_circle_score(eye_crop, baseline)

        assert score_uncalibrated == pytest.approx(100.0, abs=1.0), (
            "Sanity check on the old behavior this test guards against — "
            "if this fails, the fixed-constant fallback formula changed.")
        assert score_calibrated == pytest.approx(0.0, abs=1.0), (
            f"Eye region identical to the person's own forehead tone should "
            f"score ~0 dark-circle signal once calibrated, got {score_calibrated}")

    def test_dark_circle_genuinely_darker_eye_still_flags_with_calibration(self, analyzer):
        """Calibration must not mask a REAL dark circle — an eye region
        meaningfully darker than the person's own forehead should still
        score positively, just measured against their own baseline rather
        than a fixed constant."""
        skin_bgr = (66, 134, 198)  # medium skin
        forehead_crop = _solid_bgr(*skin_bgr)
        darker_eye_crop = _solid_bgr(15, 30, 45)  # noticeably darker than forehead

        baseline = analyzer._compute_calibration_baseline(self._region(forehead_crop))
        score_calibrated = analyzer._dark_circle_score(darker_eye_crop, baseline)

        assert score_calibrated > 30.0, (
            f"A genuinely darker under-eye region should still flag once "
            f"calibrated, got {score_calibrated}")

    # ── skin-redness scoring uses the baseline ──────────────────────────────

    def test_skin_redness_no_baseline_matches_old_fixed_constant_behavior(self, analyzer):
        crop = _solid_bgr(100, 100, 100)  # equal channels -> rel_r ~= baseline_rel_r
        pixels = crop.reshape(-1, 3).astype(float)
        score_implicit = analyzer._skin_redness(pixels)
        score_explicit_none = analyzer._skin_redness(pixels, None)
        assert score_implicit == score_explicit_none
        assert score_implicit == pytest.approx(0.0, abs=5.0)

    def test_skin_redness_unavailable_baseline_falls_back_to_fixed_constant(self, analyzer):
        from phase3_feature_analysis.feature_analyzer import CalibrationBaseline
        crop = _solid_bgr(60, 80, 160)
        pixels = crop.reshape(-1, 3).astype(float)
        unavailable = CalibrationBaseline(available=False)
        score_with_unavailable = analyzer._skin_redness(pixels, unavailable)
        score_with_none = analyzer._skin_redness(pixels, None)
        assert score_with_unavailable == score_with_none

    @pytest.mark.parametrize("bgr", [
        (35, 55, 90),    # dark brown skin
        (36, 85, 141),   # tan skin
        (125, 194, 241), # light skin
    ])
    def test_skin_redness_cheek_matching_own_forehead_tone_scores_near_zero(self, analyzer, bgr):
        """Regression guard mirroring the dark-circle case above: a cheek
        region the SAME tone as the person's own forehead (no actual
        redness) must not be inflated just because their natural rel_r
        differs from the old fixed SKIN_REDNESS.baseline_rel_r=0.33."""
        forehead_crop = _solid_bgr(*bgr)
        cheek_crop = _solid_bgr(*bgr)

        baseline = analyzer._compute_calibration_baseline(self._region(forehead_crop))
        assert baseline.available

        cheek_pixels = cheek_crop.reshape(-1, 3).astype(float)
        score_calibrated = analyzer._skin_redness(cheek_pixels, baseline)

        assert score_calibrated == pytest.approx(0.0, abs=1.0), (
            f"Cheek region identical to the person's own forehead tone "
            f"{bgr} should score ~0 redness once calibrated, got {score_calibrated}")

    def test_skin_redness_genuinely_redder_cheek_still_flags_with_calibration(self, analyzer):
        """Calibration must not mask REAL redness — a cheek noticeably more
        red than the person's own forehead should still score positively."""
        skin_bgr = (66, 134, 198)  # medium skin
        forehead_crop = _solid_bgr(*skin_bgr)
        redder_cheek_crop = _solid_bgr(50, 90, 220)  # boosted red channel

        baseline = analyzer._compute_calibration_baseline(self._region(forehead_crop))
        cheek_pixels = redder_cheek_crop.reshape(-1, 3).astype(float)
        score_calibrated = analyzer._skin_redness(cheek_pixels, baseline)

        assert score_calibrated > 20.0, (
            f"A genuinely redder cheek should still flag once calibrated, "
            f"got {score_calibrated}")

    # ── end-to-end wiring through analyze() ─────────────────────────────────

    def test_analyze_skin_accepts_optional_baseline_without_error(self, analyzer):
        """_analyze_skin's signature gained an optional baseline param —
        confirm both call styles (with/without) still work end-to-end."""
        crop = _solid_bgr(130, 110, 170)
        region = self._region(crop)
        sf_no_baseline = analyzer._analyze_skin(region)
        assert sf_no_baseline.available

        baseline = analyzer._compute_calibration_baseline(self._region(_solid_bgr(130, 110, 170)))
        sf_with_baseline = analyzer._analyze_skin(region, baseline=baseline)
        assert sf_with_baseline.available