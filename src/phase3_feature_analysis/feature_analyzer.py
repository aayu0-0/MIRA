"""
Phase 3 — Feature Analysis
Extracts measurable numerical features from each facial region.
All scores are normalized 0-100 unless stated otherwise.
"""

import cv2
import logging
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math

log = logging.getLogger(__name__)

from phase1_face_understanding.face_detector import (
    LEFT_CHEEK_PT, RIGHT_CHEEK_PT, LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER,
    MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP_CENTER, LOWER_LIP_CENTER,
)
from thresholds import (
    SWELLING, EAR, EYE_REDNESS, DARK_CIRCLE, PUFFINESS,
    SKIN_REDNESS, TEXTURE, ACNE, LIP_DRYNESS, LIP_COLOR_CONSISTENCY, LIP_PALLOR,
    SKIN_TONE_GATE, LIP_TONE_GATE, MOUTH_SHAPE, REGION_LIGHTING, CALIBRATION,
)


# ── Feature dataclasses ───────────────────────────────────────────────────────

@dataclass
class EyeFeatures:
    # FIX: default changed from 0.0 to None so the "available but EAR landmarks
    # missing" case is distinguishable from "eye closed" (ratio ≈ 0).
    # Phase 4 already gates on `is not None` before using this value.
    openness_ratio: Optional[float] = None
    redness_score: float = 0.0
    dark_circle_score: float = 0.0
    puffiness_score: float = 0.0
    available: bool = False

@dataclass
class SkinFeatures:
    acne_score: float = 0.0
    texture_irregularity: float = 0.0
    redness_score: float = 0.0
    spot_count: int = 0
    low_light: bool = False
    available: bool = False

@dataclass
class LipFeatures:
    dryness_score: float = 0.0
    color_consistency: float = 0.0
    pallor_score: float = 0.0
    low_light: bool = False
    available: bool = False

@dataclass
class SmileFeatures:
    mouth_aspect_ratio: float = 0.0
    corner_asymmetry: float = 0.0
    curvature_score: float = 0.0
    available: bool = False

@dataclass
class FaceFeatures:
    symmetry_score: float = 0.0
    symmetry_label: str = ""
    alignment_angle: float = 0.0
    face_width: int = 0
    face_height: int = 0
    aspect_ratio: float = 0.0
    cheek_eye_ratio: float = 0.0     # actual cheek-width / eye-span ratio (used for swelling)
    swelling_score: float = 0.0
    swelling_label: str = ""
    available: bool = False

@dataclass
class Phase3Result:
    left_eye: EyeFeatures = field(default_factory=EyeFeatures)
    right_eye: EyeFeatures = field(default_factory=EyeFeatures)
    skin_forehead: SkinFeatures = field(default_factory=SkinFeatures)
    skin_left_cheek: SkinFeatures = field(default_factory=SkinFeatures)
    skin_right_cheek: SkinFeatures = field(default_factory=SkinFeatures)
    skin_nose: SkinFeatures = field(default_factory=SkinFeatures)
    lips: LipFeatures = field(default_factory=LipFeatures)
    smile: SmileFeatures = field(default_factory=SmileFeatures)
    face: FaceFeatures = field(default_factory=FaceFeatures)

@dataclass(frozen=True)
class CalibrationBaseline:
    luminance: float = 0.0
    rel_r: float = 0.0
    available: bool = False


# ── EAR landmark pools ────────────────────────────────────────────────────────

_FLC = mp_vision.FaceLandmarksConnections

# Eye aspect ratio (EAR) measures how open the eye is.
# EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
# where p0/p3 are the horizontal corners and p1/p2/p4/p5 are vertical pairs.
def _eye_index_pool(connections):
    pool = set()
    for c in connections:
        pool.add(c.start)
        pool.add(c.end)
    return pool

_MP_LEFT_EYE_POOL  = _eye_index_pool(_FLC.FACE_LANDMARKS_LEFT_EYE)
_MP_RIGHT_EYE_POOL = _eye_index_pool(_FLC.FACE_LANDMARKS_RIGHT_EYE)

LEFT_EYE_EAR_POINTS  = [263, 386, 374, 362, 385, 380]
RIGHT_EYE_EAR_POINTS = [33,  159, 145, 133, 158, 153]

assert all(p in _MP_LEFT_EYE_POOL  for p in LEFT_EYE_EAR_POINTS), \
    "LEFT_EYE_EAR_POINTS contains indices outside MediaPipe's left-eye group — likely a left/right mixup."
assert all(p in _MP_RIGHT_EYE_POOL for p in RIGHT_EYE_EAR_POINTS), \
    "RIGHT_EYE_EAR_POINTS contains indices outside MediaPipe's right-eye group — likely a left/right mixup."


# ── Mask crop helper ──────────────────────────────────────────────────────────

def _crop_mask_to_bbox(mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """Return the portion of `mask` that covers `bbox` (bx, by, bw, bh).

    Handles two conventions that Phase 2 regions may use:
    - Full-image mask: mask.shape matches the source image; we slice [by:by+bh, bx:bx+bw].
    - Pre-cropped mask: mask.shape already equals (bh, bw); return as-is.

    Raises ValueError if the mask shape is inconsistent with both expectations
    so that a dimension mismatch is caught early rather than silently producing
    a wrong-shaped array.
    """
    bx, by, bw, bh = bbox
    mh, mw = mask.shape[:2]
    if mh == bh and mw == bw:
        # Mask is already cropped to the bbox — return directly.
        return mask
    # Assume full-image mask; slice to bbox.
    cropped = mask[by:by + bh, bx:bx + bw]
    if cropped.shape[:2] != (bh, bw):
        raise ValueError(
            f"Mask shape {mask.shape} does not match bbox ({bx},{by},{bw},{bh}): "
            f"sliced to {cropped.shape}, expected ({bh},{bw}). "
            f"Check that region.mask is either full-image or pre-cropped to bbox."
        )
    return cropped


# ── Feature Analyzer ──────────────────────────────────────────────────────────

class FeatureAnalyzer:
    """Single entry point to analyze all facial regions and extract measurable features."""

    def analyze(self, image_bgr: np.ndarray, phase1_result, phase2_result) -> Phase3Result:
        result = Phase3Result()

        # Face-level features from Phase 1
        result.face = self._analyze_face(phase1_result)

        regions = phase2_result.regions

        # Per-person calibration baseline derived from the forehead region.
        # Used to normalize dark-circle and redness scores across skin tones.
        # The forehead itself is NOT calibrated against this baseline (it IS the baseline).
        baseline = self._compute_calibration_baseline(regions.get("forehead"))

        # Eyes
        result.left_eye  = self._analyze_eye(
            regions.get("left_eye"),  phase1_result.landmarks, "left",  baseline=baseline)
        result.right_eye = self._analyze_eye(
            regions.get("right_eye"), phase1_result.landmarks, "right", baseline=baseline)

        # Skin — forehead uses the fixed global baseline (it IS the calibration source)
        result.skin_forehead    = self._analyze_skin(regions.get("forehead"),    region_kind="forehead")
        result.skin_left_cheek  = self._analyze_skin(regions.get("left_cheek"),  baseline=baseline, region_kind="cheek")
        result.skin_right_cheek = self._analyze_skin(regions.get("right_cheek"), baseline=baseline, region_kind="cheek")
        result.skin_nose        = self._analyze_skin(regions.get("nose"),        baseline=baseline, region_kind="nose")

        # Lips
        result.lips = self._analyze_lips(regions.get("lips"))

        # Smile — landmark geometry, not a pixel region. Pass the face's roll
        # angle so mouth geometry can be measured in a "level" frame instead
        # of the raw (possibly tilted) image frame.
        result.smile = self._analyze_smile(
            phase1_result.landmarks, phase1_result.alignment_angle
        )

        return result

    # ── Calibration ───────────────────────────────────────────────────────────

    def _compute_calibration_baseline(self, forehead_region) -> CalibrationBaseline:
        """Derive per-person skin-tone baseline from the forehead region.

        Returns an unavailable baseline (falls back to global constants) if the
        forehead region is missing, has no usable crop, or passes too few
        skin-tone-gated pixels to be reliable.
        """
        if (forehead_region is None or not forehead_region.available
                or forehead_region.masked_crop is None):
            return CalibrationBaseline(available=False)

        crop = forehead_region.masked_crop
        pixels = crop.reshape(-1, 3)
        mask_flat = (pixels.sum(axis=1) > 0)
        if mask_flat.sum() < CALIBRATION.min_pixels:
            return CalibrationBaseline(available=False)

        tone_ok_flat = self._tone_gate_mask_flat(pixels[mask_flat], SKIN_TONE_GATE)
        if tone_ok_flat.sum() < CALIBRATION.min_pixels:
            return CalibrationBaseline(available=False)

        skin_px = pixels[mask_flat][tone_ok_flat].astype(float)

        r, g, b = skin_px[:, 2], skin_px[:, 1], skin_px[:, 0]
        total = r + g + b + 1e-6
        rel_r = float((r / total).mean())

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(float)
        gray_flat = gray.reshape(-1)
        luminance = float(gray_flat[mask_flat][tone_ok_flat].mean())

        return CalibrationBaseline(luminance=luminance, rel_r=rel_r, available=True)

    # ── Face ──────────────────────────────────────────────────────────────────

    def _analyze_face(self, p1) -> FaceFeatures:
        """Populate FaceFeatures from the Phase 1 result."""
        if not p1.success:
            return FaceFeatures()
        ar = p1.face_height / p1.face_width if p1.face_width > 0 else 0
        swelling_score, swelling_label, cheek_eye_ratio = self._swelling_score(p1.landmarks)
        return FaceFeatures(
            symmetry_score=p1.symmetry_score,
            symmetry_label=p1.symmetry_label,
            alignment_angle=p1.alignment_angle,
            face_width=p1.face_width,
            face_height=p1.face_height,
            aspect_ratio=round(ar, 3),
            cheek_eye_ratio=cheek_eye_ratio,
            swelling_score=swelling_score,
            swelling_label=swelling_label,
            available=True,
        )

    def _swelling_score(self, landmarks: list) -> tuple[float, str, float]:
        """Return (score, label, cheek_eye_ratio).

        The ratio is surfaced directly in reports as 'Cheek / Eye-Span Ratio'
        so it must be returned alongside the derived score and label.

        Limitation: does not yet incorporate eye puffiness into the swelling
        score. A future pass should adjust the score based on EyeFeatures
        puffiness_score once that is computed — document here so it isn't
        forgotten.
        """
        if any(i >= len(landmarks) for i in
               (LEFT_CHEEK_PT, RIGHT_CHEEK_PT, LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER)):
            return 0.0, "Not Assessed", 0.0

        l_cheek = landmarks[LEFT_CHEEK_PT]
        r_cheek = landmarks[RIGHT_CHEEK_PT]
        l_eye   = landmarks[LEFT_EYE_OUTER_CORNER]
        r_eye   = landmarks[RIGHT_EYE_OUTER_CORNER]

        cheek_width = math.dist((l_cheek.px, l_cheek.py), (r_cheek.px, r_cheek.py))
        eye_span    = math.dist((l_eye.px,   l_eye.py),   (r_eye.px,   r_eye.py))
        if eye_span <= 0:
            return 0.0, "Not Assessed", 0.0

        ratio = cheek_width / eye_span

        if ratio <= SWELLING.typical_high:
            score = 0.0
        else:
            score = float(np.clip(
                (ratio - SWELLING.typical_high) / SWELLING.divisor * 100, 0, 100))

        if score < SWELLING.none:       label = "No Swelling Indicators"
        elif score < SWELLING.mild:     label = "Mild Swelling Indicator"
        elif score < SWELLING.moderate: label = "Moderate Swelling Indicator"
        else:                           label = "Notable Swelling Indicator"

        return round(score, 1), label, round(ratio, 3)

    # ── Smile / mouth shape ───────────────────────────────────────────────────

    def _analyze_smile(self, landmarks: list, alignment_angle: float = 0.0) -> SmileFeatures:
        """Compute mouth-shape geometry from landmarks.

        Tilt correction: alignment_angle (from face_detector._compute_angle,
        same convention as Phase 1's head-tilt) is the roll of the eye line
        relative to horizontal. The four mouth points are rotated by
        -alignment_angle around their own centroid before measuring, so a
        tilted head does not read as mouth asymmetry or curvature it does not
        actually have.

        Sign convention for curvature: positive = corners raised above the
        lip midline (smile-like), negative = corners lowered (frown-like).
        In image coordinates y increases downward, so a raised corner has a
        lower y value than the midline, making (midline_y - corner_y) > 0.
        This matches the Phase 4 narrative ("curv > 0 → smile-like").
        """
        sf = SmileFeatures()
        required = (MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP_CENTER, LOWER_LIP_CENTER)
        if any(i >= len(landmarks) for i in required):
            return sf

        l_corner = landmarks[MOUTH_LEFT]
        r_corner = landmarks[MOUTH_RIGHT]
        upper    = landmarks[UPPER_LIP_CENTER]
        lower    = landmarks[LOWER_LIP_CENTER]

        # Use sub-pixel (fx/fy) coordinates here, not the rounded int px/py.
        # A closed mouth's upper/lower-lip gap is frequently under 1px, which
        # int rounding collapses to exactly 0 -- that's what previously made
        # mouth_height hit its zero-fallback on essentially every real
        # (mostly closed-mouth) capture, pinning mouth_aspect_ratio at a
        # constant 100.0 instead of reflecting the real, slightly-varying
        # gap. See LandmarkPoint.fx/fy in face_detector.py.
        #
        # _coords() falls back to (px, py) when fx/fy are both exactly 0.0
        # (their dataclass default) -- this keeps callers that construct
        # LandmarkPoint directly without setting fx/fy (e.g. synthetic test
        # fixtures) working exactly as before, while real captures (which
        # always populate fx/fy) get the sub-pixel precision fix.
        def _coords(pt):
            return (pt.fx, pt.fy) if (pt.fx or pt.fy) else (float(pt.px), float(pt.py))

        (lx, ly), (rx, ry), (ux, uy), (dx, dy) = self._detilt_points(
            [_coords(l_corner), _coords(r_corner), _coords(upper), _coords(lower)],
            alignment_angle,
        )

        mouth_width  = math.hypot(lx - rx, ly - ry)
        mouth_height = math.hypot(ux - dx, uy - dy)
        if mouth_width <= 0:
            return sf

        sf.available = True

        # Mouth aspect ratio. Floor mouth_height at a small epsilon rather
        # than branching to a fixed sentinel value on <=0 -- a true zero
        # gap is now rare with sub-pixel coordinates, but the floor still
        # guards division-by-zero without collapsing every near-closed
        # mouth onto the exact same output value.
        mouth_height_floored = max(mouth_height, MOUTH_SHAPE.min_measurable_height)
        mar_raw = mouth_width / mouth_height_floored
        span = MOUTH_SHAPE.mar_typical_high - MOUTH_SHAPE.mar_typical_low
        sf.mouth_aspect_ratio = float(np.clip(
            (mar_raw - MOUTH_SHAPE.mar_typical_low) / span * 100, 0, 100
        ))

        # Corner asymmetry — vertical height difference between left and right
        # corners, normalized by mouth width to be scale-invariant.
        vertical_diff = abs(ly - ry)
        asym_ratio = vertical_diff / mouth_width
        sf.corner_asymmetry = float(np.clip(
            asym_ratio / MOUTH_SHAPE.asymmetry_divisor * 100, 0, 100
        ))

        # Curvature — how far the corner midpoint sits above/below the lip
        # center midline, normalized by mouth width.
        midline_y = (uy + dy) / 2.0
        corner_y  = (ly + ry) / 2.0
        curvature_raw = (midline_y - corner_y) / mouth_width
        sf.curvature_score = float(np.clip(
            curvature_raw / MOUTH_SHAPE.curvature_divisor * 100, -100, 100
        ))

        return sf

    def _detilt_points(self, points: list[tuple[float, float]],
                       angle_deg: float) -> list[tuple[float, float]]:
        """Rotate `points` by -angle_deg around their own centroid.

        Used to put mouth landmarks into a "level" frame before measuring
        width/height/curvature, so head roll does not read as mouth asymmetry.
        Sign convention matches face_detector._compute_angle (degrees,
        atan2(dy, dx) over the eye-corner vector).

        Returns points unchanged when angle_deg is zero (or effectively zero)
        to avoid floating-point drift on level captures.
        """
        if not points:
            return points
        if not angle_deg:
            return points
        theta = math.radians(-angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        out = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            out.append((cx + dx * cos_t - dy * sin_t,
                        cy + dx * sin_t + dy * cos_t))
        return out

    # ── Eyes ──────────────────────────────────────────────────────────────────

    def _analyze_eye(self, region, landmarks: list, side: str, *,
                     baseline: CalibrationBaseline = None) -> EyeFeatures:
        """Extract openness, redness, dark circles, and puffiness from one eye crop."""
        ef = EyeFeatures()
        if region is None or not region.available or region.masked_crop is None:
            return ef

        crop = region.masked_crop
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return ef

        ef.available = True
        ef.openness_ratio    = self._compute_ear(landmarks, side)
        ef.redness_score     = self._eye_redness(crop, baseline)
        ef.dark_circle_score = self._dark_circle_score(crop, baseline)
        ef.puffiness_score   = self._puffiness_score(crop)

        return ef

    def _compute_ear(self, landmarks: list, side: str) -> Optional[float]:
        """Return normalized Eye Aspect Ratio (0-100) or None if landmarks missing.

        None signals "data unavailable" rather than "eye closed" (which would
        be a near-zero float). Callers must handle None explicitly.
        """
        pts = LEFT_EYE_EAR_POINTS if side == "left" else RIGHT_EYE_EAR_POINTS
        if any(i >= len(landmarks) for i in pts):
            return None

        p = [landmarks[i] for i in pts]
        v1 = math.dist((p[1].px, p[1].py), (p[5].px, p[5].py))
        v2 = math.dist((p[2].px, p[2].py), (p[4].px, p[4].py))
        h  = math.dist((p[0].px, p[0].py), (p[3].px, p[3].py))
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.3

        return float(np.clip((ear - EAR.closed) / (EAR.open - EAR.closed) * 100, 0, 100))

    def _eye_redness(self, crop: np.ndarray,
                     baseline: CalibrationBaseline = None) -> float:
        """Score relative redness of the sclera region vs the per-person
        forehead calibration baseline (same pattern as _skin_redness /
        _dark_circle_score), falling back to the fixed generic constant
        when no baseline is available.

        This calibration is what removes the skin-tone bias a flat,
        uncalibrated formula has: without it, medium/dark skin tones read
        as false-positive redness even with a perfectly normal eye (see
        test_scoring.py::TestEyeRedness, a regression guard for that fix).
        A prior rewrite dropped the baseline parameter entirely and read
        SKIN_REDNESS.baseline_rel_r as a flat, un-calibrated constant for
        every person -- silently reintroducing the fairness bug this
        design fixes, and, on top of that, appearing to pin the score at
        0.0 for essentially every real capture (see the open calibration
        note on EYE_REDNESS in thresholds.py for why that may still be
        worth rechecking even under this corrected, tested formula).
        """
        pixels = crop.reshape(-1, 3).astype(float)
        mask_flat = pixels.sum(axis=1) > 0
        pixels = pixels[mask_flat] if mask_flat.any() else pixels
        r = pixels[:, 2]
        g = pixels[:, 1]
        b = pixels[:, 0]
        total = r + g + b + 1e-6
        rel_r = (r / total).mean()
        reference = (baseline.rel_r if baseline is not None and baseline.available
                     else SKIN_REDNESS.baseline_rel_r)
        score = (rel_r - reference) / EYE_REDNESS.divisor * 100
        return float(np.clip(score, 0, 100))

    def _dark_circle_score(self, crop: np.ndarray,
                           baseline: CalibrationBaseline = None) -> float:
        """Score darkness of the lower-third of the eye crop vs the per-person baseline."""
        if crop.shape[0] < 6:
            return 0.0
        lower = crop[crop.shape[0] * 2 // 3:, :]
        gray  = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY).astype(float)
        valid = lower.reshape(-1, 3).sum(axis=1) > 0
        vals = gray.reshape(-1)[valid]
        mean_lum = vals.mean() if vals.size > 0 else gray.mean()
        reference = (baseline.luminance if baseline is not None and baseline.available
                     else DARK_CIRCLE.bright_luminance)
        score = (reference - mean_lum) / DARK_CIRCLE.divisor * 100
        return float(np.clip(score, 0, 100))

    def _puffiness_score(self, crop: np.ndarray) -> float:
        """Score puffiness from variance of the upper-half eye crop.

        Low variance → flat/uniform surface → more puffy appearance.
        """
        upper = crop[:crop.shape[0] // 2, :]
        gray  = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY).astype(float)
        valid = upper.reshape(-1, 3).sum(axis=1) > 0
        vals = gray.reshape(-1)[valid]
        variance = vals.var() if vals.size > 0 else gray.var()
        score = max(0.0, (PUFFINESS.low_variance - min(variance, PUFFINESS.low_variance))
                    / PUFFINESS.low_variance * 100)
        return float(np.clip(score, 0, 100))

    # ── Tone-gate helpers ─────────────────────────────────────────────────────

    def _tone_gate_mask_2d(self, crop: np.ndarray, gate) -> np.ndarray:
        """Return a uint8 mask (255=keep, 0=reject) for pixels in the skin-tone band."""
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(float)
        cb = ycrcb[:, :, 2].astype(float)
        keep = (
            (cr >= gate.cr_low) & (cr <= gate.cr_high) &
            (cb >= gate.cb_low) & (cb <= gate.cb_high)
        )
        return (keep * 255).astype(np.uint8)

    def _tone_gate_mask_flat(self, bgr_pixels: np.ndarray, gate) -> np.ndarray:
        """Return a boolean mask over a flat (N, 3) BGR pixel array."""
        bgr_uint8 = np.clip(bgr_pixels, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        ycrcb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        cr = ycrcb[:, 1].astype(float)
        cb = ycrcb[:, 2].astype(float)
        return (
            (cr >= gate.cr_low) & (cr <= gate.cr_high) &
            (cb >= gate.cb_low) & (cb <= gate.cb_high)
        )

    # ── Skin ──────────────────────────────────────────────────────────────────

    def _analyze_skin(self, region, *, baseline: CalibrationBaseline = None,
                      region_kind: str = "cheek") -> SkinFeatures:
        """Extract redness, texture, and acne scores from one skin region.

        Texture and acne are suppressed (set to 0) when the region is
        underlit, because Laplacian-variance and contour-blob methods read
        sensor noise as texture/spots in low light even when the face-level
        AlignmentCheck passed (off-axis lighting can darken individual
        regions while the face average stays acceptable).

        Redness is mean-colour based and is NOT gated by the low-light flag.
        """
        sf = SkinFeatures()
        if region is None or not region.available or region.masked_crop is None:
            return sf

        crop = region.masked_crop
        pixels = crop.reshape(-1, 3)
        mask_flat = (pixels.sum(axis=1) > 0)
        if mask_flat.sum() < 50:
            return sf

        tone_ok_flat = self._tone_gate_mask_flat(pixels[mask_flat], SKIN_TONE_GATE)
        if tone_ok_flat.sum() < SKIN_TONE_GATE.min_pixels:
            return sf

        skin_px = pixels[mask_flat][tone_ok_flat].astype(float)
        sf.available = True

        # Build a combined mask (bbox-sized) for Laplacian / contour operations.
        # _crop_mask_to_bbox handles both full-image and pre-cropped region.mask
        # conventions so a Phase 2 change in mask representation doesn't silently
        # produce a wrong-shaped array.
        cm = _crop_mask_to_bbox(region.mask, region.bbox)
        tone_mask_2d = self._tone_gate_mask_2d(crop, SKIN_TONE_GATE)
        combined_mask = cv2.bitwise_and(cm, tone_mask_2d)

        sf.redness_score = self._skin_redness(skin_px, baseline)

        region_luminance = self._region_mean_luminance(crop, combined_mask)
        sf.low_light = region_luminance < REGION_LIGHTING.min_mean_luminance

        if sf.low_light:
            sf.texture_irregularity = 0.0
            sf.acne_score, sf.spot_count = 0.0, 0
        else:
            sf.texture_irregularity = self._texture_irregularity(
                crop, combined_mask, region_kind=region_kind
            )
            sf.acne_score, sf.spot_count = self._acne_score(crop, combined_mask)
            sf.spot_count = int(sf.spot_count)

        return sf

    def _skin_redness(self, skin_px: np.ndarray,
                      baseline: CalibrationBaseline = None) -> float:
        """Score redness of skin pixels relative to the per-person forehead baseline."""
        r, g, b = skin_px[:, 2], skin_px[:, 1], skin_px[:, 0]
        total = r + g + b + 1e-6
        rel_r = r / total
        reference = (baseline.rel_r if baseline is not None and baseline.available
                     else SKIN_REDNESS.baseline_rel_r)
        score = (rel_r.mean() - reference) / SKIN_REDNESS.divisor * 100
        return float(np.clip(score, 0, 100))

    def _region_mean_luminance(self, crop: np.ndarray, mask: np.ndarray) -> float:
        """Mean luminance of the masked region; returns 0.0 if mask is empty."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        vals = gray[mask > 0]
        if vals.size == 0:
            return 0.0
        return float(np.mean(vals))

    def _texture_irregularity(self, crop: np.ndarray, mask: np.ndarray,
                               region_kind: str = "cheek") -> float:
        """Score skin texture irregularity via Laplacian variance.

        Higher variance → rougher or more irregular surface. Divisor is
        region-specific because nose and forehead structurally yield higher
        Laplacian variance than cheeks even on smooth skin (nostril/tip shadow,
        surface curvature). See thresholds.TEXTURE for calibration notes.
        """
        divisor_map = {
            "forehead": TEXTURE.forehead_divisor,
            "nose":     TEXTURE.nose_divisor,
            "cheek":    TEXTURE.divisor,
        }
        divisor = divisor_map.get(region_kind, TEXTURE.divisor)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        vals = lap[mask > 0]
        if vals.size == 0:
            return 0.0
        var   = float(np.var(vals))
        score = float(np.clip(var / divisor, 0, 100))

        if log.isEnabledFor(logging.DEBUG):
            clipped = " <-- CLIPPED, divisor too low" if score >= 100.0 else ""
            log.debug(
                "region=%-9s raw_var=%10.2f divisor=%6.2f score=%6.2f%s",
                region_kind, var, divisor, score, clipped,
            )

        return score

    def _acne_score(self, crop: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
        """Detect dark spot contours as a proxy for acne/blemishes.

        Method: subtract a Gaussian-blurred version of the L channel from the
        original to highlight locally darker regions, threshold, mask to skin
        area, and count contours within the plausible size range.

        Does not yet distinguish acne (small, circular, elevated Cr) from dark
        patches (larger, irregular, Cr within normal skin range). That split is
        a planned future improvement — see inline TODO comments.

        Returns (score 0-100, spot_count).
        """
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return 0.0, 0

        lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]

        blur   = cv2.GaussianBlur(l_ch, (15, 15), 0)
        diff   = blur.astype(int) - l_ch.astype(int)
        thresh = np.clip(diff, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(thresh, 18, 255, cv2.THRESH_BINARY)

        binary = cv2.bitwise_and(binary, binary, mask=mask)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        spots = [c for c in contours
                 if ACNE.min_contour_area < cv2.contourArea(c) < ACNE.max_contour_area]
        count = len(spots)
        score = min(count / ACNE.spot_count_for_max_score * 100, 100)
        return float(score), count

    # ── Lips ──────────────────────────────────────────────────────────────────

    def _analyze_lips(self, region) -> LipFeatures:
        """Extract dryness, color consistency, and pallor from the lip region.

        Dryness (Laplacian-variance based) is suppressed under low light for
        the same reason as skin texture — sensor noise reads as surface
        irregularity. Color consistency and pallor are mean-colour based and
        are always computed when the region is available.
        """
        lf = LipFeatures()
        if region is None or not region.available or region.masked_crop is None:
            return lf

        crop = region.masked_crop
        pixels = crop.reshape(-1, 3).astype(float)
        mask_flat = pixels.sum(axis=1) > 0
        if mask_flat.sum() < 30:
            return lf

        tone_ok_flat = self._tone_gate_mask_flat(pixels[mask_flat], LIP_TONE_GATE)
        if tone_ok_flat.sum() < LIP_TONE_GATE.min_pixels:
            return lf

        lip_px = pixels[mask_flat][tone_ok_flat]
        lf.available = True

        # FIX: use _crop_mask_to_bbox to handle both full-image and
        # pre-cropped region.mask conventions (same as _analyze_skin).
        cm = _crop_mask_to_bbox(region.mask, region.bbox)
        tone_mask_2d = self._tone_gate_mask_2d(crop, LIP_TONE_GATE)
        combined_mask = cv2.bitwise_and(cm, tone_mask_2d)

        region_luminance = self._region_mean_luminance(crop, combined_mask)
        lf.low_light = region_luminance < REGION_LIGHTING.min_mean_luminance

        if lf.low_light:
            lf.dryness_score = 0.0
        else:
            lf.dryness_score = self._lip_dryness(crop, combined_mask)

        lf.color_consistency = self._lip_color_consistency(lip_px)
        lf.pallor_score      = self._lip_pallor(lip_px)

        return lf

    def _lip_dryness(self, crop: np.ndarray, mask: np.ndarray) -> float:
        """Score lip dryness via Laplacian variance — dry/cracked lips yield higher variance."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        vals = lap[mask > 0]
        if vals.size == 0:
            return 0.0
        return float(np.clip(np.var(vals) / LIP_DRYNESS.divisor, 0, 100))

    def _lip_color_consistency(self, pixels: np.ndarray) -> float:
        """Score lip color uniformity (100 = perfectly uniform, 0 = highly variable).

        Phase 4 inverts this as (100 - color_consistency) to get an
        inconsistency score that maps onto LIP_COLOR_INCONSISTENCY_SEVERITY.
        """
        std_per_channel = pixels.std(axis=0).mean()
        score = max(0.0, 100.0 - std_per_channel / LIP_COLOR_CONSISTENCY.divisor * 100)
        return float(score)

    def _lip_pallor(self, pixels: np.ndarray) -> float:
        """Score lip pallor from mean HSV saturation — lower saturation = paler lips."""
        bgr_uint8 = np.clip(pixels, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mean_sat = hsv[:, 1].mean()
        score = max(0.0, (LIP_PALLOR.divisor - mean_sat) / LIP_PALLOR.divisor * 100)
        return float(score)