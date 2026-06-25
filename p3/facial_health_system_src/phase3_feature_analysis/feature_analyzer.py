"""
Phase 3 — Feature Analysis
Extracts measurable numerical features from each facial region.
All scores are normalized 0–100 unless stated otherwise.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Data contracts ───────────────────────────────────────────────────────────

@dataclass
class EyeFeatures:
    openness_ratio: float = 0.0       # eye aspect ratio (EAR), normalized
    redness_score: float = 0.0        # 0–100
    dark_circle_score: float = 0.0    # 0–100
    puffiness_score: float = 0.0      # 0–100
    available: bool = False


@dataclass
class SkinFeatures:
    acne_score: float = 0.0           # 0–100
    texture_irregularity: float = 0.0 # 0–100
    redness_score: float = 0.0        # 0–100
    spot_count: int = 0
    available: bool = False


@dataclass
class LipFeatures:
    dryness_score: float = 0.0        # 0–100
    color_consistency: float = 0.0    # 0–100 (100 = very consistent)
    pallor_score: float = 0.0         # 0–100
    available: bool = False


@dataclass
class FaceFeatures:
    symmetry_score: float = 0.0       # from Phase 1
    symmetry_label: str = ""
    alignment_angle: float = 0.0
    face_width: int = 0
    face_height: int = 0
    aspect_ratio: float = 0.0
    available: bool = False


@dataclass
class Phase3Result:
    left_eye: EyeFeatures = field(default_factory=EyeFeatures)
    right_eye: EyeFeatures = field(default_factory=EyeFeatures)
    skin_forehead: SkinFeatures = field(default_factory=SkinFeatures)
    skin_left_cheek: SkinFeatures = field(default_factory=SkinFeatures)
    skin_right_cheek: SkinFeatures = field(default_factory=SkinFeatures)
    lips: LipFeatures = field(default_factory=LipFeatures)
    face: FaceFeatures = field(default_factory=FaceFeatures)


# ─── Eye landmark indices (MediaPipe 478) ─────────────────────────────────────
# Per-eye EAR points (vertical pairs)
LEFT_EYE_EAR_POINTS  = [33, 159, 145, 133, 158, 153]   # outer, top1, bot1, inner, top2, bot2
RIGHT_EYE_EAR_POINTS = [263, 386, 374, 362, 385, 380]


# ─── Feature analyzer ─────────────────────────────────────────────────────────

class FeatureAnalyzer:

    # ── public entry point ────────────────────────────────────────────────────

    def analyze(self, image_bgr: np.ndarray,
                phase1_result,   # Phase1Result
                phase2_result    # Phase2Result
                ) -> Phase3Result:

        result = Phase3Result()

        # Face-level features (from Phase 1)
        result.face = self._analyze_face(phase1_result)

        # Eye features
        regions = phase2_result.regions
        result.left_eye  = self._analyze_eye(
            regions.get("left_eye"),  phase1_result.landmarks, "left")
        result.right_eye = self._analyze_eye(
            regions.get("right_eye"), phase1_result.landmarks, "right")

        # Skin features
        result.skin_forehead    = self._analyze_skin(regions.get("forehead"))
        result.skin_left_cheek  = self._analyze_skin(regions.get("left_cheek"))
        result.skin_right_cheek = self._analyze_skin(regions.get("right_cheek"))

        # Lip features
        result.lips = self._analyze_lips(regions.get("lips"))

        return result

    # ── face ──────────────────────────────────────────────────────────────────

    def _analyze_face(self, p1) -> FaceFeatures:
        if not p1.success:
            return FaceFeatures()
        ar = p1.face_height / p1.face_width if p1.face_width > 0 else 0
        return FaceFeatures(
            symmetry_score=p1.symmetry_score,
            symmetry_label=p1.symmetry_label,
            alignment_angle=p1.alignment_angle,
            face_width=p1.face_width,
            face_height=p1.face_height,
            aspect_ratio=round(ar, 3),
            available=True,
        )

    # ── eyes ──────────────────────────────────────────────────────────────────

    def _analyze_eye(self, region, landmarks: list, side: str) -> EyeFeatures:
        ef = EyeFeatures()
        if region is None or not region.available or region.crop is None:
            return ef

        crop = region.crop
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return ef

        ef.available = True

        # 1. Openness (Eye Aspect Ratio)
        ef.openness_ratio = self._compute_ear(landmarks, side)

        # 2. Redness — excess red channel relative to green in the crop
        ef.redness_score = self._eye_redness(crop)

        # 3. Dark circles — compare under-eye region to cheek region
        ef.dark_circle_score = self._dark_circle_score(crop)

        # 4. Puffiness — variance of lower eye region brightness
        ef.puffiness_score = self._puffiness_score(crop)

        return ef

    def _compute_ear(self, landmarks: list, side: str) -> float:
        """Eye Aspect Ratio normalized to 0–100 (higher = more open)."""
        pts = LEFT_EYE_EAR_POINTS if side == "left" else RIGHT_EYE_EAR_POINTS
        if any(i >= len(landmarks) for i in pts):
            return 50.0
        p = [landmarks[i] for i in pts]
        # EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        v1 = math.dist((p[1].px, p[1].py), (p[5].px, p[5].py))
        v2 = math.dist((p[2].px, p[2].py), (p[4].px, p[4].py))
        h  = math.dist((p[0].px, p[0].py), (p[3].px, p[3].py))
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.3
        # Typical EAR: 0.15 (closed) – 0.40 (open). Normalize.
        return float(np.clip((ear - 0.15) / (0.40 - 0.15) * 100, 0, 100))

    def _eye_redness(self, crop: np.ndarray) -> float:
        """Redness = mean(R - G) in the eye crop, normalized."""
        r = crop[:, :, 2].astype(float)
        g = crop[:, :, 1].astype(float)
        diff = np.clip(r - g, 0, None)
        score = diff.mean() / 255 * 100 * 3   # amplify; typical range 0–30
        return float(np.clip(score, 0, 100))

    def _dark_circle_score(self, crop: np.ndarray) -> float:
        """
        Lower third of the eye crop tends to capture under-eye skin.
        Dark circles → lower luminance there.
        Score 0–100 (higher = darker / more likely dark circles).
        """
        if crop.shape[0] < 6:
            return 0.0
        lower = crop[crop.shape[0] * 2 // 3:, :]
        gray  = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY).astype(float)
        mean_lum = gray.mean()
        # Map: bright (200+) → 0, dark (<80) → 100
        score = (200 - mean_lum) / 120 * 100
        return float(np.clip(score, 0, 100))

    def _puffiness_score(self, crop: np.ndarray) -> float:
        """
        Puffiness estimated by texture flatness in the upper eyelid region
        (puffy eyelids show less texture / wrinkle variance).
        Low-variance = possibly puffy.
        Score 0–100 (higher = more puffiness signal).
        """
        upper = crop[:crop.shape[0] // 2, :]
        gray  = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY).astype(float)
        variance = gray.var()
        # Low variance (< 50) = flat/puffy, high variance (> 800) = normal texture
        score = max(0.0, (50 - min(variance, 50)) / 50 * 100)
        return float(np.clip(score, 0, 100))

    # ── skin ──────────────────────────────────────────────────────────────────

    def _analyze_skin(self, region) -> SkinFeatures:
        sf = SkinFeatures()
        if region is None or not region.available or region.masked_crop is None:
            return sf

        crop = region.masked_crop
        mask = region.mask

        # Only operate on non-black (masked) pixels
        pixels = crop.reshape(-1, 3)
        mask_flat = (pixels.sum(axis=1) > 0)
        if mask_flat.sum() < 50:
            return sf

        skin_px = pixels[mask_flat].astype(float)
        sf.available = True

        sf.redness_score      = self._skin_redness(skin_px)
        sf.texture_irregularity = self._texture_irregularity(crop, region.mask)
        sf.acne_score, sf.spot_count = self._acne_score(crop, region.mask)
        sf.spot_count = int(sf.spot_count)

        return sf

    def _skin_redness(self, skin_px: np.ndarray) -> float:
        r, g, b = skin_px[:, 2], skin_px[:, 1], skin_px[:, 0]
        # Relative red dominance
        total = r + g + b + 1e-6
        rel_r = r / total
        score = (rel_r.mean() - 0.33) / 0.10 * 100
        return float(np.clip(score, 0, 100))

    def _texture_irregularity(self, crop: np.ndarray, mask: np.ndarray) -> float:
        """Laplacian variance as texture irregularity proxy (0–100)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        # Only consider masked region
        crop_mask = mask[
            max(0, 0): min(mask.shape[0], crop.shape[0] + 0),
            max(0, 0): min(mask.shape[1], crop.shape[1] + 0)
        ]
        cm = crop_mask[:crop.shape[0], :crop.shape[1]]
        vals = lap[cm > 0]
        if vals.size == 0:
            return 0.0
        var = float(np.var(vals))
        # Typical range: 0 (smooth) – 2000 (rough/spotty)
        score = np.clip(var / 20, 0, 100)
        return float(score)

    def _acne_score(self, crop: np.ndarray, mask: np.ndarray):
        """
        Detect potential acne/spots via blob detection on the L channel (LAB).
        Returns (score_0_to_100, spot_count).
        """
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return 0.0, 0

        lab   = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch  = lab[:, :, 0]

        # Threshold: find dark spots (lower L) relative to local mean
        blur  = cv2.GaussianBlur(l_ch, (15, 15), 0)
        diff  = blur.astype(int) - l_ch.astype(int)
        thresh = np.clip(diff, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(thresh, 18, 255, cv2.THRESH_BINARY)

        # Apply region mask
        cm = mask[:crop.shape[0], :crop.shape[1]]
        binary = cv2.bitwise_and(binary, binary, mask=cm)

        # Find contours (spots)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        spots = [c for c in contours if 8 < cv2.contourArea(c) < 800]
        count = len(spots)
        # Normalize: 0 spots → 0, 10+ spots → 100
        score = min(count / 10 * 100, 100)
        return float(score), count

    # ── lips ──────────────────────────────────────────────────────────────────

    def _analyze_lips(self, region) -> LipFeatures:
        lf = LipFeatures()
        if region is None or not region.available or region.masked_crop is None:
            return lf

        crop = region.masked_crop
        pixels = crop.reshape(-1, 3).astype(float)
        mask_flat = pixels.sum(axis=1) > 0
        if mask_flat.sum() < 30:
            return lf

        skin_px = pixels[mask_flat]
        lf.available = True
        lf.dryness_score      = self._lip_dryness(crop, region.mask)
        lf.color_consistency  = self._lip_color_consistency(skin_px)
        lf.pallor_score       = self._lip_pallor(skin_px)

        return lf

    def _lip_dryness(self, crop: np.ndarray, mask: np.ndarray) -> float:
        """Texture variance in lips → higher variance = possible dryness/flaking."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        cm   = mask[:crop.shape[0], :crop.shape[1]]
        vals = lap[cm > 0]
        if vals.size == 0:
            return 0.0
        score = float(np.clip(np.var(vals) / 15, 0, 100))
        return score

    def _lip_color_consistency(self, pixels: np.ndarray) -> float:
        """Color consistency: 100 = very uniform color (healthy)."""
        std_per_channel = pixels.std(axis=0).mean()
        # Low std = consistent color. Map std 0–80 to score 100–0.
        score = max(0, 100 - std_per_channel / 80 * 100)
        return float(score)

    def _lip_pallor(self, pixels: np.ndarray) -> float:
        """
        Pallor: lips that are pale/less saturated → higher score.
        Uses saturation in HSV.
        """
        bgr_uint8 = np.clip(pixels, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mean_sat = hsv[:, 1].mean()   # 0–255
        # High saturation → healthy red/pink lips → low pallor score
        score = max(0.0, (100 - mean_sat) / 100 * 100)
        return float(score)
