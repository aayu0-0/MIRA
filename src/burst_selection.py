"""
burst_selection.py -- pick the single best frame out of a multi-photo burst.

Two stages, cheapest first:

Stage 1 (per frame, Phase 1 only):
    Reject frames that are structurally unusable before spending any time
    on Phase 2/3/4:
      - no face detected
      - blurry (Laplacian variance on the face crop, below BURST_FILTER.min_laplacian_variance)
      - eyes closed (EAR off Phase 1 landmarks, both eyes at/below
        BURST_FILTER.eyes_closed_ear_score)

Stage 2 (survivors only, full Phase 1->4):
    Run the real pipeline on whatever's left and keep the frame with the
    highest Phase 4 overall_confidence. Ties go to the sharper frame --
    overall_confidence is a region-coverage metric (see observation_engine.py)
    and is blind to focus, so it can't break ties on its own.

Phase 5 (report generation) and Phase 6 (feature export) are deliberately
NOT run here -- callers should run those once, on the winning frame only.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from phase1_face_understanding.face_detector import FaceDetector, Phase1Result
from phase2_region_extraction.region_extractor import RegionExtractor
from phase3_feature_analysis.feature_analyzer import (
    FeatureAnalyzer, LEFT_EYE_EAR_POINTS, RIGHT_EYE_EAR_POINTS,
)
from phase4_observation_engine.observation_engine import ObservationEngine
from thresholds import EAR, BURST_FILTER


# ─── Stage 1 helpers ────────────────────────────────────────────────────────

def _laplacian_variance(image_bgr: np.ndarray, bbox=None) -> float:
    """Blur metric on the face region. Falls back to the full frame if no
    bbox is available (shouldn't happen for a successful Phase 1 result)."""
    if bbox is not None:
        crop = image_bgr[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    else:
        crop = image_bgr
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ear_score(landmarks: list, points: list) -> Optional[float]:
    """Same formula as FeatureAnalyzer._compute_ear, duplicated here (rather
    than instantiating a FeatureAnalyzer) because it only needs Phase 1
    landmarks -- Phase 2/3 haven't run yet at this point in the burst filter.
    Returns None, not a fabricated mid-value, when landmarks are missing."""
    if any(i >= len(landmarks) for i in points):
        return None
    p = [landmarks[i] for i in points]
    v1 = math.dist((p[1].px, p[1].py), (p[5].px, p[5].py))
    v2 = math.dist((p[2].px, p[2].py), (p[4].px, p[4].py))
    h = math.dist((p[0].px, p[0].py), (p[3].px, p[3].py))
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.3
    return float(np.clip((ear - EAR.closed) / (EAR.open - EAR.closed) * 100, 0, 100))


@dataclass
class BurstCandidate:
    # frame is excluded from eq/repr: it's a raw numpy image array, and the
    # default dataclass equality (elementwise `==` on every field) raises
    # "truth value of an array is ambiguous" as soon as two candidates'
    # frames aren't byte-identical -- which is every real comparison. Only
    # p1/blur/ears/confidence are meaningful for equality purposes anyway.
    frame: np.ndarray = field(compare=False, repr=False)
    index: int
    p1: Phase1Result
    blur_variance: float
    left_ear: Optional[float]
    right_ear: Optional[float]
    rejected_reason: Optional[str] = None      # None == survived Stage 1 (and Stage 2, if set later)
    overall_confidence: Optional[float] = None  # filled in during Stage 2


def stage1_filter(frames: list, detector: FaceDetector) -> list:
    """Runs Phase 1 on every frame and tags the ones that fail the cheap
    blur/eyes-closed checks. Returns a BurstCandidate per input frame,
    survivors and rejects alike, so callers can report on the whole burst."""
    candidates = []
    for i, frame in enumerate(frames):
        p1 = detector.analyze(frame)

        if not p1.success:
            candidates.append(BurstCandidate(
                frame, i, p1, blur_variance=0.0, left_ear=None, right_ear=None,
                rejected_reason="no_face_detected",
            ))
            continue

        blur = _laplacian_variance(frame, p1.bbox)
        left_ear = _ear_score(p1.landmarks, LEFT_EYE_EAR_POINTS)
        right_ear = _ear_score(p1.landmarks, RIGHT_EYE_EAR_POINTS)

        reason = None
        if blur < BURST_FILTER.min_laplacian_variance:
            reason = "blurry"
        elif (
            left_ear is not None and right_ear is not None
            and left_ear <= BURST_FILTER.eyes_closed_ear_score
            and right_ear <= BURST_FILTER.eyes_closed_ear_score
        ):
            reason = "eyes_closed"

        candidates.append(BurstCandidate(
            frame, i, p1, blur, left_ear, right_ear, rejected_reason=reason,
        ))

    return candidates


# ─── Stage 2 ────────────────────────────────────────────────────────────────

def stage2_select_best(survivors: list):
    """Runs the full Phase 2->4 pipeline on every Stage-1 survivor and keeps
    the one with the highest Phase 4 overall_confidence (ties -> sharper
    frame). Returns (best_candidate, p2, p3, p4) or None if every survivor
    failed region extraction."""
    extractor = RegionExtractor()
    analyzer = FeatureAnalyzer()
    engine = ObservationEngine()

    scored = []
    for c in survivors:
        p2 = extractor.extract(c.frame, c.p1)
        if not p2.success:
            c.rejected_reason = "region_extraction_failed"
            continue
        p3 = analyzer.analyze(c.frame, c.p1, p2)
        p4 = engine.generate(p3)
        c.overall_confidence = p4.overall_confidence
        scored.append((c, p2, p3, p4))

    if not scored:
        return None

    scored.sort(key=lambda t: (t[0].overall_confidence, t[0].blur_variance), reverse=True)
    return scored[0]


# ─── Entry point ────────────────────────────────────────────────────────────

def select_best_frame(frames: list):
    """Full two-stage selection over a burst of frames.

    Returns (best_candidate, p1, p2, p3, p4) for the winning frame -- ready
    for the caller to go straight into Phase 5/6 without re-running Phase
    1-4 -- or (None, None, None, None, None) if nothing in the burst was
    usable (e.g. every frame was blurry, or no face was ever detected).
    """
    with FaceDetector() as detector:
        candidates = stage1_filter(frames, detector)

    survivors = [c for c in candidates if c.rejected_reason is None]
    if not survivors:
        return None, None, None, None, None

    result = stage2_select_best(survivors)
    if result is None:
        return None, None, None, None, None

    best, p2, p3, p4 = result
    return best, best.p1, p2, p3, p4
