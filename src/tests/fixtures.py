"""
Synthetic landmark fixtures for region geometry / symmetry tests (A6).

Builds a fake 478-point landmark set positioned on a simple parametric
"face" (an ellipse for the oval, small clusters for eyes/eyebrows/lips/
nose, named single points for nose tip / chin / forehead / eye corners /
cheeks / mouth corners) using the REAL MediaPipe index groups pulled from
face_detector.py — not arbitrary indices — so the region-extraction and
symmetry code under test is exercised with the same index sets it uses
against real MediaPipe output.

Note on asymmetry_px: the symmetry metric measures perpendicular distance
from the forehead-chin vertical midline, so a purely vertical (y-only)
shift has zero effect — asymmetry must be horizontal to move the score.
This is exactly the case the v4.1 symmetry fix targeted (bounding-box
center drift from head yaw on an otherwise symmetric face should NOT lower
the score).
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase1_face_understanding.face_detector import (
    LandmarkPoint, FaceBoundingBox, Phase1Result,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_EYEBROW_IDX, RIGHT_EYEBROW_IDX,
    LIPS_IDX, NOSE_IDX, OVAL_IDX,
    NOSE_TIP, CHIN, FOREHEAD_TOP,
    LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER,
    LEFT_EYE_INNER_CORNER, RIGHT_EYE_INNER_CORNER,
    LEFT_CHEEK_PT, RIGHT_CHEEK_PT, MOUTH_LEFT, MOUTH_RIGHT,
)

IMG_W, IMG_H = 600, 800
FACE_CX, FACE_CY = IMG_W // 2, IMG_H // 2
FACE_RX, FACE_RY = 180, 260


def _all_named_indices():
    return set(
        LEFT_EYE_IDX + RIGHT_EYE_IDX + LEFT_EYEBROW_IDX + RIGHT_EYEBROW_IDX +
        LIPS_IDX + NOSE_IDX + OVAL_IDX +
        [NOSE_TIP, CHIN, FOREHEAD_TOP,
         LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER,
         LEFT_EYE_INNER_CORNER, RIGHT_EYE_INNER_CORNER,
         LEFT_CHEEK_PT, RIGHT_CHEEK_PT,
         MOUTH_LEFT, MOUTH_RIGHT]
    )


def _set(landmarks, idx, x, y):
    landmarks[idx] = LandmarkPoint(x=x / IMG_W, y=y / IMG_H, px=int(x), py=int(y))


def build_synthetic_landmarks(asymmetry_px: int = 0, yaw_shift_px: int = 0):
    """
    Returns a list of LandmarkPoints long enough to cover every real index
    used by Phase 1/2/3, positioned on a simple symmetric (or perturbed)
    ellipse-based face.

    asymmetry_px: shrinks the *right* side's eye/cheek/mouth horizontal
        distance from the midline by this many px, creating genuine facial
        asymmetry (should lower the symmetry score).  Must be horizontal —
        see module docstring.
    yaw_shift_px: shifts the entire oval sideways only, simulating bounding-
        box centre drift from head yaw on an otherwise symmetric face.
        Should NOT lower the real-midline symmetry score.
    """
    max_idx = max(_all_named_indices())
    landmarks = [
        LandmarkPoint(x=0.5, y=0.5, px=FACE_CX, py=FACE_CY)
        for _ in range(max_idx + 1)
    ]

    # ── Face oval ─────────────────────────────────────────────────────────────
    for i, idx in enumerate(OVAL_IDX):
        theta = 2 * math.pi * i / len(OVAL_IDX)
        x = FACE_CX + yaw_shift_px + FACE_RX * math.cos(theta)
        y = FACE_CY + FACE_RY * math.sin(theta)
        _set(landmarks, idx, x, y)

    eye_y   = FACE_CY - FACE_RY * 0.25
    eye_dx  = FACE_RX * 0.45
    cheek_y = FACE_CY + FACE_RY * 0.1
    mouth_y = FACE_CY + FACE_RY * 0.55

    # Right-side distances, optionally shrunk by asymmetry_px
    r_eye_dx   = max(5, eye_dx   - asymmetry_px)
    r_cheek_dx = max(5, FACE_RX * 0.85 - asymmetry_px)
    r_mouth_dx = max(5, FACE_RX * 0.3  - asymmetry_px)

    # ── Rings (eyes / eyebrows / lips / nose) ─────────────────────────────────
    # Placed first; named single points are re-applied below so they are not
    # clobbered by the generic ring placement (several indices are shared
    # between rings and single-point roles in real MediaPipe anatomy).
    def ring(indices, cx, cy, rx, ry):
        n = len(indices)
        for i, idx in enumerate(indices):
            theta = 2 * math.pi * i / max(n, 1)
            x = cx + rx * math.cos(theta)
            y = cy + ry * math.sin(theta)
            _set(landmarks, idx, x, y)

    ring(LEFT_EYE_IDX,       FACE_CX + eye_dx * 0.7,    eye_y,       30, 14)
    ring(RIGHT_EYE_IDX,      FACE_CX - r_eye_dx * 0.7,  eye_y,       30, 14)
    ring(LEFT_EYEBROW_IDX,   FACE_CX + eye_dx * 0.7,    eye_y - 35,  35, 10)
    ring(RIGHT_EYEBROW_IDX,  FACE_CX - r_eye_dx * 0.7,  eye_y - 35,  35, 10)
    ring(LIPS_IDX,           FACE_CX,                    mouth_y,     60, 20)
    ring(NOSE_IDX,           FACE_CX,                    FACE_CY + 30, 25, 35)

    # ── Named single points (re-applied after rings) ───────────────────────────
    _set(landmarks, FOREHEAD_TOP,            FACE_CX,                    FACE_CY - FACE_RY * 0.9)
    _set(landmarks, CHIN,                    FACE_CX,                    FACE_CY + FACE_RY * 0.95)
    _set(landmarks, NOSE_TIP,                FACE_CX,                    FACE_CY)
    _set(landmarks, LEFT_EYE_OUTER_CORNER,   FACE_CX + eye_dx,           eye_y)
    _set(landmarks, LEFT_EYE_INNER_CORNER,   FACE_CX + eye_dx * 0.35,    eye_y)
    _set(landmarks, RIGHT_EYE_OUTER_CORNER,  FACE_CX - r_eye_dx,         eye_y)
    _set(landmarks, RIGHT_EYE_INNER_CORNER,  FACE_CX - r_eye_dx * 0.35,  eye_y)
    _set(landmarks, LEFT_CHEEK_PT,           FACE_CX + FACE_RX * 0.85,   cheek_y)
    _set(landmarks, RIGHT_CHEEK_PT,          FACE_CX - r_cheek_dx,        cheek_y)
    _set(landmarks, MOUTH_LEFT,              FACE_CX + FACE_RX * 0.3,    mouth_y)
    _set(landmarks, MOUTH_RIGHT,             FACE_CX - r_mouth_dx,        mouth_y)

    return landmarks


def build_phase1_result(asymmetry_px: int = 0, yaw_shift_px: int = 0) -> Phase1Result:
    """Full Phase1Result as Phase 2/3/4 expect to receive it."""
    landmarks = build_synthetic_landmarks(asymmetry_px=asymmetry_px, yaw_shift_px=yaw_shift_px)
    oval_xs = [landmarks[i].px for i in OVAL_IDX]
    oval_ys = [landmarks[i].py for i in OVAL_IDX]
    fx, fy  = min(oval_xs), min(oval_ys)
    fw, fh  = max(oval_xs) - fx, max(oval_ys) - fy
    bbox = FaceBoundingBox(x=fx, y=fy, w=fw, h=fh)
    return Phase1Result(
        success=True, bbox=bbox, landmarks=landmarks,
        face_width=fw, face_height=fh,
    )
