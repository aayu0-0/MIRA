"""
Phase 1 — Face Understanding

MediaPipe's Face Landmarker model detects the facial landmarks of a face in an Image or video.


I/P=> Image or video containing a face.
Converts into 478 landmark points of the face.
A bounding box is drawn around the face and the landmark points are plotted on the face.
It also crops out the different parts of the face like eyes, nose, lips, etc. and saves them as separate images.


Model to Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
Then add it to the path: src/phase1_face_understanding/face_landmarker.task


"""

import os
import sys
import time
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from thresholds import SYMMETRY

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")





"""
basically, the below dataclasses are used to store the results of the face detection and landmark detection process.
"""

@dataclass
class FaceBoundingBox:
    """
    Makes a box around the face detected in the image and stores the coordinates of the box.
    """
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

@dataclass
class LandmarkPoint:
    """
    The pixels coordinates of the landmark points detected on the face.
    """
    x: float
    y: float
    z: float = 0.0
    px: int = 0
    py: int = 0
    # Sub-pixel (unrounded) pixel coordinates. px/py are rounded to int for
    # code that indexes into image arrays or builds cv2 mask polygons, but
    # that rounding collapses genuinely-small real distances (e.g. a closed
    # mouth's upper/lower lip gap, often under 1px) to exactly 0 -- which
    # then collides with div-by-zero fallbacks downstream. Use fx/fy instead
    # of px/py for any calculation that measures a *distance* between two
    # nearby landmarks rather than indexing into pixel data.
    fx: float = 0.0
    fy: float = 0.0

@dataclass
class Phase1Result:
    """
    Stores the results of the face detection and landmark detection process.
    """
    success: bool
    bbox: Optional[FaceBoundingBox] = None
    landmarks: list = field(default_factory=list)
    symmetry_score: float = 0.0
    symmetry_label: str = ""
    alignment_angle: float = 0.0
    face_height: int = 0
    face_width: int = 0
    error: str = ""
    eye_rects: list = field(default_factory=list)
    nose_rect: Optional[tuple] = None
    mouth_rect: Optional[tuple] = None



_FLC = mp_vision.FaceLandmarksConnections 
# This tells the mesh coordinates of the face and tells how they are connected to each other.


def _index_pool(connections):
    """Flatten a set of (start, end) connection pairs into a sorted point list."""
    return sorted({c.start for c in connections} | {c.end for c in connections})


LEFT_EYE_IDX = _index_pool(_FLC.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE_IDX = _index_pool(_FLC.FACE_LANDMARKS_RIGHT_EYE)
LEFT_EYEBROW_IDX = _index_pool(_FLC.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_EYEBROW_IDX = _index_pool(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
LIPS_IDX = _index_pool(_FLC.FACE_LANDMARKS_LIPS)
NOSE_IDX = _index_pool(_FLC.FACE_LANDMARKS_NOSE)
OVAL_IDX = _index_pool(_FLC.FACE_LANDMARKS_FACE_OVAL)

FaceLandmarksConnections = _FLC  # so that Phase 2 can walk the graph itself if needed


NOSE_TIP = 1
CHIN = 152
FOREHEAD_TOP = 10
LEFT_EYE_OUTER_CORNER = 263    # anatomical left (viewer's right)
RIGHT_EYE_OUTER_CORNER = 33
LEFT_EYE_INNER_CORNER = 362
RIGHT_EYE_INNER_CORNER = 133
LEFT_CHEEK_PT = 234
RIGHT_CHEEK_PT = 454
MOUTH_LEFT = 291
MOUTH_RIGHT = 61
UPPER_LIP_CENTER = 13   # inner upper-lip, bottom-center
LOWER_LIP_CENTER = 14   # inner lower-lip, top-center

# Sanity check at import time: if these ever drift outside the lips group
# (e.g. after a MediaPipe index remap), fail loudly instead of silently
# scoring garbage downstream.
assert UPPER_LIP_CENTER in LIPS_IDX, "UPPER_LIP_CENTER index is outside MediaPipe's lips group."
assert LOWER_LIP_CENTER in LIPS_IDX, "LOWER_LIP_CENTER index is outside MediaPipe's lips group."


class FaceDetector:
    """Thin wrapper around MediaPipe FaceLandmarker (478-point Tasks API)."""

    def __init__(
        self,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.8,
        min_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.8,
        model_path: str = _DEFAULT_MODEL_PATH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {model_path}\n"
                "Download it from:\n"
                "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task\n"
                "and place it at that path (or pass model_path=...)."
            )

        self._running_mode = (
            mp_vision.RunningMode.IMAGE if static_image_mode else mp_vision.RunningMode.VIDEO
        )

        option_kwargs = dict(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=self._running_mode,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
        )
        # min_tracking_confidence only has an effect in VIDEO/LIVE_STREAM mode —
        # MediaPipe silently ignores it in IMAGE mode, so leave it out there to
        # avoid implying it does something it doesn't.
        if self._running_mode != mp_vision.RunningMode.IMAGE:
            option_kwargs["min_tracking_confidence"] = min_tracking_confidence

        options = mp_vision.FaceLandmarkerOptions(**option_kwargs)
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        # Monotonic clock reference for VIDEO-mode timestamps. MediaPipe just needs
        # strictly increasing millisecond timestamps that reflect real elapsed time —
        # it doesn't care what they're relative to, as long as they never go backwards.
        self._start_ns = time.perf_counter_ns()


    def analyze(self, image_bgr: np.ndarray) -> Phase1Result:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self._running_mode == mp_vision.RunningMode.IMAGE:
            result = self._landmarker.detect(mp_image)
        else:
            timestamp_ms = (time.perf_counter_ns() - self._start_ns) // 1_000_000
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return Phase1Result(success=False, error="No face detected in image.")

        raw = result.face_landmarks[0]
        landmarks = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z, px=int(lm.x * w), py=int(lm.y * h),
                          fx=lm.x * w, fy=lm.y * h)
            for lm in raw
        ]

        oval_xs = [landmarks[i].px for i in OVAL_IDX]
        oval_ys = [landmarks[i].py for i in OVAL_IDX]
        fx = max(0, min(oval_xs))
        fy = max(0, min(oval_ys))
        fw = min(w, max(oval_xs)) - fx
        fh = min(h, max(oval_ys)) - fy
        bbox = FaceBoundingBox(x=fx, y=fy, w=fw, h=fh)

        left_eye_rect = self._rect_from_indices(landmarks, LEFT_EYE_IDX, w, h, pad=6)
        right_eye_rect = self._rect_from_indices(landmarks, RIGHT_EYE_IDX, w, h, pad=6)
        eye_rects = [r for r in (left_eye_rect, right_eye_rect) if r is not None]
        mouth_rect = self._rect_from_indices(landmarks, LIPS_IDX, w, h, pad=4)
        nose_rect = self._rect_from_indices(landmarks, NOSE_IDX, w, h, pad=4)

        sym_score, sym_label = self._compute_symmetry(landmarks)
        angle = self._compute_angle(landmarks)

        return Phase1Result(
            success=True,
            bbox=bbox,
            landmarks=landmarks,
            symmetry_score=sym_score,
            symmetry_label=sym_label,
            alignment_angle=angle,
            face_height=fh,
            face_width=fw,
            eye_rects=eye_rects,
            nose_rect=nose_rect,
            mouth_rect=mouth_rect,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    # This part basically computes the bounding box around a set of landmark indices.
    def _rect_from_indices(self, landmarks, indices, img_w, img_h, pad=0):
        """Tight padded (x, y, w, h) rect around a set of landmark indices."""
        xs = [landmarks[i].px for i in indices]
        ys = [landmarks[i].py for i in indices]
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(img_w, max(xs) + pad)
        y2 = min(img_h, max(ys) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2 - x1, y2 - y1)


    # This part displays the symmetry score of the face and displays some important landmark points on the face. It also draws a bounding box around the face.
    # staticmethod: doesn't touch self._landmarker, so callers who already have
    # a Phase1Result (e.g. from burst_selection) don't need to spin up a second
    # FaceDetector (and reload the model) just to draw on the winning frame.
    @staticmethod
    def draw_landmarks(image_bgr: np.ndarray, result: Phase1Result,
                        draw_mesh=True, draw_bbox=True, draw_key_points=True) -> np.ndarray:
        out = image_bgr.copy()
        if not result.success:
            return out

        if draw_bbox:
            b = result.bbox
            cv2.rectangle(out, (b.x, b.y), (b.x+b.w, b.y+b.h), (0,255,100), 2)

        if draw_mesh:
            for lm in result.landmarks:
                cv2.circle(out, (lm.px, lm.py), 1, (100, 180, 255), -1)

        if draw_key_points:
            highlight = {NOSE_TIP, CHIN, LEFT_EYE_OUTER_CORNER,
                        RIGHT_EYE_OUTER_CORNER, MOUTH_LEFT, MOUTH_RIGHT,
                        UPPER_LIP_CENTER, LOWER_LIP_CENTER}
            for i in highlight:
                lm = result.landmarks[i]
                cv2.circle(out, (lm.px, lm.py), 3, (0, 200, 255), -1)

        b = result.bbox
        mid_x = b.x + b.w // 2
        cv2.line(out, (mid_x, b.y), (mid_x, b.y+b.h), (200,200,0), 1)

        label = f"Symmetry: {result.symmetry_score:.1f}%  |  {result.symmetry_label}"
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1)
        return out


    # This part computes the symmetry score of the face based on the landmark points detected on the face.
    def _compute_symmetry(self, landmarks) -> tuple:
        top = landmarks[FOREHEAD_TOP]
        chin = landmarks[CHIN]
        mx1, my1 = top.px, top.py
        mx2, my2 = chin.px, chin.py
        dx, dy = mx2 - mx1, my2 - my1
        norm = math.hypot(dx, dy) or 1.0

        def signed_dist(px, py):
            return ((px - mx1) * dy - (py - my1) * dx) / norm

        pairs = [
            (LEFT_EYE_OUTER_CORNER, RIGHT_EYE_OUTER_CORNER),
            (LEFT_EYE_INNER_CORNER, RIGHT_EYE_INNER_CORNER),
            (LEFT_CHEEK_PT, RIGHT_CHEEK_PT),
            (MOUTH_LEFT, MOUTH_RIGHT),
        ]

        scores = []
        for li, ri in pairs:
            if li >= len(landmarks) or ri >= len(landmarks):
                continue
            l_pt, r_pt = landmarks[li], landmarks[ri]
            dl = abs(signed_dist(l_pt.px, l_pt.py))
            dr = abs(signed_dist(r_pt.px, r_pt.py))
            if dl + dr > 0:
                scores.append(abs(dl - dr) / ((dl + dr) / 2))

        if not scores:
            scores = [0.05]

        mean_asym = float(np.mean(scores))
        score = max(0.0, min(100.0, (1 - mean_asym / SYMMETRY.divisor) * 100))

        if score >= SYMMETRY.highly_symmetric:
            label = "Highly Symmetric"
        elif score >= SYMMETRY.normal:
            label = "Normal Symmetry"
        elif score >= SYMMETRY.mild_asymmetry:
            label = "Mild Asymmetry"
        elif score >= SYMMETRY.moderate_asymmetry:
            label = "Moderate Asymmetry"
        else:
            label = "Notable Asymmetry"

        return round(score, 1), label

    # This part computes the angle of the face based on the landmark points detected on the face.
    def _compute_angle(self, landmarks) -> float:
        l_corner = landmarks[LEFT_EYE_OUTER_CORNER]
        r_corner = landmarks[RIGHT_EYE_OUTER_CORNER]
        dx = l_corner.px - r_corner.px
        dy = l_corner.py - r_corner.py
        return math.degrees(math.atan2(dy, dx))


    # end functions
    def close(self):    
        self._landmarker.close()
    def __enter__(self):   
        return self
    def __exit__(self, *_):
        self.close()