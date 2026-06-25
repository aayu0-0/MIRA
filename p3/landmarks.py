"""
Stage 2: Facial Landmarks
Wraps MediaPipe's FaceMesh solution (468 points, +10 iris points if
refine_landmarks=True -> 478 total). Outputs landmarks in pixel
coordinates (not normalized) so every downstream stage works directly
in image-space without re-deriving width/height each time.
"""

import cv2
import numpy as np
import mediapipe as mp


class FaceLandmarker:
    def __init__(self, max_faces: int = 1, refine_landmarks: bool = True, min_confidence: float = 0.5):
        self._mp_fm = mp.solutions.face_mesh
        self._mesh = self._mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,  # adds iris landmarks, needed for under-eye precision
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def get_landmarks(self, face_bgr):
        """
        Returns an (N, 3) array of (x_px, y_px, z_rel) landmark points for the
        first detected face, or None if no face/landmarks found.
        z is MediaPipe's relative depth (negative = closer to camera), not
        true metric depth -- usable for relative comparisons, not absolute distance.
        """
        h, w = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
        return pts

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- Landmark index groups: OFFICIAL regions ---
# Pulled directly from mediapipe.python.solutions.face_mesh_connections at
# import time -- not hand-typed -- so these are guaranteed correct and will
# never silently drift from whatever MediaPipe version is installed.
#
# IMPORTANT naming convention: "left"/"right" follow MediaPipe's convention,
# which is the SUBJECT's own left/right (anatomical), not image-left/image-right.
# In a normal (non-mirrored) frontal photo, the subject's left eye appears on
# the RIGHT side of the image. (Earlier draft of this file had these swapped --
# fixed by sourcing straight from the library instead of hand-typing.)


_fmc = mp.solutions.face_mesh_connections


def _point_set(connections) -> list[int]:
    """Flatten a frozenset of (a, b) edge-pairs into a sorted unique point list."""
    return sorted(set(i for pair in connections for i in pair))


LEFT_EYE = _point_set(_fmc.FACEMESH_LEFT_EYE)
RIGHT_EYE = _point_set(_fmc.FACEMESH_RIGHT_EYE)
LEFT_EYEBROW = _point_set(_fmc.FACEMESH_LEFT_EYEBROW)
RIGHT_EYEBROW = _point_set(_fmc.FACEMESH_RIGHT_EYEBROW)
NOSE = _point_set(_fmc.FACEMESH_NOSE)
MOUTH = _point_set(_fmc.FACEMESH_LIPS)
FACE_OVAL = _point_set(_fmc.FACEMESH_FACE_OVAL)
LEFT_IRIS = _point_set(_fmc.FACEMESH_LEFT_IRIS)     # only valid if refine_landmarks=True
RIGHT_IRIS = _point_set(_fmc.FACEMESH_RIGHT_IRIS)   # only valid if refine_landmarks=True

# --- Landmark index groups: HEURISTIC regions ---
# Cheeks/forehead/chin are not discrete features MediaPipe defines a contour
# for (they're skin patches, not anatomical boundaries) -- so there's no
# "official" answer here. Every point below is still taken from one of the
# official groups above (eye corners, nose-side, mouth corners, face-oval),
# composed into a small polygon per skin patch. Flagged as a design choice,
# not verified ground truth -- will sanity-check visually once rendered.

LEFT_CHEEK = [226, 31, 228, 117, 213, 187, 50, 101, 142]      # between left eye, nose, mouth corner, jaw
RIGHT_CHEEK = [446, 261, 448, 346, 433, 411, 280, 330, 371]   # mirror of the above on the right

FOREHEAD = sorted(set(FACE_OVAL))  # kept for reference only; real extraction is dynamic, see regions.py

CHIN = [152, 171, 175, 396, 148, 377]   # around the official chin point (152)
CHIN_POINT = 152
NOSE_TIP = 4

FACE_MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 17, 18, 200, 199, 175, 152]

# All extractable regions, mapped to their landmark index groups.
# regions.py builds masks/crops directly from this table.
REGION_GROUPS = {
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "left_eyebrow": LEFT_EYEBROW,
    "right_eyebrow": RIGHT_EYEBROW,
    "nose": NOSE,
    "mouth": MOUTH,
    "left_cheek": LEFT_CHEEK,
    "right_cheek": RIGHT_CHEEK,
    "forehead": FOREHEAD,
    "chin": CHIN,
}
