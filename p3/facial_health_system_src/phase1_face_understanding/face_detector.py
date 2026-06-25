"""
Phase 1 — Face Understanding
Uses OpenCV Haar cascades + geometry to produce landmarks, symmetry, and alignment.
No external model downloads required.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Data contracts ───────────────────────────────────────────────────────────

@dataclass
class FaceBoundingBox:
    x: int; y: int; w: int; h: int; confidence: float
    @property
    def center(self): return (self.x + self.w // 2, self.y + self.h // 2)

@dataclass
class LandmarkPoint:
    x: float; y: float; z: float = 0.0
    px: int = 0; py: int = 0

@dataclass
class Phase1Result:
    success: bool
    bbox: Optional[FaceBoundingBox] = None
    landmarks: list = field(default_factory=list)
    symmetry_score: float = 0.0
    symmetry_label: str = ""
    alignment_angle: float = 0.0
    face_height: int = 0
    face_width: int = 0
    error: str = ""
    # Extra: raw eye rects, mouth rect for downstream use
    eye_rects: list = field(default_factory=list)   # [(x,y,w,h), ...]
    nose_rect: Optional[tuple] = None
    mouth_rect: Optional[tuple] = None


class FaceDetector:
    """OpenCV-based face detector with 68-point approximated landmark grid."""

    def __init__(self):
        self._face_cas  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._face_alt  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self._eye_cas   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self._smile_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self._nose_cas  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

    # ── public ────────────────────────────────────────────────────────────────

    def analyze(self, image_bgr: np.ndarray) -> Phase1Result:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # Detect face
        faces = self._face_cas.detectMultiScale(gray_eq, 1.1, 5, minSize=(60,60))
        if len(faces) == 0:
            faces = self._face_alt.detectMultiScale(gray_eq, 1.1, 3, minSize=(60,60))
        if len(faces) == 0:
            return Phase1Result(success=False, error="No face detected in image.")

        # Use largest face
        fx, fy, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        bbox = FaceBoundingBox(x=fx, y=fy, w=fw, h=fh, confidence=0.9)

        face_gray = gray[fy:fy+fh, fx:fx+fw]
        face_eq   = gray_eq[fy:fy+fh, fx:fx+fw]

        # Detect eyes within face ROI
        eyes = self._eye_cas.detectMultiScale(face_eq, 1.1, 5, minSize=(20,20))
        eyes_abs = [(fx+ex, fy+ey, ew, eh) for (ex,ey,ew,eh) in eyes]

        # Detect mouth
        mouth_region = face_gray[fh//2:, :]
        smiles = self._smile_cas.detectMultiScale(mouth_region, 1.7, 20,
                                                   minSize=(fw//4, fh//8))
        mouth_abs = None
        if len(smiles) > 0:
            mx, my, mw, mh = smiles[0]
            mouth_abs = (fx+mx, fy+fh//2+my, mw, mh)

        # Build approximated 68-style landmark grid from geometry
        landmarks = self._build_landmarks(fx, fy, fw, fh, eyes_abs, mouth_abs, w, h)

        # Symmetry
        sym_score, sym_label = self._compute_symmetry(fx, fy, fw, fh, eyes_abs)

        # Alignment angle
        angle = self._compute_angle(eyes_abs, fx, fw)

        return Phase1Result(
            success=True,
            bbox=bbox,
            landmarks=landmarks,
            symmetry_score=sym_score,
            symmetry_label=sym_label,
            alignment_angle=angle,
            face_height=fh,
            face_width=fw,
            eye_rects=eyes_abs,
            mouth_rect=mouth_abs,
        )

    def draw_landmarks(self, image_bgr: np.ndarray, result: Phase1Result,
                       draw_mesh=True, draw_bbox=True, draw_key_points=True) -> np.ndarray:
        out = image_bgr.copy()
        if not result.success:
            return out

        if draw_bbox:
            b = result.bbox
            cv2.rectangle(out, (b.x, b.y), (b.x+b.w, b.y+b.h), (0,255,100), 2)

        # Draw landmark points
        if draw_key_points:
            for i, lm in enumerate(result.landmarks):
                color = (0, 200, 255) if i in (30, 8, 0, 16, 36, 45, 48, 54) else (100, 180, 255)
                cv2.circle(out, (lm.px, lm.py), 2, color, -1)

        # Draw eye rects
        for (ex, ey, ew, eh) in result.eye_rects:
            cv2.rectangle(out, (ex, ey), (ex+ew, ey+eh), (255, 100, 0), 1)

        # Draw mouth rect
        if result.mouth_rect:
            mx, my, mw, mh = result.mouth_rect
            cv2.rectangle(out, (mx, my), (mx+mw, my+mh), (0, 80, 255), 1)

        # Symmetry midline
        b = result.bbox
        mid_x = b.x + b.w // 2
        cv2.line(out, (mid_x, b.y), (mid_x, b.y+b.h), (200,200,0), 1)

        label = f"Symmetry: {result.symmetry_score:.1f}%  |  {result.symmetry_label}"
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 1)
        return out

    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # ── landmark grid ─────────────────────────────────────────────────────────

    def _build_landmarks(self, fx, fy, fw, fh, eyes, mouth, img_w, img_h) -> list:
        """
        Approximate 68-style landmark positions from face geometry.
        Returns list of LandmarkPoint (index-compatible with region extractor).
        We generate 68 geometrically-spaced points covering all facial zones.
        """
        pts = []

        # Helper
        def pt(x, y):
            nx, ny = x / img_w, y / img_h
            pts.append(LandmarkPoint(x=nx, y=ny, z=0.0, px=int(x), py=int(y)))

        # Jaw line (points 0–16): 17 points along jaw
        for i in range(17):
            t = i / 16
            x = fx + fw * (0.05 + 0.90 * t)
            y = fy + fh * (0.60 + 0.40 * math.sin(math.pi * t))
            pt(x, y)

        # Eyebrows (17–21 left, 22–26 right)
        brow_y = fy + fh * 0.28
        for i in range(5):
            pt(fx + fw*(0.12 + 0.16*i/4), brow_y)       # left brow
        for i in range(5):
            pt(fx + fw*(0.52 + 0.16*i/4), brow_y)       # right brow

        # Nose bridge (27–30) + nostrils (31–35)
        nose_cx = fx + fw * 0.50
        for i in range(4):
            pt(nose_cx, fy + fh*(0.38 + 0.12*i/3))
        for i in range(5):
            pt(fx + fw*(0.34 + 0.08*i/4), fy + fh*0.56)

        # Eyes
        if len(eyes) >= 2:
            # Sort eyes left to right
            eyes_sorted = sorted(eyes[:2], key=lambda e: e[0])
            for e in eyes_sorted:
                ex, ey, ew, eh = e
                cx, cy = ex + ew//2, ey + eh//2
                for i in range(6):
                    a = math.pi * i / 5
                    px_ = cx + int(ew//2 * math.cos(a))
                    py_ = cy + int(eh//2 * math.sin(a) * 0.5)
                    pt(px_, py_)
        else:
            # Fallback: geometric estimate
            for side in [0.22, 0.65]:
                cx = fx + int(fw * side)
                cy = fy + int(fh * 0.38)
                for i in range(6):
                    a = math.pi * i / 5
                    pt(cx + int(fw*0.08*math.cos(a)), cy + int(fh*0.04*math.sin(a)))

        # Mouth (48–59)
        if mouth:
            mx, my, mw, mh = mouth
            cx, cy = mx + mw//2, my + mh//2
            for i in range(12):
                a = 2 * math.pi * i / 12
                pt(cx + int(mw//2*math.cos(a)), cy + int(mh//2*math.sin(a)))
        else:
            cx = fx + fw//2
            cy = fy + int(fh * 0.74)
            for i in range(12):
                a = 2 * math.pi * i / 12
                pt(cx + int(fw*0.12*math.cos(a)), cy + int(fh*0.05*math.sin(a)))

        # Inner lips (60–67): 8 points inside mouth
        cx = fx + fw//2
        cy = fy + int(fh * 0.74)
        for i in range(8):
            a = 2 * math.pi * i / 8
            pt(cx + int(fw*0.08*math.cos(a)), cy + int(fh*0.03*math.sin(a)))

        return pts

    # ── symmetry ──────────────────────────────────────────────────────────────

    def _compute_symmetry(self, fx, fy, fw, fh, eyes) -> tuple:
        mid_x = fx + fw // 2
        scores = []

        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes[:2], key=lambda e: e[0])
            el = eyes_sorted[0]; er = eyes_sorted[1]
            cl_x = el[0] + el[2]//2
            cr_x = er[0] + er[2]//2
            d_left  = abs(cl_x - mid_x)
            d_right = abs(cr_x - mid_x)
            if d_left + d_right > 0:
                asym = abs(d_left - d_right) / ((d_left + d_right) / 2)
                scores.append(asym)
            # Eye size ratio
            size_ratio = min(el[2]*el[3], er[2]*er[3]) / max(el[2]*el[3], er[2]*er[3])
            scores.append(1 - size_ratio)

        # Fallback geometric score
        if not scores:
            scores = [0.05]

        mean_asym = float(np.mean(scores))
        score = max(0.0, min(100.0, (1 - mean_asym / 0.30) * 100))

        if score >= 90:   label = "Highly Symmetric"
        elif score >= 75: label = "Normal Symmetry"
        elif score >= 55: label = "Mild Asymmetry"
        elif score >= 35: label = "Moderate Asymmetry"
        else:             label = "Notable Asymmetry"

        return round(score, 1), label

    def _compute_angle(self, eyes, fx, fw) -> float:
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes[:2], key=lambda e: e[0])
            el, er = eyes_sorted
            cl = (el[0]+el[2]//2, el[1]+el[3]//2)
            cr = (er[0]+er[2]//2, er[1]+er[3]//2)
            dx, dy = cr[0]-cl[0], cr[1]-cl[1]
            return math.degrees(math.atan2(dy, dx))
        return 0.0
