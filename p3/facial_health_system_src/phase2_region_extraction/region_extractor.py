"""
Phase 2 — Region Extraction
Extracts individual facial regions using face bounding box geometry + detected features.
Works with OpenCV-based Phase1Result (no MediaPipe required).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FacialRegion:
    name: str
    crop: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    masked_crop: Optional[np.ndarray] = None
    bbox: tuple = field(default_factory=tuple)   # (x, y, w, h)
    available: bool = False
    error: str = ""


@dataclass
class Phase2Result:
    success: bool
    regions: dict = field(default_factory=dict)
    visualization: Optional[np.ndarray] = None
    error: str = ""

    def get(self, name: str) -> Optional[FacialRegion]:
        return self.regions.get(name)


class RegionExtractor:

    REGION_COLORS = {
        "left_eye":      (255, 100,   0),
        "right_eye":     (255, 100,   0),
        "left_eyebrow":  (200,   0, 200),
        "right_eyebrow": (200,   0, 200),
        "nose":          (  0, 200, 255),
        "lips":          (  0,  80, 255),
        "forehead":      ( 50, 200,  50),
        "left_cheek":    (255, 200,   0),
        "right_cheek":   (255, 200,   0),
        "chin":          (100, 255, 200),
    }

    def extract(self, image_bgr: np.ndarray, phase1_result) -> Phase2Result:
        """Extract regions from the face using bounding-box geometry."""
        h_img, w_img = image_bgr.shape[:2]

        if not phase1_result.success or phase1_result.bbox is None:
            return Phase2Result(success=False, error="Phase 1 result invalid.")

        b = phase1_result.bbox
        fx, fy, fw, fh = b.x, b.y, b.w, b.h
        eyes = phase1_result.eye_rects
        mouth = phase1_result.mouth_rect

        regions = {}

        # Sort eyes: left is lower x
        left_eye_rect = right_eye_rect = None
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes[:2], key=lambda e: e[0])
            left_eye_rect  = sorted_eyes[0]
            right_eye_rect = sorted_eyes[1]
        elif len(eyes) == 1:
            cx = eyes[0][0] + eyes[0][2]//2
            if cx < fx + fw//2:
                left_eye_rect = eyes[0]
            else:
                right_eye_rect = eyes[0]

        # Forehead
        regions["forehead"] = self._rect_region(
            image_bgr,
            (fx + int(fw*0.10), fy, int(fw*0.80), int(fh*0.28)),
            "forehead", h_img, w_img
        )

        # Eyes + eyebrows
        regions["left_eye"]      = self._eye_region(image_bgr, left_eye_rect,  "left_eye",  fx, fy, fw, fh, h_img, w_img, side="left")
        regions["right_eye"]     = self._eye_region(image_bgr, right_eye_rect, "right_eye", fx, fy, fw, fh, h_img, w_img, side="right")
        regions["left_eyebrow"]  = self._rect_region(image_bgr,
            (fx + int(fw*0.08), fy + int(fh*0.18), int(fw*0.32), int(fh*0.13)), "left_eyebrow",  h_img, w_img)
        regions["right_eyebrow"] = self._rect_region(image_bgr,
            (fx + int(fw*0.58), fy + int(fh*0.18), int(fw*0.32), int(fh*0.13)), "right_eyebrow", h_img, w_img)

        # Nose
        regions["nose"] = self._rect_region(image_bgr,
            (fx + int(fw*0.32), fy + int(fh*0.38), int(fw*0.36), int(fh*0.24)), "nose", h_img, w_img)

        # Lips / mouth
        if mouth:
            mx, my, mw, mh = mouth
            # Add padding
            regions["lips"] = self._rect_region(image_bgr,
                (max(0, mx - 5), max(0, my - 4), mw + 10, mh + 8), "lips", h_img, w_img)
        else:
            regions["lips"] = self._rect_region(image_bgr,
                (fx + int(fw*0.28), fy + int(fh*0.68), int(fw*0.44), int(fh*0.14)), "lips", h_img, w_img)

        # Cheeks
        regions["left_cheek"]  = self._rect_region(image_bgr,
            (fx + int(fw*0.04), fy + int(fh*0.44), int(fw*0.28), int(fh*0.28)), "left_cheek",  h_img, w_img)
        regions["right_cheek"] = self._rect_region(image_bgr,
            (fx + int(fw*0.68), fy + int(fh*0.44), int(fw*0.28), int(fh*0.28)), "right_cheek", h_img, w_img)

        # Chin
        regions["chin"] = self._rect_region(image_bgr,
            (fx + int(fw*0.25), fy + int(fh*0.78), int(fw*0.50), int(fh*0.20)), "chin", h_img, w_img)

        viz = self._draw_visualization(image_bgr, regions)
        return Phase2Result(success=True, regions=regions, visualization=viz)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _eye_region(self, img, eye_rect, name, fx, fy, fw, fh, h_img, w_img, side="left"):
        if eye_rect is not None:
            ex, ey, ew, eh = eye_rect
            pad = 6
            return self._rect_region(img, (max(0,ex-pad), max(0,ey-pad), ew+pad*2, eh+pad*2),
                                     name, h_img, w_img)
        # Geometric fallback
        if side == "left":
            return self._rect_region(img,
                (fx + int(fw*0.08), fy + int(fh*0.28), int(fw*0.33), int(fh*0.16)),
                name, h_img, w_img)
        else:
            return self._rect_region(img,
                (fx + int(fw*0.58), fy + int(fh*0.28), int(fw*0.33), int(fh*0.16)),
                name, h_img, w_img)

    def _rect_region(self, img, rect, name, h_img, w_img) -> FacialRegion:
        x, y, w, h = rect
        x1 = max(0, x); y1 = max(0, y)
        x2 = min(w_img, x+w); y2 = min(h_img, y+h)
        if x2 <= x1 or y2 <= y1:
            return FacialRegion(name=name, available=False, error="Empty region")

        crop = img[y1:y2, x1:x2].copy()
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        mask_full[y1:y2, x1:x2] = 255
        mask_crop = mask_full[y1:y2, x1:x2]
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask_crop)

        return FacialRegion(
            name=name, crop=crop, mask=mask_full,
            masked_crop=masked_crop,
            bbox=(x1, y1, x2-x1, y2-y1), available=True
        )

    def _draw_visualization(self, image: np.ndarray, regions: dict) -> np.ndarray:
        overlay = image.copy()
        for name, region in regions.items():
            if not region.available or region.mask is None:
                continue
            color = self.REGION_COLORS.get(name, (200,200,200))
            colored = np.zeros_like(image)
            colored[region.mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.30, 0)
            bx, by, bw, bh = region.bbox
            label = name.replace("_", " ").title()
            cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), color, 1)
            cv2.putText(overlay, label, (bx+2, by+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)
        return overlay
