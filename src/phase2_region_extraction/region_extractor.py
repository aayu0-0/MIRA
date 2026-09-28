"""

Phase 2 — Region Extraction

Basically, this module extracts the regions of interest (ROIs) from the face image.
Some of these are provided by the face detector, but some are not,so we need to compute them ourselves.
We use basic geometry and the landmarks provided by the face detector.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from phase1_face_understanding.face_detector import (FaceLandmarksConnections,)


@dataclass
class FacialRegion:
    name: str
    crop: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None          # full image-size mask (h_img × w_img); used by visualizer
    mask_crop: Optional[np.ndarray] = None     # bbox-cropped mask (same size as crop/masked_crop)
    masked_crop: Optional[np.ndarray] = None
    bbox: tuple = field(default_factory=tuple)
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

# Basically, mediapipe provides pair of indices describing connections between points
# What we want is a list of points in any order, so we can draw a polygon or a line
def _ordered_chain_points(connections) -> list:
    """
    Walk the MediaPipe connection graph and return an ordered list of unique
    point indices that form a connected chain (or chains, for multi-arc groups
    like the eyebrows).

    Improvement over the original: each connected component is now started from
    a degree-1 node (a chain endpoint) when one exists, so the walk produces
    the natural end-to-end order of the arc rather than starting mid-chain from
    wherever set-iteration happens to land.  For closed loops (no degree-1 node)
    the behaviour is unchanged — any start is equivalent because cv2.convexHull
    re-orders the points geometrically before use.
    """
    edges = {}
    nodes = set()
    for c in connections:
        edges.setdefault(c.start, []).append(c.end)
        edges.setdefault(c.end, []).append(c.start)
        nodes.add(c.start)
        nodes.add(c.end)

    visited = set()
    order = []

    # Prefer degree-1 nodes (chain endpoints) as walk origins so open arcs
    # (e.g. each eyebrow arc) are traversed end-to-end rather than mid-chain.
    def _next_start():
        # First look for an unvisited endpoint in this component
        for n in nodes:
            if n not in visited and len(edges.get(n, [])) == 1:
                return n
        # Fall back to any unvisited node (handles closed loops)
        for n in nodes:
            if n not in visited:
                return n
        return None

    while True:
        start = _next_start()
        if start is None:
            break
        cur = start
        order.append(cur)
        visited.add(cur)
        while True:
            nbrs = [n for n in edges.get(cur, []) if n not in visited]
            if not nbrs:
                break
            cur = nbrs[0]
            order.append(cur)
            visited.add(cur)

    return order

LEFT_EYE_PTS      = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE_PTS     = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
LEFT_EYEBROW_PTS  = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_EYEBROW_PTS = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
LIPS_PTS          = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
NOSE_PTS          = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_NOSE)
OVAL_PTS          = _ordered_chain_points(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)


class RegionExtractor:

# color given to each region for visualization purposes only
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
        if not phase1_result.success or not phase1_result.landmarks:
            return Phase2Result(success=False, error="Phase 1 result invalid.")

        lm = phase1_result.landmarks

# pts is a helper function to get the pixel coordinates of the landmarks given their indices
        def pts(idx_list):
            return np.array(
                [[lm[i].px, lm[i].py] for i in idx_list if i < len(lm)],
                dtype=np.int32
            )

    # eyes, eyebrows, nose and lips are provided by mediapipe, so we can just use them directly       
        regions = {}
        regions["left_eye"]      = self._poly_region(image_bgr, pts(LEFT_EYE_PTS),      "left_eye",      pad=6)
        regions["right_eye"]     = self._poly_region(image_bgr, pts(RIGHT_EYE_PTS),     "right_eye",     pad=6)
        regions["left_eyebrow"]  = self._poly_region(image_bgr, pts(LEFT_EYEBROW_PTS),  "left_eyebrow",  pad=3)
        regions["right_eyebrow"] = self._poly_region(image_bgr, pts(RIGHT_EYEBROW_PTS), "right_eyebrow", pad=3)
        regions["nose"]          = self._poly_region(image_bgr, pts(NOSE_PTS),          "nose",          pad=2)
        regions["lips"]          = self._poly_region(image_bgr, pts(LIPS_PTS),          "lips",          pad=3)

    # The following regions are not provided by mediapipe, so we need to compute them ourselves
        regions["forehead"]    = self._poly_region(image_bgr, self._build_forehead_poly(lm), "forehead", pad=0)
        regions["left_cheek"]  = self._poly_region(image_bgr, self._build_cheek_poly(lm, side="left"),  "left_cheek",  pad=0)
        regions["right_cheek"] = self._poly_region(image_bgr, self._build_cheek_poly(lm, side="right"), "right_cheek", pad=0)
        regions["chin"]        = self._poly_region(image_bgr, self._build_chin_poly(lm), "chin", pad=0)

        viz = self._draw_visualization(image_bgr, regions)
        return Phase2Result(success=True, regions=regions, visualization=viz)
    
# Helper function to get the pixel coordinates of the oval landmarks given their indices   
    def _oval_pts(self, lm) -> np.ndarray:
        return np.array(
            [[lm[i].px, lm[i].py] for i in OVAL_PTS if i < len(lm)],
            dtype=np.int32
        )

#builds forhead polygon based on the oval points and the eyebrow points. Upper part of oval, above eyebrows.
    def _build_forehead_poly(self, lm) -> np.ndarray:
        oval = self._oval_pts(lm)
        brow_top_y = min(lm[i].py for i in (LEFT_EYEBROW_PTS + RIGHT_EYEBROW_PTS) if i < len(lm))
        upper = oval[oval[:, 1] <= brow_top_y]
        if len(upper) < 3:
            upper = oval[oval[:, 1] <= np.percentile(oval[:, 1], 40)]
        return upper   

#builds chin polygon based on the oval points and the mouth points. Lower part of oval, below mouth.
    def _build_chin_poly(self, lm) -> np.ndarray:
        oval = self._oval_pts(lm)
        mouth_bottom_y = max(lm[i].py for i in LIPS_PTS if i < len(lm))
        lower = oval[oval[:, 1] >= mouth_bottom_y]
        if len(lower) < 3:
            lower = oval[oval[:, 1] >= np.percentile(oval[:, 1], 70)]
        return lower     

#builds cheek polygon based on the oval points and the eye points. Side part of oval, between eye and mouth corner.
    def _build_cheek_poly(self, lm, side: str) -> np.ndarray:
        oval = self._oval_pts(lm)
        cx = np.mean([p.px for p in lm])

        eye_idx = LEFT_EYE_PTS if side == "left" else RIGHT_EYE_PTS
        mouth_corner_idx = 291 if side == "left" else 61

        eye_pts = [lm[i] for i in eye_idx if i < len(lm)]
        eye_outer_x = max(p.px for p in eye_pts) if side == "left" else min(p.px for p in eye_pts)
        eye_cy = sum(p.py for p in eye_pts) / len(eye_pts)

        mouth_corner = lm[mouth_corner_idx] if mouth_corner_idx < len(lm) else None
        mouth_y = mouth_corner.py if mouth_corner else eye_cy + 80

        if side == "left":
            outer = oval[(oval[:, 0] >= cx) & (oval[:, 1] >= eye_cy) & (oval[:, 1] <= mouth_y)]
        else:
            outer = oval[(oval[:, 0] <= cx) & (oval[:, 1] >= eye_cy) & (oval[:, 1] <= mouth_y)]

        if len(outer) < 2:
            outer = oval[(oval[:, 1] >= eye_cy) & (oval[:, 1] <= mouth_y)]

        inner_top    = [eye_outer_x, int(eye_cy)]
        inner_bottom = [eye_outer_x, int(mouth_y)]
        return np.array(list(outer) + [inner_top, inner_bottom], dtype=np.int32)

#creates a polygonal region using the given polygon points, crops the region from the image, creates a mask, and returns a FacialRegion object.
    def _poly_region(self, img, polygon: np.ndarray, name: str, pad: int = 0) -> FacialRegion:
        h_img, w_img = img.shape[:2]

        if polygon is None or len(polygon) < 3:
            return FacialRegion(name=name, available=False, error="Insufficient polygon points")

        hull = cv2.convexHull(polygon.astype(np.int32)).reshape(-1, 2)
        if pad > 0:
            hull = self._inflate_polygon(hull, pad)
           
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mask_full, [hull], 255)

        x1, y1 = max(0, hull[:, 0].min()), max(0, hull[:, 1].min())
        x2, y2 = min(w_img, hull[:, 0].max()), min(h_img, hull[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            return FacialRegion(name=name, available=False, error="Empty region")

        crop = img[y1:y2, x1:x2].copy()
        # mask_full is kept for the visualizer (_draw_visualization indexes into
        # the full image with it).  mask_crop is the bbox-sized slice that
        # feature_analyzer needs — previously this was discarded, leaving
        # region.mask as a full h_img×w_img array that mismatched the crop dims.
        mask_crop = mask_full[y1:y2, x1:x2].copy()
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask_crop)

        return FacialRegion(
            name=name, crop=crop,
            mask=mask_full,          # full-size; visualizer uses this
            mask_crop=mask_crop,     # crop-size; feature_analyzer uses this
            masked_crop=masked_crop,
            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), available=True
        )       

# Helper function to inflate a polygon by a given padding value.
    def _inflate_polygon(self, polygon: np.ndarray, pad: int) -> np.ndarray:
        centroid = polygon.mean(axis=0)
        vecs = polygon - centroid
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        inflated = polygon + (vecs / norms) * pad
        return inflated.astype(np.int32)

# Helper function to draw the regions on the image for visualization purposes.
    def _draw_visualization(self, image: np.ndarray, regions: dict) -> np.ndarray:
        overlay = image.copy()

        for name, region in regions.items():
            if not region.available or region.mask is None:
                continue

            color = self.REGION_COLORS.get(name, (200, 200, 200))

            colored = np.zeros_like(image)
            colored[region.mask > 0] = color

            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.30, 0)

            bx, by, bw, bh = region.bbox
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.putText(overlay, name, (bx + 2, by + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

        return overlay