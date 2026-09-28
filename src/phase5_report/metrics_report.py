"""
Phase 5 — Metrics Report (File 1 of 4)
---------------------------------------
Produces a single JSON file with the raw numbers computed in Phase 1
(face geometry) and Phase 3 (feature analysis) — grouped by region,
no thresholds, no severity bands, no wrapping. Just the measured values.

Example output shape:
{
    "Face": {
        "Symmetry": 88.0,
        "Angle": 8.0,
        "Width": 512,
        "Height": 480,
        "Aspect Ratio": 0.938,
        "Swelling": 12.3
    },
    "Eyes": {
        "Left":  { "Openness": 84.0, "Redness": 5.2, "Dark Circle": 12.3, "Puffiness": 7.0 },
        "Right": { "Openness": 82.5, "Redness": 4.8, "Dark Circle": 11.9, "Puffiness": 6.4 }
    },
    "Skin": {
        "Forehead":    { "Acne": 3.0, "Redness": 6.1, "Texture": 9.4, "Spot Count": 2 },
        "Left Cheek":  { ... },
        "Right Cheek": { ... },
        "Nose":        { ... }
    },
    "Lips": { "Dryness": 14.2, "Color Consistency": 5.6, "Pallor": 8.1 },
    "Mouth": { "Aspect Ratio": 62.0, "Corner Asymmetry": 3.2, "Curvature": 1.5 }
}
"""

import json
import datetime
from pathlib import Path

import numpy as np


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _n(value, decimals=1):
    """Round a raw numeric value; pass through None untouched."""
    if value is None:
        return None
    return round(float(value), decimals)


class MetricsReportGenerator:

    def generate(self,
                 p1, p3, p4,
                 quality_warnings: list = None,
                 patient_id: str = "UNKNOWN",
                 output_dir: str = None) -> dict:
        """
        Build the flat raw-numbers JSON from Phase 1 + Phase 3 results.
        p4 and quality_warnings are passed through as plain, unscored
        context (not wrapped/banded) for an AI reading the numbers
        alongside them — see "Overall" and "quality_warnings" in the output.

        Returns the dict. If output_dir is given, also writes
        metrics_{patient_id}.json there.
        """
        quality_warnings = quality_warnings or []
        payload = self._build(p1, p3, p4, quality_warnings, patient_id)

        out_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path = f"{output_dir}/metrics_{patient_id}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, cls=_NpEncoder)

        return {"data": payload, "path": out_path}

    # ── builder ───────────────────────────────────────────────────────────────

    def _build(self, p1, p3, p4, quality_warnings, patient_id) -> dict:
        aspect_ratio = (p1.face_height / p1.face_width) if p1.face_width else None

        # ── Face ─────────────────────────────────────────────────────────────
        face = {
            "Symmetry":     _n(p1.symmetry_score),
            "Angle":        _n(p1.alignment_angle),
            "Width":        p1.face_width,
            "Height":       p1.face_height,
            "Aspect Ratio": _n(aspect_ratio, 3),
        }
        if p3.face.available:
            face["Swelling"] = _n(p3.face.swelling_score)
            face["Cheek/Eye Ratio"] = _n(p3.face.cheek_eye_ratio, 3)

        # ── Eyes ─────────────────────────────────────────────────────────────
        def _eye(eye):
            if not eye.available:
                return None
            return {
                "Openness":    _n(eye.openness_ratio),
                "Redness":     _n(eye.redness_score),
                "Dark Circle": _n(eye.dark_circle_score),
                "Puffiness":   _n(eye.puffiness_score),
            }

        eyes = {
            "Left":  _eye(p3.left_eye),
            "Right": _eye(p3.right_eye),
        }

        # ── Skin ─────────────────────────────────────────────────────────────
        def _skin(region):
            if not region.available:
                return None
            return {
                "Acne":       _n(region.acne_score),
                "Redness":    _n(region.redness_score),
                "Texture":    _n(region.texture_irregularity),
                "Spot Count": region.spot_count,
            }

        skin = {
            "Forehead":    _skin(p3.skin_forehead),
            "Left Cheek":  _skin(p3.skin_left_cheek),
            "Right Cheek": _skin(p3.skin_right_cheek),
            "Nose":        _skin(p3.skin_nose),
        }

        # ── Lips ─────────────────────────────────────────────────────────────
        lips = None
        if p3.lips.available:
            lips = {
                "Dryness":           _n(p3.lips.dryness_score),
                "Color Consistency": _n(p3.lips.color_consistency),
                "Pallor":            _n(p3.lips.pallor_score),
            }

        # ── Mouth ────────────────────────────────────────────────────────────
        mouth = None
        if p3.smile.available:
            mouth = {
                "Aspect Ratio":     _n(p3.smile.mouth_aspect_ratio),
                "Corner Asymmetry": _n(p3.smile.corner_asymmetry),
                "Curvature":        _n(p3.smile.curvature_score),
            }

        # ── Overall (from Phase 4 — plain text/labels, not scored here) ────────
        overall = {
            "Label":             p4.overall_label,
            "Observation Score": _n(p4.observation_score),
            "Summary":           list(p4.overall),   # plain summary sentences, as-is
        }

        return {
            "patient_id": patient_id,
            "timestamp":  datetime.datetime.now().isoformat(),
            "quality_warnings": list(quality_warnings),
            "Face":    face,
            "Eyes":    eyes,
            "Skin":    skin,
            "Lips":    lips,
            "Mouth":   mouth,
            "Overall": overall,
        }