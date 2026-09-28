"""
Fairness / skin-tone bias audit for SKIN_TONE_GATE and LIP_TONE_GATE.

WHY THIS FILE EXISTS
---------------------
thresholds.py's _SkinToneGate docstring asserts the Cr/Cb band was
"calibrated against a light-to-dark skin-tone gradient ... then checked
against navy fabric, black hair, white fabric, red fabric, and denim."
That calibration claim previously lived only as a comment — nothing in
the codebase re-ran it, so there was no way to catch a threshold edit
that silently reintroduced a skin-tone bias (e.g. the gate passing
light tones at a much higher rate than dark tones, or vice versa).

This file makes that audit executable and re-runnable as part of the
normal test suite, using the SAME gate math the pipeline actually runs
(_tone_gate_mask_flat's cr/cb comparison), not a reimplementation.

METHODOLOGY / LIMITATIONS
--------------------------
- Skin-tone reference swatches below are solid-colour approximations
  spanning a light-to-dark gradient (loosely modeled on the Fitzpatrick
  I-VI scale), NOT real captured faces. They are a fast regression
  check for gross bias in the gate's Cr/Cb band, not a substitute for
  auditing against real photos across diverse subjects and lighting.
- Non-skin swatches (denim, navy, black hair, white fabric, red fabric)
  match the materials named in the thresholds.py docstring, so this
  test also verifies that specific claim is still true.
- If you add real reference images later, prefer extending this file
  over replacing it — keep the synthetic gradient as a fast sanity
  check and add an image-backed suite alongside it.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase3_feature_analysis"))

from thresholds import SKIN_TONE_GATE, LIP_TONE_GATE
from feature_analyzer import FeatureAnalyzer


# ── Reference swatches (BGR, since cv2/the pipeline work in BGR) ───────────

# Light-to-dark skin-tone gradient, loosely modeled on Fitzpatrick I-VI.
# RGB source values are common illustrative skin-tone swatches; converted
# to BGR (reversed) since that's the pipeline's native pixel order.
SKIN_GRADIENT_BGR = {
    "type_I_very_light":   (196, 224, 255),
    "type_II_light":       (125, 194, 241),
    "type_III_light_med":  (105, 172, 224),
    "type_IV_medium":      (66, 134, 198),
    "type_V_med_dark":     (36, 85, 141),
    "type_VI_dark":        (22, 40, 74),
}

# Non-skin materials named explicitly in the thresholds.py docstring.
NON_SKIN_BGR = {
    "black_hair":    (15, 15, 15),
    "white_fabric":  (245, 245, 245),
    "navy_fabric":   (80, 20, 10),
    "red_fabric":    (30, 30, 200),
    "denim":         (130, 100, 90),
}


def _swatch(bgr, h=40, w=60):
    return np.full((h, w, 3), bgr, dtype=np.uint8).reshape(-1, 3)


def _keep_fraction(bgr, gate):
    analyzer = FeatureAnalyzer.__new__(FeatureAnalyzer)  # no __init__ needed for this method
    pixels = _swatch(bgr)
    keep = analyzer._tone_gate_mask_flat(pixels, gate)
    return keep.mean()


# ── 1. Every skin tone in the gradient must pass the gate ──────────────────

@pytest.mark.parametrize("label,bgr", SKIN_GRADIENT_BGR.items())
def test_skin_gradient_passes_skin_tone_gate(label, bgr):
    frac = _keep_fraction(bgr, SKIN_TONE_GATE)
    assert frac > 0.95, (
        f"{label} was rejected by SKIN_TONE_GATE (kept {frac:.0%} of pixels) — "
        f"the gate may be biased against this part of the skin-tone range."
    )


# ── 2. No part of the gradient is rejected disproportionately vs. the rest ─

def test_skin_tone_gate_has_no_gross_bias_across_gradient():
    fractions = {label: _keep_fraction(bgr, SKIN_TONE_GATE)
                 for label, bgr in SKIN_GRADIENT_BGR.items()}
    lightest = fractions["type_I_very_light"]
    darkest = fractions["type_VI_dark"]
    spread = max(fractions.values()) - min(fractions.values())
    assert spread < 0.15, (
        f"Keep-rate spread across the skin-tone gradient is {spread:.0%} "
        f"({fractions}) — one end of the gradient is passing the gate at a "
        f"meaningfully different rate than the other, which is exactly the "
        f"kind of skin-tone bias this audit exists to catch."
    )
    assert abs(lightest - darkest) < 0.15, (
        f"Lightest ({lightest:.0%}) vs darkest ({darkest:.0%}) keep-rate "
        f"differs too much — possible bias toward one end of the gradient."
    )


# ── 3. Named non-skin materials must still be rejected ─────────────────────

@pytest.mark.parametrize("label,bgr", NON_SKIN_BGR.items())
def test_non_skin_materials_are_rejected_by_skin_tone_gate(label, bgr):
    frac = _keep_fraction(bgr, SKIN_TONE_GATE)
    assert frac < 0.05, (
        f"{label} was NOT rejected by SKIN_TONE_GATE (kept {frac:.0%} of "
        f"pixels) — thresholds.py's docstring claims this material falls "
        f"outside the band; that claim no longer holds."
    )


# ── 4. Lip-tone gate: same gradient, wider Cr band (per its own docstring) ─

@pytest.mark.parametrize("label,bgr", SKIN_GRADIENT_BGR.items())
def test_skin_gradient_passes_lip_tone_gate(label, bgr):
    frac = _keep_fraction(bgr, LIP_TONE_GATE)
    assert frac > 0.90, (
        f"{label} was rejected by LIP_TONE_GATE (kept {frac:.0%} of pixels)."
    )


def test_lip_tone_gate_has_no_gross_bias_across_gradient():
    fractions = {label: _keep_fraction(bgr, LIP_TONE_GATE)
                 for label, bgr in SKIN_GRADIENT_BGR.items()}
    spread = max(fractions.values()) - min(fractions.values())
    assert spread < 0.15, (
        f"Keep-rate spread across the skin-tone gradient is {spread:.0%} "
        f"({fractions}) for LIP_TONE_GATE — possible bias."
    )
