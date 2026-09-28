"""
Centralized thresholds — single source of truth for every magic number
used in scoring and labelling across Phase 1, 3, and 4.

All values are grouped into named dataclass instances so call-sites read
as e.g. `EAR.closed` rather than bare `0.15`, making intent clear and
making tuning a one-file change.
"""

from dataclasses import dataclass


# ─── Generic severity band ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeverityBand:
    """Thresholds passed to _severity(score, band.mild, band.moderate, band.notable)."""
    mild: float
    moderate: float
    notable: float


# ─── Phase 1 — Face Detector ──────────────────────────────────────────────────

@dataclass(frozen=True)
class _Symmetry:
    # Score bands for symmetry label in _compute_symmetry
    highly_symmetric: float = 90.0
    normal:           float = 75.0
    mild_asymmetry:   float = 55.0
    moderate_asymmetry: float = 35.0
    # Divisor in the score formula: (1 - mean_asym / divisor) * 100
    divisor: float = 0.30

SYMMETRY = _Symmetry()


# ─── Phase 3 — Feature Analyzer ───────────────────────────────────────────────

@dataclass(frozen=True)
class _Swelling:
    # Cheek-width / eye-span ratio: below typical_high → score 0
    typical_low:  float = 1.30   # kept for completeness / future "too-narrow" check
    typical_high: float = 1.60
    divisor:      float = 0.30   # (ratio - typical_high) / divisor * 100
    # Severity labels (score thresholds)
    none:     float = 20.0   # below this → "No Swelling Indicators"
    mild:     float = 45.0   # below this → "Mild Swelling Indicator"
    moderate: float = 70.0   # below this → "Moderate Swelling Indicator"

SWELLING = _Swelling()


@dataclass(frozen=True)
class _EAR:
    closed: float = 0.15   # Eye Aspect Ratio at fully closed
    open:   float = 0.40   # EAR at fully open
    # Normalized: (ear - closed) / (open - closed) * 100

EAR = _EAR()


@dataclass(frozen=True)
class _BurstFilter:
    """Stage-1 pre-filter for multi-frame burst capture (see burst_selection.py).

    Runs off Phase 1 output only, before Phase 2/3/4 -- cheap enough to run
    on every candidate frame in a burst.
    """
    min_laplacian_variance: float = 60.0   # below this, the face crop is judged blurry
    eyes_closed_ear_score:  float = 20.0   # EAR-normalized (0-100); BOTH eyes must be
                                            # at/below this to reject -- a single low eye
                                            # is usually a wink or landmark wobble, not a
                                            # genuine closed-eyes frame

BURST_FILTER = _BurstFilter()


@dataclass(frozen=True)
class _EyeRedness:
    divisor: float = 0.10  # (rel_r - baseline_rel_r) / divisor * 100
                            # Matches SKIN_REDNESS.divisor so the two scores
                            # are on the same scale. The reference is the
                            # per-person forehead CalibrationBaseline (same
                            # as skin_redness) so natural skin-tone rel-R
                            # cancels out; only a shift *above* the person's
                            # own forehead reads as redness. This is what
                            # test_scoring.py::TestEyeRedness validates as a
                            # regression guard for a real, previously-fixed
                            # skin-tone fairness bug (medium/dark skin tones
                            # scoring false-positive redness under the old
                            # non-calibrated formula) -- do not remove the
                            # baseline parameter from _eye_redness() again.
                            # Falls back to SKIN_REDNESS.baseline_rel_r when
                            # no baseline is available (same fallback pattern
                            # as _skin_redness / _dark_circle_score).
                            #
                            # OPEN CALIBRATION QUESTION (unresolved, not
                            # fixed here): sclera is a different tissue than
                            # facial skin and may have a systematically
                            # lower rel_r than any given person's forehead,
                            # which would make (rel_r - baseline) read
                            # negative -> clipped to 0 -- for most real
                            # people regardless of correct per-person
                            # calibration. This divisor/reference pair is
                            # ASSERTED, not calibrated against real sclera
                            # crops -- add to the thresholds revalidation
                            # list once real capture data is available to
                            # check whether eye redness still shows healthy
                            # variance under this (correct, tested) formula.

EYE_REDNESS = _EyeRedness()


@dataclass(frozen=True)
class _DarkCircle:
    bright_luminance: float = 200.0  # fallback "not dark" reference when no
                                      # per-person CalibrationBaseline is available
                                      # (e.g. direct unit-test calls to
                                      # _dark_circle_score / _analyze_eye)
    divisor:          float = 120.0  # (reference_luminance - mean_lum) / divisor * 100

DARK_CIRCLE = _DarkCircle()


@dataclass(frozen=True)
class _Puffiness:
    low_variance: float = 50.0   # variance <= this = flat/puffy

PUFFINESS = _Puffiness()


@dataclass(frozen=True)
class _SkinRedness:
    baseline_rel_r: float = 0.33  # fallback relative-red baseline when no
                                   # per-person CalibrationBaseline is available
                                   # (e.g. direct unit-test calls to
                                   # _skin_redness / _analyze_skin)
    divisor:        float = 0.10  # (rel_r - baseline) / divisor * 100

SKIN_REDNESS = _SkinRedness()


@dataclass(frozen=True)
class _Texture:
    divisor:          float = 20.0   # Laplacian variance / divisor → 0-100 (cheeks)
                                      # Validated against real captures: raw_var 147.92→7.4,
                                      # 919.36→45.97 — spread looks healthy, no clipping.
    forehead_divisor: float = 24.0   # Calibrated from one real capture: raw_var=1948.02
                                      # was scoring 97.4 at divisor=20 (near-pinned). At 24.0
                                      # it scores 81.2 — still reads elevated but has headroom.
                                      # Based on a single sample (the high end only) — refine
                                      # with more captures, esp. a visually smoother forehead.
                                      # NOTE: previously mis-set to 45.0, which didn't match
                                      # this calibration note (scored ~43.3, not ~81.2). Fixed.
    nose_divisor:     float = 80.0    # Calibrated from one real capture: raw_var=7048.98 was
                                      # scoring 176.2 (clipped to 100) at divisor=40. At 80.0
                                      # it scores 88.1 — elevated (nose runs structurally
                                      # higher from nostril/tip shadow) but no longer clipped.
                                      # Based on a single sample (the high end only) — refine
                                      # with more captures, esp. a visually smoother nose.
                                      # NOTE: previously mis-set to 120.0, which didn't match
                                      # this calibration note (scored ~58.7, not ~88.1). Fixed.

TEXTURE = _Texture()


@dataclass(frozen=True)
class _Acne:
    min_contour_area:       float = 8.0    # px² — ignore noise below this
    max_contour_area:       float = 800.0  # px² — ignore large regions above this
    spot_count_for_max_score: float = 10.0 # count / this * 100 → score

ACNE = _Acne()


@dataclass(frozen=True)
class _LipDryness:
    divisor: float = 60.0   # Laplacian variance / divisor → 0-100
                            # HISTORY: 15.0 -> pinned 88% of a 57-sample dataset
                            # at 100. Raised to 30.0, anchored on the highest
                            # known *unclipped* sample (raw_var=1285.6, scoring
                            # 85.71 -- near-pinned); at 30.0 that sample scored
                            # ~42.9.
                            # STILL WRONG at 30.0: three fresh captures
                            # (2026-09-27, patient 260927-13C6D1) implied
                            # raw_var of ~2642, >=3000 (clipped), and ~2619 --
                            # roughly 2x the anchor -- so the same pin-at-100
                            # failure mode recurred at the new divisor. That
                            # anchor was never actually the ceiling.
                            # Raised to 60.0 as a stopgap: the known values
                            # above now score ~44.0 / >=50 / ~43.6 (headroom
                            # restored, but the clipped sample's true value is
                            # still unknown, so this is another guess, not a
                            # real calibration).
                            # OPEN CALIBRATION QUESTION (still unresolved):
                            # the CSV/pipeline only stores the post-clip
                            # score, so the true raw variance behind every
                            # clipped sample -- old and new -- is invisible.
                            # Any divisor picked this way can be blown past by
                            # the next real capture. Do not raise this again
                            # without first logging raw variance (see
                            # _lip_dryness) and calibrating off real
                            # pre-clip numbers.

LIP_DRYNESS = _LipDryness()


@dataclass(frozen=True)
class _RegionLighting:
    """
    Per-region lighting-reliability floor, gating texture_irregularity,
    acne_score, and lip dryness_score — all three are Laplacian-variance
    or contour-blob based and read sensor noise as texture/spots when a
    region is underlit, even when the AlignmentCheck whole-face brightness
    check (pipeline.MIN_BRIGHTNESS) passed. Off-axis lighting can leave
    individual regions (forehead, cheeks, nose, lips) noticeably darker
    than the face average even on an overall "good lighting" frame, so
    this is a second, region-local check rather than a substitute for
    AlignmentCheck.

    Mirrors AlignmentCheck's MIN_BRIGHTNESS=60 floor for consistency
    rather than introducing a second, differently-tuned "too dark" number.
    redness_score, color_consistency, and pallor_score are mean-color
    based (not Laplacian/contour based) and are not gated here — they
    degrade far less under low light.
    """
    min_mean_luminance: float = 60.0

REGION_LIGHTING = _RegionLighting()


@dataclass(frozen=True)
class _LipColorConsistency:
    divisor: float = 80.0   # std_per_channel / divisor * 100 → inconsistency

LIP_COLOR_CONSISTENCY = _LipColorConsistency()


@dataclass(frozen=True)
class _LipPallor:
    divisor: float = 100.0  # (divisor - mean_sat) / divisor * 100

LIP_PALLOR = _LipPallor()


@dataclass(frozen=True)
class _Smile:
    """
    Mouth-shape geometry, computed directly from landmarks (MOUTH_LEFT,
    MOUTH_RIGHT, UPPER_LIP_CENTER, LOWER_LIP_CENTER) rather than pixels —
    same category as the Phase 1 swelling/symmetry heuristics, not a
    SkinFeatures pixel-based score.
    """
    # Mouth aspect ratio (width / height of lip landmarks): typical resting
    # range is wide (mouth width >> mouth height) since lips are closed/relaxed
    # most of the time. Normalized against this typical span.
    mar_typical_low:  float = 2.0    # at/below this ratio → score 0 (very open)
    mar_typical_high: float = 6.0    # at/above this ratio → score 100 (closed/relaxed)
    min_measurable_height: float = 0.75  # px floor for the upper/lower-lip
                            # gap used as mar_raw's denominator. Guards
                            # division-by-zero on a genuinely coincident
                            # pair of landmarks without branching to a fixed
                            # sentinel value -- see _analyze_smile's use of
                            # LandmarkPoint.fx/fy for why this floor is now
                            # rarely hit at all on real (sub-pixel) captures.

    # Corner asymmetry: vertical offset between left/right mouth corners,
    # normalized by mouth width so it's scale-invariant across face sizes.
    asymmetry_divisor: float = 0.15  # (vertical_diff / mouth_width) / divisor * 100

    # Curvature: how far above/below the mid-lip-center line the mouth
    # corners sit, normalized by mouth width. Positive → corners raised
    # (smile-like), negative → corners lowered (frown-like).
    curvature_divisor: float = 0.20  # (corner_y - midline_y) / mouth_width / divisor * 100

MOUTH_SHAPE = _Smile()


@dataclass(frozen=True)
class _SkinToneGate:
    """
    YCrCb-based gate to exclude non-skin pixels (fabric, hair, accessories)
    that leak into a region polygon when its outer boundary is derived from
    face-oval landmarks rather than an actual skin/occlusion boundary —
    e.g. a turban edge intruding into the cheek or forehead crop.

    YCrCb is used (rather than RGB or HSV) because the Cr/Cb chrominance
    channels stay comparatively stable across lighting changes and across
    the full range of human skin tones, while still clearly separating
    skin from most fabrics, hair, and backgrounds.

    Band calibrated against a light-to-dark skin-tone gradient (Cr 136-166,
    Cb 83-122 across the gradient) with a small safety margin, then checked
    against navy fabric, black hair, white fabric, red fabric, and denim —
    all of which fall outside this band. It is a heuristic gate, not a
    calibrated skin-detection model, and may need revisiting if real-world
    use surfaces tones or lighting outside what was sampled here.
    """
    cr_low:  float = 130.0
    cr_high: float = 175.0
    cb_low:  float = 75.0
    cb_high: float = 130.0
    min_pixels: int = 50   # below this many gated pixels, treat region as unavailable


@dataclass(frozen=True)
class _LipToneGate:
    """
    Same calibration basis as _SkinToneGate (see that docstring), widened
    slightly on the Cr high end since lips trend more saturated/red than
    general facial skin. Still permissive enough to include pale or dry
    lips, which is exactly the condition this pipeline is trying to
    detect — an overly tight gate would filter out the very findings the
    lip analysis exists to surface.
    """
    cr_low:  float = 130.0
    cr_high: float = 210.0
    cb_low:  float = 75.0
    cb_high: float = 135.0
    min_pixels: int = 30


SKIN_TONE_GATE = _SkinToneGate()
LIP_TONE_GATE  = _LipToneGate()


@dataclass(frozen=True)
class _Calibration:
    """
    A2 — per-person skin-tone calibration baseline.

    Problem: dark-circle scoring (luminance-based) and skin-redness scoring
    (relative-red-based) each compare a region's pixels against ONE fixed
    global reference (DARK_CIRCLE.bright_luminance, SKIN_REDNESS.baseline_rel_r).
    Both raw luminance and raw relative-red dominance vary systematically
    with a person's actual skin tone, independent of whether they have dark
    circles or redness. A fixed global reference reads darker-skinned people
    as having more "dark circle" signal, and can misread lighter-skinned
    people's true neutral rel_r as artificially elevated, purely from skin
    tone, not from the underlying condition the score is meant to capture.

    Fix: derive a per-person reference from their own forehead (least
    likely region to carry dark-circle or lip-pallor-style pathology, and
    a large, easy-to-sample area) and score relative to THAT instead of
    one fixed constant. The forehead's own redness score is still computed
    in the usual way against SKIN_REDNESS.baseline_rel_r — calibration
    changes what OTHER regions are compared against, not the forehead's own
    reading of itself.

    min_pixels: forehead tone-gate-passed pixel count below which the
    baseline is considered unreliable (mirrors SKIN_TONE_GATE.min_pixels —
    same reasoning: too few pixels to trust a mean).
    """
    min_pixels: int = 50

CALIBRATION = _Calibration()


# ─── Phase 4 — Observation Engine ─────────────────────────────────────────────

# Per-feature severity bands (mild / moderate / notable thresholds)
EYE_DARK_CIRCLE_SEVERITY    = SeverityBand(mild=35, moderate=55, notable=75)
EYE_REDNESS_SEVERITY        = SeverityBand(mild=30, moderate=55, notable=75)
EYE_PUFFINESS_SEVERITY      = SeverityBand(mild=35, moderate=60, notable=78)

SKIN_ACNE_SEVERITY          = SeverityBand(mild=20, moderate=45, notable=70)
SKIN_REDNESS_SEVERITY       = SeverityBand(mild=25, moderate=50, notable=70)
SKIN_TEXTURE_SEVERITY       = SeverityBand(mild=30, moderate=55, notable=75)

# Nose uses the same SkinFeatures pipeline (_analyze_skin) as forehead/cheeks,
# so it reuses TEXTURE/ACNE/SKIN_REDNESS for scoring and gets its own severity
# band here for Phase 4 narrative thresholds — kept separate from the cheek
# bands since nose skin (more oil-prone, smaller surface area, prone to
# Laplacian-variance spikes from the nostril/tip shadow even when not
# textured) doesn't necessarily share the same "what counts as notable" line.
SKIN_NOSE_ACNE_SEVERITY     = SeverityBand(mild=20, moderate=45, notable=70)
SKIN_NOSE_REDNESS_SEVERITY  = SeverityBand(mild=25, moderate=50, notable=70)
SKIN_NOSE_TEXTURE_SEVERITY  = SeverityBand(mild=35, moderate=60, notable=80)

LIP_DRYNESS_SEVERITY           = SeverityBand(mild=30, moderate=55, notable=75)
LIP_COLOR_INCONSISTENCY_SEVERITY = SeverityBand(mild=25, moderate=50, notable=70)
LIP_PALLOR_SEVERITY            = SeverityBand(mild=35, moderate=60, notable=80)


@dataclass(frozen=True)
class _FaceObservation:
    # Symmetry narrative score bands
    symmetry_normal: float = 90.0
    symmetry_slight: float = 75.0
    symmetry_mild:   float = 55.0
    # Head tilt angle (degrees)
    tilt_level:  float = 3.0
    tilt_slight: float = 8.0
    # Face aspect ratio (height/width) normal range
    aspect_low:  float = 1.2
    aspect_high: float = 1.6
    # Swelling narrative score bands
    swelling_none:     float = 20.0
    swelling_mild:     float = 45.0
    swelling_moderate: float = 70.0

FACE_OBSERVATION = _FaceObservation()


@dataclass(frozen=True)
class _MouthObservation:
    # Corner-asymmetry score bands (0-100, from MOUTH_SHAPE.asymmetry_divisor)
    asymmetry_slight:  float = 15.0
    asymmetry_mild:    float = 35.0
    asymmetry_notable: float = 60.0
    # Curvature score bands: positive = smile-leaning, negative = frown-leaning.
    # Only the magnitude is banded; direction is reported separately as text.
    curvature_neutral_band: float = 10.0   # |curvature| below this → "neutral resting" note

MOUTH_OBSERVATION = _MouthObservation()


@dataclass(frozen=True)
class _CrossSide:
    cheek_redness_diff: float = 25.0   # min diff to flag asymmetric redness
    eye_openness_notable: float = 20.0
    eye_openness_slight:  float = 10.0

CROSS_SIDE = _CrossSide()


@dataclass(frozen=True)
class _Confidence:
    no_findings_base:  float = 85.0
    findings_base:     float = 90.0
    findings_floor:    float = 60.0
    per_finding_drop:  float = 3.0

CONFIDENCE = _Confidence()


# ─── Phase A2 — Trend Detector ────────────────────────────────────────────────

@dataclass(frozen=True)
class _Trend:
    """
    A2 — multi-day trend detection over HistoryStore (A1) data.

    All tracked metrics live on the same 0-100 scale (Phase 3's existing
    convention), so a single set of magnitude/consistency thresholds works
    across metrics rather than needing one per metric.

    window_days: how far back to look for a trend at all.
    min_days: minimum number of distinct calendar days with data inside
              the window before a trend is even evaluated -- two readings
              from one bad week shouldn't read as a "trend".
    min_total_change: minimum |last - first| value across the window to
              count as a real movement rather than score noise from
              capture-to-capture lighting/pose variance.
    min_consistency: minimum fraction of consecutive day-to-day deltas
              that must agree in sign with the overall slope direction.
              Guards against a trend being driven by one outlier day in
              an otherwise flat series.
    """
    window_days:       int   = 14
    min_days:           int   = 3
    min_total_change:   float = 8.0
    min_consistency:    float = 0.6

TREND = _Trend()