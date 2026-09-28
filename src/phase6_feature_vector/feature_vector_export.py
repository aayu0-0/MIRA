"""
Phase 6 — Feature Vector Export
---------------------------------
Consumes Phase 1 (face geometry), Phase 3 (feature analysis), and
optionally Phase 4 (observation confidence) results, and flattens them
into a single flat, all-numeric dict -- one row appended to a growing
dataset CSV per capture.

Each numeric score column is accompanied by a paired *_cat column
(category label) derived from the same thresholds used by Phase 4.
Categories describe what the score *means* in plain language rather than
just where it falls on an arbitrary 0-100 axis.  They are strings, not
ordinal integers, so downstream ML pipelines should one-hot or
label-encode them before training; exclude *_cat columns from numeric
feature matrices.

Only Phase 3 raw scores go into the vector. Phase 4 severity labels are
derived from these same scores via fixed thresholds, so including both
would duplicate signal rather than add it -- if you want severity bands
later, recompute them from this CSV instead of storing them twice.

overall_confidence (from Phase 4) is included as a trailing METADATA
column, not a feature -- it's derived from the same abnormal/moderate/
mild counts that come from these scores, so a model trained on it would
partly be predicting itself. Keep it around for filtering/QA (e.g. drop
or down-weight low-confidence rows before training), and exclude it
when you build your actual training matrix.

Usage:
    from phase6_feature_vector.feature_vector_export import (
        flatten_phase3, append_feature_row,
    )

    row = flatten_phase3(p1, p3, p4, patient_id="DEMO-001")
    append_feature_row(row, "dataset.csv")
"""

import csv
import datetime
import os
import sys

try:
    import fcntl  # Unix/Linux/macOS
except ImportError:
    fcntl = None  # Windows

# Make sure src/ is on path when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thresholds import (
    FACE_OBSERVATION, MOUTH_OBSERVATION,
    EYE_DARK_CIRCLE_SEVERITY, EYE_REDNESS_SEVERITY, EYE_PUFFINESS_SEVERITY,
    SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY,
    SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY, SKIN_NOSE_TEXTURE_SEVERITY,
    LIP_DRYNESS_SEVERITY, LIP_COLOR_INCONSISTENCY_SEVERITY, LIP_PALLOR_SEVERITY,
)


# ── Category helpers ──────────────────────────────────────────────────────────

def _severity_cat(score, band, labels=("Clear", "Mild", "Moderate", "Notable")):
    """
    Map a 0-100 score onto 4 plain-language labels using a SeverityBand.
    Default label set works for most 'higher = worse' features.
    Returns None when score is None (region unavailable).
    """
    if score is None:
        return None
    if score < band.mild:
        return labels[0]
    if score < band.moderate:
        return labels[1]
    if score < band.notable:
        return labels[2]
    return labels[3]


def _eye_openness_cat(score):
    """
    Eye openness is 0-100 where 100 = fully open, 0 = fully closed.
    Thresholds mirror CROSS_SIDE.eye_openness_notable / eye_openness_slight
    which flag a *difference* between eyes -- here we categorize the absolute
    score for a single eye instead.
    """
    if score is None:
        return None
    if score < 20:
        return "Closed"
    if score < 45:
        return "Drooping"
    if score < 75:
        return "Partially Open"
    return "Open"


def _lip_color_consistency_cat(score):
    """
    lips_color_consistency is an inverse score: 100 = perfectly consistent
    (healthy), 0 = highly inconsistent (bad).  The severity band lives on
    the *inconsistency* side, so we invert before applying it.
    """
    if score is None:
        return None
    inconsistency = 100.0 - score
    return _severity_cat(
        inconsistency,
        LIP_COLOR_INCONSISTENCY_SEVERITY,
        labels=("Consistent", "Mild Inconsistency", "Moderate Inconsistency", "Notable Inconsistency"),
    )


def _symmetry_cat(score):
    """
    Face symmetry is 0-100 where 100 = perfectly symmetric (good).
    Bands from FACE_OBSERVATION (symmetry_normal, symmetry_slight, symmetry_mild).
    """
    if score is None:
        return None
    if score >= FACE_OBSERVATION.symmetry_normal:
        return "Symmetric"
    if score >= FACE_OBSERVATION.symmetry_slight:
        return "Slight Asymmetry"
    if score >= FACE_OBSERVATION.symmetry_mild:
        return "Mild Asymmetry"
    return "Notable Asymmetry"


def _swelling_cat(score):
    """Bands from FACE_OBSERVATION swelling thresholds.

    NOTE: the "no swelling" label is deliberately "No Swelling", not the
    bare word "None" -- pandas, Excel, and most CSV readers treat the
    literal string "None" as a null value by default, which would make
    every healthy (non-swollen) row silently read back as missing data
    instead of a real, meaningful category. Do not change this back to
    "None" without also auditing every downstream reader.
    """
    if score is None:
        return None
    if score < FACE_OBSERVATION.swelling_none:
        return "No Swelling"
    if score < FACE_OBSERVATION.swelling_mild:
        return "Mild"
    if score < FACE_OBSERVATION.swelling_moderate:
        return "Moderate"
    return "Notable"


def _mouth_aspect_ratio_cat(score):
    """
    mouth_aspect_ratio: 100 = closed/relaxed, 0 = fully open.
    Simple 3-band split.
    """
    if score is None:
        return None
    if score < 30:
        return "Open"
    if score < 70:
        return "Partially Open"
    return "Closed / Relaxed"


def _mouth_asymmetry_cat(score):
    """Bands from MOUTH_OBSERVATION corner-asymmetry thresholds."""
    if score is None:
        return None
    if score < MOUTH_OBSERVATION.asymmetry_slight:
        return "Symmetric"
    if score < MOUTH_OBSERVATION.asymmetry_mild:
        return "Slight Asymmetry"
    if score < MOUTH_OBSERVATION.asymmetry_notable:
        return "Mild Asymmetry"
    return "Notable Asymmetry"


def _curvature_cat(score):
    """
    mouth_curvature_score is signed: positive = smile-leaning,
    negative = frown-leaning.  Band from MOUTH_OBSERVATION.curvature_neutral_band.
    """
    if score is None:
        return None
    band = MOUTH_OBSERVATION.curvature_neutral_band
    if abs(score) < band:
        return "Neutral"
    return "Smile-Leaning" if score > 0 else "Frown-Leaning"


# ── Column schema ─────────────────────────────────────────────────────────────

# Score columns in fixed order.  Each score column is immediately followed
# by its paired *_cat column so the CSV is human-readable left-to-right.
FEATURE_COLUMNS = [
    "patient_id", "timestamp",
    # Face
    "face_symmetry_score",      "face_symmetry_cat",
    "face_alignment_angle",
    "face_width", "face_height", "face_aspect_ratio",
    "face_swelling_score",      "face_swelling_cat",
    # Eyes
    "left_eye_openness_ratio",        "left_eye_openness_cat",
    "left_eye_redness_score",         "left_eye_redness_cat",
    "left_eye_dark_circle_score",     "left_eye_dark_circle_cat",
    "left_eye_puffiness_score",       "left_eye_puffiness_cat",
    "right_eye_openness_ratio",       "right_eye_openness_cat",
    "right_eye_redness_score",        "right_eye_redness_cat",
    "right_eye_dark_circle_score",    "right_eye_dark_circle_cat",
    "right_eye_puffiness_score",      "right_eye_puffiness_cat",
    # Skin — forehead
    "skin_forehead_acne_score",             "skin_forehead_acne_cat",
    "skin_forehead_redness_score",          "skin_forehead_redness_cat",
    "skin_forehead_texture_irregularity",   "skin_forehead_texture_cat",
    "skin_forehead_spot_count",
    # Skin — left cheek
    "skin_left_cheek_acne_score",           "skin_left_cheek_acne_cat",
    "skin_left_cheek_redness_score",        "skin_left_cheek_redness_cat",
    "skin_left_cheek_texture_irregularity", "skin_left_cheek_texture_cat",
    "skin_left_cheek_spot_count",
    # Skin — right cheek
    "skin_right_cheek_acne_score",          "skin_right_cheek_acne_cat",
    "skin_right_cheek_redness_score",       "skin_right_cheek_redness_cat",
    "skin_right_cheek_texture_irregularity","skin_right_cheek_texture_cat",
    "skin_right_cheek_spot_count",
    # Skin — nose (own severity bands)
    "skin_nose_acne_score",             "skin_nose_acne_cat",
    "skin_nose_redness_score",          "skin_nose_redness_cat",
    "skin_nose_texture_irregularity",   "skin_nose_texture_cat",
    "skin_nose_spot_count",
    # Lips
    "lips_dryness_score",       "lips_dryness_cat",
    "lips_color_consistency",   "lips_color_consistency_cat",
    "lips_pallor_score",        "lips_pallor_cat",
    # Mouth / smile geometry
    "mouth_aspect_ratio",       "mouth_aspect_ratio_cat",
    "mouth_corner_asymmetry",   "mouth_corner_asymmetry_cat",
    "mouth_curvature_score",    "mouth_curvature_cat",
]

# Trailing metadata — exclude from training matrices.
METADATA_COLUMNS = [
    "overall_confidence",
    "photo_path",   # filename of the captured frame, saved alongside this CSV
]

ALL_COLUMNS = FEATURE_COLUMNS + METADATA_COLUMNS


# ── Utilities ─────────────────────────────────────────────────────────────────

def _n(value, decimals=2):
    """Round a numeric value; pass through None (missing) untouched."""
    if value is None:
        return None
    return round(float(value), decimals)


# ── Main export function ──────────────────────────────────────────────────────

def flatten_phase3(
    p1,
    p3,
    p4=None,
    patient_id: str = "UNKNOWN",
    timestamp: str = None,
    photo_path: str = None,
) -> dict:
    """
    Build one flat feature-vector row from a Phase1Result + Phase3Result,
    plus trailing metadata columns from Phase4Report (left None if p4 isn't
    passed) and an optional photo_path.

    photo_path should be the filename (or relative path) of the captured
    frame saved alongside this CSV.  Pass it here rather than injecting it
    after the call so the returned row is always self-contained for any caller.

    Unavailable regions come through as None (CSV: empty cell) rather than
    0, so a missing capture never masquerades as a real "no signs" score.
    Paired *_cat columns are derived from the same thresholds used by Phase 4.
    """
    row = {col: None for col in ALL_COLUMNS}
    row["patient_id"]  = patient_id
    row["timestamp"]   = timestamp or datetime.datetime.now().isoformat()
    row["photo_path"]  = photo_path  # None when not supplied — caller can still
                                     # overwrite, but the field is always present

    if p4 is not None:
        row["overall_confidence"] = _n(p4.overall_confidence)

    # ── Face ──────────────────────────────────────────────────────────────
    if p1 is not None and getattr(p1, "success", False):
        sym = _n(p1.symmetry_score)
        row["face_symmetry_score"] = sym
        row["face_symmetry_cat"]   = _symmetry_cat(sym)
        row["face_alignment_angle"] = _n(p1.alignment_angle)
        row["face_width"]           = p1.face_width
        row["face_height"]          = p1.face_height
        row["face_aspect_ratio"]    = _n(p1.face_height / p1.face_width, 3) if p1.face_width else None

    if p3.face.available:
        sw = _n(p3.face.swelling_score)
        row["face_swelling_score"] = sw
        row["face_swelling_cat"]   = _swelling_cat(sw)

    # ── Eyes ──────────────────────────────────────────────────────────────
    for side, eye in (("left_eye", p3.left_eye), ("right_eye", p3.right_eye)):
        if eye.available:
            op = _n(eye.openness_ratio)
            re = _n(eye.redness_score)
            dc = _n(eye.dark_circle_score)
            pu = _n(eye.puffiness_score)

            row[f"{side}_openness_ratio"]    = op
            row[f"{side}_openness_cat"]      = _eye_openness_cat(op)
            row[f"{side}_redness_score"]     = re
            row[f"{side}_redness_cat"]       = _severity_cat(re, EYE_REDNESS_SEVERITY)
            row[f"{side}_dark_circle_score"] = dc
            row[f"{side}_dark_circle_cat"]   = _severity_cat(dc, EYE_DARK_CIRCLE_SEVERITY)
            row[f"{side}_puffiness_score"]   = pu
            row[f"{side}_puffiness_cat"]     = _severity_cat(pu, EYE_PUFFINESS_SEVERITY)

    # ── Skin ──────────────────────────────────────────────────────────────
    # Nose uses its own (stricter) bands; cheeks and forehead share standard bands.
    skin_regions = (
        ("skin_forehead",    p3.skin_forehead,    SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY),
        ("skin_left_cheek",  p3.skin_left_cheek,  SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY),
        ("skin_right_cheek", p3.skin_right_cheek, SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY),
        ("skin_nose",        p3.skin_nose,         SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY, SKIN_NOSE_TEXTURE_SEVERITY),
    )
    for prefix, region, acne_band, red_band, tex_band in skin_regions:
        if region.available:
            ac = _n(region.acne_score)
            re = _n(region.redness_score)
            tx = _n(region.texture_irregularity)

            row[f"{prefix}_acne_score"]           = ac
            row[f"{prefix}_acne_cat"]             = _severity_cat(ac, acne_band,
                                                        labels=("Clear", "Mild", "Moderate", "Notable"))
            row[f"{prefix}_redness_score"]        = re
            row[f"{prefix}_redness_cat"]          = _severity_cat(re, red_band)
            row[f"{prefix}_texture_irregularity"] = tx
            row[f"{prefix}_texture_cat"]          = _severity_cat(tx, tex_band,
                                                        labels=("Smooth", "Mild", "Moderate", "Notable"))
            row[f"{prefix}_spot_count"]           = region.spot_count

    # ── Lips ──────────────────────────────────────────────────────────────
    if p3.lips.available:
        dr = _n(p3.lips.dryness_score)
        cc = _n(p3.lips.color_consistency)
        pa = _n(p3.lips.pallor_score)

        row["lips_dryness_score"]           = dr
        row["lips_dryness_cat"]             = _severity_cat(dr, LIP_DRYNESS_SEVERITY,
                                                labels=("Hydrated", "Mild", "Moderate", "Notable"))
        row["lips_color_consistency"]       = cc
        row["lips_color_consistency_cat"]   = _lip_color_consistency_cat(cc)
        row["lips_pallor_score"]            = pa
        row["lips_pallor_cat"]              = _severity_cat(pa, LIP_PALLOR_SEVERITY,
                                                labels=("Normal", "Mild", "Moderate", "Notable"))

    # ── Mouth / smile geometry ─────────────────────────────────────────────
    if p3.smile.available:
        mar = _n(p3.smile.mouth_aspect_ratio)
        asym = _n(p3.smile.corner_asymmetry)
        curv = _n(p3.smile.curvature_score)

        row["mouth_aspect_ratio"]       = mar
        row["mouth_aspect_ratio_cat"]   = _mouth_aspect_ratio_cat(mar)
        row["mouth_corner_asymmetry"]   = asym
        row["mouth_corner_asymmetry_cat"] = _mouth_asymmetry_cat(asym)
        row["mouth_curvature_score"]    = curv
        row["mouth_curvature_cat"]      = _curvature_cat(curv)

    return row


# Maps every derived *_cat column to the raw score column it's computed
# from and the function that computes it. Used only during schema
# migration (below) so that a *_cat column added to the schema AFTER some
# rows were already written gets recomputed from the raw score that WAS
# already present in those rows, instead of being blank-filled even though
# the information needed was sitting right there. Kept in sync manually
# with the row[...] assignments in flatten_phase3 above -- if you add a new
# *_cat column there, add its recompute rule here too, or a future schema
# change will silently blank it out for every pre-existing row again.
_CAT_RECOMPUTE = {
    "face_symmetry_cat":  ("face_symmetry_score",  lambda v: _symmetry_cat(v)),
    "face_swelling_cat":  ("face_swelling_score",  lambda v: _swelling_cat(v)),

    "left_eye_openness_cat":     ("left_eye_openness_ratio",     lambda v: _eye_openness_cat(v)),
    "right_eye_openness_cat":    ("right_eye_openness_ratio",    lambda v: _eye_openness_cat(v)),
    "left_eye_redness_cat":      ("left_eye_redness_score",      lambda v: _severity_cat(v, EYE_REDNESS_SEVERITY)),
    "right_eye_redness_cat":     ("right_eye_redness_score",     lambda v: _severity_cat(v, EYE_REDNESS_SEVERITY)),
    "left_eye_dark_circle_cat":  ("left_eye_dark_circle_score",  lambda v: _severity_cat(v, EYE_DARK_CIRCLE_SEVERITY)),
    "right_eye_dark_circle_cat": ("right_eye_dark_circle_score", lambda v: _severity_cat(v, EYE_DARK_CIRCLE_SEVERITY)),
    "left_eye_puffiness_cat":    ("left_eye_puffiness_score",    lambda v: _severity_cat(v, EYE_PUFFINESS_SEVERITY)),
    "right_eye_puffiness_cat":   ("right_eye_puffiness_score",   lambda v: _severity_cat(v, EYE_PUFFINESS_SEVERITY)),

    "skin_forehead_acne_cat":    ("skin_forehead_acne_score",    lambda v: _severity_cat(v, SKIN_ACNE_SEVERITY, labels=("Clear", "Mild", "Moderate", "Notable"))),
    "skin_forehead_redness_cat": ("skin_forehead_redness_score", lambda v: _severity_cat(v, SKIN_REDNESS_SEVERITY)),
    "skin_forehead_texture_cat": ("skin_forehead_texture_irregularity", lambda v: _severity_cat(v, SKIN_TEXTURE_SEVERITY, labels=("Smooth", "Mild", "Moderate", "Notable"))),
    "skin_left_cheek_acne_cat":    ("skin_left_cheek_acne_score",    lambda v: _severity_cat(v, SKIN_ACNE_SEVERITY, labels=("Clear", "Mild", "Moderate", "Notable"))),
    "skin_left_cheek_redness_cat": ("skin_left_cheek_redness_score", lambda v: _severity_cat(v, SKIN_REDNESS_SEVERITY)),
    "skin_left_cheek_texture_cat": ("skin_left_cheek_texture_irregularity", lambda v: _severity_cat(v, SKIN_TEXTURE_SEVERITY, labels=("Smooth", "Mild", "Moderate", "Notable"))),
    "skin_right_cheek_acne_cat":    ("skin_right_cheek_acne_score",    lambda v: _severity_cat(v, SKIN_ACNE_SEVERITY, labels=("Clear", "Mild", "Moderate", "Notable"))),
    "skin_right_cheek_redness_cat": ("skin_right_cheek_redness_score", lambda v: _severity_cat(v, SKIN_REDNESS_SEVERITY)),
    "skin_right_cheek_texture_cat": ("skin_right_cheek_texture_irregularity", lambda v: _severity_cat(v, SKIN_TEXTURE_SEVERITY, labels=("Smooth", "Mild", "Moderate", "Notable"))),
    "skin_nose_acne_cat":    ("skin_nose_acne_score",    lambda v: _severity_cat(v, SKIN_NOSE_ACNE_SEVERITY, labels=("Clear", "Mild", "Moderate", "Notable"))),
    "skin_nose_redness_cat": ("skin_nose_redness_score", lambda v: _severity_cat(v, SKIN_NOSE_REDNESS_SEVERITY)),
    "skin_nose_texture_cat": ("skin_nose_texture_irregularity", lambda v: _severity_cat(v, SKIN_NOSE_TEXTURE_SEVERITY, labels=("Smooth", "Mild", "Moderate", "Notable"))),

    "lips_dryness_cat":           ("lips_dryness_score",       lambda v: _severity_cat(v, LIP_DRYNESS_SEVERITY, labels=("Hydrated", "Mild", "Moderate", "Notable"))),
    "lips_color_consistency_cat": ("lips_color_consistency",   lambda v: _lip_color_consistency_cat(v)),
    "lips_pallor_cat":            ("lips_pallor_score",        lambda v: _severity_cat(v, LIP_PALLOR_SEVERITY, labels=("Normal", "Mild", "Moderate", "Notable"))),

    "mouth_aspect_ratio_cat":     ("mouth_aspect_ratio",      lambda v: _mouth_aspect_ratio_cat(v)),
    "mouth_corner_asymmetry_cat": ("mouth_corner_asymmetry",  lambda v: _mouth_asymmetry_cat(v)),
    "mouth_curvature_cat":        ("mouth_curvature_score",   lambda v: _curvature_cat(v)),
}


def _migrate_if_schema_changed(csv_path: str) -> None:
    """
    If csv_path already exists but its header doesn't match the current
    ALL_COLUMNS (e.g. the schema grew a *_cat column or photo_path since
    this file was created), rewrite the file in place: every old row is
    remapped by column NAME onto the current schema.

    A column that didn't exist under the old schema is recomputed from its
    paired raw-score column when possible (see _CAT_RECOMPUTE) rather than
    silently blank-filled -- the previous blank-fill behavior meant any
    *_cat column added after some rows were already captured came back
    empty for every one of those rows forever, even though the raw score
    needed to compute it was sitting right there in the same row.
    """
    if not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0):
        return  # new file — nothing to migrate

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    old_header, old_data = rows[0], rows[1:]
    if old_header == ALL_COLUMNS:
        return  # already current schema

    fixed = []
    for r in old_data:
        if len(r) == len(old_header):
            d = dict(zip(old_header, r))
        else:
            # Row width doesn't match its own file's header either (a rarer,
            # already-corrupted case) — best effort: pair by position against
            # the row's OWN header (old_header), truncating/padding as needed,
            # so each value at least maps to its original column name before
            # being remapped onto the current schema below.
            paired_len = min(len(r), len(old_header))
            d = dict(zip(old_header[:paired_len], r[:paired_len]))

        new_row = {}
        for col in ALL_COLUMNS:
            if col in d:
                new_row[col] = d[col]
                continue
            recompute = _CAT_RECOMPUTE.get(col)
            raw_val = d.get(recompute[0], "") if recompute else ""
            if recompute and raw_val not in ("", None):
                try:
                    new_row[col] = recompute[1](float(raw_val))
                    continue
                except (TypeError, ValueError):
                    pass  # fall through to blank -- raw value wasn't numeric
            new_row[col] = ""
        fixed.append(new_row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(fixed)


def append_feature_row(row: dict, csv_path: str) -> None:
    """Append one feature-vector row to the dataset CSV.

    Uses an advisory lock on Unix. On Windows, the lock dependency is
    unavailable; MIRA launches one pipeline process per capture, so the
    append remains safe for the normal application workflow.
    """
    lock_path = csv_path + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        if fcntl is not None:
            fcntl.flock(lock_f, fcntl.LOCK_EX)

        try:
            _migrate_if_schema_changed(csv_path)
            file_exists = (
                os.path.isfile(csv_path)
                and os.path.getsize(csv_path) > 0
            )

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_f, fcntl.LOCK_UN)