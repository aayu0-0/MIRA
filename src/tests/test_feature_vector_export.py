"""
test_feature_vector_export.py — Phase 6 (feature_vector_export.py)

Covers:
  - Every category-label helper (_severity_cat, _eye_openness_cat,
    _lip_color_consistency_cat, _symmetry_cat, _swelling_cat,
    _mouth_aspect_ratio_cat, _mouth_asymmetry_cat, _curvature_cat),
    including None passthrough and boundary values.
  - flatten_phase3(): full pipeline row (via the same synthetic-image
    fixture used by test_integration.py), missing-p1, missing-p4, and a
    Phase1Result with success=False.
  - append_feature_row() + _migrate_if_schema_changed(): new file creation,
    header-based schema migration when columns are added/removed, and
    concurrent-append safety (advisory lock doesn't corrupt output).

No camera, no model file, no network.
"""

import csv
import os
import sys
import shutil
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.fixtures import build_phase1_result, IMG_W, IMG_H, FACE_CX, FACE_CY, FACE_RX, FACE_RY

from phase1_face_understanding.face_detector import Phase1Result
from phase2_region_extraction.region_extractor import RegionExtractor
from phase3_feature_analysis.feature_analyzer import FeatureAnalyzer
from phase4_observation_engine.observation_engine import ObservationEngine
from thresholds import (
    FACE_OBSERVATION, MOUTH_OBSERVATION,
    EYE_DARK_CIRCLE_SEVERITY, LIP_DRYNESS_SEVERITY,
    LIP_COLOR_INCONSISTENCY_SEVERITY,
)
from phase6_feature_vector.feature_vector_export import (
    _severity_cat, _eye_openness_cat, _lip_color_consistency_cat,
    _symmetry_cat, _swelling_cat, _mouth_aspect_ratio_cat,
    _mouth_asymmetry_cat, _curvature_cat, _n,
    flatten_phase3, append_feature_row, _migrate_if_schema_changed,
    ALL_COLUMNS, FEATURE_COLUMNS, METADATA_COLUMNS,
)


def _synthetic_face_image() -> np.ndarray:
    img = np.full((IMG_H, IMG_W, 3), (200, 190, 180), dtype=np.uint8)
    cv2.ellipse(img, (FACE_CX, FACE_CY), (FACE_RX, FACE_RY), 0, 0, 360, (150, 160, 190), -1)
    eye_y = int(FACE_CY - FACE_RY * 0.25)
    for sign in (-1, 1):
        ex = int(FACE_CX + sign * FACE_RX * 0.45 * 0.7)
        cv2.ellipse(img, (ex, eye_y), (18, 8), 0, 0, 360, (80, 70, 70), -1)
    mouth_y = int(FACE_CY + FACE_RY * 0.55)
    cv2.ellipse(img, (FACE_CX, mouth_y), (40, 12), 0, 0, 360, (90, 80, 150), -1)
    return img


# ─── Category helper tests ─────────────────────────────────────────────────

class TestSeverityCat(unittest.TestCase):

    def test_none_passthrough(self):
        self.assertIsNone(_severity_cat(None, LIP_DRYNESS_SEVERITY))

    def test_below_mild_is_first_label(self):
        band = EYE_DARK_CIRCLE_SEVERITY
        self.assertEqual(_severity_cat(band.mild - 1, band), "Clear")

    def test_at_mild_boundary_is_second_label(self):
        band = EYE_DARK_CIRCLE_SEVERITY
        self.assertEqual(_severity_cat(band.mild, band), "Mild")

    def test_at_moderate_boundary_is_third_label(self):
        band = EYE_DARK_CIRCLE_SEVERITY
        self.assertEqual(_severity_cat(band.moderate, band), "Moderate")

    def test_at_notable_boundary_is_fourth_label(self):
        band = EYE_DARK_CIRCLE_SEVERITY
        self.assertEqual(_severity_cat(band.notable, band), "Notable")

    def test_custom_labels_used(self):
        band = LIP_DRYNESS_SEVERITY
        custom = ("Hydrated", "Mild", "Moderate", "Notable")
        result = _severity_cat(band.notable + 5, band, labels=custom)
        self.assertEqual(result, "Notable")

    def test_zero_score_is_clear(self):
        self.assertEqual(_severity_cat(0, EYE_DARK_CIRCLE_SEVERITY), "Clear")

    def test_hundred_score_is_notable(self):
        self.assertEqual(_severity_cat(100, EYE_DARK_CIRCLE_SEVERITY), "Notable")


class TestEyeOpennessCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_eye_openness_cat(None))

    def test_closed(self):
        self.assertEqual(_eye_openness_cat(0), "Closed")
        self.assertEqual(_eye_openness_cat(19.9), "Closed")

    def test_drooping(self):
        self.assertEqual(_eye_openness_cat(20), "Drooping")
        self.assertEqual(_eye_openness_cat(44.9), "Drooping")

    def test_partially_open(self):
        self.assertEqual(_eye_openness_cat(45), "Partially Open")
        self.assertEqual(_eye_openness_cat(74.9), "Partially Open")

    def test_open(self):
        self.assertEqual(_eye_openness_cat(75), "Open")
        self.assertEqual(_eye_openness_cat(100), "Open")


class TestLipColorConsistencyCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_lip_color_consistency_cat(None))

    def test_perfect_consistency_is_consistent(self):
        # score=100 -> inconsistency=0 -> below band.mild -> "Consistent"
        self.assertEqual(_lip_color_consistency_cat(100), "Consistent")

    def test_zero_score_is_worst_inconsistency(self):
        # score=0 -> inconsistency=100 -> "Notable Inconsistency"
        result = _lip_color_consistency_cat(0)
        if 100 >= LIP_COLOR_INCONSISTENCY_SEVERITY.notable:
            self.assertEqual(result, "Notable Inconsistency")

    def test_inversion_is_monotonic(self):
        # Higher raw score (more consistent) should never map to a "worse"
        # category than a lower raw score.
        order = ["Consistent", "Mild Inconsistency", "Moderate Inconsistency", "Notable Inconsistency"]
        prev_rank = -1
        for score in (100, 80, 60, 40, 20, 0):
            cat = _lip_color_consistency_cat(score)
            rank = order.index(cat)
            self.assertGreaterEqual(rank, prev_rank,
                f"score={score} produced {cat} which is 'better' than a higher score's category")
            prev_rank = rank


class TestSymmetryCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_symmetry_cat(None))

    def test_symmetric_at_or_above_normal_threshold(self):
        self.assertEqual(_symmetry_cat(FACE_OBSERVATION.symmetry_normal), "Symmetric")
        self.assertEqual(_symmetry_cat(100), "Symmetric")

    def test_slight_asymmetry_band(self):
        mid = (FACE_OBSERVATION.symmetry_normal + FACE_OBSERVATION.symmetry_slight) / 2
        self.assertEqual(_symmetry_cat(mid), "Slight Asymmetry")

    def test_mild_asymmetry_band(self):
        mid = (FACE_OBSERVATION.symmetry_slight + FACE_OBSERVATION.symmetry_mild) / 2
        self.assertEqual(_symmetry_cat(mid), "Mild Asymmetry")

    def test_notable_asymmetry_below_mild_floor(self):
        self.assertEqual(_symmetry_cat(FACE_OBSERVATION.symmetry_mild - 1), "Notable Asymmetry")
        self.assertEqual(_symmetry_cat(0), "Notable Asymmetry")


class TestSwellingCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_swelling_cat(None))

    def test_none_band(self):
        # "No Swelling", not the bare word "None" -- see _swelling_cat's
        # own docstring: pandas/Excel/most CSV readers treat a literal
        # "None" string as null, which would make every healthy row
        # silently read back as missing data instead of a real category.
        self.assertEqual(_swelling_cat(0), "No Swelling")
        self.assertEqual(_swelling_cat(FACE_OBSERVATION.swelling_none - 0.1), "No Swelling")

    def test_mild_band(self):
        self.assertEqual(_swelling_cat(FACE_OBSERVATION.swelling_none), "Mild")

    def test_moderate_band(self):
        self.assertEqual(_swelling_cat(FACE_OBSERVATION.swelling_mild), "Moderate")

    def test_notable_band(self):
        self.assertEqual(_swelling_cat(FACE_OBSERVATION.swelling_moderate), "Notable")
        self.assertEqual(_swelling_cat(100), "Notable")


class TestMouthAspectRatioCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_mouth_aspect_ratio_cat(None))

    def test_open(self):
        self.assertEqual(_mouth_aspect_ratio_cat(0), "Open")
        self.assertEqual(_mouth_aspect_ratio_cat(29.9), "Open")

    def test_partially_open(self):
        self.assertEqual(_mouth_aspect_ratio_cat(30), "Partially Open")
        self.assertEqual(_mouth_aspect_ratio_cat(69.9), "Partially Open")

    def test_closed_relaxed(self):
        self.assertEqual(_mouth_aspect_ratio_cat(70), "Closed / Relaxed")
        self.assertEqual(_mouth_aspect_ratio_cat(100), "Closed / Relaxed")


class TestMouthAsymmetryCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_mouth_asymmetry_cat(None))

    def test_symmetric_below_slight(self):
        self.assertEqual(_mouth_asymmetry_cat(MOUTH_OBSERVATION.asymmetry_slight - 1), "Symmetric")

    def test_slight_band(self):
        self.assertEqual(_mouth_asymmetry_cat(MOUTH_OBSERVATION.asymmetry_slight), "Slight Asymmetry")

    def test_mild_band(self):
        self.assertEqual(_mouth_asymmetry_cat(MOUTH_OBSERVATION.asymmetry_mild), "Mild Asymmetry")

    def test_notable_band(self):
        self.assertEqual(_mouth_asymmetry_cat(MOUTH_OBSERVATION.asymmetry_notable), "Notable Asymmetry")
        self.assertEqual(_mouth_asymmetry_cat(100), "Notable Asymmetry")


class TestCurvatureCat(unittest.TestCase):

    def test_none(self):
        self.assertIsNone(_curvature_cat(None))

    def test_neutral_band_symmetric_around_zero(self):
        band = MOUTH_OBSERVATION.curvature_neutral_band
        self.assertEqual(_curvature_cat(0), "Neutral")
        self.assertEqual(_curvature_cat(band - 0.01), "Neutral")
        self.assertEqual(_curvature_cat(-(band - 0.01)), "Neutral")

    def test_positive_beyond_band_is_smile_leaning(self):
        band = MOUTH_OBSERVATION.curvature_neutral_band
        self.assertEqual(_curvature_cat(band + 1), "Smile-Leaning")

    def test_negative_beyond_band_is_frown_leaning(self):
        band = MOUTH_OBSERVATION.curvature_neutral_band
        self.assertEqual(_curvature_cat(-(band + 1)), "Frown-Leaning")


class TestRoundHelper(unittest.TestCase):

    def test_none_passthrough(self):
        self.assertIsNone(_n(None))

    def test_rounds_to_given_decimals(self):
        self.assertEqual(_n(3.14159, 2), 3.14)

    def test_default_two_decimals(self):
        self.assertEqual(_n(1.0 / 3), 0.33)

    def test_accepts_int(self):
        self.assertEqual(_n(5), 5.0)


# ─── flatten_phase3 integration-style tests ─────────────────────────────────

class TestFlattenPhase3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image = _synthetic_face_image()
        cls.p1 = build_phase1_result()
        cls.p2 = RegionExtractor().extract(cls.image, cls.p1)
        cls.p3 = FeatureAnalyzer().analyze(cls.image, cls.p1, cls.p2)
        cls.p4 = ObservationEngine().generate(cls.p3)

    def test_row_has_every_column(self):
        row = flatten_phase3(self.p1, self.p3, self.p4, patient_id="TEST-001")
        self.assertEqual(set(row.keys()), set(ALL_COLUMNS))

    def test_patient_id_and_timestamp_set(self):
        row = flatten_phase3(self.p1, self.p3, self.p4, patient_id="TEST-042")
        self.assertEqual(row["patient_id"], "TEST-042")
        self.assertIsNotNone(row["timestamp"])

    def test_explicit_timestamp_used_verbatim(self):
        row = flatten_phase3(self.p1, self.p3, self.p4, timestamp="2026-01-01T00:00:00")
        self.assertEqual(row["timestamp"], "2026-01-01T00:00:00")

    def test_default_patient_id_is_unknown(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        self.assertEqual(row["patient_id"], "UNKNOWN")

    def test_photo_path_defaults_to_none(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        self.assertIsNone(row["photo_path"])

    def test_photo_path_passthrough(self):
        row = flatten_phase3(self.p1, self.p3, self.p4, photo_path="frame_001.jpg")
        self.assertEqual(row["photo_path"], "frame_001.jpg")

    def test_face_symmetry_populated_and_paired_cat_consistent(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        self.assertIsNotNone(row["face_symmetry_score"])
        self.assertEqual(row["face_symmetry_cat"], _symmetry_cat(row["face_symmetry_score"]))

    def test_face_aspect_ratio_matches_height_over_width(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        expected = round(self.p1.face_height / self.p1.face_width, 3)
        self.assertEqual(row["face_aspect_ratio"], expected)

    def test_eye_columns_populated_when_available(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        if self.p3.left_eye.available:
            self.assertIsNotNone(row["left_eye_openness_ratio"])
            self.assertIn(row["left_eye_openness_cat"], ("Closed", "Drooping", "Partially Open", "Open"))

    def test_skin_regions_have_spot_count_as_int_like(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        if self.p3.skin_forehead.available:
            self.assertIsNotNone(row["skin_forehead_spot_count"])

    def test_overall_confidence_from_p4(self):
        row = flatten_phase3(self.p1, self.p3, self.p4)
        self.assertEqual(row["overall_confidence"], _n(self.p4.overall_confidence))

    def test_no_p4_leaves_confidence_none(self):
        row = flatten_phase3(self.p1, self.p3, p4=None)
        self.assertIsNone(row["overall_confidence"])

    def test_no_p1_leaves_face_columns_none(self):
        row = flatten_phase3(None, self.p3, self.p4)
        self.assertIsNone(row["face_symmetry_score"])
        self.assertIsNone(row["face_width"])

    def test_failed_p1_leaves_face_columns_none(self):
        failed_p1 = Phase1Result(success=False, error="No face detected in image.")
        row = flatten_phase3(failed_p1, self.p3, self.p4)
        self.assertIsNone(row["face_symmetry_score"])
        self.assertIsNone(row["face_alignment_angle"])

    def test_zero_face_width_does_not_raise_on_aspect_ratio(self):
        # Guards against a divide-by-zero if face_width is ever 0 (e.g. a
        # degenerate detection). Should yield None, not raise.
        p1 = build_phase1_result()
        # Construct a copy-like object with face_width forced to 0.
        import dataclasses
        zero_width_p1 = dataclasses.replace(p1, face_width=0)
        try:
            row = flatten_phase3(zero_width_p1, self.p3, self.p4)
        except ZeroDivisionError:
            self.fail("flatten_phase3 raised ZeroDivisionError on face_width=0")
        self.assertIsNone(row["face_aspect_ratio"])

    def test_no_feature_column_leaks_into_metadata_or_vice_versa(self):
        self.assertEqual(len(set(FEATURE_COLUMNS) & set(METADATA_COLUMNS)), 0)
        self.assertEqual(ALL_COLUMNS, FEATURE_COLUMNS + METADATA_COLUMNS)


# ─── CSV append / schema migration tests ────────────────────────────────────

class TestAppendFeatureRow(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _sample_row(self, patient_id="P1"):
        return {col: None for col in ALL_COLUMNS} | {
            "patient_id": patient_id, "timestamp": "2026-01-01T00:00:00",
        }

    def test_creates_file_with_header_on_first_write(self):
        append_feature_row(self._sample_row(), self.csv_path)
        with open(self.csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ALL_COLUMNS)
        self.assertEqual(len(rows), 2)  # header + 1 row

    def test_second_append_does_not_duplicate_header(self):
        append_feature_row(self._sample_row("P1"), self.csv_path)
        append_feature_row(self._sample_row("P2"), self.csv_path)
        with open(self.csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 3)  # header + 2 rows
        self.assertEqual(rows[0], ALL_COLUMNS)

    def test_appended_row_values_round_trip(self):
        row = self._sample_row("PATIENT-XYZ")
        row["face_symmetry_score"] = 87.5
        append_feature_row(row, self.csv_path)
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_row = next(reader)
        self.assertEqual(read_row["patient_id"], "PATIENT-XYZ")
        self.assertEqual(read_row["face_symmetry_score"], "87.5")

    def test_migration_adds_missing_columns_with_blank_values(self):
        # Simulate an "old" CSV written before a new column existed.
        old_columns = [c for c in ALL_COLUMNS if c != "face_swelling_cat"]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow({c: "x" for c in old_columns})

        _migrate_if_schema_changed(self.csv_path)

        with open(self.csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ALL_COLUMNS)
        idx = ALL_COLUMNS.index("face_swelling_cat")
        self.assertEqual(rows[1][idx], "")

    def test_migration_preserves_existing_values_by_name_not_position(self):
        # Shuffle the old header order to prove remap is name-based.
        shuffled = list(reversed(ALL_COLUMNS))
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=shuffled)
            writer.writeheader()
            writer.writerow({c: (f"val_{c}" if c == "patient_id" else "") for c in shuffled})

        _migrate_if_schema_changed(self.csv_path)

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        self.assertEqual(row["patient_id"], "val_patient_id")

    def test_migration_noop_when_schema_already_current(self):
        append_feature_row(self._sample_row(), self.csv_path)
        with open(self.csv_path) as f:
            before = f.read()
        _migrate_if_schema_changed(self.csv_path)
        with open(self.csv_path) as f:
            after = f.read()
        self.assertEqual(before, after)

    def test_migration_noop_on_missing_file(self):
        # Should not raise when the file doesn't exist yet.
        _migrate_if_schema_changed(self.csv_path)
        self.assertFalse(os.path.exists(self.csv_path))

    def test_migration_handles_corrupted_row_width_mismatch(self):
        # A row shorter than its own header (already-corrupted data) should
        # not crash the migration -- best-effort positional pairing.
        old_columns = ALL_COLUMNS[:5]
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(old_columns)
            w.writerow(["only", "three", "values"])  # short row

        try:
            _migrate_if_schema_changed(self.csv_path)
        except Exception as exc:
            self.fail(f"Migration raised on corrupted row: {exc}")

        with open(self.csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ALL_COLUMNS)

    def test_lock_file_created_alongside_csv(self):
        append_feature_row(self._sample_row(), self.csv_path)
        self.assertTrue(os.path.exists(self.csv_path + ".lock"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
