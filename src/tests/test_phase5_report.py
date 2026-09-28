"""
test_phase5_report.py

Coverage for the four Phase 5 report generators (metrics_report.py,
doctor_report.py, user_report.py, tts_report.py) — the current split of
what used to be a single phase5_report/report_generator.py (see note in
test_integration.py). Uses the same synthetic Phase 1 fixture as
test_integration.py so Phase 2/3/4 are exercised with a realistic,
non-trivial mix of severities (the default fixture already produces
None/Mild/Moderate/Notable findings — see setUpClass), which matters
here specifically: every severity level is rendered through the same
color-lookup path in the PDF generators, so running generate() against
this fixture is a real regression guard against the
`Color.hexval()[1:]` vs `[2:]` bug (ReportLab's hexval() returns
"0xRRGGBB", not "#RRGGBB" — slicing off only the first character left a
stray "x" in the hex string and crashed PDF generation for every
observation, at every severity, including "None").

No camera, no model file, no network.
"""

import io
import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.fixtures import build_phase1_result, IMG_W, IMG_H, FACE_CX, FACE_CY, FACE_RX, FACE_RY

from phase2_region_extraction.region_extractor import RegionExtractor
from phase3_feature_analysis.feature_analyzer import FeatureAnalyzer
from phase4_observation_engine.observation_engine import ObservationEngine
from phase5_report.metrics_report import MetricsReportGenerator
from phase5_report.doctor_report import DoctorReportGenerator
from phase5_report.user_report import UserReportGenerator
from phase5_report.tts_report import TTSReportGenerator


def _synthetic_face_image() -> np.ndarray:
    """Same synthetic face image as test_integration.py — geometry matches
    the fixture's FACE_CX/CY/RX/RY so masks land on it."""
    img = np.full((IMG_H, IMG_W, 3), (200, 190, 180), dtype=np.uint8)
    cv2.ellipse(img, (FACE_CX, FACE_CY), (FACE_RX, FACE_RY), 0, 0, 360, (150, 160, 190), -1)
    eye_y = int(FACE_CY - FACE_RY * 0.25)
    for sign in (-1, 1):
        ex = int(FACE_CX + sign * FACE_RX * 0.45 * 0.7)
        cv2.ellipse(img, (ex, eye_y), (18, 8), 0, 0, 360, (80, 70, 70), -1)
    mouth_y = int(FACE_CY + FACE_RY * 0.55)
    cv2.ellipse(img, (FACE_CX, mouth_y), (40, 12), 0, 0, 360, (90, 80, 150), -1)
    return img


def _pdf_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return " ".join(page.extract_text() for page in reader.pages)


class TestPhase5Reports(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image = _synthetic_face_image()
        cls.p1 = build_phase1_result()
        cls.p2 = RegionExtractor().extract(cls.image, cls.p1)
        cls.p3 = FeatureAnalyzer().analyze(cls.image, cls.p1, cls.p2)
        cls.p4 = ObservationEngine().generate(cls.p3)

        # Sanity check on the fixture itself: if this ever stops being
        # true, the color-regression coverage below silently loses value
        # (would only exercise a subset of severities).
        all_obs = (
            cls.p4.eyes.left + cls.p4.eyes.right +
            cls.p4.skin.forehead + cls.p4.skin.left_cheek +
            cls.p4.skin.right_cheek + cls.p4.skin.nose + cls.p4.skin.overall +
            cls.p4.lips.findings + cls.p4.mouth.findings
        )
        cls._all_severities = {o.severity.value for o in all_obs}

    def test_fixture_covers_all_severity_levels(self):
        # Guards the regression-test premise above: None/Mild/Moderate/Notable
        # must all appear so the PDF color path is exercised for every band.
        self.assertEqual(
            self._all_severities, {"None", "Mild", "Moderate", "Notable"},
            "fixture no longer produces all severities — color-regression "
            "coverage in this file would be weaker than intended",
        )

    # ── Metrics JSON (File 1) ───────────────────────────────────────────────

    def test_metrics_report_returns_data_and_writes_json(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = MetricsReportGenerator().generate(
                self.p1, self.p3, self.p4,
                patient_id="TEST-001", output_dir=out_dir,
            )
            self.assertIn("data", result)
            self.assertIn("path", result)
            self.assertTrue(os.path.exists(result["path"]))
            with open(result["path"]) as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk["patient_id"], "TEST-001")

    def test_metrics_report_without_output_dir_skips_file(self):
        result = MetricsReportGenerator().generate(self.p1, self.p3, self.p4)
        self.assertIsNone(result["path"])
        self.assertIsInstance(result["data"], dict)

    def test_metrics_schema_has_expected_top_level_sections(self):
        # NOTE: metrics_report.py was refactored from a phase-prefixed,
        # snake_case schema (phase1_geometry / phase3_face / phase4_summary /
        # schema_version) to a flat, Title Case, region-grouped schema. This
        # test now asserts the current shape.
        result = MetricsReportGenerator().generate(self.p1, self.p3, self.p4)
        expected_keys = {
            "patient_id", "timestamp", "quality_warnings",
            "Face", "Eyes", "Skin", "Lips", "Mouth", "Overall",
        }
        self.assertTrue(expected_keys.issubset(result["data"].keys()))

    def test_metrics_json_is_actually_serializable(self):
        # Catches numpy scalar leakage (np.float32 etc. are not
        # JSON-serializable by default) before it reaches disk.
        result = MetricsReportGenerator().generate(self.p1, self.p3, self.p4)
        json.dumps(result["data"])  # raises TypeError if anything leaked through

    # ── Doctor PDF (File 2) ──────────────────────────────────────────────────

    def test_doctor_report_generates_valid_pdf_bytes(self):
        # Regression coverage for the hexval() slicing bug: this call spans
        # every severity band (see test_fixture_covers_all_severity_levels)
        # and previously raised ValueError: Invalid color value '#x......'
        pdf_bytes = DoctorReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-001",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_doctor_report_writes_to_output_path(self):
        with tempfile.TemporaryDirectory() as out_dir:
            path = os.path.join(out_dir, "doctor.pdf")
            DoctorReportGenerator().generate(
                self.p1, self.p2, self.p3, self.p4,
                annotated_image_bgr=self.image,
                patient_id="TEST-001", output_path=path,
            )
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)

    def test_doctor_report_quality_warning_banner_present(self):
        pdf_bytes = DoctorReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-WARN",
            quality_warnings=["Find better lighting — too dark", "Keep your head level"],
        )
        text = _pdf_text(pdf_bytes)
        self.assertIn("capture quality warning", text.lower())
        self.assertIn("Find better lighting", text)
        self.assertIn("Keep your head level", text)

    def test_doctor_report_no_banner_when_warnings_empty(self):
        pdf_bytes = DoctorReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-CLEAN", quality_warnings=[],
        )
        text = _pdf_text(pdf_bytes)
        self.assertNotIn("capture quality warning", text.lower())

    # ── User PDF (File 3) ────────────────────────────────────────────────────

    def test_user_report_generates_valid_pdf_bytes(self):
        pdf_bytes = UserReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-001",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_user_report_writes_to_output_path(self):
        with tempfile.TemporaryDirectory() as out_dir:
            path = os.path.join(out_dir, "user.pdf")
            UserReportGenerator().generate(
                self.p1, self.p2, self.p3, self.p4,
                annotated_image_bgr=self.image,
                patient_id="TEST-001", output_path=path,
            )
            self.assertTrue(os.path.exists(path))

    def test_user_report_quality_note_present(self):
        # User PDF uses different banner copy ("Photo Quality Note") than
        # the doctor PDF ("CAPTURE QUALITY WARNING") — check its own wording.
        pdf_bytes = UserReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-WARN",
            quality_warnings=["Find better lighting — too dark"],
        )
        text = _pdf_text(pdf_bytes)
        self.assertIn("photo quality note", text.lower())
        self.assertIn("Find better lighting", text)

    def test_user_report_no_banner_when_warnings_empty(self):
        pdf_bytes = UserReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-CLEAN", quality_warnings=[],
        )
        text = _pdf_text(pdf_bytes)
        self.assertNotIn("photo quality note", text.lower())

    # ── TTS text (File 4) ────────────────────────────────────────────────────

    def test_tts_report_returns_text_and_writes_file(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = TTSReportGenerator().generate(
                self.p1, self.p3, self.p4,
                patient_id="TEST-001", output_dir=out_dir,
            )
            self.assertIsInstance(result["text"], str)
            self.assertGreater(len(result["text"]), 0)
            self.assertTrue(os.path.exists(result["path"]))
            with open(result["path"]) as f:
                self.assertEqual(f.read(), result["text"])

    def test_tts_report_uses_patient_id_for_filename_not_speech(self):
        # By design (see TTSReportGenerator.generate docstring): patient_id
        # names the output file but is never spoken -- a voice report
        # reading an internal ID like "TEST-007" aloud to the patient
        # would be jarring. The doctor/user PDF reports DO print it (it's
        # a written record); only the spoken script omits it.
        with tempfile.TemporaryDirectory() as out_dir:
            result = TTSReportGenerator().generate(
                self.p1, self.p3, self.p4,
                patient_id="TEST-007", output_dir=out_dir,
            )
            self.assertNotIn("TEST-007", result["text"])
            self.assertIn("TEST-007", result["path"])

    def test_tts_report_without_output_dir_skips_file(self):
        result = TTSReportGenerator().generate(self.p1, self.p3, self.p4)
        self.assertIsNone(result["path"])

    # ── All four together (mirrors the manual Phase 1-5 smoke test) ─────────

    def test_all_four_reports_generate_together(self):
        with tempfile.TemporaryDirectory() as out_dir:
            metrics = MetricsReportGenerator().generate(
                self.p1, self.p3, self.p4, patient_id="TEST-ALL", output_dir=out_dir)
            doctor_path = os.path.join(out_dir, "report_doctor_TEST-ALL.pdf")
            DoctorReportGenerator().generate(
                self.p1, self.p2, self.p3, self.p4,
                annotated_image_bgr=self.image, patient_id="TEST-ALL",
                output_path=doctor_path)
            user_path = os.path.join(out_dir, "report_user_TEST-ALL.pdf")
            UserReportGenerator().generate(
                self.p1, self.p2, self.p3, self.p4,
                annotated_image_bgr=self.image, patient_id="TEST-ALL",
                output_path=user_path)
            tts = TTSReportGenerator().generate(
                self.p1, self.p3, self.p4, patient_id="TEST-ALL", output_dir=out_dir)

            self.assertTrue(os.path.exists(metrics["path"]))
            self.assertTrue(os.path.exists(doctor_path))
            self.assertTrue(os.path.exists(user_path))
            self.assertTrue(os.path.exists(tts["path"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)