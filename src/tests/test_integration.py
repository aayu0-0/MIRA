"""
test_integration.py — Phase 1 through Phase 5 cross-phase contract test.

End-to-end pipeline test: Phase 1 result (synthetic, real-MediaPipe-index
fixture) -> Phase 2 (region extraction) -> Phase 3 (feature analysis) ->
Phase 4 (observation engine) -> Phase 5 (all 4 report outputs), exercising
the actual cross-phase data contracts instead of each phase in isolation.

Skips only real MediaPipe face detection (Phase 1's model inference itself),
since that requires the .task model file and is not what this test is
checking — fixtures.build_phase1_result() already produces a Phase1Result
shaped exactly like FaceDetector.analyze() would, built from the same named
MediaPipe index groups, so phases 2-5 are exercised with a fully realistic
contract.

Scope note: this is deliberately narrower than an earlier version of this
file. That version imported `phase5_report.report_generator.ReportGenerator`
(a single unified generator) and drove `pipeline._run_analysis` /
`pipeline.AlignmentCheck` directly — neither exists in the current source
tree. Phase 5 is now four separate generators (metrics/doctor/user/tts —
see test_phase5_report.py for dedicated coverage of those), and the
history/quality-gating wiring (_run_analysis, AlignmentCheck,
HistoryStore-backed history checks) now lives in main.py, which imports the
phase_a_* (history/trends/voice) modules that are out of scope for this
five-phase test tree. If/when main.py's own imports are fixed, that wiring
deserves its own integration test importing main.py directly rather than
being bolted onto this file.

No camera, no model file, no network. Runs with plain unittest (pytest not
required, though tests are pytest-discoverable too).
"""

import os
import sys
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


def _synthetic_face_image() -> np.ndarray:
    """
    A plain BGR image with a filled ellipse standing in for a face, plus
    darker ellipses for eyes/mouth so Phase 3's region-based pixel stats
    (redness, dark circles, etc.) have non-degenerate, non-crashing input.
    Geometry matches the fixture's FACE_CX/CY/RX/RY so masks land on it.
    """
    img = np.full((IMG_H, IMG_W, 3), (200, 190, 180), dtype=np.uint8)  # skin-ish BGR
    cv2.ellipse(img, (FACE_CX, FACE_CY), (FACE_RX, FACE_RY), 0, 0, 360, (150, 160, 190), -1)
    eye_y = int(FACE_CY - FACE_RY * 0.25)
    for sign in (-1, 1):
        ex = int(FACE_CX + sign * FACE_RX * 0.45 * 0.7)
        cv2.ellipse(img, (ex, eye_y), (18, 8), 0, 0, 360, (80, 70, 70), -1)
    mouth_y = int(FACE_CY + FACE_RY * 0.55)
    cv2.ellipse(img, (FACE_CX, mouth_y), (40, 12), 0, 0, 360, (90, 80, 150), -1)
    return img


class TestPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image = _synthetic_face_image()
        cls.p1 = build_phase1_result()
        cls.p2 = RegionExtractor().extract(cls.image, cls.p1)
        cls.p3 = FeatureAnalyzer().analyze(cls.image, cls.p1, cls.p2)
        cls.p4 = ObservationEngine().generate(cls.p3)

    # ── Phase 1 -> 2 ─────────────────────────────────────────────────────────

    def test_phase2_receives_valid_phase1(self):
        self.assertTrue(self.p1.success)
        self.assertIsNotNone(self.p1.bbox)

    def test_phase2_extracts_expected_regions(self):
        # Core regions Phase 3/4 depend on must always be present, regardless
        # of however many extra regions Phase 2 also extracts.
        required = {"forehead", "left_cheek", "right_cheek", "lips",
                    "left_eye", "right_eye", "nose"}
        self.assertTrue(required.issubset(set(self.p2.regions.keys())))
        for name in ("forehead", "left_cheek", "right_cheek", "lips"):
            self.assertTrue(self.p2.regions[name].available, f"{name} not available")

    def test_phase2_masked_crops_are_nonempty_arrays(self):
        for name, region in self.p2.regions.items():
            if region.available:
                self.assertIsNotNone(region.masked_crop)
                self.assertGreater(region.masked_crop.size, 0)

    # ── Phase 2 -> 3 ─────────────────────────────────────────────────────────

    def test_phase3_consumes_phase2_regions_without_error(self):
        # If Phase 2/3's region-name or array-shape contract drifted, .analyze
        # would raise (KeyError / shape mismatch) before reaching here.
        self.assertIsNotNone(self.p3)

    def test_phase3_scores_are_in_valid_range(self):
        score_fields = [
            self.p3.left_eye.dark_circle_score, self.p3.left_eye.redness_score,
            self.p3.left_eye.puffiness_score,
            self.p3.right_eye.dark_circle_score, self.p3.right_eye.redness_score,
            self.p3.right_eye.puffiness_score,
            self.p3.skin_forehead.acne_score, self.p3.skin_forehead.redness_score,
            self.p3.skin_left_cheek.acne_score, self.p3.skin_right_cheek.acne_score,
            self.p3.lips.dryness_score, self.p3.lips.pallor_score,
        ]
        for s in score_fields:
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 100)

    # ── Phase 3 -> 4 ─────────────────────────────────────────────────────────

    def test_phase4_consumes_phase3_without_error(self):
        self.assertIsNotNone(self.p4)

    def test_phase4_overall_confidence_in_range(self):
        self.assertGreaterEqual(self.p4.overall_confidence, 0)
        self.assertLessEqual(self.p4.overall_confidence, 100)

    def test_phase4_produces_findings_for_every_section(self):
        # Empty finding lists would silently produce a blank-looking report
        # without any phase raising an error.
        self.assertTrue(len(self.p4.eyes.left) > 0 or len(self.p4.eyes.right) > 0)
        self.assertTrue(len(self.p4.lips.findings) > 0)
        self.assertTrue(len(self.p4.overall) > 0)

    def test_phase4_no_diagnostic_language_leaks(self):
        # Cheap guard against regressions in the non-diagnostic language
        # requirement: scan every generated *finding* string (not the
        # intentional "this is not a diagnosis" disclaimer, which legitimately
        # contains the word) for obvious diagnostic/medical-verdict terms.
        banned = ("you have ", "disease", "disorder", "condition is", "diagnosed with")
        all_text = " ".join(
            obs.finding.lower()
            for obs in (self.p4.eyes.left + self.p4.eyes.right +
                        self.p4.skin.forehead + self.p4.skin.left_cheek +
                        self.p4.skin.right_cheek + self.p4.skin.overall +
                        self.p4.lips.findings)
        )
        for term in banned:
            self.assertNotIn(term, all_text, f"possible diagnostic language: {term!r}")

    # ── Phase 4 -> 5 ─────────────────────────────────────────────────────────
    # (Broader Phase 5 coverage — all 4 generators, quality-warning banners,
    # severity/color regression — lives in test_phase5_report.py. These two
    # stay here as a minimal "the chain doesn't break" cross-phase check.)

    def test_phase5_metrics_consumes_phase4_without_error(self):
        # "phase4_summary" was the key name under the old phase-prefixed
        # metrics schema; the current flat schema surfaces the same Phase 4
        # data under "Overall" (see test_metrics_schema_has_expected_top_level_sections
        # in test_phase5_report.py for the full current key set).
        result = MetricsReportGenerator().generate(self.p1, self.p3, self.p4, patient_id="TEST-001")
        self.assertIn("Overall", result["data"])

    def test_phase5_doctor_pdf_generates_nonempty_bytes(self):
        pdf_bytes = DoctorReportGenerator().generate(
            self.p1, self.p2, self.p3, self.p4,
            annotated_image_bgr=self.image,
            patient_id="TEST-001",
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main(verbosity=2)