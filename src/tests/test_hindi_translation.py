"""
test_hindi_translation.py

Coverage for the Phase 6 Hindi translation toggle:
    - phase5_report/hindi_translation.py's dictionary-based translator in
      isolation (unit tests on known English -> Hindi mappings).
    - TTSReportGenerator.generate(..., language=...) end-to-end, using the
      same synthetic Phase 1-4 fixture as test_phase5_report.py, so the
      toggle is exercised against a realistic, non-trivial report rather
      than hand-built strings only.

No camera, no model file, no network.
"""

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
from phase5_report.tts_report import TTSReportGenerator
from phase5_report.hindi_translation import translate_tts_text


def _synthetic_face_image() -> np.ndarray:
    """Same synthetic face image as test_phase5_report.py / test_integration.py."""
    img = np.full((IMG_H, IMG_W, 3), (200, 190, 180), dtype=np.uint8)
    cv2.ellipse(img, (FACE_CX, FACE_CY), (FACE_RX, FACE_RY), 0, 0, 360, (150, 160, 190), -1)
    eye_y = int(FACE_CY - FACE_RY * 0.25)
    for sign in (-1, 1):
        ex = int(FACE_CX + sign * FACE_RX * 0.45 * 0.7)
        cv2.ellipse(img, (ex, eye_y), (18, 8), 0, 0, 360, (80, 70, 70), -1)
    mouth_y = int(FACE_CY + FACE_RY * 0.55)
    cv2.ellipse(img, (FACE_CX, mouth_y), (40, 12), 0, 0, 360, (90, 80, 150), -1)
    return img


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


class TestHindiTranslationUnit(unittest.TestCase):
    """Unit-level coverage of translate_tts_text() in isolation."""

    def test_empty_and_none_pass_through(self):
        self.assertEqual(translate_tts_text(""), "")
        self.assertIsNone(translate_tts_text(None))

    def test_plain_english_with_no_matches_is_unchanged(self):
        text = "xyz unmatched filler zzz 12345"
        self.assertEqual(translate_tts_text(text), text)

    def test_scaffold_phrase_translated(self):
        out = translate_tts_text("For your eyes, everything looked fine.")
        self.assertIn("आपकी आंखों की बात करें तो", out)
        self.assertNotIn("For your eyes,", out)

    def test_greeting_preserves_name(self):
        out = translate_tts_text(
            "Hi Aayu, here is your facial health summary for 27 September 2026."
        )
        self.assertIn("नमस्ते Aayu", out)
        self.assertIn("यह रहा आपका चेहरे की सेहत का सारांश", out)

    def test_closing_template_preserves_sign_off(self):
        closing = (
            "That wraps up your summary, Aayu. Everything looks good, keep it up. "
            "Check back in a few days to keep tracking your progress."
        )
        out = translate_tts_text(closing)
        self.assertIn("यह रहा आपका पूरा सारांश, Aayu", out)
        self.assertIn("सब कुछ ठीक दिख रहा है", out)

    def test_closing_template_without_name(self):
        closing = (
            "That wraps up your summary. Everything looks good, keep it up. "
            "Check back in a few days to keep tracking your progress."
        )
        out = translate_tts_text(closing)
        # No stray ", " left over when sign_off was empty.
        self.assertIn("यह रहा आपका पूरा सारांश।", out)

    def test_word_phrase_layer_translates_region_and_severity_words(self):
        out = translate_tts_text("Redness was noticeable on the left cheek.")
        self.assertIn("लालिमा", out)
        self.assertIn("ध्यान देने योग्य", out)
        self.assertIn("बाएं गाल", out)

    def test_longer_phrase_wins_over_shorter_word(self):
        # "uneven skin texture" should match as one unit, not split into
        # separate "skin" + leftover "uneven ... texture" fragments.
        out = translate_tts_text("There is uneven skin texture on the forehead.")
        self.assertIn("त्वचा की असमान बनावट", out)
        # The word-level "skin" entry should not have fired separately
        # inside the already-matched longer phrase.
        self.assertNotIn("त्वचात्वचा", out)

    def test_translation_is_idempotent_on_repeated_calls(self):
        text = "For your eyes, redness was noticeable on the left cheek."
        once = translate_tts_text(text)
        # Calling again on the already-Hindi output shouldn't crash or
        # double-translate anything further (nothing new left to match).
        twice = translate_tts_text(once)
        self.assertEqual(once, twice)


class TestTTSReportLanguageToggle(unittest.TestCase):
    """End-to-end: TTSReportGenerator.generate(language=...) against a
    real Phase 1-4 pipeline run."""

    @classmethod
    def setUpClass(cls):
        cls.image = _synthetic_face_image()
        cls.p1 = build_phase1_result()
        cls.p2 = RegionExtractor().extract(cls.image, cls.p1)
        cls.p3 = FeatureAnalyzer().analyze(cls.image, cls.p1, cls.p2)
        cls.p4 = ObservationEngine().generate(cls.p3)

    def test_default_language_is_english(self):
        result = TTSReportGenerator().generate(
            self.p1, self.p3, self.p4, patient_id="TEST-HI-001",
        )
        self.assertEqual(result["language"], "en")
        self.assertFalse(_contains_devanagari(result["text"]))

    def test_hindi_language_produces_devanagari_text(self):
        result = TTSReportGenerator().generate(
            self.p1, self.p3, self.p4, patient_id="TEST-HI-002", language="hi",
        )
        self.assertEqual(result["language"], "hi")
        self.assertTrue(_contains_devanagari(result["text"]))
        # Nothing should be dropped -- translated text is still non-trivial
        # in length relative to the English original.
        english = TTSReportGenerator().generate(
            self.p1, self.p3, self.p4, patient_id="TEST-HI-002",
        )["text"]
        self.assertGreater(len(result["text"]), len(english) * 0.5)

    def test_unrecognized_language_falls_back_to_english(self):
        result = TTSReportGenerator().generate(
            self.p1, self.p3, self.p4, patient_id="TEST-HI-003", language="fr",
        )
        self.assertEqual(result["language"], "en")
        self.assertFalse(_contains_devanagari(result["text"]))

    def test_hindi_output_written_to_distinct_file(self):
        with tempfile.TemporaryDirectory() as out_dir:
            en_result = TTSReportGenerator().generate(
                self.p1, self.p3, self.p4,
                patient_id="TEST-HI-004", output_dir=out_dir,
            )
            hi_result = TTSReportGenerator().generate(
                self.p1, self.p3, self.p4,
                patient_id="TEST-HI-004", output_dir=out_dir, language="hi",
            )
            self.assertNotEqual(en_result["path"], hi_result["path"])
            self.assertTrue(os.path.exists(en_result["path"]))
            self.assertTrue(os.path.exists(hi_result["path"]))
            with open(hi_result["path"], encoding="utf-8") as f:
                on_disk = f.read()
            self.assertTrue(_contains_devanagari(on_disk))


if __name__ == "__main__":
    unittest.main()
