"""
test_pipeline_helpers.py — pure, tkinter/camera-free helpers in pipeline.py:

  - sanitize_filename_part(): making user-entered names/IDs filesystem-safe
  - next_capture_number(): finding the next free session number for a patient
  - _mean_brightness() / _laplacian_variance(): cheap frame-quality metrics
  - _quick_quality_score(): the live-preview ranking score

No camera, no model file, no network.
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import (
    sanitize_filename_part, next_capture_number,
    _mean_brightness, _laplacian_variance, _quick_quality_score,
)


class TestSanitizeFilenamePart(unittest.TestCase):

    def test_plain_alnum_unchanged(self):
        self.assertEqual(sanitize_filename_part("Aayush123"), "Aayush123")

    def test_illegal_windows_chars_replaced(self):
        self.assertEqual(sanitize_filename_part('a<b>c:d"e/f\\g|h?i*j'), "a_b_c_d_e_f_g_h_i_j")

    def test_control_chars_replaced(self):
        self.assertEqual(sanitize_filename_part("a\x01b\x1fc"), "a_b_c")

    def test_whitespace_collapsed_to_single_underscore(self):
        self.assertEqual(sanitize_filename_part("John   Doe"), "John_Doe")

    def test_tabs_and_newlines_collapsed(self):
        self.assertEqual(sanitize_filename_part("John\t\nDoe"), "John_Doe")

    def test_multiple_underscores_collapsed(self):
        self.assertEqual(sanitize_filename_part("a___b"), "a_b")

    def test_leading_trailing_dots_underscores_spaces_stripped(self):
        self.assertEqual(sanitize_filename_part("  ._John_Doe._  "), "John_Doe")

    def test_empty_string_becomes_unknown(self):
        self.assertEqual(sanitize_filename_part(""), "UNKNOWN")

    def test_only_illegal_chars_becomes_unknown(self):
        self.assertEqual(sanitize_filename_part("???"), "UNKNOWN")

    def test_only_whitespace_becomes_unknown(self):
        self.assertEqual(sanitize_filename_part("   "), "UNKNOWN")

    def test_non_string_input_is_stringified(self):
        self.assertEqual(sanitize_filename_part(12345), "12345")

    def test_idempotent(self):
        once = sanitize_filename_part("  Weird///Name**  ")
        twice = sanitize_filename_part(once)
        self.assertEqual(once, twice)


class TestNextCaptureNumber(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.photos_dir = os.path.join(self.tmpdir, "photos")
        self.reports_dir = os.path.join(self.tmpdir, "reports")
        os.makedirs(self.photos_dir)
        os.makedirs(self.reports_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_folders_start_at_one(self):
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 1)

    def test_missing_folders_start_at_one(self):
        n = next_capture_number("Jane_Doe", "001", "/no/such/photos", "/no/such/reports")
        self.assertEqual(n, 1)

    def test_finds_highest_existing_photo_number(self):
        open(os.path.join(self.photos_dir, "Jane_Doe_001_01.jpg"), "w").close()
        open(os.path.join(self.photos_dir, "Jane_Doe_001_03.jpg"), "w").close()
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 4)

    def test_reports_folder_also_considered(self):
        open(os.path.join(self.photos_dir, "Jane_Doe_001_01.jpg"), "w").close()
        open(os.path.join(self.reports_dir, "Jane_Doe_001_05_doctor.pdf"), "w").close()
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 6)

    def test_unrelated_files_ignored(self):
        open(os.path.join(self.photos_dir, "SomeoneElse_002_09.jpg"), "w").close()
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 1)

    def test_name_and_id_are_sanitized_before_matching(self):
        # A name containing illegal filename chars should match files that
        # were written using the sanitized form.
        open(os.path.join(self.photos_dir, "John_Doe_001_02.jpg"), "w").close()
        n = next_capture_number("John/Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 3)

    def test_two_digit_zero_padded_numbers_recognized(self):
        open(os.path.join(self.photos_dir, "Jane_Doe_001_09.jpg"), "w").close()
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 10)

    def test_non_numeric_suffix_after_prefix_ignored(self):
        open(os.path.join(self.photos_dir, "Jane_Doe_001_notanumber.jpg"), "w").close()
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 1)

    def test_prefix_must_match_exactly_not_substring(self):
        # "Jane_Doe_0012_01" should not be treated as patient "001"'s files
        # (prefix is "Jane_Doe_001_", which is not a prefix of "Jane_Doe_0012_...").
        open(os.path.join(self.photos_dir, "Jane_Doe_0012_07.jpg"), "w").close()
        n = next_capture_number("Jane_Doe", "001", self.photos_dir, self.reports_dir)
        self.assertEqual(n, 1)


class TestMeanBrightness(unittest.TestCase):

    def test_uniform_gray_returns_that_value(self):
        frame = np.full((50, 50), 128, dtype=np.uint8)
        self.assertAlmostEqual(_mean_brightness(frame), 128.0, places=3)

    def test_all_black_is_zero(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        self.assertEqual(_mean_brightness(frame), 0.0)

    def test_all_white_is_255(self):
        frame = np.full((10, 10), 255, dtype=np.uint8)
        self.assertEqual(_mean_brightness(frame), 255.0)

    def test_returns_python_float(self):
        frame = np.zeros((4, 4), dtype=np.uint8)
        self.assertIsInstance(_mean_brightness(frame), float)


class TestLaplacianVariance(unittest.TestCase):

    def test_uniform_frame_has_zero_variance(self):
        frame = np.full((60, 60), 100, dtype=np.uint8)
        self.assertAlmostEqual(_laplacian_variance(frame), 0.0, places=3)

    def test_noisy_frame_has_higher_variance_than_uniform(self):
        rng = np.random.default_rng(42)
        uniform = np.full((60, 60), 100, dtype=np.uint8)
        noisy = rng.integers(0, 256, (60, 60), dtype=np.uint8)
        self.assertGreater(_laplacian_variance(noisy), _laplacian_variance(uniform))

    def test_returns_python_float(self):
        frame = np.zeros((10, 10), dtype=np.uint8)
        self.assertIsInstance(_laplacian_variance(frame), float)


class TestQuickQualityScore(unittest.TestCase):

    def test_ideal_inputs_score_high(self):
        # Sharp (blur_var >= 300 saturates sharpness), perfect symmetry,
        # level alignment, ideal exposure (127.5).
        score = _quick_quality_score(blur_var=300, symmetry_score=100, alignment_angle=0, brightness=127.5)
        self.assertAlmostEqual(score, 100.0, places=3)

    def test_worst_inputs_score_low(self):
        score = _quick_quality_score(blur_var=0, symmetry_score=0, alignment_angle=45, brightness=127.5)
        self.assertLess(score, 40.0)

    def test_blur_above_saturation_point_does_not_exceed_cap(self):
        low = _quick_quality_score(300, 100, 0, 127.5)
        high = _quick_quality_score(10_000, 100, 0, 127.5)
        self.assertAlmostEqual(low, high, places=3)

    def test_symmetry_score_is_clamped_above_100(self):
        # Passing an out-of-range symmetry (shouldn't normally happen) must
        # not push the overall score above what 100 would give.
        normal = _quick_quality_score(300, 100, 0, 127.5)
        clamped = _quick_quality_score(300, 150, 0, 127.5)
        self.assertAlmostEqual(normal, clamped, places=3)

    def test_symmetry_score_is_clamped_below_zero(self):
        normal = _quick_quality_score(300, 0, 0, 127.5)
        clamped = _quick_quality_score(300, -50, 0, 127.5)
        self.assertAlmostEqual(normal, clamped, places=3)

    def test_larger_alignment_angle_reduces_score(self):
        s1 = _quick_quality_score(300, 100, 0, 127.5)
        s2 = _quick_quality_score(300, 100, 10, 127.5)
        self.assertGreater(s1, s2)

    def test_negative_alignment_angle_penalized_same_as_positive(self):
        s_pos = _quick_quality_score(300, 100, 12, 127.5)
        s_neg = _quick_quality_score(300, 100, -12, 127.5)
        self.assertAlmostEqual(s_pos, s_neg, places=6)

    def test_extreme_alignment_angle_does_not_go_negative_score(self):
        score = _quick_quality_score(300, 100, 90, 127.5)
        self.assertGreaterEqual(score, 0.0)

    def test_brightness_far_from_ideal_reduces_score(self):
        ideal = _quick_quality_score(300, 100, 0, 127.5)
        dark = _quick_quality_score(300, 100, 0, 10)
        bright = _quick_quality_score(300, 100, 0, 250)
        self.assertGreater(ideal, dark)
        self.assertGreater(ideal, bright)

    def test_score_is_weighted_sum_matches_manual_calc(self):
        blur_var, sym, angle, bright = 150, 80, 5, 100
        sharpness = min(blur_var / 300.0, 1.0) * 100.0
        symmetry = max(0.0, min(float(sym), 100.0))
        align_score = max(0.0, 100.0 - abs(angle) * 4.0)
        exposure = max(0.0, 100.0 - abs(bright - 127.5) / 1.275)
        expected = 0.40 * sharpness + 0.25 * symmetry + 0.20 * align_score + 0.15 * exposure
        self.assertAlmostEqual(_quick_quality_score(blur_var, sym, angle, bright), expected, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
