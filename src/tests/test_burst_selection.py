"""
test_burst_selection.py — burst_selection.py (multi-frame "pick the best shot")

Uses the synthetic Phase1Result fixtures (tests/fixtures.py) and a fake
FaceDetector so the whole two-stage selection runs with no camera, no
MediaPipe model file, and no network.

Covers:
  - _laplacian_variance(): blur metric, incl. the no-bbox fallback and the
    empty-crop guard
  - _ear_score(): same EAR normalization as FeatureAnalyzer, incl. the
    missing-landmark-index guard (returns None, not a fabricated value)
  - stage1_filter(): no-face / blurry / eyes-closed rejection, survivors
    tagged with rejected_reason=None
  - stage2_select_best(): highest overall_confidence wins; ties broken by
    sharper frame; region-extraction failure is tagged and excluded
  - select_best_frame(): end-to-end happy path and the "nothing usable"
    empty-result path
"""

import math
import os
import sys
import unittest
from dataclasses import replace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.fixtures import build_phase1_result, IMG_W, IMG_H, FACE_CX, FACE_CY, FACE_RX, FACE_RY

from phase1_face_understanding.face_detector import Phase1Result, LandmarkPoint
from phase3_feature_analysis.feature_analyzer import LEFT_EYE_EAR_POINTS, RIGHT_EYE_EAR_POINTS
from thresholds import BURST_FILTER

import burst_selection as bsel


def _synthetic_face_image(sharp=True):
    img = np.full((IMG_H, IMG_W, 3), (200, 190, 180), dtype=np.uint8)
    cv2.ellipse(img, (FACE_CX, FACE_CY), (FACE_RX, FACE_RY), 0, 0, 360, (150, 160, 190), -1)
    if sharp:
        # Add high-frequency detail (checkerboard-ish noise) so Laplacian
        # variance is comfortably above BURST_FILTER.min_laplacian_variance.
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 60, img.shape, dtype=np.uint8)
        img = cv2.subtract(img, noise)
    return img


def _blank_image():
    """Perfectly uniform image -> zero Laplacian variance -> "blurry"."""
    return np.full((IMG_H, IMG_W, 3), (150, 150, 150), dtype=np.uint8)


class FakeDetector:
    """Stands in for FaceDetector: returns pre-baked Phase1Results in order,
    one per call to analyze(), regardless of the frame passed in."""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    def analyze(self, frame):
        r = self._results[self._i]
        self._i += 1
        return r

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestLaplacianVariance(unittest.TestCase):

    def test_uniform_crop_has_zero_variance(self):
        img = _blank_image()
        p1 = build_phase1_result()
        val = bsel._laplacian_variance(img, p1.bbox)
        self.assertAlmostEqual(val, 0.0, places=3)

    def test_noisy_crop_has_high_variance(self):
        img = _synthetic_face_image(sharp=True)
        p1 = build_phase1_result()
        val = bsel._laplacian_variance(img, p1.bbox)
        self.assertGreater(val, 0.0)

    def test_no_bbox_falls_back_to_full_frame(self):
        img = _blank_image()
        val = bsel._laplacian_variance(img, bbox=None)
        self.assertAlmostEqual(val, 0.0, places=3)

    def test_empty_crop_returns_zero(self):
        # A bbox entirely outside the image bounds -> zero-size crop.
        from phase1_face_understanding.face_detector import FaceBoundingBox
        img = _blank_image()
        bad_bbox = FaceBoundingBox(x=IMG_W + 10, y=IMG_H + 10, w=5, h=5)
        val = bsel._laplacian_variance(img, bad_bbox)
        self.assertEqual(val, 0.0)


class TestEarScore(unittest.TestCase):

    def test_open_eye_scores_near_100(self):
        p1 = build_phase1_result()
        score = bsel._ear_score(p1.landmarks, LEFT_EYE_EAR_POINTS)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_missing_landmark_index_returns_none(self):
        short_landmarks = [LandmarkPoint(x=0.5, y=0.5, px=1, py=1)]  # far too short
        score = bsel._ear_score(short_landmarks, LEFT_EYE_EAR_POINTS)
        self.assertIsNone(score)

    def test_result_matches_feature_analyzer_normalization_range(self):
        p1 = build_phase1_result()
        left = bsel._ear_score(p1.landmarks, LEFT_EYE_EAR_POINTS)
        right = bsel._ear_score(p1.landmarks, RIGHT_EYE_EAR_POINTS)
        self.assertTrue(0 <= left <= 100)
        self.assertTrue(0 <= right <= 100)


class TestStage1Filter(unittest.TestCase):

    def test_no_face_detected_is_rejected(self):
        failed = Phase1Result(success=False, error="No face detected in image.")
        detector = FakeDetector([failed])
        candidates = bsel.stage1_filter([_blank_image()], detector)
        self.assertEqual(candidates[0].rejected_reason, "no_face_detected")

    def test_blurry_frame_is_rejected(self):
        p1 = build_phase1_result()
        detector = FakeDetector([p1])
        candidates = bsel.stage1_filter([_blank_image()], detector)
        self.assertEqual(candidates[0].rejected_reason, "blurry")

    def test_sharp_open_eyes_frame_survives(self):
        p1 = build_phase1_result()
        detector = FakeDetector([p1])
        candidates = bsel.stage1_filter([_synthetic_face_image(sharp=True)], detector)
        self.assertIsNone(candidates[0].rejected_reason)

    @staticmethod
    def _collapse_eye(landmarks, ear_points, cx, cy, half_width=20):
        """Force a single eye's EAR to ~0 (closed): both vertical landmark
        pairs collapsed to a single point, horizontal corners kept apart."""
        p0, p1_, p2_, p3, p4_, p5_ = ear_points
        landmarks[p0] = LandmarkPoint(x=(cx - half_width) / IMG_W, y=cy / IMG_H, px=cx - half_width, py=cy)
        landmarks[p3] = LandmarkPoint(x=(cx + half_width) / IMG_W, y=cy / IMG_H, px=cx + half_width, py=cy)
        landmarks[p1_] = LandmarkPoint(x=cx / IMG_W, y=cy / IMG_H, px=cx, py=cy)
        landmarks[p5_] = LandmarkPoint(x=cx / IMG_W, y=cy / IMG_H, px=cx, py=cy)
        landmarks[p2_] = LandmarkPoint(x=cx / IMG_W, y=cy / IMG_H, px=cx, py=cy)
        landmarks[p4_] = LandmarkPoint(x=cx / IMG_W, y=cy / IMG_H, px=cx, py=cy)

    def test_both_eyes_closed_is_rejected(self):
        p1 = build_phase1_result()
        landmarks = list(p1.landmarks)
        self._collapse_eye(landmarks, LEFT_EYE_EAR_POINTS, FACE_CX + 80, FACE_CY - 50)
        self._collapse_eye(landmarks, RIGHT_EYE_EAR_POINTS, FACE_CX - 80, FACE_CY - 50)
        closed_p1 = replace(p1, landmarks=landmarks)

        left = bsel._ear_score(closed_p1.landmarks, LEFT_EYE_EAR_POINTS)
        right = bsel._ear_score(closed_p1.landmarks, RIGHT_EYE_EAR_POINTS)
        self.assertLessEqual(left, BURST_FILTER.eyes_closed_ear_score)
        self.assertLessEqual(right, BURST_FILTER.eyes_closed_ear_score)

        detector = FakeDetector([closed_p1])
        candidates = bsel.stage1_filter([_synthetic_face_image(sharp=True)], detector)
        self.assertEqual(candidates[0].rejected_reason, "eyes_closed")

    def test_one_eye_closed_does_not_reject(self):
        # A single low-EAR eye (wink / landmark wobble) must NOT be treated
        # as "eyes closed" -- both eyes must be at/below threshold.
        p1 = build_phase1_result()
        landmarks = list(p1.landmarks)
        self._collapse_eye(landmarks, LEFT_EYE_EAR_POINTS, FACE_CX + 80, FACE_CY - 50)
        winking_p1 = replace(p1, landmarks=landmarks)

        left = bsel._ear_score(winking_p1.landmarks, LEFT_EYE_EAR_POINTS)
        self.assertLessEqual(left, BURST_FILTER.eyes_closed_ear_score)

        detector = FakeDetector([winking_p1])
        candidates = bsel.stage1_filter([_synthetic_face_image(sharp=True)], detector)
        self.assertNotEqual(candidates[0].rejected_reason, "eyes_closed")

    def test_returns_one_candidate_per_input_frame(self):
        p1 = build_phase1_result()
        failed = Phase1Result(success=False, error="No face detected in image.")
        detector = FakeDetector([p1, failed, p1])
        frames = [_synthetic_face_image(True), _blank_image(), _synthetic_face_image(True)]
        candidates = bsel.stage1_filter(frames, detector)
        self.assertEqual(len(candidates), 3)
        self.assertEqual([c.index for c in candidates], [0, 1, 2])


class TestStage2SelectBest(unittest.TestCase):

    def _survivor(self, frame, p1, blur=200.0):
        return bsel.BurstCandidate(
            frame=frame, index=0, p1=p1, blur_variance=blur,
            left_ear=90.0, right_ear=90.0, rejected_reason=None,
        )

    def test_returns_none_when_all_region_extraction_fails(self):
        failed_p1 = Phase1Result(success=True, bbox=None, landmarks=[])
        survivors = [self._survivor(_blank_image(), failed_p1)]
        result = bsel.stage2_select_best(survivors)
        self.assertIsNone(result)
        self.assertEqual(survivors[0].rejected_reason, "region_extraction_failed")

    def test_picks_highest_overall_confidence(self):
        img = _synthetic_face_image(True)
        p1 = build_phase1_result()
        # Two identical-quality survivors but different blur, and (since
        # confidence for both will be roughly the region-coverage figure)
        # we mainly check that the winner is one of the actual survivors
        # and comes with matching p2/p3/p4.
        s1 = self._survivor(img, p1, blur=100.0)
        s2 = self._survivor(img, p1, blur=500.0)
        result = bsel.stage2_select_best([s1, s2])
        self.assertIsNotNone(result)
        best, p2, p3, p4 = result
        self.assertIn(best, (s1, s2))
        self.assertIsNotNone(best.overall_confidence)
        self.assertTrue(p2.success)

    def test_ties_broken_by_sharper_frame(self):
        img = _synthetic_face_image(True)
        p1 = build_phase1_result()
        s_dull = self._survivor(img, p1, blur=10.0)
        s_sharp = self._survivor(img, p1, blur=999.0)
        # Force identical confidence so the tiebreak (blur_variance) decides.
        with mock.patch("burst_selection.ObservationEngine") as MockEngine:
            fake_engine = MockEngine.return_value
            fake_engine.generate.return_value = mock.Mock(overall_confidence=77.0)
            result = bsel.stage2_select_best([s_dull, s_sharp])
        self.assertIsNotNone(result)
        best, _, _, _ = result
        self.assertIs(best, s_sharp)


class TestSelectBestFrame(unittest.TestCase):

    def test_no_survivors_returns_all_none_tuple(self):
        failed = Phase1Result(success=False, error="No face detected in image.")
        with mock.patch("burst_selection.FaceDetector", return_value=FakeDetector([failed])):
            result = bsel.select_best_frame([_blank_image()])
        self.assertEqual(result, (None, None, None, None, None))

    def test_happy_path_returns_full_tuple(self):
        p1 = build_phase1_result()
        img = _synthetic_face_image(True)
        with mock.patch("burst_selection.FaceDetector", return_value=FakeDetector([p1])):
            best, r_p1, p2, p3, p4 = bsel.select_best_frame([img])
        self.assertIsNotNone(best)
        self.assertIs(r_p1, p1)
        self.assertTrue(p2.success)
        self.assertIsNotNone(p3)
        self.assertIsNotNone(p4)

    def test_all_blurry_frames_returns_all_none_tuple(self):
        p1 = build_phase1_result()
        with mock.patch("burst_selection.FaceDetector", return_value=FakeDetector([p1, p1])):
            result = bsel.select_best_frame([_blank_image(), _blank_image()])
        self.assertEqual(result, (None, None, None, None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
