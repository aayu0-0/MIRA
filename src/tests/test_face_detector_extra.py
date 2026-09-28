"""
test_face_detector_extra.py — phase1_face_understanding/face_detector.py

test_symmetry.py already covers _compute_symmetry() and _compute_angle() in
depth. This file covers everything else that doesn't need the real
MediaPipe model file:

  - FaceDetector.__init__(): missing model file raises FileNotFoundError
    with a helpful message, before touching MediaPipe at all
  - _rect_from_indices(): padded bbox geometry, image-edge clamping,
    degenerate (zero-area) rejection
  - FaceBoundingBox.center property
  - draw_landmarks(): no-op passthrough on a failed result; doesn't raise
    on a successful one; respects the draw_* toggles
  - Index-group sanity: named single-point constants fall inside their
    claimed MediaPipe groups (mirrors the module's own import-time asserts)
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.fixtures import build_phase1_result, IMG_W, IMG_H

from phase1_face_understanding.face_detector import (
    FaceDetector, FaceBoundingBox, Phase1Result,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, LIPS_IDX, OVAL_IDX,
    UPPER_LIP_CENTER, LOWER_LIP_CENTER,
)


def _synthetic_image():
    return np.full((IMG_H, IMG_W, 3), (180, 170, 160), dtype=np.uint8)


class TestFaceDetectorInit(unittest.TestCase):

    def test_missing_model_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            FaceDetector(model_path="/no/such/model/at/all.task")
        self.assertIn("FaceLandmarker model not found", str(ctx.exception))

    def test_missing_model_error_mentions_download_url(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            FaceDetector(model_path="/definitely/missing.task")
        self.assertIn("storage.googleapis.com", str(ctx.exception))


class TestFaceBoundingBox(unittest.TestCase):

    def test_center_property(self):
        bbox = FaceBoundingBox(x=10, y=20, w=100, h=50)
        self.assertEqual(bbox.center, (60, 45))

    def test_center_with_zero_size(self):
        bbox = FaceBoundingBox(x=5, y=5, w=0, h=0)
        self.assertEqual(bbox.center, (5, 5))


class TestRectFromIndices(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector.__new__(FaceDetector)  # bypass __init__
        self.p1 = build_phase1_result()

    def test_returns_tuple_for_valid_indices(self):
        rect = self.detector._rect_from_indices(self.p1.landmarks, LEFT_EYE_IDX, IMG_W, IMG_H, pad=6)
        self.assertIsNotNone(rect)
        x, y, w, h = rect
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_padding_increases_rect_size(self):
        no_pad = self.detector._rect_from_indices(self.p1.landmarks, LEFT_EYE_IDX, IMG_W, IMG_H, pad=0)
        padded = self.detector._rect_from_indices(self.p1.landmarks, LEFT_EYE_IDX, IMG_W, IMG_H, pad=10)
        self.assertGreaterEqual(padded[2], no_pad[2])
        self.assertGreaterEqual(padded[3], no_pad[3])

    def test_clamped_to_image_bounds(self):
        # Enormous padding should clamp to the image edges, not go negative
        # or exceed image dimensions.
        rect = self.detector._rect_from_indices(self.p1.landmarks, LEFT_EYE_IDX, IMG_W, IMG_H, pad=10_000)
        x, y, w, h = rect
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, IMG_W)
        self.assertLessEqual(y + h, IMG_H)

    def test_single_point_index_list_returns_none_or_degenerate(self):
        # A single repeated index has zero width/height -> should not
        # produce a rect (x2<=x1 or y2<=y1 guard).
        rect = self.detector._rect_from_indices(self.p1.landmarks, [0, 0], IMG_W, IMG_H, pad=0)
        self.assertIsNone(rect)


class TestDrawLandmarks(unittest.TestCase):

    def test_failed_result_returns_unmodified_copy(self):
        img = _synthetic_image()
        failed = Phase1Result(success=False, error="No face detected in image.")
        out = FaceDetector.draw_landmarks(img, failed)
        np.testing.assert_array_equal(out, img)
        self.assertIsNot(out, img)  # must be a copy, not the same array

    def test_successful_result_does_not_raise(self):
        img = _synthetic_image()
        p1 = build_phase1_result()
        try:
            out = FaceDetector.draw_landmarks(img, p1)
        except Exception as exc:
            self.fail(f"draw_landmarks raised on a valid result: {exc}")
        self.assertEqual(out.shape, img.shape)

    def test_output_differs_from_input_when_drawing_enabled(self):
        img = _synthetic_image()
        p1 = build_phase1_result()
        out = FaceDetector.draw_landmarks(img, p1, draw_mesh=True, draw_bbox=True, draw_key_points=True)
        self.assertFalse(np.array_equal(out, img))

    def test_all_draw_flags_disabled_still_draws_symmetry_label_and_midline(self):
        # The symmetry-label text and midline are drawn unconditionally
        # (not gated by draw_mesh/draw_bbox/draw_key_points), so the image
        # should still differ from the input even with everything disabled.
        img = _synthetic_image()
        p1 = build_phase1_result()
        out = FaceDetector.draw_landmarks(img, p1, draw_mesh=False, draw_bbox=False, draw_key_points=False)
        self.assertFalse(np.array_equal(out, img))

    def test_does_not_mutate_input_image(self):
        img = _synthetic_image()
        original = img.copy()
        p1 = build_phase1_result()
        FaceDetector.draw_landmarks(img, p1)
        np.testing.assert_array_equal(img, original)


class TestIndexGroupSanity(unittest.TestCase):
    """Mirrors the module's own import-time assertions -- these should never
    fail unless a MediaPipe upgrade silently remaps landmark indices."""

    def test_upper_lower_lip_center_within_lips_group(self):
        self.assertIn(UPPER_LIP_CENTER, LIPS_IDX)
        self.assertIn(LOWER_LIP_CENTER, LIPS_IDX)

    def test_index_groups_are_disjoint_from_each_other_where_expected(self):
        # Eyes and oval shouldn't overlap in a normal MediaPipe topology.
        self.assertEqual(set(LEFT_EYE_IDX) & set(OVAL_IDX), set())
        self.assertEqual(set(RIGHT_EYE_IDX) & set(OVAL_IDX), set())

    def test_index_groups_are_nonempty(self):
        for group in (LEFT_EYE_IDX, RIGHT_EYE_IDX, LIPS_IDX, OVAL_IDX):
            self.assertGreater(len(group), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
