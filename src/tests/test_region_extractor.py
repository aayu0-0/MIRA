"""
test_region_extractor.py — phase2_region_extraction/region_extractor.py

test_integration.py already checks that RegionExtractor.extract() produces
the expected regions end-to-end on a full pipeline run. This file drills
into the internals that integration coverage doesn't exercise directly:

  - _ordered_chain_points(): chain endpoints are preferred as walk origins
  - _poly_region(): insufficient points / degenerate (empty) region guards,
    padding inflation, correct crop/mask/bbox geometry
  - _inflate_polygon(): points move outward from the centroid by ~pad
  - _build_forehead_poly / _build_cheek_poly / _build_chin_poly: percentile
    fallback when too few oval points land in the primary band
  - extract(): invalid Phase 1 result short-circuits with success=False
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.fixtures import build_phase1_result, IMG_W, IMG_H, FACE_CX, FACE_CY, FACE_RX, FACE_RY

from phase1_face_understanding.face_detector import Phase1Result
from phase2_region_extraction.region_extractor import (
    RegionExtractor, FacialRegion, Phase2Result, _ordered_chain_points,
)
from mediapipe.tasks.python import vision as mp_vision


def _synthetic_face_image():
    img = np.full((IMG_H, IMG_W, 3), (200, 190, 180), dtype=np.uint8)
    cv2.ellipse(img, (FACE_CX, FACE_CY), (FACE_RX, FACE_RY), 0, 0, 360, (150, 160, 190), -1)
    return img


class _FakeConn:
    def __init__(self, start, end):
        self.start, self.end = start, end


class TestOrderedChainPoints(unittest.TestCase):

    def test_simple_open_chain_starts_at_endpoint(self):
        # 0-1-2-3 chain: degree-1 endpoints are 0 and 3.
        conns = [_FakeConn(0, 1), _FakeConn(1, 2), _FakeConn(2, 3)]
        order = _ordered_chain_points(conns)
        self.assertIn(order[0], (0, 3))
        self.assertEqual(set(order), {0, 1, 2, 3})
        self.assertEqual(len(order), 4)

    def test_chain_order_is_contiguous(self):
        conns = [_FakeConn(5, 6), _FakeConn(6, 7), _FakeConn(7, 8)]
        order = _ordered_chain_points(conns)
        # Walking from either end should produce a path where consecutive
        # points are actually connected.
        edges = {(5, 6), (6, 5), (6, 7), (7, 6), (7, 8), (8, 7)}
        for a, b in zip(order, order[1:]):
            self.assertIn((a, b), edges)

    def test_closed_loop_visits_every_node_once(self):
        # A 4-cycle: 0-1-2-3-0 (no degree-1 node).
        conns = [_FakeConn(0, 1), _FakeConn(1, 2), _FakeConn(2, 3), _FakeConn(3, 0)]
        order = _ordered_chain_points(conns)
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertEqual(len(order), 4)

    def test_multiple_disconnected_components_all_visited(self):
        conns = [_FakeConn(0, 1), _FakeConn(10, 11)]
        order = _ordered_chain_points(conns)
        self.assertEqual(sorted(order), [0, 1, 10, 11])

    def test_real_mediapipe_groups_produce_full_index_sets(self):
        # Sanity check against the real MediaPipe connection graph used in
        # production -- every referenced index should show up exactly once.
        conns = mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
        expected = {c.start for c in conns} | {c.end for c in conns}
        order = _ordered_chain_points(conns)
        self.assertEqual(set(order), expected)
        self.assertEqual(len(order), len(set(order)))  # no duplicates


class TestPolyRegion(unittest.TestCase):

    def setUp(self):
        self.extractor = RegionExtractor()
        self.img = _synthetic_face_image()

    def test_none_polygon_is_unavailable(self):
        region = self.extractor._poly_region(self.img, None, "test")
        self.assertFalse(region.available)
        self.assertIn("Insufficient", region.error)

    def test_fewer_than_three_points_is_unavailable(self):
        pts = np.array([[10, 10], [20, 20]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "test")
        self.assertFalse(region.available)

    def test_valid_triangle_produces_available_region(self):
        pts = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "triangle")
        self.assertTrue(region.available)
        self.assertIsNotNone(region.crop)
        self.assertIsNotNone(region.mask)
        self.assertIsNotNone(region.mask_crop)
        self.assertIsNotNone(region.masked_crop)

    def test_crop_and_mask_crop_have_matching_shape(self):
        pts = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "triangle")
        self.assertEqual(region.crop.shape[:2], region.mask_crop.shape[:2])

    def test_bbox_matches_crop_dimensions(self):
        pts = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "triangle")
        _, _, bw, bh = region.bbox
        self.assertEqual((bh, bw), region.crop.shape[:2])

    def test_mask_full_is_full_image_size(self):
        pts = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "triangle")
        self.assertEqual(region.mask.shape, self.img.shape[:2])

    def test_padding_enlarges_the_bbox(self):
        pts = np.array([[100, 100], [200, 100], [150, 200]], dtype=np.int32)
        no_pad = self.extractor._poly_region(self.img, pts, "t1", pad=0)
        padded = self.extractor._poly_region(self.img, pts, "t2", pad=15)
        self.assertGreaterEqual(padded.bbox[2] * padded.bbox[3], no_pad.bbox[2] * no_pad.bbox[3])

    def test_degenerate_zero_area_polygon_is_unavailable(self):
        # Three collinear points along a horizontal line -> zero-height hull.
        pts = np.array([[10, 50], [20, 50], [30, 50]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "flat")
        self.assertFalse(region.available)

    def test_polygon_fully_outside_image_is_unavailable(self):
        pts = np.array([[-100, -100], [-90, -100], [-95, -90]], dtype=np.int32)
        region = self.extractor._poly_region(self.img, pts, "outside")
        self.assertFalse(region.available)


class TestInflatePolygon(unittest.TestCase):

    def setUp(self):
        self.extractor = RegionExtractor()

    def test_points_move_away_from_centroid(self):
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        inflated = self.extractor._inflate_polygon(square, pad=5)
        centroid = square.mean(axis=0)
        for orig, new in zip(square, inflated):
            orig_dist = np.linalg.norm(orig - centroid)
            new_dist = np.linalg.norm(new - centroid)
            self.assertGreater(new_dist, orig_dist)

    def test_zero_pad_leaves_points_near_original(self):
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        inflated = self.extractor._inflate_polygon(square, pad=0)
        np.testing.assert_array_almost_equal(inflated, square, decimal=0)

    def test_coincident_points_do_not_crash_on_zero_norm(self):
        # All points identical -> centroid == points -> zero-norm vectors,
        # which the implementation guards against (norms[norms==0] = 1.0).
        pts = np.array([[5, 5], [5, 5], [5, 5]], dtype=np.int32)
        try:
            self.extractor._inflate_polygon(pts, pad=10)
        except Exception as exc:
            self.fail(f"_inflate_polygon raised on coincident points: {exc}")


class TestBuildRegionPolygons(unittest.TestCase):

    def setUp(self):
        self.extractor = RegionExtractor()
        self.p1 = build_phase1_result()
        self.lm = self.p1.landmarks

    def test_forehead_poly_has_at_least_three_points(self):
        poly = self.extractor._build_forehead_poly(self.lm)
        self.assertGreaterEqual(len(poly), 3)

    def test_forehead_poly_is_upper_part_of_face(self):
        poly = self.extractor._build_forehead_poly(self.lm)
        # All forehead points should be above (smaller y than) the face center.
        self.assertTrue(all(y <= FACE_CY for _, y in poly))

    def test_chin_poly_has_at_least_three_points(self):
        poly = self.extractor._build_chin_poly(self.lm)
        self.assertGreaterEqual(len(poly), 3)

    def test_chin_poly_is_lower_part_of_face(self):
        poly = self.extractor._build_chin_poly(self.lm)
        self.assertTrue(all(y >= FACE_CY for _, y in poly))

    def test_left_cheek_poly_has_points(self):
        poly = self.extractor._build_cheek_poly(self.lm, side="left")
        self.assertGreaterEqual(len(poly), 3)

    def test_right_cheek_poly_has_points(self):
        poly = self.extractor._build_cheek_poly(self.lm, side="right")
        self.assertGreaterEqual(len(poly), 3)

    def test_left_and_right_cheek_polys_are_on_opposite_sides(self):
        left = self.extractor._build_cheek_poly(self.lm, side="left")
        right = self.extractor._build_cheek_poly(self.lm, side="right")
        left_mean_x = np.mean([x for x, _ in left])
        right_mean_x = np.mean([x for x, _ in right])
        self.assertGreater(left_mean_x, right_mean_x)


class TestExtractEndToEnd(unittest.TestCase):

    def setUp(self):
        self.extractor = RegionExtractor()
        self.img = _synthetic_face_image()

    def test_failed_phase1_result_short_circuits(self):
        failed = Phase1Result(success=False, error="No face detected in image.")
        result = self.extractor.extract(self.img, failed)
        self.assertFalse(result.success)
        self.assertEqual(result.regions, {})

    def test_success_but_no_landmarks_short_circuits(self):
        empty = Phase1Result(success=True, landmarks=[])
        result = self.extractor.extract(self.img, empty)
        self.assertFalse(result.success)

    def test_successful_extraction_returns_all_expected_region_names(self):
        p1 = build_phase1_result()
        result = self.extractor.extract(self.img, p1)
        self.assertTrue(result.success)
        expected_names = {
            "left_eye", "right_eye", "left_eyebrow", "right_eyebrow",
            "nose", "lips", "forehead", "left_cheek", "right_cheek", "chin",
        }
        self.assertEqual(set(result.regions.keys()), expected_names)

    def test_get_helper_returns_named_region(self):
        p1 = build_phase1_result()
        result = self.extractor.extract(self.img, p1)
        region = result.get("forehead")
        self.assertIsInstance(region, FacialRegion)

    def test_get_helper_returns_none_for_unknown_name(self):
        p1 = build_phase1_result()
        result = self.extractor.extract(self.img, p1)
        self.assertIsNone(result.get("does_not_exist"))

    def test_visualization_image_matches_input_dimensions(self):
        p1 = build_phase1_result()
        result = self.extractor.extract(self.img, p1)
        self.assertEqual(result.visualization.shape, self.img.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
