"""
test_trend_window.py — trend_window.py pure data functions

Covers:
  - collect_metric_keys(): union of metric keys across sessions, score-like first
  - pretty_metric_label(): snake_case -> "Capitalized label"
  - filter_window(): last-N-days windowing relative to the most recent entry
  - build_metric_series(): (date, value) extraction, sorted oldest-first
  - compute_trend_summary(): plain-language, judgment-free summary text

No tkinter widgets are built or exercised here.
"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from history_window import SessionEntry
from trend_window import (
    collect_metric_keys, pretty_metric_label, filter_window,
    build_metric_series, compute_trend_summary, _direction,
)


def _entry(date, **metrics):
    return SessionEntry(date=date, raw_row={}, metrics=metrics)


class TestCollectMetricKeys(unittest.TestCase):

    def test_union_across_entries(self):
        entries = [_entry(None, a=1), _entry(None, b=2)]
        self.assertEqual(collect_metric_keys(entries), ["a", "b"])

    def test_score_like_keys_come_first(self):
        entries = [_entry(None, z_value=1, a_score=2, m_score=3)]
        keys = collect_metric_keys(entries)
        score_idx = [keys.index(k) for k in keys if "score" in k.lower()]
        non_score_idx = [keys.index(k) for k in keys if "score" not in k.lower()]
        self.assertTrue(all(si < ni for si in score_idx for ni in non_score_idx))

    def test_alphabetical_within_group(self):
        entries = [_entry(None, b_score=1, a_score=2)]
        self.assertEqual(collect_metric_keys(entries), ["a_score", "b_score"])

    def test_empty_entries_returns_empty_list(self):
        self.assertEqual(collect_metric_keys([]), [])

    def test_duplicate_keys_across_entries_not_repeated(self):
        entries = [_entry(None, x=1), _entry(None, x=2)]
        self.assertEqual(collect_metric_keys(entries), ["x"])


class TestPrettyMetricLabel(unittest.TestCase):

    def test_underscores_become_spaces_and_capitalized(self):
        self.assertEqual(pretty_metric_label("left_eye_redness_score"), "Left eye redness score")

    def test_single_word(self):
        self.assertEqual(pretty_metric_label("symmetry"), "Symmetry")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(pretty_metric_label("  face_score  "), "Face score")

    def test_all_caps_input_lowercased_except_first_letter(self):
        self.assertEqual(pretty_metric_label("ABC_SCORE"), "Abc score")


class TestFilterWindow(unittest.TestCase):

    def setUp(self):
        self.entries = [
            _entry(datetime(2026, 1, 1), score=1),
            _entry(datetime(2026, 1, 15), score=2),
            _entry(datetime(2026, 1, 30), score=3),
        ]

    def test_none_window_returns_all(self):
        self.assertEqual(filter_window(self.entries, None), self.entries)

    def test_window_relative_to_most_recent_dated_entry(self):
        # Most recent = Jan 30. A 7-day window should keep only Jan 30's cutoff
        # (Jan 24 - Jan 30), excluding Jan 1 and Jan 15.
        result = filter_window(self.entries, 7)
        self.assertEqual([e.date for e in result], [datetime(2026, 1, 30)])

    def test_wide_window_includes_all(self):
        result = filter_window(self.entries, 90)
        self.assertEqual(len(result), 3)

    def test_boundary_day_included(self):
        # 30-day window from Jan 30 -> cutoff Jan 1 inclusive (30-1=29 day span).
        result = filter_window(self.entries, 30)
        self.assertIn(datetime(2026, 1, 1), [e.date for e in result])

    def test_no_dated_entries_returns_all_unfiltered(self):
        undated = [_entry(None, score=1), _entry(None, score=2)]
        self.assertEqual(filter_window(undated, 7), undated)

    def test_mixed_dated_and_undated_keeps_only_undated_out_of_window(self):
        mixed = self.entries + [_entry(None, score=99)]
        result = filter_window(mixed, 7)
        # The undated entry has no `.date`, so it's excluded by the `e.date and`
        # check in filter_window (only entries with a date within-window survive).
        self.assertNotIn(None, [e.date for e in result])


class TestBuildMetricSeries(unittest.TestCase):

    def test_series_sorted_oldest_first(self):
        entries = [
            _entry(datetime(2026, 3, 1), score=30),
            _entry(datetime(2026, 1, 1), score=10),
            _entry(datetime(2026, 2, 1), score=20),
        ]
        series = build_metric_series(entries, "score")
        self.assertEqual([v for _, v in series], [10, 20, 30])

    def test_entries_without_date_excluded(self):
        entries = [_entry(None, score=1), _entry(datetime(2026, 1, 1), score=2)]
        series = build_metric_series(entries, "score")
        self.assertEqual(len(series), 1)

    def test_entries_missing_the_metric_excluded(self):
        entries = [_entry(datetime(2026, 1, 1), other=1)]
        series = build_metric_series(entries, "score")
        self.assertEqual(series, [])

    def test_empty_entries(self):
        self.assertEqual(build_metric_series([], "score"), [])


class TestDirection(unittest.TestCase):

    def test_stable_when_pct_below_2_percent(self):
        self.assertEqual(_direction(0.5, 1.5), "roughly stable")

    def test_up_when_positive_delta_and_pct(self):
        self.assertEqual(_direction(5, 10.0), "trending up")

    def test_down_when_negative_delta_and_pct(self):
        self.assertEqual(_direction(-5, -10.0), "trending down")

    def test_no_pct_falls_back_to_absolute_delta_threshold(self):
        self.assertEqual(_direction(0.005, None), "roughly stable")
        self.assertEqual(_direction(0.5, None), "trending up")
        self.assertEqual(_direction(-0.5, None), "trending down")


class TestComputeTrendSummary(unittest.TestCase):

    def test_empty_series_message(self):
        msg = compute_trend_summary("Symmetry score", [])
        self.assertIn("No dated values", msg)
        self.assertIn("Symmetry score", msg)

    def test_single_point_message(self):
        msg = compute_trend_summary("Symmetry score", [(datetime(2026, 1, 5), 88.0)])
        self.assertIn("Only one session", msg)
        self.assertIn("88", msg)
        self.assertIn("Jan 05, 2026", msg)

    def test_multi_point_includes_first_and_last_values(self):
        series = [(datetime(2026, 1, 1), 50.0), (datetime(2026, 2, 1), 70.0)]
        msg = compute_trend_summary("Score", series)
        self.assertIn("50", msg)
        self.assertIn("70", msg)
        self.assertIn("trending up", msg)

    def test_multi_point_shows_session_count(self):
        series = [(datetime(2026, 1, 1), 10.0), (datetime(2026, 1, 2), 10.0), (datetime(2026, 1, 3), 20.0)]
        msg = compute_trend_summary("Score", series)
        self.assertIn("Across 3 sessions", msg)

    def test_percent_change_shown_when_first_value_nonzero(self):
        series = [(datetime(2026, 1, 1), 50.0), (datetime(2026, 1, 2), 75.0)]
        msg = compute_trend_summary("Score", series)
        self.assertIn("+50.0%", msg)

    def test_no_percent_shown_when_first_value_zero(self):
        series = [(datetime(2026, 1, 1), 0.0), (datetime(2026, 1, 2), 10.0)]
        msg = compute_trend_summary("Score", series)
        self.assertNotIn("%", msg)

    def test_downward_trend_labeled_correctly(self):
        series = [(datetime(2026, 1, 1), 90.0), (datetime(2026, 1, 2), 40.0)]
        msg = compute_trend_summary("Score", series)
        self.assertIn("trending down", msg)

    def test_no_medical_judgment_language(self):
        # compute_trend_summary must stay judgment-free (per module docstring):
        # no "good"/"bad"/"improving"/"worsening" language.
        series = [(datetime(2026, 1, 1), 90.0), (datetime(2026, 1, 2), 40.0)]
        msg = compute_trend_summary("Score", series).lower()
        for banned in ("good", "bad", "improving", "worsening", "worse", "better"):
            self.assertNotIn(banned, msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
