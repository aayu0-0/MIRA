"""
test_history_window.py — history_window.py (CSV-based patient history)

Covers the pure, tkinter-free surface:
  - resolve_patient_folder(): registry.json lookup, bare-id fallback,
    "Name_<id>" suffix fallback, not-found
  - _read_csv / _detect_column / _parse_date / _coerce_numeric
  - _find_photo / _find_reports / _find_tts_script
  - load_patient_sessions(): end-to-end CSV -> SessionEntry list, newest-first
  - highlight_points() / format_timestamp()
  - paginate() / total_pages()
  - synthesize_and_play(): with a fake engine (no real Piper/audio)

No tkinter widgets are built or exercised here (those need a Tk root and are
out of scope for a headless unit test suite).
"""

import csv
import json
import os
import sys
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import history_window as hw
from history_window import (
    SessionEntry, resolve_patient_folder, _read_csv, _detect_column,
    _parse_date, _coerce_numeric, _find_photo, _find_reports,
    _find_tts_script, load_patient_sessions, highlight_points,
    format_timestamp, paginate, total_pages, synthesize_and_play,
    PAGE_SIZE,
)


# ─── resolve_patient_folder ─────────────────────────────────────────────────

class TestResolvePatientFolder(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_resolves_via_registry_json(self):
        folder = os.path.join(self.root, "Aayush_Sardana_001")
        os.makedirs(folder)
        registry = {"patients": [{"name": "Aayush Sardana", "id": "001", "folder": "Aayush_Sardana_001"}]}
        with open(os.path.join(self.root, "registry.json"), "w") as f:
            json.dump(registry, f)

        result = resolve_patient_folder("001", self.root)
        self.assertEqual(result, Path(folder))

    def test_registry_id_matched_as_string(self):
        # registry has id as int, lookup passes a string -- should still match.
        folder = os.path.join(self.root, "Someone_7")
        os.makedirs(folder)
        registry = {"patients": [{"name": "Someone", "id": 7, "folder": "Someone_7"}]}
        with open(os.path.join(self.root, "registry.json"), "w") as f:
            json.dump(registry, f)
        result = resolve_patient_folder("7", self.root)
        self.assertEqual(result, Path(folder))

    def test_registry_entry_points_at_missing_folder_falls_through(self):
        registry = {"patients": [{"name": "Ghost", "id": "999", "folder": "does_not_exist"}]}
        with open(os.path.join(self.root, "registry.json"), "w") as f:
            json.dump(registry, f)
        # No fallback folder either -> should raise.
        with self.assertRaises(FileNotFoundError):
            resolve_patient_folder("999", self.root)

    def test_corrupted_registry_json_falls_back_gracefully(self):
        with open(os.path.join(self.root, "registry.json"), "w") as f:
            f.write("{not valid json")
        folder = os.path.join(self.root, "42")
        os.makedirs(folder)
        result = resolve_patient_folder("42", self.root)
        self.assertEqual(result, Path(folder))

    def test_fallback_folder_named_literally_after_id(self):
        folder = os.path.join(self.root, "001")
        os.makedirs(folder)
        result = resolve_patient_folder("001", self.root)
        self.assertEqual(result, Path(folder))

    def test_fallback_folder_ending_in_underscore_id(self):
        folder = os.path.join(self.root, "Jane_Doe_777")
        os.makedirs(folder)
        result = resolve_patient_folder("777", self.root)
        self.assertEqual(result, Path(folder))

    def test_no_match_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            resolve_patient_folder("nonexistent", self.root)

    def test_root_itself_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_patient_folder("001", os.path.join(self.root, "no_such_root"))


# ─── CSV / column / date / numeric helpers ──────────────────────────────────

class TestReadCsv(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(_read_csv(Path(self.tmpdir) / "nope.csv"), [])

    def test_reads_rows_as_dicts(self):
        p = Path(self.tmpdir) / "d.csv"
        with open(p, "w", newline="") as f:
            f.write("a,b\n1,2\n3,4\n")
        rows = _read_csv(p)
        self.assertEqual(rows, [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])


class TestDetectColumn(unittest.TestCase):

    def test_finds_case_insensitive_match(self):
        self.assertEqual(_detect_column(["Timestamp", "score"], ["timestamp"]), "Timestamp")

    def test_prefers_first_candidate_present(self):
        self.assertEqual(_detect_column(["date", "timestamp"], ["timestamp", "date"]), "timestamp")

    def test_returns_none_when_no_candidate_matches(self):
        self.assertIsNone(_detect_column(["foo", "bar"], ["timestamp", "date"]))


class TestParseDate(unittest.TestCase):

    def test_empty_string_returns_none(self):
        self.assertIsNone(_parse_date(""))

    def test_iso_datetime_with_microseconds(self):
        self.assertEqual(_parse_date("2026-01-15T10:30:00.123456"),
                          datetime(2026, 1, 15, 10, 30, 0, 123456))

    def test_space_separated_datetime(self):
        self.assertEqual(_parse_date("2026-01-15 10:30:00"), datetime(2026, 1, 15, 10, 30, 0))

    def test_date_only(self):
        self.assertEqual(_parse_date("2026-01-15"), datetime(2026, 1, 15))

    def test_day_month_year_format(self):
        self.assertEqual(_parse_date("15-01-2026"), datetime(2026, 1, 15))

    def test_us_slash_format(self):
        self.assertEqual(_parse_date("01/15/2026"), datetime(2026, 1, 15))

    def test_unparseable_string_returns_none(self):
        self.assertIsNone(_parse_date("not a date at all"))


class TestCoerceNumeric(unittest.TestCase):

    def test_valid_int_string(self):
        self.assertEqual(_coerce_numeric("42"), 42.0)

    def test_valid_float_string(self):
        self.assertEqual(_coerce_numeric("3.14"), 3.14)

    def test_non_numeric_returns_none(self):
        self.assertIsNone(_coerce_numeric("Normal Symmetry"))

    def test_none_input_returns_none(self):
        self.assertIsNone(_coerce_numeric(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_coerce_numeric(""))


# ─── _find_photo / _find_reports / _find_tts_script ─────────────────────────

class TestFindPhoto(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patient_folder = Path(self.tmpdir) / "Patient_001"
        self.photos_dir = self.patient_folder / "photos"
        self.photos_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_relative_path_in_photos_dir(self):
        (self.photos_dir / "shot.jpg").touch()
        result = _find_photo({"photo": "shot.jpg"}, "photo", self.patient_folder, None)
        self.assertEqual(result, self.photos_dir / "shot.jpg")

    def test_relative_path_directly_under_patient_folder(self):
        (self.patient_folder / "shot2.jpg").touch()
        result = _find_photo({"photo": "shot2.jpg"}, "photo", self.patient_folder, None)
        self.assertEqual(result, self.patient_folder / "shot2.jpg")

    def test_column_present_but_file_missing_returns_none(self):
        result = _find_photo({"photo": "ghost.jpg"}, "photo", self.patient_folder, None)
        self.assertIsNone(result)

    def test_no_column_falls_back_to_date_match(self):
        (self.photos_dir / "2026-03-01_capture.jpg").touch()
        result = _find_photo({}, None, self.patient_folder, datetime(2026, 3, 1, 9, 0))
        self.assertEqual(result, self.photos_dir / "2026-03-01_capture.jpg")

    def test_no_column_and_no_date_match_returns_none(self):
        result = _find_photo({}, None, self.patient_folder, datetime(2026, 3, 1))
        self.assertIsNone(result)

    def test_absolute_existing_path_used_directly(self):
        abs_photo = Path(self.tmpdir) / "elsewhere.jpg"
        abs_photo.touch()
        result = _find_photo({"photo": str(abs_photo)}, "photo", self.patient_folder, None)
        self.assertEqual(result, abs_photo)


class TestFindReports(unittest.TestCase):
    """
    NOTE ON NAMING: the patient-prefix used here deliberately does NOT
    itself look like a 3-digit sequence number (see
    TestFindReportsSeqTokenCollisionBug below for why that matters --
    _find_reports's zero-padded-to-3-digits token can collide with a
    3-digit patient ID embedded earlier in the filename).
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.reports_dir = Path(self.tmpdir) / "reports"
        self.reports_dir.mkdir()
        names = [
            "Patient_A_01_doctor.pdf", "Patient_A_01_user.pdf",
            "Patient_A_01_tts.txt", "Patient_A_02_doctor.pdf",
        ]
        for n in names:
            (self.reports_dir / n).touch()
        self.all_reports = sorted(self.reports_dir.glob("*"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_matches_by_sequence_number_zero_padded(self):
        result = _find_reports({}, None, self.all_reports, None, seq=1)
        names = sorted(p.name for p in result)
        self.assertEqual(names, ["Patient_A_01_doctor.pdf", "Patient_A_01_tts.txt", "Patient_A_01_user.pdf"])

    def test_different_sequence_number_isolated(self):
        result = _find_reports({}, None, self.all_reports, None, seq=2)
        names = [p.name for p in result]
        self.assertEqual(names, ["Patient_A_02_doctor.pdf"])

    def test_explicit_report_column_takes_priority(self):
        result = _find_reports({"report_id": "tts"}, "report_id", self.all_reports, None, seq=2)
        names = [p.name for p in result]
        self.assertEqual(names, ["Patient_A_01_tts.txt"])

    def test_falls_back_to_date_when_no_seq_match(self):
        (self.reports_dir / "2026-05-01_extra.pdf").touch()
        all_reports = sorted(self.reports_dir.glob("*"))
        result = _find_reports({}, None, all_reports, datetime(2026, 5, 1), seq=999)
        names = [p.name for p in result]
        self.assertEqual(names, ["2026-05-01_extra.pdf"])

    def test_no_match_anywhere_returns_empty(self):
        result = _find_reports({}, None, self.all_reports, None, seq=None)
        self.assertEqual(result, [])


class TestFindReportsSeqTokenCollisionBug(unittest.TestCase):
    """
    BUG (found while writing this suite): _find_reports() builds its
    sequence-number match tokens as (f"_{seq:02d}_", f"_{seq:03d}_",
    f"_{seq}_") and checks `token in filename`. For seq=1, the 3-digit
    token is "_001_". If the patient ID itself is 3 digits (a very
    realistic case -- pipeline.py's own registry uses "001", "002", ...),
    every report filename embeds that ID as "_001_" regardless of its
    OWN session number, e.g. "Patient_001_02_doctor.pdf" contains the
    substring "_001_" purely from the patient ID, not from a session-1
    report. So a lookup for session 1 spuriously matches session 2's
    (and every other session's) reports too, once there are >=100 patients
    or any 3-digit patient ID sharing a folder with a real report.

    This test documents the CORRECT expected behaviour (only session 1's
    own reports should match) and is expected to FAIL against the current
    implementation, pinpointing the bug for a future fix -- e.g. anchoring
    the token to the position right after the sanitized name+id prefix
    instead of a bare substring search over the whole filename.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.reports_dir = Path(self.tmpdir) / "reports"
        self.reports_dir.mkdir()
        # Patient ID "001" is embedded in every filename; session 2's own
        # report should NOT show up when looking up session 1.
        (self.reports_dir / "Patient_001_01_doctor.pdf").touch()
        (self.reports_dir / "Patient_001_02_doctor.pdf").touch()
        self.all_reports = sorted(self.reports_dir.glob("*"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_session_1_lookup_should_not_return_session_2_report(self):
        result = _find_reports({}, None, self.all_reports, None, seq=1)
        names = [p.name for p in result]
        self.assertNotIn(
            "Patient_001_02_doctor.pdf", names,
            "seq=1 lookup matched session 2's report -- the '_001_' 3-digit-padded "
            "token collided with the patient ID '001' embedded in the filename, "
            "not with an actual session-1 marker.",
        )


class TestFindTtsScript(unittest.TestCase):

    def test_finds_tts_txt_file(self):
        reports = [Path("a_01_doctor.pdf"), Path("a_01_tts.txt"), Path("a_01_user.pdf")]
        self.assertEqual(_find_tts_script(reports), Path("a_01_tts.txt"))

    def test_returns_none_when_absent(self):
        reports = [Path("a_01_doctor.pdf"), Path("a_01_user.pdf")]
        self.assertIsNone(_find_tts_script(reports))

    def test_case_insensitive_suffix_and_stem(self):
        reports = [Path("a_01_TTS.TXT")]
        self.assertEqual(_find_tts_script(reports), Path("a_01_TTS.TXT"))

    def test_txt_file_not_ending_in_tts_is_ignored(self):
        reports = [Path("readme.txt")]
        self.assertIsNone(_find_tts_script(reports))


# ─── load_patient_sessions (end-to-end) ─────────────────────────────────────

class TestLoadPatientSessions(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.folder = Path(self.root) / "Patient_001"
        self.folder.mkdir()
        registry = {"patients": [{"name": "Patient", "id": "001", "folder": "Patient_001"}]}
        with open(Path(self.root) / "registry.json", "w") as f:
            json.dump(registry, f)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_csv(self, rows, fieldnames):
        with open(self.folder / "dataset.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def test_no_csv_returns_empty_list(self):
        self.assertEqual(load_patient_sessions("001", self.root), [])

    def test_sessions_sorted_newest_first(self):
        self._write_csv(
            [
                {"timestamp": "2026-01-01", "score": "50"},
                {"timestamp": "2026-03-01", "score": "70"},
                {"timestamp": "2026-02-01", "score": "60"},
            ],
            ["timestamp", "score"],
        )
        sessions = load_patient_sessions("001", self.root)
        dates = [s.date for s in sessions]
        self.assertEqual(dates, sorted(dates, reverse=True))
        self.assertEqual(sessions[0].date, datetime(2026, 3, 1))

    def test_numeric_metrics_extracted_non_numeric_excluded(self):
        self._write_csv(
            [{"timestamp": "2026-01-01", "symmetry_score": "88.5", "overall_label": "Good"}],
            ["timestamp", "symmetry_score", "overall_label"],
        )
        sessions = load_patient_sessions("001", self.root)
        self.assertEqual(sessions[0].metrics, {"symmetry_score": 88.5})
        # overall_label is a NON_METRIC_COLUMN -> not in metrics, but kept in raw_row
        self.assertEqual(sessions[0].raw_row["overall_label"], "Good")

    def test_rows_without_date_use_datetime_min_for_sorting(self):
        self._write_csv(
            [{"timestamp": "", "score": "1"}, {"timestamp": "2026-01-01", "score": "2"}],
            ["timestamp", "score"],
        )
        sessions = load_patient_sessions("001", self.root)
        self.assertEqual(sessions[0].date, datetime(2026, 1, 1))
        self.assertIsNone(sessions[1].date)

    def test_reports_linked_by_sequence_number(self):
        reports_dir = self.folder / "reports"
        reports_dir.mkdir()
        (reports_dir / "Patient_001_01_tts.txt").touch()
        (reports_dir / "Patient_001_02_tts.txt").touch()
        self._write_csv(
            [{"timestamp": "2026-01-01", "score": "1"}, {"timestamp": "2026-01-02", "score": "2"}],
            ["timestamp", "score"],
        )
        sessions = load_patient_sessions("001", self.root)
        # newest first: row 2 (seq=2) then row 1 (seq=1)
        self.assertEqual(sessions[0].tts_text_path.name, "Patient_001_02_tts.txt")
        self.assertEqual(sessions[1].tts_text_path.name, "Patient_001_01_tts.txt")

    def test_missing_patient_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_patient_sessions("no-such-id", self.root)


# ─── highlight_points / format_timestamp ────────────────────────────────────

class TestHighlightPoints(unittest.TestCase):

    def test_no_data_gives_placeholder(self):
        entry = SessionEntry(date=None, raw_row={}, metrics={})
        self.assertEqual(highlight_points(entry), ["No numeric data recorded for this session."])

    def test_overall_label_included_first(self):
        entry = SessionEntry(date=None, raw_row={"overall_label": "Good"}, metrics={"x_score": 90})
        points = highlight_points(entry)
        self.assertTrue(points[0].startswith("Overall: Good"))

    def test_overall_label_with_confidence(self):
        entry = SessionEntry(date=None, raw_row={"overall_label": "Good", "overall_confidence": "85"}, metrics={})
        points = highlight_points(entry)
        self.assertEqual(points[0], "Overall: Good (85% confidence)")

    def test_score_metrics_sorted_before_non_score_metrics(self):
        entry = SessionEntry(
            date=None, raw_row={},
            metrics={"z_value": 1, "a_score": 2, "b_score": 3},
        )
        points = highlight_points(entry)
        # both *_score entries should appear before z_value
        score_positions = [i for i, p in enumerate(points) if "score" in p.lower()]
        value_position = next(i for i, p in enumerate(points) if p.startswith("Z value"))
        self.assertTrue(all(i < value_position for i in score_positions))

    def test_metric_key_prettified(self):
        entry = SessionEntry(date=None, raw_row={}, metrics={"left_eye_redness_score": 12.0})
        points = highlight_points(entry)
        self.assertIn("Left eye redness score: 12", points)

    def test_max_points_respected(self):
        metrics = {f"metric_{i}": i for i in range(20)}
        entry = SessionEntry(date=None, raw_row={}, metrics=metrics)
        points = highlight_points(entry, max_points=5)
        self.assertEqual(len(points), 5)

    def test_g_format_drops_trailing_zeros(self):
        entry = SessionEntry(date=None, raw_row={}, metrics={"a_score": 90.0})
        points = highlight_points(entry)
        self.assertIn("A score: 90", points)


class TestFormatTimestamp(unittest.TestCase):

    def test_with_date(self):
        entry = SessionEntry(date=datetime(2026, 3, 5, 14, 30), raw_row={})
        self.assertEqual(format_timestamp(entry), "Mar 05, 2026 - 02:30 PM")

    def test_without_date(self):
        entry = SessionEntry(date=None, raw_row={})
        self.assertEqual(format_timestamp(entry), "Unknown time")


# ─── paginate / total_pages ──────────────────────────────────────────────────

class TestPagination(unittest.TestCase):

    def test_total_pages_empty(self):
        self.assertEqual(total_pages([]), 1)

    def test_total_pages_exact_multiple(self):
        entries = list(range(PAGE_SIZE * 2))
        self.assertEqual(total_pages(entries), 2)

    def test_total_pages_with_remainder(self):
        entries = list(range(PAGE_SIZE * 2 + 1))
        self.assertEqual(total_pages(entries), 3)

    def test_paginate_first_page(self):
        entries = list(range(25))
        self.assertEqual(paginate(entries, 0, page_size=10), list(range(10)))

    def test_paginate_middle_page(self):
        entries = list(range(25))
        self.assertEqual(paginate(entries, 1, page_size=10), list(range(10, 20)))

    def test_paginate_last_partial_page(self):
        entries = list(range(25))
        self.assertEqual(paginate(entries, 2, page_size=10), list(range(20, 25)))

    def test_paginate_out_of_range_page_returns_empty(self):
        entries = list(range(5))
        self.assertEqual(paginate(entries, 5, page_size=10), [])


# ─── synthesize_and_play ─────────────────────────────────────────────────────

class TestSynthesizeAndPlay(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Reset the lazily-cached engine/import-error globals between tests.
        hw._tts_engine = None
        hw._tts_import_error = None

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        hw._tts_engine = None
        hw._tts_import_error = None

    def test_engine_unavailable_returns_false(self):
        with mock.patch.object(hw, "_get_tts_engine", return_value=None), \
             mock.patch.object(hw, "_tts_import_error", "no piper installed"):
            ok, msg = synthesize_and_play(Path(self.tmpdir) / "script.txt")
        self.assertFalse(ok)
        self.assertIn("piper_tts.py not found", msg)

    def test_missing_script_file_returns_false(self):
        fake_engine = mock.Mock()
        with mock.patch.object(hw, "_get_tts_engine", return_value=fake_engine):
            ok, msg = synthesize_and_play(Path(self.tmpdir) / "nope.txt")
        self.assertFalse(ok)
        self.assertIn("Couldn't read script", msg)

    def test_empty_script_returns_false(self):
        script = Path(self.tmpdir) / "empty.txt"
        script.write_text("   ")
        fake_engine = mock.Mock()
        with mock.patch.object(hw, "_get_tts_engine", return_value=fake_engine):
            ok, msg = synthesize_and_play(script)
        self.assertFalse(ok)
        self.assertIn("empty", msg.lower())

    def test_successful_fresh_synthesis(self):
        script = Path(self.tmpdir) / "session.txt"
        script.write_text("Hello, this is your report.")
        fake_engine = mock.Mock()
        fake_result = mock.Mock(success=True, played=True)
        fake_engine.speak.return_value = fake_result
        with mock.patch.object(hw, "_get_tts_engine", return_value=fake_engine):
            ok, msg = synthesize_and_play(script)
        self.assertTrue(ok)
        fake_engine.speak.assert_called_once()

    def test_synthesis_failure_propagates_error(self):
        script = Path(self.tmpdir) / "session.txt"
        script.write_text("Hello.")
        fake_engine = mock.Mock()
        fake_result = mock.Mock(success=False, error="synth exploded")
        fake_engine.speak.return_value = fake_result
        with mock.patch.object(hw, "_get_tts_engine", return_value=fake_engine):
            ok, msg = synthesize_and_play(script)
        self.assertFalse(ok)
        self.assertEqual(msg, "synth exploded")

    def test_cached_wav_played_without_resynthesizing(self):
        script = Path(self.tmpdir) / "session.txt"
        script.write_text("Hello.")
        wav = script.with_name(script.stem + ".wav")
        wav.write_bytes(b"RIFF....WAVEfake")

        fake_engine = mock.Mock()
        fake_engine.timeout = 5.0
        fake_engine._runner = mock.Mock()

        with mock.patch.object(hw, "_get_tts_engine", return_value=fake_engine), \
             mock.patch("piper_tts._player_command", return_value=["aplay", str(wav)]):
            ok, msg = synthesize_and_play(script)

        self.assertTrue(ok)
        self.assertEqual(msg, "Played cached audio.")
        fake_engine.speak.assert_not_called()

    def test_cached_wav_playback_failure_falls_through_to_resynthesize(self):
        script = Path(self.tmpdir) / "session.txt"
        script.write_text("Hello.")
        wav = script.with_name(script.stem + ".wav")
        wav.write_bytes(b"RIFF....WAVEfake")

        fake_engine = mock.Mock()
        fake_engine.timeout = 5.0
        fake_engine._runner = mock.Mock(side_effect=RuntimeError("player crashed"))
        fake_result = mock.Mock(success=True, played=True)
        fake_engine.speak.return_value = fake_result

        with mock.patch.object(hw, "_get_tts_engine", return_value=fake_engine), \
             mock.patch("piper_tts._player_command", return_value=["aplay", str(wav)]):
            ok, msg = synthesize_and_play(script)

        self.assertTrue(ok)
        fake_engine.speak.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
