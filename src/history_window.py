"""
history_window.py
------------------
MIRA -- History window (CSV-based).

Phase 1 (phase_a_history.HistoryStore / phase_a_trends.TrendDetector,
SQLite) no longer exists in this project. History now lives exactly the
way pipeline.py already writes it, per patient:

    database/
        registry.json                    <- {"patients": [{"name","id","folder"}, ...]}
        <folder>/                        e.g. "Aayush_Sardana_001"
            photos/                      <- session photos (may be missing per-row)
            reports/                     <- <folder>_01_doctor.pdf, <folder>_01_user.pdf,
                                             <folder>_01_metrics.json, <folder>_01_tts.txt,
                                             <folder>_02_..., etc. (2-digit session number)
            dataset.csv                  <- one row per session

This file reads that CSV directly -- no SQLite, no phase_a_* imports, no
TrendDetector. It shows each session as a card: photo (or "No photo"),
a handful of headline points, and a placeholder audio/TTS control (Piper
TTS audio isn't wired up yet -- pipeline currently only writes the
tts_report_*.txt script, not an audio file). Cards are paginated 10 per
page with an inner scrollbar in case a page still overflows the window.

CONFIGURE THESE if your dataset.csv column names differ -- best-guess
defaults with graceful fallbacks, same convention as before.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATE_COLUMN_CANDIDATES = [
    "timestamp", "date", "session_date", "datetime", "created_at", "session_time",
]
PHOTO_COLUMN_CANDIDATES = [
    "photo_path", "photo_file", "photo", "image_path", "image",
]
REPORT_LINK_COLUMN_CANDIDATES = [
    "report_file", "report_path", "report_id", "session_id",
]
NON_METRIC_COLUMNS = {
    "user_id", "patient_id", "id", "session_id", "photo_path", "photo_file",
    "photo", "image_path", "image", "report_file", "report_path",
    "overall_label", "quality_warnings",
}

REGISTRY_FILENAME = "registry.json"
DATASET_FILENAME = "dataset.csv"
PHOTOS_SUBFOLDER = "photos"
REPORTS_SUBFOLDER = "reports"
PAGE_SIZE = 10

# --- Piper TTS (A5) ---------------------------------------------------------
# piper_tts.py sits alongside this file, but the "piper" folder itself lives
# one level up from src/ (MIRA_doc+clean/piper), not inside src/, so that
# src stays free of the binary/model distribution. Extract the piper
# distribution (piper.exe + its DLLs + espeak-ng-data/ + the .onnx models)
# into a "piper" folder next to src/, i.e.:
#   <MIRA_doc+clean>/piper/piper.exe
#   <MIRA_doc+clean>/piper/en_US-amy-medium.onnx (+ .onnx.json)
#   <MIRA_doc+clean>/piper/hi_IN-priyamvada-medium.onnx (+ .onnx.json)
#   <MIRA_doc+clean>/piper/espeak-ng-data/...
# piper.exe needs those DLLs and espeak-ng-data alongside it to run, so the
# whole folder from the zip has to be extracted intact -- don't move just
# the .exe on its own. Each .onnx model needs its matching .onnx.json
# config sitting right next to it (piper.exe looks for "<model>.onnx.json"
# automatically) -- without it, synthesis fails for that voice specifically.
_PIPER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "piper"
)

# One voice per report language -- keyed the same way translate_tts_text()'s
# caller decides "hi" vs "en" (see _detect_report_language below). Swap the
# filenames here if you use different voice models.
PIPER_VOICE_PATHS = {
    "en": os.path.join(_PIPER_DIR, "en_US-amy-medium.onnx"),
    "hi": os.path.join(_PIPER_DIR, "hi_IN-priyamvada-medium.onnx"),
}
PIPER_BINARY_PATH = os.path.join(_PIPER_DIR, "piper.exe")

# Devanagari Unicode block -- any character in this range means the script
# is a Hindi report. Cheap and reliable since tts_report.py's Hindi output
# always contains Devanagari (see phase5_report/hindi_translation.py), and
# the English scaffold vocabulary never does.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _detect_report_language(text: str) -> str:
    """Returns "hi" if the TTS script text contains any Devanagari
    characters, else "en". Detecting from content rather than trusting a
    filename convention means this keeps working even though pipeline.py
    writes every session's script to the same "<base>_tts.txt" name
    regardless of language."""
    return "hi" if _DEVANAGARI_RE.search(text or "") else "en"

# Theme -- matches mira_app.py
BG = "#1a1a2e"
CARD = "#16213e"
CARD_ALT = "#0f1b35"
PRIMARY = "#e94560"
PRIMARY_HOVER = "#c73652"
SECONDARY = "#0f3460"
TEXT = "#e0e0e0"
MUTED = "#a8a8b3"
SUCCESS = "#27ae60"
WARNING = "#e67e22"

THUMB_SIZE = (160, 160)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SessionEntry:
    date: Optional[datetime]
    raw_row: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    photo_path: Optional[Path] = None
    report_paths: list[Path] = field(default_factory=list)
    tts_text_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Registry / folder resolution -- matches mira_app.py's actual registry.json
# shape: {"patients": [{"name": ..., "id": ..., "folder": ...}, ...]}
# ---------------------------------------------------------------------------

def resolve_patient_folder(patient_id: str, database_root: str) -> Path:
    root = Path(database_root)
    registry_path = root / REGISTRY_FILENAME

    if registry_path.is_file():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("patients", []):
                if str(entry.get("id")) == str(patient_id):
                    folder = root / entry.get("folder", "")
                    if folder.is_dir():
                        return folder
        except (OSError, json.JSONDecodeError):
            pass

    # Fallback: folder literally named after the id, or "Name_<id>".
    if root.is_dir():
        candidate = root / str(patient_id)
        if candidate.is_dir():
            return candidate
        for folder_name in os.listdir(root):
            folder_path = root / folder_name
            if folder_path.is_dir() and folder_name.endswith(f"_{patient_id}"):
                return folder_path

    raise FileNotFoundError(
        f"No patient folder found for id={patient_id!r} under {root}"
    )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _read_csv(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _detect_column(fieldnames, candidates) -> Optional[str]:
    lower_map = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _coerce_numeric(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_photo(row, photo_col, patient_folder, date_val) -> Optional[Path]:
    photos_dir = patient_folder / PHOTOS_SUBFOLDER

    if photo_col and row.get(photo_col):
        raw = row[photo_col]
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return p
        candidate = photos_dir / raw
        if candidate.exists():
            return candidate
        candidate = patient_folder / raw
        if candidate.exists():
            return candidate
        return None  # column says there should be one, but it's missing on disk

    if date_val and photos_dir.is_dir():
        day_str = date_val.strftime("%Y-%m-%d")
        matches = [p for p in sorted(photos_dir.glob("*")) if day_str in p.name]
        if matches:
            return matches[0]

    return None


def _find_reports(row, report_col, all_reports, date_val, seq: Optional[int]) -> list[Path]:
    """Reports linked to one CSV row. Tries, in order: an explicit CSV
    column (report_id / session_id, if your dataset.csv has one), the
    session's sequence number (matches pipeline.py's real naming --
    "<folder>_01_doctor.pdf", "<folder>_01_tts.txt", ... -- one/two-digit,
    zero-padded), then a same-day date fallback."""
    if report_col and row.get(report_col):
        key = row[report_col]
        matches = [r for r in all_reports if key in r.name]
        if matches:
            return matches

    if seq is not None:
        # Anchored to the actual session-number slot in
        # "<Name>_<ID>_<NN>_doctor.pdf" (second-to-last underscore segment
        # of the stem), not a bare substring search over the whole
        # filename -- a substring search on "_{seq:03d}_" false-matches
        # any 3-digit patient ID that happens to equal the session number,
        # e.g. seq=1 spuriously matching "Patient_001_02_doctor.pdf" via
        # the "_001_" patient-ID portion rather than a real session 1.
        tokens = {f"{seq:02d}", f"{seq:03d}", str(seq)}
        matches = []
        for r in all_reports:
            parts = r.stem.split("_")
            if len(parts) >= 2 and parts[-2] in tokens:
                matches.append(r)
        if matches:
            return matches

    if date_val:
        day_str = date_val.strftime("%Y-%m-%d")
        matches = [r for r in all_reports if day_str in r.name]
        if matches:
            return matches

    return []


def _find_tts_script(reports: list[Path]) -> Optional[Path]:
    """The plain-text TTS script for a session, e.g.
    "Aayush_Sardana_001_01_tts.txt" -- matches pipeline.py's real
    "<..>_tts.txt" naming rather than the earlier "tts_report_*" guess."""
    for r in reports:
        if r.suffix.lower() == ".txt" and r.stem.lower().endswith("_tts"):
            return r
    return None


def load_patient_sessions(patient_id: str, database_root: str) -> list[SessionEntry]:
    """All sessions for this patient, newest first."""
    patient_folder = resolve_patient_folder(patient_id, database_root)
    csv_path = patient_folder / DATASET_FILENAME
    reports_dir = patient_folder / REPORTS_SUBFOLDER

    rows = _read_csv(csv_path)
    all_reports = sorted(reports_dir.glob("*")) if reports_dir.is_dir() else []

    if not rows:
        return []

    fieldnames = list(rows[0].keys())
    date_col = _detect_column(fieldnames, DATE_COLUMN_CANDIDATES)
    photo_col = _detect_column(fieldnames, PHOTO_COLUMN_CANDIDATES)
    report_col = _detect_column(fieldnames, REPORT_LINK_COLUMN_CANDIDATES)

    sessions: list[SessionEntry] = []
    for seq, row in enumerate(rows, start=1):  # 1-indexed, matches "<folder>_01_...", "_02_...", etc.
        date_val = _parse_date(row.get(date_col, "")) if date_col else None

        metrics = {}
        for key, val in row.items():
            if key in NON_METRIC_COLUMNS or key == date_col:
                continue
            num = _coerce_numeric(val)
            if num is not None:
                metrics[key] = num

        reports = _find_reports(row, report_col, all_reports, date_val, seq)
        tts_path = _find_tts_script(reports)

        sessions.append(SessionEntry(
            date=date_val,
            raw_row=row,
            metrics=metrics,
            photo_path=_find_photo(row, photo_col, patient_folder, date_val),
            report_paths=reports,
            tts_text_path=tts_path,
        ))

    sessions.sort(key=lambda s: s.date or datetime.min, reverse=True)  # newest first
    return sessions


# ---------------------------------------------------------------------------
# Headline points shown on each card (pure function -- easy to unit test)
# ---------------------------------------------------------------------------

def highlight_points(entry: SessionEntry, max_points: int = 8) -> list[str]:
    points = []
    label = entry.raw_row.get("overall_label")
    if label:
        conf = entry.raw_row.get("overall_confidence") or entry.raw_row.get("confidence")
        conf_str = f" ({conf}% confidence)" if conf else ""
        points.append(f"Overall: {label}{conf_str}")

    # Score-like metrics first (symmetry score, overall score, etc.), then
    # everything else, so the most meaningful numbers aren't pushed off the
    # card by less interesting ones when there are more than max_points.
    ordered_keys = sorted(
        entry.metrics.keys(),
        key=lambda k: (0 if "score" in k.lower() else 1, k.lower()),
    )

    for key in ordered_keys:
        if len(points) >= max_points:
            break
        pretty = key.replace("_", " ").strip().capitalize()
        points.append(f"{pretty}: {entry.metrics[key]:g}")

    if not points:
        points.append("No numeric data recorded for this session.")
    return points[:max_points]


def _find_report(entry: SessionEntry, *suffixes: str) -> Optional[Path]:
    """First report path for this session whose filename stem ends with
    one of the given suffixes, e.g. _find_report(entry, "_doctor") for
    "<folder>_01_doctor.pdf". Case-insensitive."""
    for suffix in suffixes:
        for r in entry.report_paths:
            if r.stem.lower().endswith(suffix.lower()):
                return r
    return None


def all_summary_points(entry: SessionEntry) -> list[str]:
    """Every recorded metric for this session, unlike highlight_points()
    which caps at max_points for the card view. Used by the 'Full Summary'
    popup so nothing is hidden behind the card's short preview."""
    points = []
    label = entry.raw_row.get("overall_label")
    if label:
        conf = entry.raw_row.get("overall_confidence") or entry.raw_row.get("confidence")
        conf_str = f" ({conf}% confidence)" if conf else ""
        points.append(f"Overall: {label}{conf_str}")

    ordered_keys = sorted(
        entry.metrics.keys(),
        key=lambda k: (0 if "score" in k.lower() else 1, k.lower()),
    )
    for key in ordered_keys:
        pretty = key.replace("_", " ").strip().capitalize()
        points.append(f"{pretty}: {entry.metrics[key]:g}")

    if not points:
        points.append("No numeric data recorded for this session.")
    return points


def format_timestamp(entry: SessionEntry) -> str:
    if entry.date:
        return entry.date.strftime("%b %d, %Y - %I:%M %p")
    return "Unknown time"


# ---------------------------------------------------------------------------
# Pagination (pure -- no tkinter)
# ---------------------------------------------------------------------------

def paginate(entries: list, page: int, page_size: int = PAGE_SIZE) -> list:
    start = page * page_size
    return entries[start:start + page_size]


def total_pages(entries: list, page_size: int = PAGE_SIZE) -> int:
    if not entries:
        return 1
    return (len(entries) - 1) // page_size + 1


# ---------------------------------------------------------------------------
# Piper TTS integration (A5 -- piper_tts.py)
# ---------------------------------------------------------------------------

_tts_engines = {}           # lazy per-language singletons: {"en": PiperTTS, "hi": PiperTTS}
_tts_import_error = None
_piper_importable = None    # None = not checked yet, else True/False


def _ensure_piper_importable():
    """One-time check that piper_tts.py can be imported at all, independent
    of which voice/language ends up being used. Cached so this only costs
    an import attempt once, not once per language."""
    global _tts_import_error, _piper_importable
    if _piper_importable is not None:
        return _piper_importable
    try:
        import piper_tts  # noqa: F401
    except ImportError as exc:
        _tts_import_error = str(exc)
        _piper_importable = False
        return False
    _piper_importable = True
    return True


def _get_tts_engine(language: str = "en"):
    """Returns a shared PiperTTS instance for the given language ("en" or
    "hi"), or None if piper_tts.py can't be imported. Cached per language
    so each voice's binary/model resolution only happens once."""
    global _tts_engines
    if language not in PIPER_VOICE_PATHS:
        language = "en"
    if language in _tts_engines:
        return _tts_engines[language]
    if not _ensure_piper_importable():
        return None
    from piper_tts import PiperTTS
    engine = PiperTTS(
        model_path=PIPER_VOICE_PATHS[language],
        binary_path=PIPER_BINARY_PATH,
        play=True,
    )
    _tts_engines[language] = engine
    return engine


def _load_script_and_engine(script_path: Path):
    """Shared setup for both synthesize_only() and synthesize_and_play():
    read the script, detect its language, and resolve the matching engine.
    Returns (text, engine, wav_path, error_message). On failure, text/engine
    are None and error_message explains why."""
    if not _ensure_piper_importable():
        return None, None, None, f"piper_tts.py not found: {_tts_import_error}"

    # Cheap up-front probe: if no engine can be built at all (binary/model
    # missing, etc.), say so now rather than masking it behind an unrelated
    # file-read error below. Real per-language resolution (which needs the
    # script's actual text) still happens after the file is read.
    if _get_tts_engine("en") is None:
        return None, None, None, f"piper_tts.py not found: {_tts_import_error}"

    try:
        text = script_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return None, None, None, f"Couldn't read script: {exc}"
    if not text:
        return None, None, None, "TTS script is empty."

    language = _detect_report_language(text)
    engine = _get_tts_engine(language)
    if engine is None:
        return None, None, None, f"piper_tts.py not found: {_tts_import_error}"

    wav_path = script_path.with_name(script_path.stem + ".wav")
    return text, engine, wav_path, None


_prefetch_queued: set = set()   # script paths already handed to a background
                                 # prefetch worker, so re-loading the History
                                 # screen doesn't re-queue the same session
_prefetch_lock = threading.Lock()


def prefetch_missing_audio(entries):
    """Kick off a single background daemon thread that pre-synthesizes the
    WAV for every session in `entries` whose audio isn't cached yet (already-
    cached ones are skipped instantly inside synthesize_only()). Returns
    immediately -- the caller, and the rest of the app, keep running right
    away; this just gets audio ready ahead of someone clicking "Play Audio".
    Safe to call repeatedly (e.g. every time the History screen reloads):
    a script already handed to a worker isn't queued a second time."""
    if not _ensure_piper_importable():
        return

    to_queue = []
    with _prefetch_lock:
        for entry in entries:
            script_path = entry.tts_text_path
            if not script_path:
                continue
            key = str(script_path)
            if key in _prefetch_queued:
                continue
            _prefetch_queued.add(key)
            to_queue.append(script_path)

    if not to_queue:
        return

    def worker():
        for script_path in to_queue:
            try:
                synthesize_only(script_path)
            except Exception:
                pass  # best-effort background prefetch -- a later click on
                      # "Play Audio" will retry and surface any real error

    threading.Thread(target=worker, daemon=True).start()


def synthesize_only(script_path: Path):
    """Blocking call: pre-generate the WAV for a session's *_tts.txt script
    without playing it back. A no-op (returns immediately) if the WAV is
    already cached from an earlier synthesize_only() or synthesize_and_play()
    call. Meant to be kicked off from a background daemon thread right after
    a report is created, so the audio is already sitting on disk by the time
    someone clicks "Play Audio" -- instead of paying the synthesis cost at
    that click. Never touches playback, so it can safely run in the
    background while the user keeps using the rest of the app. Returns
    (ok: bool, message: str)."""
    text, engine, wav_path, error = _load_script_and_engine(script_path)
    if error:
        return False, error

    if wav_path.exists():
        return True, "Already cached."

    result = engine.synthesize(text, output_path=str(wav_path))
    if not result.success:
        return False, result.error or "Synthesis failed."
    return True, "Synthesized."


def synthesize_and_play(script_path: Path):
    """Blocking call: read a session's *_tts.txt script, detect whether
    it's English or Hindi from its content, and speak it via the matching
    Piper voice. Caches the WAV next to the script (same stem, ".wav") so
    a re-play doesn't re-synthesize -- and so a prior background
    synthesize_only() call (see above) makes this instant. Meant to be run
    off the Tk main thread (see the button handler below) -- returns
    (ok: bool, message: str).
    """
    text, engine, wav_path, error = _load_script_and_engine(script_path)
    if error:
        return False, error

    if wav_path.exists():
        from piper_tts import _player_command, wav_duration_seconds
        cmd = _player_command(str(wav_path))
        if cmd:
            # Same fix as PiperTTS.speak(): size the playback timeout to
            # the clip itself rather than reusing engine.timeout (a
            # synthesis-sized ~30s budget), which was killing playback of
            # any cached clip longer than that -- the exact "plays for a
            # bit, cuts off, has to be re-clicked from 0:00" symptom.
            duration = wav_duration_seconds(str(wav_path))
            playback_timeout = (duration + engine.playback_margin) if duration else None
            try:
                # Go through engine._runner (not subprocess.run directly) so
                # this playback process is tracked and can be killed by
                # kill_all_active() if the app closes mid-playback.
                engine._runner(cmd, "", playback_timeout)
                return True, "Played cached audio."
            except Exception:
                pass  # fall through and re-synthesize

    result = engine.speak(text, output_path=str(wav_path))
    if not result.success:
        return False, result.error or "Synthesis failed."
    if result.played:
        return True, "Played."
    if result.playback_error:
        return True, f"Synthesized, but playback didn't finish: {result.playback_error}"
    return True, "Synthesized (no audio player found to play it)."


# ---------------------------------------------------------------------------
# Audio activity animation -- small equalizer shown while Piper is
# synthesizing/playing. Pure Canvas + Tk .after() loop, no extra deps.
# ---------------------------------------------------------------------------

_ANIM_BAR_COUNT = 5
_ANIM_W, _ANIM_H = 42, 20
_ANIM_BAR_W = 4


def _make_audio_animation(parent, bg):
    """Creates a small equalizer-style Canvas inside `parent` (packed
    side='left' right where it's created) and returns (start, stop).

    start() begins animating bars pulsing at slightly different phases,
    stop() freezes them back down to a flat baseline. Safe to call stop()
    multiple times or after the widget is gone.
    """
    canvas = tk.Canvas(parent, width=_ANIM_W, height=_ANIM_H, bg=bg,
                        highlightthickness=0)
    canvas.pack(side="left", padx=(10, 0))

    gap = (_ANIM_W - _ANIM_BAR_COUNT * _ANIM_BAR_W) // (_ANIM_BAR_COUNT + 1)
    bar_ids = []
    for i in range(_ANIM_BAR_COUNT):
        x0 = gap + i * (_ANIM_BAR_W + gap)
        bar_ids.append(canvas.create_rectangle(
            x0, _ANIM_H - 2, x0 + _ANIM_BAR_W, _ANIM_H, fill=PRIMARY, width=0,
        ))

    anim_state = {"running": False, "phase": 0.0}

    def _tick():
        if not anim_state["running"]:
            return
        anim_state["phase"] += 0.4
        for i, bar_id in enumerate(bar_ids):
            # each bar rides its own phase offset so they don't move in lockstep
            level = (math.sin(anim_state["phase"] + i * 1.4) + 1) / 2  # 0..1
            h = 2 + level * (_ANIM_H - 4)
            x0 = gap + i * (_ANIM_BAR_W + gap)
            try:
                canvas.coords(bar_id, x0, _ANIM_H - h, x0 + _ANIM_BAR_W, _ANIM_H)
            except tk.TclError:
                anim_state["running"] = False
                return
        try:
            canvas.after(80, _tick)
        except tk.TclError:
            anim_state["running"] = False

    def start():
        if anim_state["running"]:
            return
        anim_state["running"] = True
        _tick()

    def stop():
        anim_state["running"] = False
        try:
            for bar_id in bar_ids:
                x0, _, x1, _ = canvas.coords(bar_id)
                canvas.coords(bar_id, x0, _ANIM_H - 2, x1, _ANIM_H)
        except tk.TclError:
            pass

    return start, stop


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def open_file(path: Path) -> None:
    if not path.exists():
        return
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)
    elif sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.as_uri())


def _show_full_summary(parent, entry: SessionEntry) -> None:
    """Popup listing every recorded metric for one session (see
    all_summary_points) -- the card view only shows a handful."""
    win = tk.Toplevel(parent)
    win.title(f"Full Summary -- {format_timestamp(entry)}")
    win.configure(bg=BG)
    win.geometry("520x600")
    win.transient(parent.winfo_toplevel())

    tk.Label(win, text=format_timestamp(entry), font=("Helvetica", 16, "bold"),
              bg=BG, fg=PRIMARY).pack(anchor="w", padx=20, pady=(18, 4))

    body = tk.Frame(win, bg=BG)
    body.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
    scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=BG)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for point in all_summary_points(entry):
        tk.Label(inner, text=f"\u2022 {point}", font=("Helvetica", 12),
                  bg=BG, fg=TEXT, justify="left", anchor="w",
                  wraplength=460).pack(anchor="w", fill="x", pady=(3, 0))

    tk.Button(
        win, text="Close", command=win.destroy,
        font=("Helvetica", 11, "bold"), bg=CARD, fg="white",
        activebackground=SECONDARY, activeforeground="white",
        relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
    ).pack(pady=(0, 16))



# ---------------------------------------------------------------------------
# Tkinter UI -- embedded screen (not a separate window)
#
# mira_app.py uses one Tk root with a single `self.container` frame that
# every screen clears and rebuilds into (show_registration, show_guidelines,
# show_success, ...). This follows the same pattern instead of opening a
# Toplevel, so "History" behaves like any other screen/tab in the app and
# fills the whole (already-maximized) app window.
# ---------------------------------------------------------------------------

def build_history_screen(parent, patient_id: str, database_root: str,
                          patient_label: str = "", on_back=None,
                          on_view_trends=None):
    """Builds the History screen directly into `parent` (normally
    mira_app.self.container, right after self.clear()). Fills the full
    width/height of whatever `parent` gives it."""

    root_frame = tk.Frame(parent, bg=BG)
    root_frame.pack(fill="both", expand=True)

    state = {"entries": [], "page": 0, "error": None}
    state["_thumb_refs"] = []      # keep PhotoImage refs alive
    state["_wrap_labels"] = []     # bullet labels whose wraplength tracks window width

    # -- header --
    header = tk.Frame(root_frame, bg=BG)
    header.pack(fill="x", padx=30, pady=(20, 10))

    if on_back:
        tk.Button(
            header, text="\u2190 Back", command=on_back,
            font=("Helvetica", 11, "bold"), bg=CARD, fg="white",
            activebackground=SECONDARY, activeforeground="white",
            relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
        ).pack(side="left", padx=(0, 18))

    tk.Label(header, text="History", font=("Helvetica", 26, "bold"),
              bg=BG, fg=PRIMARY).pack(side="left")
    if patient_label:
        tk.Label(header, text=patient_label, font=("Helvetica", 13),
                  bg=BG, fg=MUTED).pack(side="left", padx=(14, 0))

    if on_view_trends:
        tk.Button(
            header, text="View Trends \u2192", command=on_view_trends,
            font=("Helvetica", 11, "bold"), bg=CARD, fg="white",
            activebackground=SECONDARY, activeforeground="white",
            relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
        ).pack(side="right")

    # -- scrollable card list (fills full width) --
    body = tk.Frame(root_frame, bg=BG)
    body.pack(fill="both", expand=True, padx=30, pady=(0, 10))

    canvas = tk.Canvas(body, bg=BG, highlightthickness=0)
    scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
    cards_frame = tk.Frame(canvas, bg=BG)
    canvas_window = canvas.create_window((0, 0), window=cards_frame, anchor="nw")

    def _on_cards_configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        # Stretch the inner frame to the canvas's full width so cards span
        # left-to-right instead of hugging the left side, and re-wrap the
        # bullet-point labels to the newly available width.
        canvas.itemconfig(canvas_window, width=event.width)
        avail = max(event.width - THUMB_SIZE[0] - 90, 240)
        for lbl in state["_wrap_labels"]:
            lbl.config(wraplength=avail)

    cards_frame.bind("<Configure>", _on_cards_configure)
    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        delta = -1 if event.delta > 0 else 1
        canvas.yview_scroll(delta, "units")

    # Only capture the mouse wheel while the pointer is actually over this
    # screen's canvas -- bind/unbind on enter/leave instead of bind_all for
    # the whole app, since this frame gets destroyed on every screen switch.
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # -- pager --
    pager = tk.Frame(root_frame, bg=BG)
    pager.pack(fill="x", padx=30, pady=(0, 20))
    prev_btn = tk.Button(pager, text="< Prev", font=("Helvetica", 12, "bold"),
                          bg=CARD, fg="white", activebackground=SECONDARY,
                          activeforeground="white", relief="flat", bd=0,
                          padx=18, pady=8, cursor="hand2")
    prev_btn.pack(side="left")
    page_label = tk.Label(pager, text="", font=("Helvetica", 12),
                           bg=BG, fg=MUTED)
    page_label.pack(side="left", expand=True)
    next_btn = tk.Button(pager, text="Next >", font=("Helvetica", 12, "bold"),
                          bg=CARD, fg="white", activebackground=SECONDARY,
                          activeforeground="white", relief="flat", bd=0,
                          padx=18, pady=8, cursor="hand2")
    next_btn.pack(side="right")

    def _make_card(parent_frame, entry: SessionEntry, wrap_width: int):
        card = tk.Frame(parent_frame, bg=CARD, padx=20, pady=18)
        card.pack(fill="x", pady=8, padx=2)

        # -- photo (or empty "No photo" box) --
        photo_box = tk.Frame(card, bg=CARD_ALT, width=THUMB_SIZE[0],
                              height=THUMB_SIZE[1])
        photo_box.pack(side="left", padx=(0, 24))
        photo_box.pack_propagate(False)

        shown = False
        if entry.photo_path and _PIL_OK:
            try:
                img = Image.open(entry.photo_path)
                img.thumbnail(THUMB_SIZE)
                tk_img = ImageTk.PhotoImage(img)
                state["_thumb_refs"].append(tk_img)
                tk.Label(photo_box, image=tk_img, bg=CARD_ALT).place(
                    relx=0.5, rely=0.5, anchor="center")
                shown = True
            except Exception:
                shown = False
        if not shown:
            tk.Label(photo_box, text="No photo", font=("Helvetica", 10),
                      bg=CARD_ALT, fg=MUTED).place(relx=0.5, rely=0.5,
                                                     anchor="center")

        # -- text column: fills all remaining width left to right --
        text_col = tk.Frame(card, bg=CARD)
        text_col.pack(side="left", fill="both", expand=True)

        tk.Label(text_col, text=format_timestamp(entry),
                  font=("Helvetica", 16, "bold"), bg=CARD, fg=TEXT).pack(
            anchor="w", fill="x")

        for point in highlight_points(entry):
            lbl = tk.Label(
                text_col, text=f"\u2022 {point}", font=("Helvetica", 13),
                bg=CARD, fg=MUTED, wraplength=wrap_width, justify="left",
                anchor="w",
            )
            lbl.pack(anchor="w", fill="x", pady=(4, 0))
            state["_wrap_labels"].append(lbl)

        # -- audio/TTS control --
        controls = tk.Frame(text_col, bg=CARD)
        controls.pack(anchor="w", fill="x", pady=(12, 0))

        if entry.tts_text_path:
            audio_btn = tk.Button(
                controls, text="\U0001F50A Play Audio",
                font=("Helvetica", 11), bg=SECONDARY, fg="white",
                activebackground=PRIMARY, activeforeground="white",
                relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
            )
            audio_btn.pack(side="left")

            anim_start, anim_stop = _make_audio_animation(controls, CARD)

            status_lbl = tk.Label(controls, text="", font=("Helvetica", 10),
                                    bg=CARD, fg=MUTED)
            status_lbl.pack(side="left", padx=(10, 0))

            def _play(script_path=entry.tts_text_path, btn=audio_btn, lbl=status_lbl):
                btn.config(state="disabled", text="\u23f3 Synthesizing...")
                lbl.config(text="")
                anim_start()

                def worker():
                    ok, message = synthesize_and_play(script_path)

                    def _update():
                        anim_stop()
                        if not btn.winfo_exists():
                            return
                        btn.config(
                            state="normal",
                            text="\U0001F50A Play Audio" if ok else "\u26a0 Retry",
                        )
                        lbl.config(text=message, fg=MUTED if ok else WARNING)
                    try:
                        btn.after(0, _update)
                    except tk.TclError:
                        pass

                threading.Thread(target=worker, daemon=True).start()

            audio_btn.config(command=_play)
        else:
            tk.Button(
                controls, text="\U0001F507 No audio script for this session",
                font=("Helvetica", 11), bg=SECONDARY, fg=MUTED,
                relief="flat", bd=0, padx=14, pady=6, state="disabled",
            ).pack(side="left")

        # -- summary / PDF controls (second row) --
        doc_controls = tk.Frame(text_col, bg=CARD)
        doc_controls.pack(anchor="w", fill="x", pady=(8, 0))

        tk.Button(
            doc_controls, text="\U0001F4CB Full Summary",
            command=lambda e=entry: _show_full_summary(card, e),
            font=("Helvetica", 11), bg=SECONDARY, fg="white",
            activebackground=PRIMARY, activeforeground="white",
            relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
        ).pack(side="left")

        doctor_pdf = _find_report(entry, "_doctor")
        if doctor_pdf:
            tk.Button(
                doc_controls, text="\U0001FA7A Doctor PDF",
                command=lambda p=doctor_pdf: open_file(p),
                font=("Helvetica", 11), bg=SECONDARY, fg="white",
                activebackground=PRIMARY, activeforeground="white",
                relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
            ).pack(side="left", padx=(10, 0))

        user_pdf = _find_report(entry, "_user")
        if user_pdf:
            tk.Button(
                doc_controls, text="\U0001F4C4 Patient PDF",
                command=lambda p=user_pdf: open_file(p),
                font=("Helvetica", 11), bg=SECONDARY, fg="white",
                activebackground=PRIMARY, activeforeground="white",
                relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
            ).pack(side="left", padx=(10, 0))

        if not doctor_pdf and not user_pdf:
            tk.Label(
                doc_controls, text="No PDF report for this session",
                font=("Helvetica", 10), bg=CARD, fg=MUTED,
            ).pack(side="left", padx=(10, 0))

    def _clear_cards():
        for w in cards_frame.winfo_children():
            w.destroy()
        state["_thumb_refs"].clear()
        state["_wrap_labels"].clear()

    def _redraw():
        _clear_cards()
        canvas.yview_moveto(0)

        canvas_w = canvas.winfo_width() or body.winfo_width() or 900
        wrap_width = max(canvas_w - THUMB_SIZE[0] - 90, 240)

        if state["error"]:
            tk.Label(cards_frame, text=state["error"], font=("Helvetica", 13),
                      bg=BG, fg=WARNING, wraplength=wrap_width,
                      justify="left").pack(anchor="w", pady=30, fill="x")
            page_label.config(text="")
            prev_btn.config(state="disabled")
            next_btn.config(state="disabled")
            return

        entries = state["entries"]
        if not entries:
            tk.Label(cards_frame, text="No history yet for this patient.",
                      font=("Helvetica", 13), bg=BG, fg=MUTED).pack(
                anchor="w", pady=30, fill="x")
            page_label.config(text="")
            prev_btn.config(state="disabled")
            next_btn.config(state="disabled")
            return

        pages = total_pages(entries)
        state["page"] = max(0, min(state["page"], pages - 1))
        page_entries = paginate(entries, state["page"])

        for entry in page_entries:
            _make_card(cards_frame, entry, wrap_width)

        page_label.config(
            text=f"Page {state['page'] + 1} of {pages}  "
                 f"({len(entries)} session(s) total)")
        prev_btn.config(state="normal" if state["page"] > 0 else "disabled")
        next_btn.config(state="normal" if state["page"] < pages - 1 else "disabled")

    def _go_prev():
        state["page"] -= 1
        _redraw()

    def _go_next():
        state["page"] += 1
        _redraw()

    prev_btn.config(command=_go_prev)
    next_btn.config(command=_go_next)

    def _reload():
        try:
            state["entries"] = load_patient_sessions(patient_id, database_root)
            state["error"] = None
        except Exception as exc:
            state["entries"] = []
            state["error"] = f"Could not load history:\n{exc}"
        state["page"] = 0
        _redraw()
        # Get every session's audio synthesizing in the background as soon
        # as History loads, rather than waiting for a "Play Audio" click --
        # this runs on a daemon thread and never blocks the UI.
        prefetch_missing_audio(state["entries"])

    _reload()

    root_frame._history_screen_state = state
    root_frame._history_screen_reload = _reload
    return root_frame