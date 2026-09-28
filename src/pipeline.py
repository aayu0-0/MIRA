"""
MIRA pipeline — Phase 1 → Phase 6

This file owns ALL non-UI work.

mira_app.py only launches this file with:
    python pipeline.py --name "<NAME>" --id "<ID>"

This pipeline:
    1. Opens the webcam and runs a fully automatic LIVE burst capture: no
       key press is needed. Frames are streamed continuously; only frames
       with no face or a genuinely blurry face are rejected outright.
       Everything else (lighting, angle, symmetry) is scored, not gated,
       and the best-scoring frames are kept as a candidate pool (Stage 1).
    2. Runs the full analysis (Phases 1-4) on the top finalists of that pool
       and auto-selects the single frame with the highest `overall_confidence`
       as the tiebreaker (Stage 2) — no manual picking.
    3. Extracts facial regions (already computed for the winning frame).
    4. Calculates feature scores (already computed for the winning frame).
    5. Generates observations (already computed for the winning frame).
    6. Generates doctor/user reports and metrics/TTS outputs.
    7. Saves the auto-selected photo in Database/photos/.
    8. Saves reports in Database/reports/.
    9. Appends one feature-vector row to Database/dataset.csv.

File naming:
    <Name>_<ID>_<NN>.jpg
    <Name>_<ID>_<NN>_doctor.pdf
    <Name>_<ID>_<NN>_user.pdf
    <Name>_<ID>_<NN>_metrics.json
    <Name>_<ID>_<NN>_tts.txt

Example:
    Aayush_Sardana_001_01.jpg
    Aayush_Sardana_001_01_doctor.pdf
    Aayush_Sardana_001_01_user.pdf

NN automatically increments: 01, 02, 03, ...
"""

import argparse
import csv
import datetime
import os
import re
import sys
import time

# Windows terminals may default to cp1252/charmap. The pipeline is launched
# by mira_app.py and its stdout is read as UTF-8, so force UTF-8 here.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import cv2

from phase1_face_understanding.face_detector import FaceDetector
from phase2_region_extraction.region_extractor import RegionExtractor
from phase3_feature_analysis.feature_analyzer import FeatureAnalyzer
from phase4_observation_engine.observation_engine import ObservationEngine
from phase5_report.metrics_report import MetricsReportGenerator
from phase5_report.doctor_report import DoctorReportGenerator
from phase5_report.user_report import UserReportGenerator
from phase5_report.tts_report import TTSReportGenerator
from phase6_feature_vector.feature_vector_export import (
    ALL_COLUMNS,
    append_feature_row,
    flatten_phase3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Database/ lives one level up from src/ (MIRA_doc+clean/Database), not
# inside src/, so that src stays free of generated/data folders.
DATABASE_DIR = os.path.join(os.path.dirname(SRC_DIR), "Database")


# ─────────────────────────────────────────────────────────────────────────────
# Identity / filename helpers
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_filename_part(value: str) -> str:
    """Make a user-entered name/ID safe to use in a Windows filename."""
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip(" ._")

    return value or "UNKNOWN"


def normalize_patient_id(value: str, width: int = 3) -> str:
    """Zero-pad a purely numeric patient ID to a fixed width.

    Without this, the same patient gets stored/foldered under a different
    string depending on how many leading zeros they happened to type that
    session ("1" vs "001") -- same person, inconsistent identity across
    rows/folders. Non-numeric IDs are returned unchanged (still passed
    through sanitize_filename_part as before wherever a filename-safe
    version is needed).
    """
    value = str(value).strip()
    return value.zfill(width) if value.isdigit() else value


def next_capture_number(name: str, patient_id: str, photos_dir: str, reports_dir: str) -> int:
    """
    Find the next available NN for this Name_ID combination.

    The photos folder is the source of truth for existing captures.
    We also inspect the reports folder so a partially completed capture
    cannot accidentally reuse an existing report number.
    """
    safe_name = sanitize_filename_part(name)
    safe_id = sanitize_filename_part(patient_id)
    prefix = f"{safe_name}_{safe_id}_"

    highest = 0

    for folder in (photos_dir, reports_dir):
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            stem, _ = os.path.splitext(filename)

            if not stem.startswith(prefix):
                continue

            remainder = stem[len(prefix):]
            match = re.match(r"(\d+)(?:_|$)", remainder)

            if match:
                try:
                    highest = max(highest, int(match.group(1)))
                except ValueError:
                    pass

    return highest + 1


# ─────────────────────────────────────────────────────────────────────────────
# Face positioning guide overlay (live camera preview only)
#
# This mirrors the static guide drawn in mira_app.py (oval outline, eye-line,
# corner brackets, centre crosshair) but is rendered per-frame with OpenCV on
# top of the live webcam preview. It is purely visual: it is drawn on a COPY
# of each frame right before cv2.imshow, and is never present in the frame
# that actually gets captured/saved/analyzed.
# ─────────────────────────────────────────────────────────────────────────────

# BGR colours (converted from the GUI's hex palette: PRIMARY=#e94560, MUTED=#a8a8b3).
GUIDE_PRIMARY_BGR = (96, 69, 233)
GUIDE_MUTED_BGR = (179, 168, 168)
GUIDE_EYELINE_BGR = (0, 210, 255)   # bright amber — high contrast on any skin tone/background


def draw_face_guide_overlay(frame):
    """Return a copy of `frame` with a face-positioning guide drawn on it."""
    display = frame.copy()
    h, w = display.shape[:2]
    cx, cy = w // 2, h // 2

    # Corner viewfinder brackets.
    bracket = int(min(w, h) * 0.06)
    margin = int(min(w, h) * 0.04)
    corners = [(margin, margin, 1, 1), (w - margin, margin, -1, 1),
               (margin, h - margin, 1, -1), (w - margin, h - margin, -1, -1)]

    for x, y, dx, dy in corners:
        cv2.line(display, (x, y), (x + dx * bracket, y), GUIDE_MUTED_BGR, 2)
        cv2.line(display, (x, y), (x, y + dy * bracket), GUIDE_MUTED_BGR, 2)

    # Face oval guide (dashed look via short segments).
    oval_axes = (int(w * 0.16), int(h * 0.33))
    _draw_dashed_ellipse(display, (cx, cy), oval_axes, GUIDE_PRIMARY_BGR, 2)

    # Eye-line, roughly upper-third of the oval. Solid (not dashed) and
    # brighter/thicker than the rest of the guide so it reads clearly even
    # against busy backgrounds — plus two small markers showing exactly
    # where each eye should land.
    eye_y = int(cy - oval_axes[1] * 0.35)
    eye_half_span = int(oval_axes[0] * 0.55)

    cv2.line(
        display,
        (cx - int(oval_axes[0] * 0.9), eye_y),
        (cx + int(oval_axes[0] * 0.9), eye_y),
        GUIDE_EYELINE_BGR,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(display, (cx - eye_half_span, eye_y), 5, GUIDE_EYELINE_BGR, -1, cv2.LINE_AA)
    cv2.circle(display, (cx + eye_half_span, eye_y), 5, GUIDE_EYELINE_BGR, -1, cv2.LINE_AA)
    cv2.putText(
        display,
        "Eyes here",
        (cx + eye_half_span + 12, eye_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        GUIDE_EYELINE_BGR,
        1,
        cv2.LINE_AA,
    )

    # Centre crosshair.
    cross = 10
    cv2.line(display, (cx - cross, cy), (cx + cross, cy), GUIDE_MUTED_BGR, 1)
    cv2.line(display, (cx, cy - cross), (cx, cy + cross), GUIDE_MUTED_BGR, 1)

    cv2.putText(
        display,
        "Align your face inside the outline",
        (int(w * 0.5) - 150, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GUIDE_MUTED_BGR,
        1,
        cv2.LINE_AA,
    )

    return display


def _draw_dashed_ellipse(img, center, axes, color, thickness, dash_deg=10, gap_deg=8):
    angle = 0
    while angle < 360:
        cv2.ellipse(
            img, center, axes, 0, angle, min(angle + dash_deg, 360),
            color, thickness, cv2.LINE_AA,
        )
        angle += dash_deg + gap_deg


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len=6, gap_len=5):
    import math

    x1, y1 = pt1
    x2, y2 = pt2
    length = math.hypot(x2 - x1, y2 - y1)

    if length == 0:
        return

    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0

    while pos < length:
        seg_end = min(pos + dash_len, length)
        sx, sy = x1 + dx * pos, y1 + dy * pos
        ex, ey = x1 + dx * seg_end, y1 + dy * seg_end
        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), color, thickness, cv2.LINE_AA)
        pos += dash_len + gap_len


# ─────────────────────────────────────────────────────────────────────────────
# Live burst capture + automatic best-frame selection
#
# Simplified gating (no more brightness/highlight/alignment/symmetry hard
# rejects — they were computed on the wrong region and/or stacked into a
# wall that killed every frame). Only two hard rejects remain:
#   - no face detected
#   - face crop too blurry (Laplacian variance on the face bbox, not the
#     full frame — background clutter no longer counts against you)
#
# Lighting, alignment angle, and symmetry now only affect ranking via
# _quick_quality_score, so a bad-but-usable frame can still get captured
# and just loses out to a better one in the pool instead of blocking
# capture outright.
#
# Stage 1 (cheap, per live frame): reject no-face / blurry frames, score
# everything else, and keep a candidate pool of the BURST_POOL_SIZE
# best-scoring frames.
#
# Stage 2 (after the burst ends): run the real analysis (Phase 2-4) on the
# top finalists and auto-select the single frame with the highest
# `overall_confidence` as the tiebreaker. Phase 1 is reused from Stage 1
# (not recomputed), and Phase 2-4 results for the winning frame are reused
# downstream in run_pipeline (not recomputed either) — so nothing here is
# thrown away.
#
# Everything is automatic: there is no SPACE-to-capture step any more. ESC
# still lets the user abort the whole capture.
# ─────────────────────────────────────────────────────────────────────────────

BURST_POOL_SIZE = 20          # how many top-scoring candidates Stage 1 keeps
BURST_MIN_SECONDS = 2.0       # don't stop early even if the pool fills fast
BURST_MAX_SECONDS = 10.0      # hard cap on how long the live burst can run
BURST_MAX_FRAMES = 200        # hard cap on raw frames read from the webcam

BLUR_VARIANCE_MIN = 60.0      # face-crop Laplacian variance floor — only hard
                               # quality reject left, besides "no face"

FINALIST_COUNT = 5             # only the top N of the pool get full Phase 2-4
                                # analysis in Stage 2 (was: all 20 — wasteful
                                # and let low-sharpness frames win on confidence)
CONFIDENCE_TIEBREAK_MARGIN = 5.0  # a finalist only competes on overall_confidence
                                   # if its quick_score is within this many points
                                   # of the sharpest/best-aligned finalist

# NOTE: BLUR_VARIANCE_MIN is heuristic and camera-dependent. If real captures
# are getting rejected too often (or not often enough), this is the number
# to tune first.


def _laplacian_variance(gray_frame) -> float:
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _mean_brightness(gray_frame) -> float:
    return float(gray_frame.mean())


def _quick_quality_score(
    blur_var: float, symmetry_score: float, alignment_angle: float, brightness: float
) -> float:
    """
    Cheap, Phase-1-only quality score used to rank the live stream and pick
    the Stage 1 candidate pool. 0-100 scale (soft-capped, not a hard bound).
    Lighting is folded in here (as a soft penalty) instead of being a hard
    reject.
    """
    sharpness = min(blur_var / 300.0, 1.0) * 100.0
    symmetry = max(0.0, min(float(symmetry_score), 100.0))
    align_score = max(0.0, 100.0 - abs(alignment_angle) * 4.0)
    exposure = max(0.0, 100.0 - abs(brightness - 127.5) / 1.275)

    return 0.40 * sharpness + 0.25 * symmetry + 0.20 * align_score + 0.15 * exposure


def _draw_burst_hud(display, raw_count: int, candidate_count: int, status):
    accepted = bool(status) and status.startswith("Accepted")
    status_color = (90, 200, 90) if accepted else GUIDE_MUTED_BGR

    cv2.putText(
        display, f"Frames scanned: {raw_count}", (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GUIDE_MUTED_BGR, 1, cv2.LINE_AA,
    )
    cv2.putText(
        display, f"Candidates: {candidate_count}/{BURST_POOL_SIZE}", (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GUIDE_MUTED_BGR, 1, cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            display, status, (16, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA,
        )


def _draw_waiting_hud(display):
    cv2.putText(
        display, "Get positioned, then press SPACE to start", (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 90), 2, cv2.LINE_AA,
    )
    cv2.putText(
        display, "ESC to cancel", (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GUIDE_MUTED_BGR, 1, cv2.LINE_AA,
    )


WEBCAM_WINDOW_TITLE = "MIRA — Live Capture"


def capture_and_select_best_frame(detector, extractor, analyzer, engine):
    """
    Wait for SPACE, then run a fully automatic live burst capture and return
    the auto-selected best frame, already analyzed.

    Returns a dict with keys: frame, p1, p2, p3, p4, quick_score,
    overall_confidence, raw_frame_count, candidate_count, pool_size.
    Returns None if the user cancelled (ESC) or no usable frame was found.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("WEBCAM: Camera opened.")
    print("WEBCAM: Press SPACE to start the automatic burst capture, ESC to cancel.")

    candidates = []  # each: {"frame", "p1", "quick_score"}
    raw_count = 0
    start_time = None
    burst_started = False

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError("Failed to read frame from webcam.")

            # ── Waiting phase: show the live preview + face guide, but do
            # not process frames or start the burst timer until SPACE. ──
            if not burst_started:
                display_frame = draw_face_guide_overlay(frame)
                _draw_waiting_hud(display_frame)
                cv2.imshow(WEBCAM_WINDOW_TITLE, display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print("CANCELLED: Capture aborted before starting.")
                    return None

                if key == 32:
                    burst_started = True
                    start_time = time.time()
                    print("WEBCAM: Burst capture started.")

                continue

            raw_count += 1

            p1 = detector.analyze(frame)

            if not p1.success:
                status = "No face detected"
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_crop = gray[
                    p1.bbox.y:p1.bbox.y + p1.bbox.h,
                    p1.bbox.x:p1.bbox.x + p1.bbox.w,
                ]

                if face_crop.size:
                    blur_var = _laplacian_variance(face_crop)
                    brightness = _mean_brightness(face_crop)
                else:
                    blur_var = 0.0
                    brightness = 127.5

                if blur_var < BLUR_VARIANCE_MIN:
                    status = "Too blurry — hold still"
                else:
                    score = _quick_quality_score(
                        blur_var, p1.symmetry_score, p1.alignment_angle, brightness
                    )
                    candidates.append(
                        {"frame": frame.copy(), "p1": p1, "quick_score": score}
                    )
                    status = f"Accepted (score {score:.0f})"

            # Display-only overlay; the frames appended to `candidates` above
            # are unmodified copies of the raw camera frame.
            display_frame = draw_face_guide_overlay(frame)
            _draw_burst_hud(display_frame, raw_count, len(candidates), status)

            cv2.imshow(WEBCAM_WINDOW_TITLE, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("CANCELLED: Burst capture aborted by user.")
                return None

            elapsed = time.time() - start_time
            pool_ready = len(candidates) >= BURST_POOL_SIZE and elapsed >= BURST_MIN_SECONDS

            if pool_ready or elapsed >= BURST_MAX_SECONDS or raw_count >= BURST_MAX_FRAMES:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not candidates:
        print(
            "WEBCAM: No usable frames captured — no face was detected, or "
            "every frame was too blurry. Check lighting/focus and try again."
        )
        return None

    # Stage 1 result: keep the top BURST_POOL_SIZE by quick score (sharpness /
    # symmetry / alignment / exposure). This ranking is authoritative for
    # "how good does this frame look" — Stage 2 below is only allowed to
    # override it within a small tie margin, using overall_confidence as a
    # genuine tiebreaker, never as the primary criterion.
    candidates.sort(key=lambda c: c["quick_score"], reverse=True)
    pool = candidates[:BURST_POOL_SIZE]

    print(
        f"BURST: {raw_count} frames scanned, {len(candidates)} passed quality "
        f"filters, top {len(pool)} kept as the pool."
    )

    # Stage 2: run full analysis ONLY on the sharpest/best-aligned finalists
    # (not the whole pool — no reason to spend Phase 2-4 compute on frames
    # that already lost on quick_score). Among the finalists, only frames
    # within CONFIDENCE_TIEBREAK_MARGIN of the top quick_score are allowed to
    # win via overall_confidence; a lower-quality frame can no longer beat a
    # sharper, better-aligned one just because it scored a higher confidence.
    finalists = pool[:FINALIST_COUNT]

    if not finalists:
        print("WEBCAM: No finalists to analyze.")
        return None

    top_quick_score = finalists[0]["quick_score"]
    best = None

    for candidate in finalists:
        frame_c = candidate["frame"]
        p1 = candidate["p1"]

        p2 = extractor.extract(frame_c, p1)
        if not p2.success:
            continue

        p3 = analyzer.analyze(frame_c, p1, p2)
        p4 = engine.generate(p3)

        candidate["p2"], candidate["p3"], candidate["p4"] = p2, p3, p4
        candidate["overall_confidence"] = p4.overall_confidence

        within_tie_margin = (
            top_quick_score - candidate["quick_score"]
        ) <= CONFIDENCE_TIEBREAK_MARGIN

        if not within_tie_margin:
            # Meaningfully blurrier/less aligned than the best finalist —
            # not a contender regardless of what overall_confidence says.
            continue

        if best is None or candidate["overall_confidence"] > best["overall_confidence"]:
            best = candidate

    if best is None:
        # Nothing passed Phase 2 or cleared the tie margin — fall back to
        # the single sharpest/best-aligned finalist rather than failing.
        fallback = finalists[0]
        if "p4" not in fallback:
            frame_c = fallback["frame"]
            p1 = fallback["p1"]
            p2 = extractor.extract(frame_c, p1)
            if p2.success:
                p3 = analyzer.analyze(frame_c, p1, p2)
                p4 = engine.generate(p3)
                fallback["p2"], fallback["p3"], fallback["p4"] = p2, p3, p4
                fallback["overall_confidence"] = p4.overall_confidence

        best = fallback if "p4" in fallback else None

    if best is None:
        print("WEBCAM: None of the candidate frames passed full analysis.")
        return None

    best["raw_frame_count"] = raw_count
    best["candidate_count"] = len(candidates)
    best["pool_size"] = len(pool)

    print(
        f"AUTO-SELECTED: quick_score={best['quick_score']:.0f}, "
        f"overall_confidence={best['overall_confidence']:.0f}%"
    )

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Console progress
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'-' * 55}")
    print(f"  {title}")
    print(f"{'-' * 55}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(name: str, patient_id: str, language: str = "en") -> int:
    # Normalize once, up front, so every downstream use -- the patient
    # folder name, capture filenames, and the patient_id column written to
    # dataset.csv -- sees the same canonical ID regardless of how many
    # leading zeros were typed this particular session.
    patient_id = normalize_patient_id(patient_id)
    safe_name = sanitize_filename_part(name)
    safe_id = sanitize_filename_part(patient_id)

    # Create a separate database folder for each patient.
    patient_folder = f"{safe_name}_{safe_id}"
    output_dir = os.path.join(DATABASE_DIR, patient_folder)
    photos_dir = os.path.join(output_dir, "photos")
    reports_dir = os.path.join(output_dir, "reports")
    dataset_csv = os.path.join(output_dir, "dataset.csv")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    capture_number = next_capture_number(safe_name, safe_id, photos_dir, reports_dir)

    # Keep NN at least two digits: 01, 02, 03...
    capture_tag = f"{capture_number:02d}"
    base_name = f"{safe_name}_{safe_id}_{capture_tag}"

    photo_filename = f"{base_name}.jpg"
    photo_path = os.path.join(photos_dir, photo_filename)

    doctor_pdf_path = os.path.join(
        reports_dir,
        f"{base_name}_doctor.pdf",
    )
    user_pdf_path = os.path.join(
        reports_dir,
        f"{base_name}_user.pdf",
    )
    metrics_path = os.path.join(
        reports_dir,
        f"{base_name}_metrics.json",
    )
    tts_path = os.path.join(
        reports_dir,
        f"{base_name}_tts.txt",
    )

    print(f"PATIENT: {name}")
    print(f"ID:      {patient_id}")
    print(f"CAPTURE: {capture_tag}")

    # ── Capture: live burst + auto-select (Phases 1-4 run inside) ──────────
    print_section("Camera — live burst capture (automatic)")

    with FaceDetector() as detector:
        extractor = RegionExtractor()
        analyzer = FeatureAnalyzer()
        engine = ObservationEngine()

        best = capture_and_select_best_frame(detector, extractor, analyzer, engine)

        if best is None:
            return 1

        frame = best["frame"]
        p1, p2, p3, p4 = best["p1"], best["p2"], best["p3"], best["p4"]

        p1_viz = detector.draw_landmarks(frame, p1)

    # Save the auto-selected photo.
    if not cv2.imwrite(photo_path, frame):
        raise RuntimeError(f"Could not save captured photo: {photo_path}")

    print(f"PHOTO SAVED: {photo_path}")

    # Store a portable path in the CSV rather than an absolute machine path.
    csv_photo_path = os.path.join(
        "photos",
        photo_filename,
    ).replace("\\", "/")

    print_section("Phase 1 — Face Detection (winning frame)")
    print(f"Symmetry  : {p1.symmetry_score} ({p1.symmetry_label})")
    print(f"Alignment : {p1.alignment_angle:.2f} deg")
    print(
        f"Face bbox : x={p1.bbox.x} y={p1.bbox.y} "
        f"w={p1.bbox.w} h={p1.bbox.h}"
    )

    print_section("Phase 2 — Region Extraction (winning frame)")
    available = [k for k, v in p2.regions.items() if v.available]
    failed = [k for k, v in p2.regions.items() if not v.available]

    print(f"Regions extracted: {len(available)}/{len(p2.regions)}")

    if failed:
        print(f"Failed regions: {failed}")

    print_section("Phase 3 — Feature Analysis (winning frame)")
    print(
        f"Left eye  - dark circles: "
        f"{p3.left_eye.dark_circle_score:.1f}, "
        f"redness: {p3.left_eye.redness_score:.1f}, "
        f"puffiness: {p3.left_eye.puffiness_score:.1f}"
    )

    print(
        f"Right eye - dark circles: "
        f"{p3.right_eye.dark_circle_score:.1f}, "
        f"redness: {p3.right_eye.redness_score:.1f}"
    )

    print(
        f"Skin (L cheek) - acne: "
        f"{p3.skin_left_cheek.acne_score:.1f}, "
        f"redness: {p3.skin_left_cheek.redness_score:.1f}"
    )

    print(
        f"Skin (nose) - acne: "
        f"{p3.skin_nose.acne_score:.1f}, "
        f"texture: {p3.skin_nose.texture_irregularity:.1f}"
    )

    print(
        f"Lips - dryness: {p3.lips.dryness_score:.1f}, "
        f"pallor: {p3.lips.pallor_score:.1f}"
    )

    print(
        f"Smile - MAR: {p3.smile.mouth_aspect_ratio:.1f}, "
        f"corner asymmetry: {p3.smile.corner_asymmetry:.1f}, "
        f"curvature: {p3.smile.curvature_score:+.1f}"
    )

    print_section("Phase 4 — Observation Engine (winning frame)")
    print(f"Overall: {p4.overall_label}")
    # quick_score (this capture's sharpness/alignment/coverage score from
    # burst selection) is shown to the user as "Photo quality" -- it's the
    # number that actually varies capture to capture. overall_confidence is
    # kept separate as "Analysis confidence": it's a capture/data-quality
    # metric (see observation_engine.py's Phase4Report docs), NOT a
    # judgment of the findings, and reads ~100% on nearly every real
    # capture since burst selection already discarded any poorly-covered
    # frame before this point -- it isn't meant to answer "how much should
    # I trust this report," which observation_score (blended with the
    # findings) is closer to.
    print(f"Photo quality: {best['quick_score']:.0f}")
    print(f"Analysis confidence: {p4.overall_confidence:.0f}%")

    for line in p4.overall:
        print(f"  - {line}")

    print_section("Burst summary")
    print(f"Raw frames scanned : {best['raw_frame_count']}")
    print(f"Candidates passed  : {best['candidate_count']}")
    print(f"Pool scored (top)  : {best['pool_size']}")
    print(f"Winning quick score: {best['quick_score']:.0f}")

    # ── Phase 5 ───────────────────────────────────────────────────────────
    print_section("Phase 5 — Report Generation")

    quality_warnings = []

    # Metrics JSON
    metrics_reporter = MetricsReportGenerator()

    metrics_result = metrics_reporter.generate(
        p1,
        p3,
        p4,
        quality_warnings=quality_warnings,
        patient_id=patient_id,
        output_dir=None,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        json.dump(
            metrics_result["data"],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Metrics JSON saved : {metrics_path}")

    # Doctor PDF
    doctor_reporter = DoctorReportGenerator()

    doctor_bytes = doctor_reporter.generate(
        p1,
        p2,
        p3,
        p4,
        annotated_image_bgr=p1_viz,
        patient_id=patient_id,
        output_path=doctor_pdf_path,
        quality_warnings=quality_warnings,
    )

    print(
        f"Doctor PDF saved   : "
        f"{doctor_pdf_path} ({len(doctor_bytes) // 1024} KB)"
    )

    # User PDF
    user_reporter = UserReportGenerator()

    user_bytes = user_reporter.generate(
        p1,
        p2,
        p3,
        p4,
        annotated_image_bgr=p1_viz,
        patient_id=patient_id,
        output_path=user_pdf_path,
        quality_warnings=quality_warnings,
    )

    print(
        f"User PDF saved     : "
        f"{user_pdf_path} ({len(user_bytes) // 1024} KB)"
    )

    # TTS text
    tts_reporter = TTSReportGenerator()

    tts_result = tts_reporter.generate(
        p1,
        p3,
        p4,
        quality_warnings=quality_warnings,
        patient_id=patient_id,
        output_dir=None,
        language=language,
    )

    with open(tts_path, "w", encoding="utf-8") as f:
        f.write(tts_result["text"])

    print(f"TTS text saved     : {tts_path}")

    # ── Phase 6 ───────────────────────────────────────────────────────────
    print_section("Phase 6 — Feature Vector Export")

    row = flatten_phase3(
        p1,
        p3,
        p4,
        patient_id=patient_id,
        timestamp=datetime.datetime.now().isoformat(),
        photo_path=csv_photo_path,
    )

    # append_feature_row creates the CSV/header when needed and appends
    # exactly one row for this capture.
    append_feature_row(row, dataset_csv)

    filled = sum(
        1
        for key, value in row.items()
        if value is not None
        and key not in ("patient_id", "timestamp")
    )

    total = len(row) - 2

    print(f"Feature vector : {filled}/{total} fields populated")
    print(f"Dataset CSV    : {dataset_csv}")
    print("CSV ACTION     : Appended 1 new row.")

    # ── Final ─────────────────────────────────────────────────────────────
    print_section("COMPLETE")

    print(f"PHOTO   : {photo_path}")
    print(f"DOCTOR  : {doctor_pdf_path}")
    print(f"USER    : {user_pdf_path}")
    print(f"METRICS : {metrics_path}")
    print(f"TTS     : {tts_path}")
    print(f"CSV     : {dataset_csv}")
    print(f"CAPTURE : {base_name}")

    print("PIPELINE_STATUS: SUCCESS")

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the complete MIRA capture and analysis pipeline."
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Participant name.",
    )

    parser.add_argument(
        "--id",
        dest="patient_id",
        required=True,
        help="Participant ID/code.",
    )

    parser.add_argument(
        "--lang",
        dest="language",
        choices=["en", "hi"],
        default="en",
        help="Language for the spoken TTS report text (default: en).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        return run_pipeline(
            name=args.name,
            patient_id=args.patient_id,
            language=args.language,
        )

    except KeyboardInterrupt:
        print("\nPIPELINE_STATUS: CANCELLED")
        return 1

    except Exception as exc:
        print("\nPIPELINE_STATUS: FAILED")
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())