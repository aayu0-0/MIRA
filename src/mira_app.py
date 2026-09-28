"""
MIRA App — UI/UX, plus patient login/signup and face verification.

The GUI is responsible for:
    1. Login / Sign Up
    2. Showing the preparation guidelines
    3. Starting pipeline.py when the user clicks "Take Photo" -- the same
       burst-capture flow is reused for both a normal scan AND the photo
       that becomes (sign up) or verifies against (log in) a patient's
       face encoding
    4. Showing pipeline progress/results/errors, and -- after login -- a
       face-match check before anything analysis-related is shown

All camera handling, face *analysis*, saving photos, CSV/database work,
report generation, TTS, etc. belongs to pipeline.py. Identity (patients,
passwords, face encodings) belongs to patient_auth.py. This module wires
the two together: it never touches a camera frame or a face encoding
directly.

Expected pipeline.py invocation:
    python pipeline.py --name "<NAME>" --id "<ID>"
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import patient_auth
import security


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_W = 1000
WINDOW_H = 700

BG = "#1a1a2e"
CARD = "#16213e"
PRIMARY = "#e94560"
PRIMARY_HOVER = "#c73652"
SECONDARY = "#0f3460"
TEXT = "#e0e0e0"
MUTED = "#a8a8b3"
SUCCESS = "#27ae60"
WARNING = "#e67e22"
EYELINE = "#ffcc00"  # bright amber — high contrast for the eye-level guide

SEX_OPTIONS = ["Male", "Female", "Other"]
BLOOD_GROUPS = ["Unknown", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Database/ lives one level up from src/ (MIRA_doc+clean/Database), not
# inside src/, so that src stays free of generated/data folders. patient_auth
# computes the same path independently for mira_patients.db, and pipeline.py
# for its own per-patient folders -- all three need to agree on this location.
DATABASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Database",
)

# History -- Phase 1's SQLite HistoryStore (phase_a_history/phase_a_trends)
# no longer exists. History is read straight from each patient's
# database/<folder>/dataset.csv instead, so the window below just needs
# DATABASE_DIR (already defined above) -- no separate history db path.


# ─────────────────────────────────────────────────────────────────────────────
# Turning pipeline.py's raw stdout into something a patient should see
#
# pipeline.py's log is written for a developer debugging it: TF Lite/
# MediaPipe init warnings, per-frame bounding boxes, and absolute file
# paths on every "saved" line. Two places show that log to the actual
# user -- the live status line while a scan runs, and the results screen
# afterwards -- and both were dumping it in raw. The helpers below turn it
# into a short, plain-language version in both spots; the unfiltered log
# is still kept (as self._last_pipeline_output) behind "Show technical
# details" on the results screen for troubleshooting.
# ─────────────────────────────────────────────────────────────────────────────

# (line-start prefix, friendly message). First match wins. A friendly
# value of None means "show this line as-is" (already short, no paths --
# just cancellation/error messages). Anything that matches nothing here
# (warnings, per-frame metrics, coordinates, saved-file paths, ...) is
# dropped from the live status entirely.
_STATUS_LINE_MAP = [
    ("WEBCAM: Camera opened.", "Camera ready."),
    ("WEBCAM: Press SPACE", "Get positioned, then press SPACE to start."),
    ("WEBCAM: Burst capture started.", "Capturing \u2014 hold still\u2026"),
    ("BURST:", "Photos captured \u2014 picking the best one\u2026"),
    ("AUTO-SELECTED:", "Best photo selected."),
    ("PHOTO SAVED:", "Photo saved."),
    ("Doctor PDF saved", "Doctor report generated."),
    ("User PDF saved", "Your report generated."),
    ("TTS text saved", "Preparing audio narration\u2026"),
    ("CSV ACTION", "Session recorded."),
    ("CANCELLED:", None),
    ("PIPELINE_STATUS: FAILED", None),
    ("ERROR:", None),
]


def _friendly_status_line(line):
    """A short, plain-language version of one pipeline.py stdout line for
    the live status label, or None to drop the line (most of them --
    they're for troubleshooting, not for the person watching the scan)."""
    stripped = line.strip()
    for prefix, friendly in _STATUS_LINE_MAP:
        if stripped.startswith(prefix):
            return friendly if friendly is not None else stripped
    return None


def _paren_suffix(line):
    """Pulls the trailing '(515 KB)' off a "... saved : <path> (515 KB)"
    line, without the path in front of it."""
    idx = line.rfind("(")
    return f" {line[idx:]}" if idx != -1 else ""


def _summarize_pipeline_output(output):
    """Turns pipeline.py's full raw log into a short summary for the
    results screen: patient/capture, symmetry, overall observation, and a
    plain confirmation of what got saved -- deliberately no file paths."""
    patient = patient_id = capture = symmetry = overall = disclaimer = ""
    photo_quality = analysis_confidence = ""
    notable = mild = None
    saved = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith("PATIENT:"):
            patient = line.split(":", 1)[1].strip()
        elif line.startswith("ID:"):
            patient_id = line.split(":", 1)[1].strip()
        elif line.startswith("CAPTURE:") and not capture:
            capture = line.split(":", 1)[1].strip()
        elif line.startswith("Symmetry"):
            symmetry = line.split(":", 1)[1].strip()
        elif line.startswith("Overall:"):
            overall = line.split(":", 1)[1].strip()
        elif line.startswith("Photo quality:"):
            photo_quality = line.split(":", 1)[1].strip()
        elif line.startswith("Analysis confidence:"):
            analysis_confidence = line.split(":", 1)[1].strip()
        elif "notable observation" in line:
            notable = "".join(ch for ch in line.split("notable")[0] if ch.isdigit()) or notable
        elif "mild observation" in line:
            mild = "".join(ch for ch in line.split("mild")[0] if ch.isdigit()) or mild
        elif "does not constitute a medical diagnosis" in line:
            disclaimer = line.lstrip("- ").strip()
        elif line.startswith("PHOTO SAVED"):
            saved.append("Photo")
        elif line.startswith("Doctor PDF saved"):
            saved.append("Doctor report" + _paren_suffix(line))
        elif line.startswith("User PDF saved"):
            saved.append("Your report" + _paren_suffix(line))
        elif line.startswith("TTS text saved"):
            saved.append("Audio narration")

    header = f"Capture {capture}" if capture else "Analysis"
    if patient:
        header += f" \u2014 {patient}"
        if patient_id:
            header += f" (ID {patient_id})"

    parts = [f"\u2713 {header} complete"]
    if symmetry:
        parts.append(f"Face symmetry: {symmetry}")
    if overall:
        parts.append(f"Overall: {overall}")
    if photo_quality or analysis_confidence:
        bits = []
        if photo_quality:
            bits.append(f"Photo quality: {photo_quality}")
        if analysis_confidence:
            bits.append(f"Analysis confidence: {analysis_confidence}")
        parts.append("  ·  ".join(bits))
    if notable or mild:
        counts = [f"{n} {label}" for n, label in ((notable, "notable"), (mild, "mild")) if n]
        parts.append("Observations: " + ", ".join(counts))
    if saved:
        parts.append("Saved: " + ", ".join(saved) + ".")
    if disclaimer:
        parts.append(disclaimer)

    if len(parts) <= 1:
        return output if output else "Pipeline completed successfully."
    return "\n\n".join(parts)


GUIDELINES = [
    (
        "1. Sit in a well-lit room with light facing your face.",
        "1. अच्छी रोशनी वाले कमरे में बैठें, रोशनी आपके चेहरे की ओर हो।",
    ),
    (
        "2. Remove glasses, hats, or anything covering your face.",
        "2. चश्मा, टोपी या चेहरे को ढकने वाली कोई भी चीज़ हटा दें।",
    ),
    (
        "3. Keep your face centred and look straight at the camera.",
        "3. अपने चेहरे को बीच में रखें और कैमरे की ओर सीधा देखें।",
    ),
    (
        "4. Hold still for 3–5 seconds while the scan runs.",
        "4. स्कैन चलने के दौरान 3–5 सेकंड तक स्थिर रहें।",
    ),
    (
        "5. Keep a neutral expression — no smiling or frowning.",
        "5. चेहरे के भाव सामान्य रखें — मुस्कुराएं या भौंहें न सिकोड़ें।",
    ),
    (
        "6. Make sure the camera lens is clean.",
        "6. सुनिश्चित करें कि कैमरा लेंस साफ़ है।",
    ),
    (
        "7. Stay roughly arm's length from the camera.",
        "7. कैमरे से लगभग एक बांह की दूरी पर रहें।",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

class MIRAApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("MIRA — Facial Analysis")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(850, 600)
        self.configure(bg=BG)

        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        patient_auth.init_db()

        # UI state only -- the patient row itself lives in patient_auth's
        # SQLite table.
        self.name = ""
        self.patient_id = ""
        self.language = "en"  # report/TTS output language: "en" or "hi"
        self.pipeline_process = None
        self._last_pipeline_output = ""

        # What the upcoming capture is *for*: "signup" (turn the photo into
        # a new fingerprint) or "login" (verify against a stored one). Set
        # right before show_guidelines(), read in pipeline_finished().
        self._pending_action = None
        self._pending_signup_values = None   # form values, held until capture succeeds
        self._login_patient = None           # sqlite3.Row, set by do_login()

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.show_landing()

    # ─────────────────────────────────────────────────────────────────────
    # Generic UI helpers
    # ─────────────────────────────────────────────────────────────────────

    def clear(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def make_button(
        self,
        parent,
        text,
        command,
        bg=PRIMARY,
        active_bg=PRIMARY_HOVER,
        width=None,
    ):
        kwargs = {
            "text": text,
            "command": command,
            "font": ("Helvetica", 13, "bold"),
            "bg": bg,
            "fg": "white",
            "activebackground": active_bg,
            "activeforeground": "white",
            "relief": "flat",
            "bd": 0,
            "padx": 24,
            "pady": 11,
            "cursor": "hand2",
        }

        if width is not None:
            kwargs["width"] = width

        return tk.Button(parent, **kwargs)

    def title_block(self, parent, title, subtitle=None):
        tk.Label(
            parent,
            text=title,
            font=("Helvetica", 30, "bold"),
            bg=BG,
            fg=PRIMARY,
        ).pack(pady=(35, 5))

        if subtitle:
            tk.Label(
                parent,
                text=subtitle,
                font=("Helvetica", 12),
                bg=BG,
                fg=MUTED,
            ).pack(pady=(0, 25))

    # ─────────────────────────────────────────────────────────────────────
    # Face positioning guide (static illustration)
    #
    # This is a purely decorative Canvas drawing — an oval face outline with
    # eye-line and centre-crosshair markers — used to show the user how to
    # frame their face before/while the pipeline captures the real photo.
    # It does NOT read from a camera; it is drawn once with static
    # coordinates. A live version (drawn over an actual camera feed) belongs
    # in pipeline.py, since that module owns all camera access.
    # ─────────────────────────────────────────────────────────────────────

    def make_face_guide_canvas(self, parent, width=260, height=300):
        canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg="#0f1b35",
            highlightthickness=0,
        )
        self._draw_face_guide(canvas, width, height)
        return canvas

    @staticmethod
    def _draw_face_guide(canvas, width, height):
        cx, cy = width / 2, height / 2

        # Outer frame corner brackets (camera-viewfinder style).
        bracket = 22
        margin = 10
        pts = [
            (margin, margin, margin + bracket, margin),
            (margin, margin, margin, margin + bracket),
            (width - margin, margin, width - margin - bracket, margin),
            (width - margin, margin, width - margin, margin + bracket),
            (margin, height - margin, margin + bracket, height - margin),
            (margin, height - margin, margin, height - margin - bracket),
            (width - margin, height - margin, width - margin - bracket, height - margin),
            (width - margin, height - margin, width - margin, height - margin - bracket),
        ]
        for x1, y1, x2, y2 in pts:
            canvas.create_line(x1, y1, x2, y2, fill=MUTED, width=2)

        # Face oval guide.
        oval_w, oval_h = width * 0.52, height * 0.66
        canvas.create_oval(
            cx - oval_w / 2,
            cy - oval_h / 2,
            cx + oval_w / 2,
            cy + oval_h / 2,
            outline=PRIMARY,
            width=3,
            dash=(6, 4),
        )

        # Eye-line: solid (not dashed) and brighter/thicker than the rest of
        # the guide so it's easy to see at a glance, with small markers
        # showing exactly where each eye should land.
        eye_y = cy - oval_h * 0.10
        eye_half_span = oval_w * 0.32

        canvas.create_line(
            cx - eye_half_span, eye_y,
            cx + eye_half_span, eye_y,
            fill=EYELINE,
            width=2,
        )

        marker_r = 3
        for eye_x in (cx - eye_half_span, cx + eye_half_span):
            canvas.create_oval(
                eye_x - marker_r, eye_y - marker_r,
                eye_x + marker_r, eye_y + marker_r,
                fill=EYELINE,
                outline="",
            )

        canvas.create_text(
            cx + eye_half_span + 16,
            eye_y,
            text="Eyes here",
            fill=EYELINE,
            font=("Helvetica", 8),
            anchor="w",
        )

        # Centre crosshair.
        cross = 8
        canvas.create_line(cx - cross, cy, cx + cross, cy, fill=MUTED, width=1)
        canvas.create_line(cx, cy - cross, cx, cy + cross, fill=MUTED, width=1)

        canvas.create_text(
            cx,
            height - 18,
            text="Align your face inside the outline",
            fill=MUTED,
            font=("Helvetica", 9),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Screen 1 — Landing (Log In / Sign Up)
    # ─────────────────────────────────────────────────────────────────────

    def show_landing(self):
        self.clear()
        self.unbind("<Return>")

        # Reset any in-progress auth state -- landing here always means
        # starting fresh, whether from app launch, "Log Out", or "Start Over".
        self.name = ""
        self.patient_id = ""
        self._pending_action = None
        self._pending_signup_values = None
        self._login_patient = None

        wrapper = tk.Frame(self.container, bg=BG)
        wrapper.pack(expand=True)

        self.title_block(
            wrapper,
            "MIRA",
            "Multi-dimensional Intelligent Response Analysis",
        )

        card = tk.Frame(wrapper, bg=CARD, padx=40, pady=32)
        card.pack(pady=10)

        tk.Label(
            card,
            text="Welcome",
            font=("Helvetica", 17, "bold"),
            bg=CARD,
            fg=TEXT,
        ).pack(pady=(0, 18))

        self.make_button(
            card,
            "Log In",
            self.show_login_form,
            width=22,
        ).pack(pady=(0, 10))

        self.make_button(
            card,
            "Sign Up",
            self.show_signup_form,
            bg=SECONDARY,
            active_bg=CARD,
            width=22,
        ).pack()

    # ─────────────────────────────────────────────────────────────────────
    # Screen 1a — Log In (name + password, then a capture verifies the face)
    # ─────────────────────────────────────────────────────────────────────

    def show_login_form(self):
        self.clear()

        wrapper = tk.Frame(self.container, bg=BG)
        wrapper.pack(expand=True)

        self.title_block(wrapper, "MIRA", "Log In")

        card = tk.Frame(wrapper, bg=CARD, padx=40, pady=32)
        card.pack()

        self._login_name_entry = self._labeled_entry(card, "Full Name")
        self._login_name_entry.focus_set()
        self._login_password_entry = self._labeled_entry(card, "Password", show="•")

        self._login_error = tk.Label(
            card, text="", font=("Helvetica", 10), bg=CARD, fg=PRIMARY, wraplength=320,
        )
        self._login_error.pack(pady=(4, 10))

        self._login_password_entry.bind("<Return>", lambda e: self.do_login())
        self.make_button(card, "Log In  →", self.do_login).pack(fill="x")

        self.make_button(
            wrapper, "← Back", self.show_landing, bg=CARD, active_bg=SECONDARY,
        ).pack(pady=(16, 0))

    def do_login(self):
        name = self._login_name_entry.get().strip()
        password = self._login_password_entry.get()

        if not name or not password:
            self._login_error.config(text="Enter both your name and password.")
            return

        patient = patient_auth.check_login(name, password)
        if not patient:
            self._login_error.config(text="Invalid name or password.")
            return

        self._login_patient = patient
        self._pending_action = "login"
        self.name = patient["name"]
        self.patient_id = patient["patient_id"]

        self.unbind("<Return>")
        self.show_guidelines()

    # ─────────────────────────────────────────────────────────────────────
    # Screen 1b — Sign Up (full intake form; the photo comes later, via the
    # same capture flow as a normal scan)
    # ─────────────────────────────────────────────────────────────────────

    def _labeled_entry(self, parent, label, show=None, width=34):
        wrap = tk.Frame(parent, bg=CARD)
        wrap.pack(fill="x", pady=(0, 14))
        tk.Label(wrap, text=label, font=("Helvetica", 10, "bold"), bg=CARD, fg=TEXT, anchor="w").pack(fill="x")
        entry = tk.Entry(
            wrap, font=("Helvetica", 12), width=width, show=show,
            bg="#0f1b35", fg=TEXT, insertbackground=TEXT, relief="flat",
        )
        entry.pack(fill="x", ipady=6, pady=(4, 0))
        return entry

    def _labeled_dropdown(self, parent, label, options, default=None):
        wrap = tk.Frame(parent, bg=CARD)
        wrap.pack(fill="x", pady=(0, 14))
        tk.Label(wrap, text=label, font=("Helvetica", 10, "bold"), bg=CARD, fg=TEXT, anchor="w").pack(fill="x")
        var = tk.StringVar(value=default or options[0])
        dropdown = ttk.Combobox(wrap, textvariable=var, values=options, state="readonly", font=("Helvetica", 12))
        dropdown.pack(fill="x", pady=(4, 0), ipady=3)
        return var

    def show_signup_form(self):
        self.clear()
        self._signup_entries = {}

        header = tk.Frame(self.container, bg=BG)
        header.pack(fill="x", padx=50, pady=(30, 10))
        tk.Label(header, text="Sign Up", font=("Helvetica", 22, "bold"), bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(
            header,
            text="Fill in your details and set a password. On the next screen you'll "
                 "take a photo the same way MIRA always does -- that photo becomes "
                 "your fingerprint for logging in next time.",
            font=("Helvetica", 11), bg=BG, fg=MUTED, wraplength=880, justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # Footer packed before the scrollable body so the submit button
        # always has its space reserved, same reasoning as every other
        # long-form screen in this app.
        footer = tk.Frame(self.container, bg=BG)
        footer.pack(side="bottom", fill="x", padx=50, pady=(6, 24))
        self.make_button(footer, "Continue to Photo Capture  →", self.do_signup_continue).pack(side="left")
        self.make_button(
            footer, "Already have an account? Log In", self.show_login_form,
            bg=CARD, active_bg=SECONDARY,
        ).pack(side="left", padx=10)
        self.make_button(
            footer, "← Back", self.show_landing, bg=CARD, active_bg=SECONDARY,
        ).pack(side="left")

        body_wrap = tk.Frame(self.container, bg=BG)
        body_wrap.pack(fill="both", expand=True, padx=50, pady=(0, 10))

        canvas = tk.Canvas(body_wrap, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(body_wrap, orient="vertical", command=canvas.yview)
        card = tk.Frame(canvas, bg=CARD, padx=32, pady=28)

        card.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=card, anchor="nw", width=WINDOW_W - 140)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _mousewheel)

        e = self._signup_entries

        tk.Label(card, text="Identity", font=("Helvetica", 13, "bold"), bg=CARD, fg=PRIMARY, anchor="w").pack(fill="x", pady=(0, 10))
        e["name"] = self._labeled_entry(card, "Full name")
        row1 = tk.Frame(card, bg=CARD); row1.pack(fill="x")
        left1 = tk.Frame(row1, bg=CARD); left1.pack(side="left", fill="x", expand=True, padx=(0, 10))
        right1 = tk.Frame(row1, bg=CARD); right1.pack(side="left", fill="x", expand=True)
        e["age"] = self._labeled_entry(left1, "Age", width=16)
        e["sex"] = self._labeled_dropdown(right1, "Sex", SEX_OPTIONS)
        e["guardian_name"] = self._labeled_entry(card, "Spouse / Father's name")
        e["address"] = self._labeled_entry(card, "Address")
        e["contact_number"] = self._labeled_entry(card, "Contact number")

        tk.Label(card, text="Medical details", font=("Helvetica", 13, "bold"), bg=CARD, fg=PRIMARY, anchor="w").pack(fill="x", pady=(14, 10))
        row2 = tk.Frame(card, bg=CARD); row2.pack(fill="x")
        left2 = tk.Frame(row2, bg=CARD); left2.pack(side="left", fill="x", expand=True, padx=(0, 10))
        right2 = tk.Frame(row2, bg=CARD); right2.pack(side="left", fill="x", expand=True)
        e["blood_group"] = self._labeled_dropdown(left2, "Blood group", BLOOD_GROUPS)
        e["registration_fee"] = self._labeled_entry(right2, "Registration fee", width=16)
        e["known_allergies"] = self._labeled_entry(card, "Known allergies (e.g. Penicillin, none known)")
        e["existing_conditions"] = self._labeled_entry(card, "Existing conditions (e.g. Diabetes, hypertension)")

        tk.Label(card, text="Emergency contact", font=("Helvetica", 13, "bold"), bg=CARD, fg=PRIMARY, anchor="w").pack(fill="x", pady=(14, 10))
        row3 = tk.Frame(card, bg=CARD); row3.pack(fill="x")
        left3 = tk.Frame(row3, bg=CARD); left3.pack(side="left", fill="x", expand=True, padx=(0, 10))
        right3 = tk.Frame(row3, bg=CARD); right3.pack(side="left", fill="x", expand=True)
        e["emergency_contact_name"] = self._labeled_entry(left3, "Name")
        e["emergency_contact_number"] = self._labeled_entry(right3, "Number")

        tk.Label(card, text="Account", font=("Helvetica", 13, "bold"), bg=CARD, fg=PRIMARY, anchor="w").pack(fill="x", pady=(14, 10))
        row4 = tk.Frame(card, bg=CARD); row4.pack(fill="x")
        left4 = tk.Frame(row4, bg=CARD); left4.pack(side="left", fill="x", expand=True, padx=(0, 10))
        right4 = tk.Frame(row4, bg=CARD); right4.pack(side="left", fill="x", expand=True)
        e["password"] = self._labeled_entry(left4, "Password", show="•")
        e["confirm_password"] = self._labeled_entry(right4, "Confirm password", show="•")

        tk.Label(card, text="Report Language", font=("Helvetica", 13, "bold"), bg=CARD, fg=PRIMARY, anchor="w").pack(fill="x", pady=(14, 10))
        lang_frame = tk.Frame(card, bg=CARD); lang_frame.pack(fill="x")
        e["language"] = tk.StringVar(value=self.language)
        tk.Radiobutton(
            lang_frame, text="English", variable=e["language"], value="en",
            font=("Helvetica", 12), bg=CARD, fg=TEXT, selectcolor=SECONDARY,
            activebackground=CARD, activeforeground=TEXT,
        ).pack(side="left", padx=(0, 15))
        tk.Radiobutton(
            lang_frame, text="हिन्दी (Hindi)", variable=e["language"], value="hi",
            font=("Helvetica", 12), bg=CARD, fg=TEXT, selectcolor=SECONDARY,
            activebackground=CARD, activeforeground=TEXT,
        ).pack(side="left")

        self._signup_error = tk.Label(
            card, text="", font=("Helvetica", 10), bg=CARD, fg=PRIMARY, wraplength=780, justify="left",
        )
        self._signup_error.pack(fill="x", pady=(14, 0))

    def _collect_and_validate_signup(self):
        e = self._signup_entries
        values = {
            "name": e["name"].get().strip(),
            "age": e["age"].get().strip(),
            "sex": e["sex"].get(),
            "guardian_name": e["guardian_name"].get().strip(),
            "address": e["address"].get().strip(),
            "contact_number": e["contact_number"].get().strip(),
            "blood_group": e["blood_group"].get(),
            "registration_fee": e["registration_fee"].get().strip() or "0",
            "known_allergies": e["known_allergies"].get().strip(),
            "existing_conditions": e["existing_conditions"].get().strip(),
            "emergency_contact_name": e["emergency_contact_name"].get().strip(),
            "emergency_contact_number": e["emergency_contact_number"].get().strip(),
            "password": e["password"].get(),
            "confirm_password": e["confirm_password"].get(),
            "language": e["language"].get(),
        }

        errors = []
        if not values["name"]:
            errors.append("Name is required.")
        elif patient_auth.name_exists(values["name"]):
            errors.append(f"'{values['name']}' is already registered — log in instead.")
        if not values["age"].isdigit() or not (0 < int(values["age"]) < 130):
            errors.append("Enter a valid age.")
        if values["sex"] not in SEX_OPTIONS:
            errors.append("Select a sex.")
        try:
            values["registration_fee"] = float(values["registration_fee"])
        except ValueError:
            errors.append("Registration fee must be a number.")
            values["registration_fee"] = 0.0
        if len(values["password"]) < 6:
            errors.append("Password must be at least 6 characters.")
        if values["password"] != values["confirm_password"]:
            errors.append("Passwords do not match.")

        return values, errors

    def do_signup_continue(self):
        values, errors = self._collect_and_validate_signup()
        if errors:
            self._signup_error.config(text="  •  ".join(errors))
            return

        self._pending_signup_values = values
        self._pending_action = "signup"
        self.name = values["name"]
        self.patient_id = patient_auth.generate_patient_id()
        self.language = values["language"]

        self.show_guidelines()

    # ─────────────────────────────────────────────────────────────────────
    # Screen 2 — Guidelines
    # ─────────────────────────────────────────────────────────────────────

    def show_guidelines(self):
        self.clear()

        header = tk.Frame(self.container, bg=BG)
        header.pack(fill="x", padx=45, pady=(25, 5))

        tk.Label(
            header,
            text=f"Hello, {self.name}!",
            font=("Helvetica", 24, "bold"),
            bg=BG,
            fg=PRIMARY,
        ).pack()

        tk.Label(
            header,
            text=f"ID: {self.patient_id}",
            font=("Helvetica", 11),
            bg=BG,
            fg=MUTED,
        ).pack(pady=(3, 0))

        tk.Label(
            self.container,
            text="Please read the guidelines before starting your scan.",
            font=("Helvetica", 13),
            bg=BG,
            fg=MUTED,
        ).pack(pady=(12, 15))

        # Body: guidelines list on the left, face positioning guide on the
        # right, side by side.
        body = tk.Frame(self.container, bg=BG)
        body.pack(fill="x", padx=70)

        card = tk.Frame(body, bg=CARD, padx=28, pady=20)
        card.pack(side="left", fill="both", expand=True)

        for english, hindi in GUIDELINES:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=5)

            tk.Label(
                row,
                text=english,
                font=("Helvetica", 11),
                bg=CARD,
                fg=TEXT,
                anchor="w",
                justify="left",
            ).pack(fill="x")

            tk.Label(
                row,
                text=hindi,
                font=("Nirmala UI", 11),
                bg=CARD,
                fg=MUTED,
                anchor="w",
                justify="left",
            ).pack(fill="x")

        guide_card = tk.Frame(body, bg=CARD, padx=16, pady=16)
        guide_card.pack(side="left", padx=(16, 0))

        tk.Label(
            guide_card,
            text="Positioning Guide",
            font=("Helvetica", 12, "bold"),
            bg=CARD,
            fg=TEXT,
        ).pack(pady=(0, 10))

        self.make_face_guide_canvas(guide_card).pack()

        buttons = tk.Frame(self.container, bg=BG)
        buttons.pack(pady=28)

        self.make_button(
            buttons,
            "📷  Take Photo & Start Analysis",
            self.start_pipeline,
            bg=PRIMARY,
            active_bg=PRIMARY_HOVER,
        ).pack(side="left", padx=8)

        self.make_button(
            buttons,
            "← Back",
            self.show_landing,
            bg=CARD,
            active_bg=SECONDARY,
        ).pack(side="left", padx=8)

    # ─────────────────────────────────────────────────────────────────────
    # Screen 3 — Pipeline running
    # ─────────────────────────────────────────────────────────────────────

    def show_running(self):
        self.clear()

        wrapper = tk.Frame(self.container, bg=BG)
        wrapper.pack(expand=True)

        tk.Label(
            wrapper,
            text="MIRA",
            font=("Helvetica", 34, "bold"),
            bg=BG,
            fg=PRIMARY,
        ).pack(pady=(0, 8))

        tk.Label(
            wrapper,
            text="Analysis in progress",
            font=("Helvetica", 21, "bold"),
            bg=BG,
            fg=TEXT,
        ).pack()

        tk.Label(
            wrapper,
            text=f"{self.name}  •  ID: {self.patient_id}",
            font=("Helvetica", 11),
            bg=BG,
            fg=MUTED,
        ).pack(pady=(6, 30))

        card = tk.Frame(wrapper, bg=CARD, padx=30, pady=30)
        card.pack()

        # Face positioning guide replaces the old static camera emoji, so
        # the user has something actionable to look at while pipeline.py
        # (which owns the real camera feed) is running.
        self.make_face_guide_canvas(card, width=220, height=260).pack()

        self.status_var = tk.StringVar(
            value="Starting pipeline.py…"
        )

        tk.Label(
            card,
            textvariable=self.status_var,
            font=("Helvetica", 14),
            bg=CARD,
            fg=TEXT,
            wraplength=650,
            justify="center",
        ).pack(pady=(15, 8))

        tk.Label(
            card,
            text="The camera and analysis are being handled by pipeline.py.",
            font=("Helvetica", 10),
            bg=CARD,
            fg=MUTED,
        ).pack()

        self.spinner_var = tk.StringVar(value="●")
        tk.Label(
            wrapper,
            textvariable=self.spinner_var,
            font=("Helvetica", 20),
            bg=BG,
            fg=PRIMARY,
        ).pack(pady=25)

        self._animate_spinner(0)

    def _animate_spinner(self, index):
        if not self.pipeline_process:
            return

        dots = ["●", "● ●", "● ● ●"]
        self.spinner_var.set(dots[index % len(dots)])
        self.after(350, lambda: self._animate_spinner(index + 1))

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline launcher
    # ─────────────────────────────────────────────────────────────────────

    def start_pipeline(self):
        if self.pipeline_process is not None:
            return

        self.show_running()

        threading.Thread(
            target=self._run_pipeline_process,
            daemon=True,
        ).start()

    def _run_pipeline_process(self):
        """
        UI/process boundary.

        The GUI deliberately does not import any MIRA analysis modules and
        does not manipulate frames, photos, reports, CSVs, databases, etc.

        It simply launches pipeline.py and waits for it to finish.
        """

        pipeline_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pipeline.py",
        )

        if not os.path.isfile(pipeline_path):
            self.after(
                0,
                lambda: self.pipeline_failed(
                    f"pipeline.py was not found:\n{pipeline_path}"
                ),
            )
            return

        command = [
            sys.executable,
            pipeline_path,
            "--name",
            self.name,
            "--id",
            self.patient_id,
            "--lang",
            self.language,
        ]

        try:
            creationflags = 0

            # On Windows, prevent an additional persistent console window.
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                command,
                cwd=os.path.dirname(pipeline_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )

            self.pipeline_process = process

            output_lines = []

            # Read pipeline output while it runs.
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        # Full line kept for the technical log and TTS
                        # prefetch (_start_tts_prefetch parses "TTS  :").
                        output_lines.append(line)

                        # Only a short, plain-language version -- if any --
                        # reaches the live status label. Most raw lines
                        # (warnings, per-frame metrics, saved-file paths)
                        # aren't meant for the person watching the scan.
                        friendly = _friendly_status_line(line)
                        if friendly:
                            self.after(
                                0,
                                lambda msg=friendly: self.status_var.set(msg),
                            )

            return_code = process.wait()

            output = "\n".join(output_lines)

            self.after(
                0,
                lambda: self.pipeline_finished(return_code, output),
            )

        except Exception as exc:
            # `exc` is auto-deleted by Python at the end of this except block,
            # but self.after() runs the lambda later on the Tk main loop --
            # by then `exc` no longer exists, so capturing it directly in the
            # closure raises NameError instead of showing the real error.
            error_message = str(exc)
            self.after(
                0,
                lambda msg=error_message: self.pipeline_failed(msg),
            )

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline result screens
    # ─────────────────────────────────────────────────────────────────────

    def pipeline_finished(self, return_code, output):
        self.pipeline_process = None

        if return_code != 0:
            self.pipeline_failed(
                output or f"pipeline.py exited with code {return_code}"
            )
            return

        photo_path = self._extract_photo_path(output)

        if self._pending_action == "signup":
            self._finish_signup(output, photo_path)
        elif self._pending_action == "login":
            self._finish_login(output, photo_path)
        else:
            # A rescan by an already-verified session ("Run Another Scan") --
            # nothing to enroll or verify, just show the result as usual.
            self._start_tts_prefetch(output)
            self.show_success(output)

    @staticmethod
    def _extract_photo_path(output):
        """Pulls the path off pipeline.py's "PHOTO SAVED: <path>" line --
        the same photo used for analysis becomes the one turned into a
        face encoding, so there's no second capture."""
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if line.startswith("PHOTO SAVED:"):
                return line.split(":", 1)[1].strip()
        return None

    def _finish_signup(self, output, photo_path):
        """Turns the photo pipeline.py just captured into this new
        patient's face encoding ("digital fingerprint") and saves the full
        record -- deferred until now because we needed a real photo to
        encode, not just the form fields."""
        values = self._pending_signup_values

        if not photo_path:
            self.pipeline_failed(
                "The scan finished, but no photo path was reported, so your "
                "fingerprint couldn't be saved. Please try again."
            )
            return

        try:
            encoding = patient_auth.extract_face_encoding(photo_path)
        except patient_auth.FaceMatchError as exc:
            self.pipeline_failed(str(exc))
            return

        patient_auth.insert_patient(
            patient_id=self.patient_id,
            name=values["name"],
            password_hash=security.hash_password(values["password"]),
            age=int(values["age"]),
            sex=values["sex"],
            guardian_name=values["guardian_name"],
            address=values["address"],
            contact_number=values["contact_number"],
            registration_fee=values["registration_fee"],
            fee_paid=values["registration_fee"] == 0,
            blood_group=values["blood_group"] if values["blood_group"] != "Unknown" else "",
            known_allergies=values["known_allergies"],
            existing_conditions=values["existing_conditions"],
            emergency_contact_name=values["emergency_contact_name"],
            emergency_contact_number=values["emergency_contact_number"],
            face_encoding=patient_auth.encoding_to_json(encoding),
            photo_path=photo_path,
        )

        self._pending_signup_values = None
        self._pending_action = None
        self._start_tts_prefetch(output)
        self.show_success(output)

    def _finish_login(self, output, photo_path):
        """Compares the photo just captured against the fingerprint stored
        for the patient found at login. Analysis results are only shown if
        the face matches -- the password only decided *whose* fingerprint
        to check against, not that this is really them."""
        patient = self._login_patient

        if not photo_path:
            self.show_verification_failed(
                "The scan finished, but no photo was captured to verify your "
                "identity. Please try again."
            )
            return

        try:
            encoding = patient_auth.extract_face_encoding(photo_path)
        except patient_auth.FaceMatchError as exc:
            self.show_verification_failed(str(exc))
            return

        if not patient_auth.match_encoding(encoding, patient["face_encoding"]):
            self.show_verification_failed(
                f"This doesn't match the photo on file for {patient['name']}. "
                "If this is you, try the capture again in good lighting; "
                "otherwise, this account isn't yours to log into."
            )
            return

        self._pending_action = None
        self._start_tts_prefetch(output)
        self.show_success(output)

    def show_verification_failed(self, message):
        """Login's face check failed -- shown instead of any analysis
        result, whether or not pipeline.py itself computed one."""
        self.clear()

        wrapper = tk.Frame(self.container, bg=BG)
        wrapper.pack(expand=True, fill="both", padx=70, pady=50)

        tk.Label(
            wrapper,
            text="⚠  Wrong Person",
            font=("Helvetica", 25, "bold"),
            bg=BG,
            fg=PRIMARY,
        ).pack(pady=(20, 10))

        tk.Label(
            wrapper,
            text=message,
            font=("Helvetica", 13),
            bg=BG,
            fg=MUTED,
            wraplength=700,
            justify="left",
        ).pack(pady=(0, 20))

        buttons = tk.Frame(wrapper, bg=BG)
        buttons.pack(pady=20)

        self.make_button(
            buttons,
            "Try Again",
            self.show_guidelines,
            bg=PRIMARY,
            active_bg=PRIMARY_HOVER,
        ).pack(side="left", padx=8)

        self.make_button(
            buttons,
            "Log Out",
            self.show_landing,
            bg=CARD,
            active_bg=SECONDARY,
        ).pack(side="left", padx=8)

    def _start_tts_prefetch(self, output):
        """Kick off Piper synthesis for the report pipeline.py just wrote,
        right away instead of waiting for the user to open History and
        click "Play Audio" -- so the WAV is usually already sitting on disk
        by the time they get there. Runs on a background daemon thread and
        never blocks this screen transition or anything else in the app;
        a failure here (missing piper, bad model, etc.) is silently
        swallowed since "Play Audio" will retry and report it normally.
        """
        tts_path = None
        for line in output.splitlines():
            if line.startswith("TTS     :"):
                tts_path = line.split(":", 1)[1].strip()
                break
        if not tts_path or not os.path.isfile(tts_path):
            return

        def worker():
            try:
                from pathlib import Path
                from history_window import synthesize_only
                synthesize_only(Path(tts_path))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def pipeline_failed(self, error):
        self.pipeline_process = None

        self.clear()

        wrapper = tk.Frame(self.container, bg=BG)
        wrapper.pack(expand=True, fill="both", padx=70, pady=50)

        tk.Label(
            wrapper,
            text="⚠  Analysis Failed",
            font=("Helvetica", 25, "bold"),
            bg=BG,
            fg=PRIMARY,
        ).pack(pady=(20, 10))

        tk.Label(
            wrapper,
            text="pipeline.py reported an error.",
            font=("Helvetica", 13),
            bg=BG,
            fg=MUTED,
        ).pack(pady=(0, 20))

        output_box = tk.Text(
            wrapper,
            height=16,
            bg="#0f1b35",
            fg=TEXT,
            insertbackground="white",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        output_box.pack(fill="both", expand=True)
        output_box.insert("1.0", error)
        output_box.configure(state="disabled")

        buttons = tk.Frame(wrapper, bg=BG)
        buttons.pack(pady=20)

        self.make_button(
            buttons,
            "Try Again",
            self.show_guidelines,
            bg=PRIMARY,
            active_bg=PRIMARY_HOVER,
        ).pack(side="left", padx=8)

        self.make_button(
            buttons,
            "Start Over",
            self.show_landing,
            bg=CARD,
            active_bg=SECONDARY,
        ).pack(side="left", padx=8)

    def show_success(self, output):
        self.clear()

        wrapper = tk.Frame(self.container, bg=BG)
        wrapper.pack(expand=True, fill="both", padx=70, pady=45)

        tk.Label(
            wrapper,
            text="✓  Analysis Complete",
            font=("Helvetica", 26, "bold"),
            bg=BG,
            fg=SUCCESS,
        ).pack(pady=(10, 5))

        tk.Label(
            wrapper,
            text=f"{self.name}  •  ID: {self.patient_id}",
            font=("Helvetica", 11),
            bg=BG,
            fg=MUTED,
        ).pack(pady=(0, 20))

        card = tk.Frame(wrapper, bg=CARD, padx=25, pady=20)
        card.pack(fill="both", expand=True)

        tk.Label(
            card,
            text=_summarize_pipeline_output(output),
            font=("Helvetica", 13),
            bg=CARD,
            fg=TEXT,
            justify="left",
            anchor="w",
            wraplength=760,
        ).pack(anchor="w", fill="x")

        # Full raw pipeline.py log (warnings, per-frame metrics, file
        # paths) is kept for troubleshooting, but collapsed by default --
        # nobody but a developer needs to see it.
        details_state = {"shown": False}

        details_box = tk.Text(
            card,
            bg="#0f1b35",
            fg=TEXT,
            insertbackground="white",
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
            height=14,
        )
        details_box.insert(
            "1.0",
            output if output else "Pipeline completed successfully.",
        )
        details_box.configure(state="disabled")

        def _toggle_details():
            if details_state["shown"]:
                details_box.pack_forget()
                details_toggle.config(text="Show technical details \u25be")
            else:
                details_box.pack(fill="both", expand=True, pady=(14, 0))
                details_toggle.config(text="Hide technical details \u25b4")
            details_state["shown"] = not details_state["shown"]

        details_toggle = tk.Button(
            card,
            text="Show technical details \u25be",
            command=_toggle_details,
            font=("Helvetica", 10),
            bg=CARD,
            fg=MUTED,
            activebackground=SECONDARY,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=8,
            pady=4,
            cursor="hand2",
        )
        details_toggle.pack(anchor="w", pady=(16, 0))

        # kept so "View History" -> "Back" can return here with the same output
        self._last_pipeline_output = output

        buttons = tk.Frame(wrapper, bg=BG)
        buttons.pack(pady=18)

        self.make_button(
            buttons,
            "Run Another Scan",
            self.show_guidelines,
            bg=PRIMARY,
            active_bg=PRIMARY_HOVER,
        ).pack(side="left", padx=8)

        self.make_button(
            buttons,
            "View History",
            self.show_history,
            bg=CARD,
            active_bg=SECONDARY,
        ).pack(side="left", padx=8)

        self.make_button(
            buttons,
            "View Trends",
            self.show_trends,
            bg=CARD,
            active_bg=SECONDARY,
        ).pack(side="left", padx=8)

        self.make_button(
            buttons,
            "Start Over",
            self.show_landing,
            bg=CARD,
            active_bg=SECONDARY,
        ).pack(side="left", padx=8)

    # ─────────────────────────────────────────────────────────────────────
    # History (same window, same tab-like navigation as every other screen)
    # ─────────────────────────────────────────────────────────────────────

    def show_history(self):
        """Replaces the success screen with the History screen, in-place --
        same self.container every other screen uses, so it fills the whole
        (already-maximized) app window instead of opening a separate
        window. Import is local (not at module top) so a missing
        history_window.py / Pillow never breaks the rest of the app --
        only this button stops working, with a clear error instead of a
        crash on launch."""
        try:
            from history_window import build_history_screen
        except ImportError as exc:
            messagebox.showerror(
                "History unavailable",
                f"Couldn't load the history module:\n{exc}",
            )
            return

        self.clear()
        try:
            build_history_screen(
                self.container,
                patient_id=self.patient_id,
                database_root=DATABASE_DIR,
                patient_label=f"{self.name} · ID: {self.patient_id}",
                on_back=lambda: self.show_success(self._last_pipeline_output),
                on_view_trends=self.show_trends,
            )
        except Exception as exc:
            messagebox.showerror(
                "Couldn't open history",
                f"Something went wrong opening history:\n{exc}",
            )
            self.show_success(self._last_pipeline_output)

    def show_trends(self):
        """Same idea as show_history, but for the Trends screen."""
        try:
            from trend_window import build_trend_screen
        except ImportError as exc:
            messagebox.showerror(
                "Trends unavailable",
                f"Couldn't load the trends module:\n{exc}",
            )
            return

        self.clear()
        try:
            build_trend_screen(
                self.container,
                patient_id=self.patient_id,
                database_root=DATABASE_DIR,
                patient_label=f"{self.name} · ID: {self.patient_id}",
                on_back=lambda: self.show_success(self._last_pipeline_output),
                on_view_history=self.show_history,
            )
        except Exception as exc:
            messagebox.showerror(
                "Couldn't open trends",
                f"Something went wrong opening trends:\n{exc}",
            )
            self.show_success(self._last_pipeline_output)

    # ─────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────

    def _on_close(self):
        if self.pipeline_process is not None:
            try:
                self.pipeline_process.terminate()
            except Exception:
                pass

        # Kill any in-flight Piper synthesis or audio playback (subprocess.run
        # doesn't tie a child's lifetime to ours, so without this an audio
        # clip keeps playing after the window is gone).
        try:
            from piper_tts import kill_all_active
            kill_all_active()
        except Exception:
            pass

        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MIRAApp()
    app.mainloop()