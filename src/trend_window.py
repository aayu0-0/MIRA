"""
trend_window.py
----------------
MIRA -- Trends screen (CSV-based).

Companion to history_window.py, built the same way: no phase_a_trends /
TrendDetector (that package no longer exists), everything comes straight
from load_patient_sessions() in history_window.py, which already reads
each patient's database/<folder>/dataset.csv. This file adds nothing new
on disk -- it just aggregates the same SessionEntry list over time and
plots one metric across sessions.

Unlike the old TrendDetector, this has no built-in medical judgment about
which direction is "good" or "bad" per metric (that mapping lived in
phase_a_trends and isn't available anymore) -- it reports the plain
direction of change (up / down / roughly stable) and leaves interpretation
to the person reading it.

Embeds into mira_app.py's self.container exactly like build_history_screen,
so "Trends" behaves as another tab in the same window rather than a
separate popup.
"""

from __future__ import annotations

import datetime
import math
import tkinter as tk
from tkinter import ttk
from typing import Optional

from history_window import (
    SessionEntry,
    load_patient_sessions,
    BG, CARD, CARD_ALT, PRIMARY, PRIMARY_HOVER, SECONDARY, TEXT, MUTED,
    SUCCESS, WARNING,
)

WINDOW_CHOICES = (7, 30, 60, 90, None)   # None = all history
WINDOW_LABELS = {7: "7 days", 30: "30 days", 60: "60 days", 90: "90 days", None: "All time"}
DEFAULT_WINDOW_DAYS = 30

# X-axis mode: "session" spaces points evenly by session order (so a long
# gap between two sessions -- e.g. 2 months with nothing logged -- doesn't
# stretch the chart and squash every other point into a corner). "date"
# spaces points by real elapsed time, same as the original behavior.
X_MODE_CHOICES = ("session", "date")
X_MODE_LABELS = {"session": "By Session", "date": "By Date"}
DEFAULT_X_MODE = "session"
MAX_X_TICK_LABELS = 15   # thin labels past this many sessions so they don't overlap

# "Show last" picker labels: in "date" mode this filters by real calendar
# days (original behavior); in "session" mode the same numbers mean "last
# N sessions" instead, since several sessions can land on the same day
# (or none for weeks) and a calendar-day cutoff doesn't map to anything
# meaningful once the x-axis itself is session-ordered.
WINDOW_LABELS_BY_MODE = {
    "date": WINDOW_LABELS,
    "session": {7: "7 sessions", 30: "30 sessions", 60: "60 sessions",
                90: "90 sessions", None: "All time"},
}


# ---------------------------------------------------------------------------
# Pure data functions (no tkinter -- easy to unit test)
# ---------------------------------------------------------------------------

def collect_metric_keys(entries: list[SessionEntry]) -> list[str]:
    """All metric column names seen across any session, score-like ones
    first (same ordering convention as history_window.highlight_points)."""
    keys = {k for e in entries for k in e.metrics}
    return sorted(keys, key=lambda k: (0 if "score" in k.lower() else 1, k.lower()))


def pretty_metric_label(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def filter_window(entries: list[SessionEntry], window_days: Optional[int]) -> list[SessionEntry]:
    """Entries within the last `window_days` days of the most recent dated
    entry. window_days=None returns everything unfiltered."""
    dated = [e for e in entries if e.date]
    if window_days is None or not dated:
        return entries
    last_date = max(e.date for e in dated).date()
    cutoff = last_date - datetime.timedelta(days=window_days - 1)
    return [e for e in entries if e.date and e.date.date() >= cutoff]


def filter_last_n_sessions(entries: list[SessionEntry], n: Optional[int]) -> list[SessionEntry]:
    """The most recent `n` dated sessions (n=None returns everything
    dated). Used instead of filter_window() in session x-axis mode, where
    the cutoff should be a session count, not a calendar-day range."""
    dated = sorted((e for e in entries if e.date), key=lambda e: e.date)
    if n is None:
        return dated
    return dated[-n:]


def parse_ddmmyyyy(text: str) -> Optional[datetime.date]:
    """Parses a 'DD/MM/YYYY' string the user typed. Returns None on any
    malformed input rather than guessing -- the caller decides whether
    that's "field left blank" or "bad input" to report."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return datetime.datetime.strptime(text, "%d/%m/%Y").date()
    except ValueError:
        return None


def filter_date_range(entries: list[SessionEntry], start: Optional[datetime.date],
                       end: Optional[datetime.date]) -> list[SessionEntry]:
    """Sessions whose date falls within [start, end] (either side open if
    None). A session outside the range is just left out -- nothing is
    fabricated or clamped into range to fill empty space."""
    return [e for e in entries if e.date and
            (start is None or e.date.date() >= start) and
            (end is None or e.date.date() <= end)]


def filter_session_range(entries: list[SessionEntry], start: Optional[int],
                          end: Optional[int]) -> list[SessionEntry]:
    """1-indexed inclusive slice over dated sessions in chronological
    (oldest-first) order -- session 1 is the earliest session, matching
    the left-to-right order used on the session x-axis. A range that
    reaches past the real session count is simply clipped to what exists;
    no placeholder points are added for numbers with no session."""
    dated = sorted((e for e in entries if e.date), key=lambda e: e.date)
    n = len(dated)
    if n == 0:
        return []
    lo = 1 if start is None else max(1, start)
    hi = n if end is None else min(n, end)
    if lo > hi:
        return []
    return dated[lo - 1:hi]


def build_metric_series(entries: list[SessionEntry], metric_key: str) -> list[tuple[datetime.datetime, float]]:
    """(date, value) pairs for one metric, oldest first -- what the chart plots."""
    series = [(e.date, e.metrics[metric_key]) for e in entries
              if e.date is not None and metric_key in e.metrics]
    series.sort(key=lambda t: t[0])
    return series


def _direction(delta: float, pct: Optional[float]) -> str:
    if pct is not None:
        return "roughly stable" if abs(pct) < 2 else ("trending up" if delta > 0 else "trending down")
    return "roughly stable" if abs(delta) < 0.01 else ("trending up" if delta > 0 else "trending down")


def format_session_tick(dt: datetime.datetime) -> str:
    """Two-line tick label for session-mode x-axis: time, then
    DD/MM/YYYY, e.g. '14:32\n26/09/2026'."""
    return dt.strftime("%H:%M\n%d/%m/%Y")


def thin_tick_indices(n: int, max_labels: int = MAX_X_TICK_LABELS) -> list[int]:
    """Evenly spaced indices into range(n) to label, always including the
    first and last session, capped at max_labels so labels don't overlap
    when there are many sessions."""
    if n <= 0:
        return []
    if n <= max_labels:
        return list(range(n))
    step = math.ceil((n - 1) / (max_labels - 1))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def compute_trend_summary(metric_label: str, series: list[tuple[datetime.datetime, float]]) -> str:
    """Plain-language, judgment-free description of how a metric moved
    across the sessions in `series` (oldest -> newest)."""
    if not series:
        return f"No dated values for \u201c{metric_label}\u201d in this window."
    if len(series) == 1:
        d, v = series[0]
        return f"Only one session with \u201c{metric_label}\u201d in this window ({d.strftime('%b %d, %Y')}): {v:g}."

    first_d, first_v = series[0]
    last_d, last_v = series[-1]
    delta = last_v - first_v
    pct = (delta / first_v * 100) if first_v else None
    pct_str = f", {pct:+.1f}%" if pct is not None else ""

    return (
        f"{metric_label}: {first_v:g} \u2192 {last_v:g}  ({delta:+.2f}{pct_str})\n"
        f"Across {len(series)} sessions, {first_d.strftime('%b %d')} \u2013 "
        f"{last_d.strftime('%b %d, %Y')} \u2014 {_direction(delta, pct)}."
    )


# ---------------------------------------------------------------------------
# Tkinter UI -- embedded screen (same pattern as build_history_screen)
# ---------------------------------------------------------------------------

def build_trend_screen(parent, patient_id: str, database_root: str,
                        patient_label: str = "", on_back=None,
                        on_view_history=None):
    """Builds the Trends screen directly into `parent` (normally
    mira_app.self.container, right after self.clear())."""

    root_frame = tk.Frame(parent, bg=BG)
    root_frame.pack(fill="both", expand=True)

    state = {"entries": [], "window_days": DEFAULT_WINDOW_DAYS, "metric_key": None,
              "error": None, "x_mode": DEFAULT_X_MODE,
              "range_mode": "preset", "custom_start": None, "custom_end": None}

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

    tk.Label(header, text="Trends", font=("Helvetica", 26, "bold"),
              bg=BG, fg=PRIMARY).pack(side="left")
    if patient_label:
        tk.Label(header, text=patient_label, font=("Helvetica", 13),
                  bg=BG, fg=MUTED).pack(side="left", padx=(14, 0))

    if on_view_history:
        tk.Button(
            header, text="View History \u2192", command=on_view_history,
            font=("Helvetica", 11, "bold"), bg=CARD, fg="white",
            activebackground=SECONDARY, activeforeground="white",
            relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
        ).pack(side="right")

    body = tk.Frame(root_frame, bg=BG)
    body.pack(fill="both", expand=True, padx=30, pady=(0, 20))

    # -- window-length picker --
    picker_row = tk.Frame(body, bg=BG)
    picker_row.pack(fill="x", pady=(0, 12))
    tk.Label(picker_row, text="Show last:", font=("Helvetica", 12),
              bg=BG, fg=MUTED).pack(side="left", padx=(0, 10))
    window_var = tk.StringVar(
        value=WINDOW_LABELS_BY_MODE[DEFAULT_X_MODE][DEFAULT_WINDOW_DAYS])

    window_buttons = {}

    def _on_window_pick(days):
        state["window_days"] = days
        state["range_mode"] = "preset"
        window_var.set(WINDOW_LABELS_BY_MODE[state["x_mode"]][days])
        for d, b in window_buttons.items():
            b.config(bg=PRIMARY if d == days else CARD)
        _clear_custom_fields_silently()
        _redraw()

    def _refresh_window_button_labels():
        labels = WINDOW_LABELS_BY_MODE[state["x_mode"]]
        for d, b in window_buttons.items():
            b.config(text=labels[d])
        window_var.set(labels[state["window_days"]])

    for days in WINDOW_CHOICES:
        b = tk.Button(
            picker_row, text=WINDOW_LABELS_BY_MODE[DEFAULT_X_MODE][days],
            font=("Helvetica", 11, "bold"),
            bg=PRIMARY if days == DEFAULT_WINDOW_DAYS else CARD, fg="white",
            activebackground=PRIMARY_HOVER, activeforeground="white",
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
            command=lambda d=days: _on_window_pick(d),
        )
        b.pack(side="left", padx=(0, 6))
        window_buttons[days] = b

    # -- x-axis mode picker (session order vs. real calendar date) --
    xmode_row = tk.Frame(body, bg=BG)
    xmode_row.pack(fill="x", pady=(0, 12))
    tk.Label(xmode_row, text="X-axis:", font=("Helvetica", 12),
              bg=BG, fg=MUTED).pack(side="left", padx=(0, 10))

    xmode_buttons = {}

    def _on_xmode_pick(mode):
        state["x_mode"] = mode
        for m, b in xmode_buttons.items():
            b.config(bg=PRIMARY if m == mode else CARD)
        _refresh_window_button_labels()   # "7 days" <-> "7 sessions" etc.
        # A custom range typed for one mode (dates or session numbers)
        # isn't meaningful in the other -- clear it and fall back to the
        # preset picker rather than silently reinterpreting the numbers.
        state["range_mode"] = "preset"
        state["custom_start"] = None
        state["custom_end"] = None
        for d, b in window_buttons.items():
            b.config(bg=PRIMARY if d == state["window_days"] else CARD)
        _clear_custom_fields_silently()
        _refresh_custom_range_labels()
        _redraw()

    for mode in X_MODE_CHOICES:
        b = tk.Button(
            xmode_row, text=X_MODE_LABELS[mode],
            font=("Helvetica", 11, "bold"),
            bg=PRIMARY if mode == DEFAULT_X_MODE else CARD, fg="white",
            activebackground=PRIMARY_HOVER, activeforeground="white",
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
            command=lambda m=mode: _on_xmode_pick(m),
        )
        b.pack(side="left", padx=(0, 6))
        xmode_buttons[mode] = b

    # -- custom range: explicit dates or session numbers, instead of only
    # the preset buckets above. A range with nothing in it (or reaching
    # past the real session count) simply shows no points for that part --
    # nothing is fabricated to fill the gap. --
    custom_row = tk.Frame(body, bg=BG)
    custom_row.pack(fill="x", pady=(0, 12))

    custom_label_var = tk.StringVar()
    tk.Label(custom_row, textvariable=custom_label_var, font=("Helvetica", 12),
              bg=BG, fg=MUTED).pack(side="left", padx=(0, 10))

    custom_start_entry = tk.Entry(custom_row, font=("Helvetica", 11), width=12,
                                    bg=CARD_ALT, fg=TEXT, insertbackground=TEXT,
                                    relief="flat")
    custom_start_entry.pack(side="left", padx=(0, 4))
    tk.Label(custom_row, text="\u2013", font=("Helvetica", 12),
              bg=BG, fg=MUTED).pack(side="left")
    custom_end_entry = tk.Entry(custom_row, font=("Helvetica", 11), width=12,
                                  bg=CARD_ALT, fg=TEXT, insertbackground=TEXT,
                                  relief="flat")
    custom_end_entry.pack(side="left", padx=(4, 10))

    custom_status_var = tk.StringVar()
    custom_status_label = tk.Label(custom_row, textvariable=custom_status_var,
                                     font=("Helvetica", 11), bg=BG, fg=MUTED)

    def _refresh_custom_range_labels():
        if state["x_mode"] == "session":
            custom_label_var.set("Custom range (session # \u2013 #):")
        else:
            custom_label_var.set("Custom range (DD/MM/YYYY \u2013 DD/MM/YYYY):")

    def _clear_custom_fields_silently():
        """Clears the entry boxes/status text without touching state or
        redrawing -- used when a preset button or the x-axis toggle takes
        over, so stale custom text doesn't linger looking active."""
        custom_start_entry.delete(0, "end")
        custom_end_entry.delete(0, "end")
        custom_status_var.set("")
        custom_status_label.pack_forget()

    def _show_custom_status(text, is_error):
        custom_status_var.set(text)
        custom_status_label.config(fg=WARNING if is_error else PRIMARY)
        custom_status_label.pack(side="left")

    def _apply_custom_range():
        start_raw = custom_start_entry.get()
        end_raw = custom_end_entry.get()

        if state["x_mode"] == "session":
            try:
                start_val = int(start_raw.strip()) if start_raw.strip() else None
                end_val = int(end_raw.strip()) if end_raw.strip() else None
            except ValueError:
                _show_custom_status("Session numbers must be whole numbers.", True)
                return
        else:
            start_val = parse_ddmmyyyy(start_raw)
            end_val = parse_ddmmyyyy(end_raw)
            if (start_raw.strip() and start_val is None) or (end_raw.strip() and end_val is None):
                _show_custom_status("Use DD/MM/YYYY for both dates.", True)
                return

        if start_val is None and end_val is None:
            _show_custom_status("Enter a start, an end, or both.", True)
            return

        state["range_mode"] = "custom"
        state["custom_start"] = start_val
        state["custom_end"] = end_val
        for d, b in window_buttons.items():
            b.config(bg=CARD)
        _show_custom_status("Custom range active.", False)
        _redraw()

    def _clear_custom_range():
        state["range_mode"] = "preset"
        state["custom_start"] = None
        state["custom_end"] = None
        _clear_custom_fields_silently()
        for d, b in window_buttons.items():
            b.config(bg=PRIMARY if d == state["window_days"] else CARD)
        _redraw()

    tk.Button(
        custom_row, text="Apply", command=_apply_custom_range,
        font=("Helvetica", 11, "bold"), bg=SECONDARY, fg="white",
        activebackground=PRIMARY_HOVER, activeforeground="white",
        relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
    ).pack(side="left", padx=(0, 6))

    tk.Button(
        custom_row, text="Clear", command=_clear_custom_range,
        font=("Helvetica", 11, "bold"), bg=CARD, fg="white",
        activebackground=SECONDARY, activeforeground="white",
        relief="flat", bd=0, padx=14, pady=6, cursor="hand2",
    ).pack(side="left")

    _refresh_custom_range_labels()

    # -- metric picker --
    metric_row = tk.Frame(body, bg=BG)
    metric_row.pack(fill="x", pady=(0, 12))
    tk.Label(metric_row, text="Metric:", font=("Helvetica", 12),
              bg=BG, fg=MUTED).pack(side="left", padx=(0, 10))
    metric_var = tk.StringVar(value="")
    metric_picker = ttk.Combobox(metric_row, textvariable=metric_var,
                                  state="readonly", width=36,
                                  font=("Helvetica", 11))
    metric_picker.pack(side="left")

    # -- summary --
    summary_label = tk.Label(
        body, text="", font=("Helvetica", 14), bg=CARD, fg=TEXT,
        justify="left", anchor="w", padx=18, pady=14,
    )
    summary_label.pack(fill="x", pady=(0, 14))

    def _on_body_configure(event):
        summary_label.config(wraplength=max(event.width - 40, 240))

    body.bind("<Configure>", _on_body_configure)

    # -- chart --
    chart_frame = tk.Frame(body, bg=BG)
    chart_frame.pack(fill="both", expand=True)

    fig = ax = canvas_widget = mdates = None
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.dates as mdates
        fig = Figure(figsize=(7, 3.4), dpi=100, facecolor=BG)
        ax = fig.add_subplot(111)
        canvas_widget = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas_widget.get_tk_widget().pack(fill="both", expand=True)
    except ImportError:
        tk.Label(chart_frame,
                  text="matplotlib is not installed -- chart unavailable.\n"
                       "pip install matplotlib to see the graph here; the "
                       "summary above still works.",
                  bg=BG, fg=MUTED, font=("Helvetica", 11),
                  justify="left").pack(anchor="w", pady=20)

    # -- redraw --

    def _redraw():
        entries = state["entries"]

        if state["error"]:
            summary_label.config(text=state["error"], fg=WARNING)
            metric_picker.config(values=[])
            if ax is not None:
                ax.clear()
                canvas_widget.draw()
            return

        if not entries:
            summary_label.config(text="No history yet for this patient.", fg=MUTED)
            metric_picker.config(values=[])
            if ax is not None:
                ax.clear()
                canvas_widget.draw()
            return

        if state["range_mode"] == "custom":
            if state["x_mode"] == "session":
                window_entries = filter_session_range(
                    entries, state["custom_start"], state["custom_end"])
            else:
                window_entries = filter_date_range(
                    entries, state["custom_start"], state["custom_end"])
        elif state["x_mode"] == "session":
            window_entries = filter_last_n_sessions(entries, state["window_days"])
        else:
            window_entries = filter_window(entries, state["window_days"])

        metric_keys = collect_metric_keys(entries)
        labels = [pretty_metric_label(k) for k in metric_keys]
        metric_picker.config(values=labels)

        if state["metric_key"] is None and metric_keys:
            state["metric_key"] = metric_keys[0]
        if state["metric_key"] in metric_keys:
            metric_var.set(pretty_metric_label(state["metric_key"]))

        if not metric_keys:
            summary_label.config(text="No numeric metrics found in dataset.csv.", fg=MUTED)
            if ax is not None:
                ax.clear()
                canvas_widget.draw()
            return

        series = build_metric_series(window_entries, state["metric_key"])
        summary_label.config(
            text=compute_trend_summary(pretty_metric_label(state["metric_key"]), series),
            fg=TEXT,
        )

        if ax is not None:
            ax.clear()
            ax.set_facecolor(BG)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.tick_params(colors=TEXT, labelsize=9)
            if series:
                dates = [d for d, _ in series]
                values = [v for _, v in series]

                if state["x_mode"] == "session":
                    # Evenly spaced by session order, not by real elapsed
                    # time -- a 2-month gap between two sessions no longer
                    # stretches the axis and squashes every other point
                    # into a corner.
                    xs = list(range(len(series)))
                    ax.plot(xs, values, marker="o", color=PRIMARY, linewidth=2)
                    tick_idx = thin_tick_indices(len(series))
                    ax.set_xticks([xs[i] for i in tick_idx])
                    ax.set_xticklabels(
                        [format_session_tick(dates[i]) for i in tick_idx],
                        rotation=90, fontsize=10, color=TEXT, linespacing=1.4,
                    )
                    ax.set_xlim(-0.5, len(series) - 0.5)
                    fig.subplots_adjust(bottom=0.32)
                else:
                    # Real calendar-date spacing (original behavior) --
                    # gaps between sessions show up as blank stretches.
                    # Tick labels use the same DD/MM/YYYY convention as
                    # session mode instead of matplotlib's default
                    # MM-DD/hour format.
                    ax.plot(dates, values, marker="o", color=PRIMARY, linewidth=2)
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
                    fig.autofmt_xdate(rotation=30)
                    for lbl in ax.get_xticklabels():
                        lbl.set_fontsize(10)

                ax.set_ylabel(pretty_metric_label(state["metric_key"]), color=TEXT, fontsize=10)
            else:
                ax.text(0.5, 0.5, "No data for this metric in this window",
                        ha="center", va="center", transform=ax.transAxes, color=MUTED)

            if state["x_mode"] != "session" or not series:
                fig.tight_layout()
            canvas_widget.draw()

    def _on_metric_change(_event=None):
        label = metric_var.get()
        metric_keys = collect_metric_keys(state["entries"])
        state["metric_key"] = next(
            (k for k in metric_keys if pretty_metric_label(k) == label), state["metric_key"])
        _redraw()

    metric_picker.bind("<<ComboboxSelected>>", _on_metric_change)

    def _reload():
        try:
            state["entries"] = load_patient_sessions(patient_id, database_root)
            state["error"] = None
        except Exception as exc:
            state["entries"] = []
            state["error"] = f"Could not load history:\n{exc}"
        _redraw()

    _reload()

    root_frame._trend_screen_state = state
    root_frame._trend_screen_reload = _reload
    return root_frame