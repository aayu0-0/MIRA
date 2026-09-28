"""
Phase 5 — Doctor Report (File 2 of 4)
---------------------------------------
Generates a clinical-grade PDF intended for a healthcare professional.

Design decisions:
- Full raw scores alongside every observation (the doctor wants numbers)
- Severity tables per region (acne score: 42/100, band: mild 20–45)
- Capture quality warnings surfaced prominently at the top
- Per-metric normal range printed in the table so the doctor can judge
  without looking up source docs
- No softening language — reports facts and scores as-is
- Annotated face image included for visual reference
- Full disclaimer + generation metadata in footer
"""

import io
import datetime
import cv2
import numpy as np
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)

from thresholds import (
    SYMMETRY, SWELLING, EAR,
    EYE_DARK_CIRCLE_SEVERITY, EYE_REDNESS_SEVERITY, EYE_PUFFINESS_SEVERITY,
    SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY,
    SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY, SKIN_NOSE_TEXTURE_SEVERITY,
    LIP_DRYNESS_SEVERITY, LIP_COLOR_INCONSISTENCY_SEVERITY, LIP_PALLOR_SEVERITY,
    FACE_OBSERVATION, MOUTH_OBSERVATION,
)


# ── Palette ───────────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#0D1B2A")
C_ACCENT  = colors.HexColor("#1B4F72")
C_LIGHT   = colors.HexColor("#EAF2FB")
C_GREEN   = colors.HexColor("#1E8449")
C_YELLOW  = colors.HexColor("#B7950B")
C_ORANGE  = colors.HexColor("#CA6F1E")
C_RED     = colors.HexColor("#922B21")
C_GRAY    = colors.HexColor("#566573")
C_LGRAY   = colors.HexColor("#D5D8DC")
C_WHITE   = colors.white
C_ROW_ALT = colors.HexColor("#F2F3F4")

SEV_COLORS = {
    "none":     C_GREEN,
    "mild":     C_YELLOW,
    "moderate": C_ORANGE,
    "notable":  C_RED,
}

SEV_BG = {
    "none":     colors.HexColor("#EAFAF1"),
    "mild":     colors.HexColor("#FEF9E7"),
    "moderate": colors.HexColor("#FEF5EC"),
    "notable":  colors.HexColor("#FDEDEC"),
}


def _sev_color(s):
    return SEV_COLORS.get((s or "").lower(), C_GRAY)


def _sev_bg(s):
    return SEV_BG.get((s or "").lower(), C_WHITE)


class DoctorReportGenerator:

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    COL_W  = PAGE_W - 2 * MARGIN   # usable width

    def generate(self,
                 p1, p2, p3, p4,
                 annotated_image_bgr=None,
                 patient_id: str = "UNKNOWN",
                 output_path: str = None,
                 quality_warnings: list = None) -> bytes:
        """
        Generate clinical PDF. Returns PDF bytes; writes to output_path if given.
        """
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=self.MARGIN,
            bottomMargin=self.MARGIN + 6 * mm,
        )
        s = self._styles()
        story = []

        story += self._header(s, patient_id)
        story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
        story.append(Spacer(1, 3 * mm))
        story += self._quality_section(quality_warnings, s)
        story += self._image_section(annotated_image_bgr, s)
        story += self._face_geo_full(p1, p3, p4, s)
        story += self._eye_section(p3, p4, s)
        story += self._skin_section(p3, p4, s)
        story += self._lip_section(p3, p4, s)
        story += self._mouth_section(p3, p4, s)
        story += self._assessment_section(p4, s)
        story += self._disclaimer(s)

        doc.build(story,
                  onFirstPage=self._footer,
                  onLaterPages=self._footer)
        pdf_bytes = buf.getvalue()
        if output_path:
            Path(output_path).write_bytes(pdf_bytes)
        return pdf_bytes

    # ── Styles ────────────────────────────────────────────────────────────────

    def _styles(self):
        s = {}
        s["title"] = ParagraphStyle("dr_title",
            fontName="Helvetica-Bold", fontSize=17, leading=21,
            textColor=C_DARK, spaceAfter=2 * mm)
        s["subtitle"] = ParagraphStyle("dr_subtitle",
            fontName="Helvetica", fontSize=9, leading=12,
            textColor=C_GRAY, spaceAfter=1 * mm)
        s["section"] = ParagraphStyle("dr_section",
            fontName="Helvetica-Bold", fontSize=12, textColor=C_ACCENT,
            spaceBefore=6 * mm, spaceAfter=2 * mm)
        s["subsection"] = ParagraphStyle("dr_subsection",
            fontName="Helvetica-Bold", fontSize=10, textColor=C_DARK,
            spaceBefore=3 * mm, spaceAfter=1 * mm)
        s["body"] = ParagraphStyle("dr_body",
            fontName="Helvetica", fontSize=9, leading=13, textColor=C_DARK,
            spaceAfter=1 * mm)
        s["obs"] = ParagraphStyle("dr_obs",
            fontName="Helvetica", fontSize=9, leading=12, textColor=C_DARK,
            leftIndent=4 * mm, spaceAfter=0.5 * mm)
        s["warn_body"] = ParagraphStyle("dr_warn_body",
            fontName="Helvetica", fontSize=8.5, leading=11,
            textColor=colors.HexColor("#7A4A00"))
        s["cell"] = ParagraphStyle("dr_cell",
            fontName="Helvetica", fontSize=8.5, leading=11, textColor=C_DARK)
        s["cell_bold"] = ParagraphStyle("dr_cell_bold",
            fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=C_DARK)
        s["cell_hdr"] = ParagraphStyle("dr_cell_hdr",
            fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=C_WHITE)
        s["disclaimer"] = ParagraphStyle("dr_disclaimer",
            fontName="Helvetica-Oblique", fontSize=7, leading=10,
            textColor=C_GRAY)
        return s

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self, s, patient_id):
        now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        return [
            Paragraph("Facial Health Observation Report — Clinical Review", s["title"]),
            Paragraph(
                f"Patient ID: <b>{patient_id}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Generated: {now}&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"System: Mira AI Facial Analysis v1.0&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<i>FOR CLINICAL USE — CONTAINS RAW METRIC DATA</i>",
                s["subtitle"]),
            Spacer(1, 2 * mm),
        ]

    # ── Quality warning ───────────────────────────────────────────────────────

    def _quality_section(self, warnings, s):
        if not warnings:
            return []
        lines = "<br/>".join(f"• {w}" for w in warnings)
        banner = Table(
            [[Paragraph(
                f"<b>CAPTURE QUALITY WARNING</b><br/>{lines}<br/>"
                f"<i>Luminance-sensitive scores (dark circles, redness) may be "
                f"skewed by lighting rather than physiology.</i>",
                s["warn_body"]
            )]],
            colWidths=[self.COL_W],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFF3CD")),
            ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#FFC107")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return [banner, Spacer(1, 4 * mm)]

    # ── Annotated image ───────────────────────────────────────────────────────

    def _image_section(self, img_bgr, s):
        if img_bgr is None:
            return []
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, png = cv2.imencode(".png", rgb)
        img_io = io.BytesIO(png.tobytes())
        h, w = img_bgr.shape[:2]
        max_w = 90 * mm
        scale = min(max_w / w, 65 * mm / h)
        rl = RLImage(img_io, width=w * scale, height=h * scale)
        return [
            Paragraph("Facial Analysis Visualization", s["section"]),
            rl,
            Spacer(1, 2 * mm),
            Paragraph("Annotated mesh with Phase 2 region overlays.", s["body"]),
        ]

    # ── Metric table helper ───────────────────────────────────────────────────

    def _metric_table(self, rows, s):
        """
        rows: list of (label, value_str, normal_range_str, severity_str)
        """
        header = [
            Paragraph("Metric", s["cell_hdr"]),
            Paragraph("Value", s["cell_hdr"]),
            Paragraph("Normal Range", s["cell_hdr"]),
            Paragraph("Severity", s["cell_hdr"]),
        ]
        col_w = [self.COL_W * 0.40,
                 self.COL_W * 0.18,
                 self.COL_W * 0.27,
                 self.COL_W * 0.15]

        table_data = [header]
        style_cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0), C_ACCENT),
            ("GRID",          (0, 0), (-1, -1), 0.4, C_LGRAY),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]

        for i, (label, value, normal, severity) in enumerate(rows, start=1):
            sev_str = (severity or "—").capitalize()
            row = [
                Paragraph(label, s["cell"]),
                Paragraph(str(value), s["cell_bold"]),
                Paragraph(str(normal), s["cell"]),
                Paragraph(sev_str, s["cell_bold"]),
            ]
            table_data.append(row)
            bg = _sev_bg(severity) if severity else (C_ROW_ALT if i % 2 == 0 else C_WHITE)
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
            if severity and severity != "none":
                style_cmds.append(("TEXTCOLOR", (3, i), (3, i), _sev_color(severity)))

        t = Table(table_data, colWidths=col_w)
        t.setStyle(TableStyle(style_cmds))
        return t

    # ── Observation list helper ───────────────────────────────────────────────

    def _obs_list(self, observations, s):
        items = []
        for obs in observations:
            sev = obs.severity.value.lower()
            col = _sev_color(sev)
            score_str = f" [score: {obs.score:.1f}]" if obs.score is not None else ""
            items.append(Paragraph(
                f"<font color='#{col.hexval()[2:]}'>●</font> "
                f"<b>[{obs.severity.value.upper()}]</b>{score_str} {obs.finding}",
                s["obs"]
            ))
        return items

    # ── Section 1: Face Geometry ──────────────────────────────────────────────

    def _face_geo_full(self, p1, p3, p4, s):
        items = [Paragraph("1. Face Geometry", s["section"])]

        if p1.symmetry_score >= SYMMETRY.highly_symmetric:
            sym_sev = None
        elif p1.symmetry_score >= SYMMETRY.normal:
            sym_sev = None
        elif p1.symmetry_score >= SYMMETRY.mild_asymmetry:
            sym_sev = "mild"
        else:
            sym_sev = "moderate"

        tilt_sev = (
            "none" if abs(p1.alignment_angle) <= FACE_OBSERVATION.tilt_level
            else "mild" if abs(p1.alignment_angle) <= FACE_OBSERVATION.tilt_slight
            else "moderate"
        )
        if tilt_sev == "none":
            tilt_sev = None

        ar = p1.face_height / p1.face_width if p1.face_width else 0
        ar_ok = FACE_OBSERVATION.aspect_low <= ar <= FACE_OBSERVATION.aspect_high

        rows = [
            ("Symmetry Score",
             f"{p1.symmetry_score:.1f} / 100",
             f">= {SYMMETRY.normal:.0f}",
             sym_sev),
            ("Head Tilt",
             f"{p1.alignment_angle:.1f}°",
             f"± {FACE_OBSERVATION.tilt_level:.0f}°",
             tilt_sev),
            ("Face Aspect Ratio (H/W)",
             f"{ar:.3f}",
             f"{FACE_OBSERVATION.aspect_low}–{FACE_OBSERVATION.aspect_high}",
             None if ar_ok else "mild"),
        ]
        if p3.face.available:
            swelling_raw = p3.face.swelling_label or ""
            sw_word = swelling_raw.lower().split()[0] if swelling_raw else "none"
            rows += [
                ("Swelling Score",
                 f"{p3.face.swelling_score:.1f} / 100",
                 f"< {SWELLING.none:.0f}",
                 sw_word if sw_word != "no" else None),
                ("Cheek / Eye-Span Ratio",
                 f"{p3.face.cheek_eye_ratio:.3f}",
                 f"<= {SWELLING.typical_high:.2f}",
                 None),
            ]
        items.append(self._metric_table(rows, s))
        items.append(Spacer(1, 2 * mm))
        items.append(Paragraph("Clinical Observations:", s["subsection"]))
        for text in [p4.face.symmetry, p4.face.alignment,
                     p4.face.proportion, p4.face.swelling]:
            if text:
                items.append(Paragraph(f"• {text}", s["obs"]))
        return items

    # ── Section 2: Eyes ───────────────────────────────────────────────────────

    def _eye_section(self, p3, p4, s):
        items = [Paragraph("2. Eyes", s["section"])]

        for side, eye, obs_list in [
            ("Left Eye",  p3.left_eye,  p4.eyes.left),
            ("Right Eye", p3.right_eye, p4.eyes.right),
        ]:
            items.append(Paragraph(side, s["subsection"]))
            if not eye.available:
                items.append(Paragraph("Region unavailable.", s["body"]))
                continue

            rows = [
                ("Dark Circle Score",
                 f"{eye.dark_circle_score:.1f} / 100",
                 f"< {EYE_DARK_CIRCLE_SEVERITY.mild:.0f}",
                 self._sev_str(eye.dark_circle_score, EYE_DARK_CIRCLE_SEVERITY)),
                ("Redness Score",
                 f"{eye.redness_score:.1f} / 100",
                 f"< {EYE_REDNESS_SEVERITY.mild:.0f}",
                 self._sev_str(eye.redness_score, EYE_REDNESS_SEVERITY)),
                ("Puffiness Score",
                 f"{eye.puffiness_score:.1f} / 100",
                 f"< {EYE_PUFFINESS_SEVERITY.mild:.0f}",
                 self._sev_str(eye.puffiness_score, EYE_PUFFINESS_SEVERITY)),
                ("Openness Ratio",
                 f"{eye.openness_ratio:.1f} / 100" if eye.openness_ratio is not None
                     else "N/A (landmarks unavailable)",
                 f">= 20",
                 None),
            ]
            items.append(self._metric_table(rows, s))
            items.append(Spacer(1, 1 * mm))
            items += self._obs_list(obs_list, s)

        if p4.eyes.openness_note:
            items.append(Paragraph(f"• {p4.eyes.openness_note}", s["obs"]))

        return items

    # ── Section 3: Skin ───────────────────────────────────────────────────────

    def _skin_section(self, p3, p4, s):
        items = [Paragraph("3. Skin", s["section"])]

        regions = [
            ("Forehead",    p3.skin_forehead,    p4.skin.forehead,
             SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY),
            ("Left Cheek",  p3.skin_left_cheek,  p4.skin.left_cheek,
             SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY),
            ("Right Cheek", p3.skin_right_cheek, p4.skin.right_cheek,
             SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY),
            ("Nose",        p3.skin_nose,        p4.skin.nose,
             SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY, SKIN_NOSE_TEXTURE_SEVERITY),
        ]

        for label, region, obs_list, acne_b, red_b, tex_b in regions:
            items.append(Paragraph(label, s["subsection"]))
            if not region.available:
                items.append(Paragraph("Region unavailable.", s["body"]))
                continue
            ll = f"  <i>(⚠ low light — acne/texture suppressed)</i>" if region.low_light else ""
            rows = [
                (f"Acne Score{ll}",
                 f"{region.acne_score:.1f} / 100",
                 f"< {acne_b.mild:.0f}",
                 self._sev_str(region.acne_score, acne_b)),
                ("Redness Score",
                 f"{region.redness_score:.1f} / 100",
                 f"< {red_b.mild:.0f}",
                 self._sev_str(region.redness_score, red_b)),
                ("Texture Irregularity",
                 f"{region.texture_irregularity:.1f} / 100",
                 f"< {tex_b.mild:.0f}",
                 self._sev_str(region.texture_irregularity, tex_b)),
            ]
            items.append(self._metric_table(rows, s))
            items.append(Spacer(1, 1 * mm))
            items += self._obs_list(obs_list, s)

        # Overall cross-region skin observations
        if p4.skin.overall:
            items.append(Paragraph("Cross-Region Skin Findings:", s["subsection"]))
            items += self._obs_list(p4.skin.overall, s)

        return items

    # ── Section 4: Lips ───────────────────────────────────────────────────────

    def _lip_section(self, p3, p4, s):
        items = [Paragraph("4. Lips", s["section"])]

        if not p3.lips.available:
            items.append(Paragraph("Lip region unavailable.", s["body"]))
            return items

        ll = "  <i>(⚠ low light — dryness suppressed)</i>" if p3.lips.low_light else ""
        rows = [
            (f"Dryness Score{ll}",
             f"{p3.lips.dryness_score:.1f} / 100",
             f"< {LIP_DRYNESS_SEVERITY.mild:.0f}",
             self._sev_str(p3.lips.dryness_score, LIP_DRYNESS_SEVERITY)),
            ("Color Inconsistency",
             f"{p3.lips.color_consistency:.1f} / 100",
             f"< {LIP_COLOR_INCONSISTENCY_SEVERITY.mild:.0f}",
             self._sev_str(p3.lips.color_consistency, LIP_COLOR_INCONSISTENCY_SEVERITY)),
            ("Pallor Score",
             f"{p3.lips.pallor_score:.1f} / 100",
             f"< {LIP_PALLOR_SEVERITY.mild:.0f}",
             self._sev_str(p3.lips.pallor_score, LIP_PALLOR_SEVERITY)),
        ]
        items.append(self._metric_table(rows, s))
        items.append(Spacer(1, 1 * mm))
        items += self._obs_list(p4.lips.findings, s)
        return items

    # ── Section 5: Mouth Geometry ─────────────────────────────────────────────

    def _mouth_section(self, p3, p4, s):
        items = [Paragraph("5. Mouth Geometry", s["section"])]

        if not p3.smile.available:
            items.append(Paragraph("Mouth landmark data unavailable.", s["body"]))
            return items

        casym_sev = (
            None if p3.smile.corner_asymmetry < MOUTH_OBSERVATION.asymmetry_slight
            else "mild" if p3.smile.corner_asymmetry < MOUTH_OBSERVATION.asymmetry_mild
            else "moderate" if p3.smile.corner_asymmetry < MOUTH_OBSERVATION.asymmetry_notable
            else "notable"
        )
        curv_neutral = abs(p3.smile.curvature_score) < MOUTH_OBSERVATION.curvature_neutral_band

        rows = [
            ("Mouth Aspect Ratio (normalized)",
             f"{p3.smile.mouth_aspect_ratio:.1f} / 100",
             "50–100 (resting closed)",
             None),
            ("Corner Asymmetry Score",
             f"{p3.smile.corner_asymmetry:.1f} / 100",
             f"< {MOUTH_OBSERVATION.asymmetry_slight:.0f}",
             casym_sev),
            ("Curvature Score (signed)",
             f"{p3.smile.curvature_score:.1f}",
             f"± {MOUTH_OBSERVATION.curvature_neutral_band:.0f} (neutral)",
             None if curv_neutral else "mild"),
        ]
        items.append(self._metric_table(rows, s))
        items.append(Spacer(1, 1 * mm))
        items += self._obs_list(p4.mouth.findings, s)
        return items

    # ── Section 6: Overall Assessment ────────────────────────────────────────

    def _assessment_section(self, p4, s):
        items = [Paragraph("6. Overall Assessment", s["section"])]

        conf_color = (C_GREEN if p4.observation_score >= 80
                      else C_YELLOW if p4.observation_score >= 65
                      else C_ORANGE)

        items.append(Paragraph(
            f"<b>Assessment Label:</b> {p4.overall_label}&nbsp;&nbsp;&nbsp;"
            f"<b>Observation Score:</b> {p4.observation_score:.0f}%",
            s["body"]
        ))
        items.append(Spacer(1, 1 * mm))
        for text in p4.overall:
            items.append(Paragraph(f"• {text}", s["obs"]))
        return items

    # ── Disclaimer ────────────────────────────────────────────────────────────

    def _disclaimer(self, s):
        return [
            Spacer(1, 6 * mm),
            HRFlowable(width="100%", thickness=1, color=C_LGRAY),
            Spacer(1, 2 * mm),
            Paragraph(
                "DISCLAIMER: This report is generated by an automated computer vision system "
                "(Mira AI Facial Analysis). It contains visual and geometric observations only "
                "and does NOT constitute a medical diagnosis, clinical assessment, or treatment "
                "recommendation. All findings must be interpreted by a qualified healthcare "
                "professional in the context of the patient's clinical history. Scores are "
                "derived from a single photographic frame and may be affected by lighting, "
                "pose, and image quality.",
                s["disclaimer"]
            ),
        ]

    # ── Footer ────────────────────────────────────────────────────────────────

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_GRAY)
        canvas.drawString(self.MARGIN,
                          8 * mm,
                          "Mira AI — Clinical Report — CONFIDENTIAL")
        canvas.drawRightString(
            self.PAGE_W - self.MARGIN,
            8 * mm,
            f"Page {canvas.getPageNumber()}"
        )
        canvas.restoreState()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sev_str(value, band) -> str:
        if value is None:
            return None
        if value < band.mild:
            return "none"
        if value < band.moderate:
            return "mild"
        if value < band.notable:
            return "moderate"
        return "notable"