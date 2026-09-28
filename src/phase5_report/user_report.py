"""
Phase 5 — User Report (File 3 of 4)
--------------------------------------
Generates a plain-language PDF for the person whose face was analyzed.

Design decisions:
- No raw scores, no technical jargon, no medical terminology
- Severity communicated through plain words: "looks healthy", "slightly",
  "noticeably", "significant" — never "mild/moderate/notable"
- Colour-coded visual indicators (green dot = good, yellow = watch,
  orange = notable, red = significant) — icons the user can scan quickly
- Each section: one summary sentence + bullet-point findings
- Ends with a clear "what to do next" block
- Prominent disclaimer styled as a friendly note, not legalese
- Capture quality warning in plain language if present
"""

import io
import datetime
import cv2
import numpy as np
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)


# ── Palette: warmer, friendlier than the doctor palette ───────────────────────
C_DARK    = colors.HexColor("#1C2833")
C_ACCENT  = colors.HexColor("#1A5276")
C_MINT    = colors.HexColor("#D5F5E3")
C_WARM    = colors.HexColor("#FAF3E0")
C_GRAY    = colors.HexColor("#717D7E")
C_LGRAY   = colors.HexColor("#D7DBDD")
C_WHITE   = colors.white

# Friendly traffic-light colours
DOT_GREEN  = colors.HexColor("#27AE60")
DOT_YELLOW = colors.HexColor("#F39C12")
DOT_ORANGE = colors.HexColor("#E67E22")
DOT_RED    = colors.HexColor("#C0392B")

SEV_TO_DOT = {
    "none":     ("●", DOT_GREEN,  "Looks healthy"),
    "mild":     ("●", DOT_YELLOW, "Slightly noticeable"),
    "moderate": ("●", DOT_ORANGE, "Worth keeping an eye on"),
    "notable":  ("●", DOT_RED,    "Noticeably present"),
}

# Plain-language substitution map for finding text cleanup
_JARGON = {
    "notable": "significant",
    "moderate": "noticeable",
    "mild": "slight",
    "puffiness": "slight swelling",
    "pallor": "paleness",
    "erythema": "redness",
    "asymmetry": "unevenness",
    "texture irregularity": "uneven skin texture",
    "acne score": "breakout level",
}


def _plain(text: str) -> str:
    """Lightly clean up clinical terms in observation text."""
    for k, v in _JARGON.items():
        text = text.replace(k, v)
    return text


def _dot_color(severity: str):
    return SEV_TO_DOT.get(severity.lower(), ("●", C_GRAY, "—"))


class UserReportGenerator:

    PAGE_W, PAGE_H = A4
    MARGIN = 20 * mm
    COL_W  = PAGE_W - 2 * MARGIN

    def generate(self,
                 p1, p2, p3, p4,
                 annotated_image_bgr=None,
                 patient_id: str = "UNKNOWN",
                 output_path: str = None,
                 quality_warnings: list = None) -> bytes:
        """
        Generate user-friendly PDF. Returns bytes; writes to output_path if given.
        """
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=self.MARGIN,
            bottomMargin=self.MARGIN + 8 * mm,
        )
        s = self._styles()
        story = []

        story += self._header(s, patient_id)
        story.append(Spacer(1, 3 * mm))
        story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
        story.append(Spacer(1, 4 * mm))

        story += self._quality_section(quality_warnings, s)
        story += self._image_section(annotated_image_bgr, s)
        story += self._legend(s)
        story += self._overview_section(p4, s)
        story += self._face_section(p1, p3, p4, s)
        story += self._eye_section(p3, p4, s)
        story += self._skin_section(p3, p4, s)
        story += self._lip_section(p3, p4, s)
        story += self._mouth_section(p3, p4, s)
        story += self._next_steps(p4, s)
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
        s["title"] = ParagraphStyle("u_title",
            fontName="Helvetica-Bold", fontSize=18, leading=22,
            textColor=C_ACCENT, spaceAfter=2 * mm)
        s["subtitle"] = ParagraphStyle("u_subtitle",
            fontName="Helvetica", fontSize=9, leading=12,
            textColor=C_GRAY, spaceAfter=1 * mm)
        s["section"] = ParagraphStyle("u_section",
            fontName="Helvetica-Bold", fontSize=12, textColor=C_ACCENT,
            spaceBefore=6 * mm, spaceAfter=2 * mm)
        s["summary"] = ParagraphStyle("u_summary",
            fontName="Helvetica", fontSize=10, leading=14,
            textColor=C_DARK, spaceAfter=2 * mm)
        s["body"] = ParagraphStyle("u_body",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=C_DARK, spaceAfter=1 * mm)
        s["finding_good"] = ParagraphStyle("u_good",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=colors.HexColor("#1E8449"), leftIndent=6 * mm)
        s["finding_watch"] = ParagraphStyle("u_watch",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=colors.HexColor("#9A7D0A"), leftIndent=6 * mm)
        s["finding_notable"] = ParagraphStyle("u_notable",
            fontName="Helvetica-Bold", fontSize=9, leading=13,
            textColor=colors.HexColor("#CA6F1E"), leftIndent=6 * mm)
        s["finding_red"] = ParagraphStyle("u_red",
            fontName="Helvetica-Bold", fontSize=9, leading=13,
            textColor=colors.HexColor("#922B21"), leftIndent=6 * mm)
        s["note"] = ParagraphStyle("u_note",
            fontName="Helvetica-Oblique", fontSize=8, leading=11,
            textColor=C_GRAY)
        s["next"] = ParagraphStyle("u_next",
            fontName="Helvetica", fontSize=9, leading=13,
            textColor=C_DARK, leftIndent=4 * mm, spaceAfter=1 * mm)
        s["disclaimer"] = ParagraphStyle("u_disclaimer",
            fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
            textColor=C_GRAY)
        s["warn_body"] = ParagraphStyle("u_warn_body",
            fontName="Helvetica", fontSize=8.5, leading=11,
            textColor=colors.HexColor("#7A4A00"))
        s["legend_cell"] = ParagraphStyle("u_legend_cell",
            fontName="Helvetica", fontSize=8, leading=10, textColor=C_DARK)
        return s

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self, s, patient_id):
        now = datetime.datetime.now().strftime("%d %B %Y")
        return [
            Paragraph("Your Facial Health Summary", s["title"]),
            Paragraph(
                f"Report for: <b>{patient_id}</b>&nbsp;&nbsp;|&nbsp;&nbsp;Date: {now}",
                s["subtitle"]
            ),
        ]

    # ── Quality warning ───────────────────────────────────────────────────────

    def _quality_section(self, warnings, s):
        if not warnings:
            return []
        lines = "<br/>".join(f"• {w}" for w in warnings)
        banner = Table(
            [[Paragraph(
                f"<b>📷 Photo Quality Note</b><br/>{lines}<br/>"
                f"<i>Some results may be less accurate due to the above. "
                f"For best results, retake the photo in good lighting, "
                f"facing the camera directly.</i>",
                s["warn_body"]
            )]],
            colWidths=[self.COL_W],
        )
        banner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
            ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#FFD54F")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
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
        max_w = 85 * mm
        scale = min(max_w / w, 60 * mm / h)
        rl = RLImage(img_io, width=w * scale, height=h * scale)
        return [
            Paragraph("Analysis Visualization", s["section"]),
            rl,
            Spacer(1, 1 * mm),
            Paragraph("The highlighted areas show the regions we analyzed.", s["note"]),
            Spacer(1, 3 * mm),
        ]

    # ── Colour legend ─────────────────────────────────────────────────────────

    def _legend(self, s):
        items = [Paragraph("How to read this report:", s["body"])]
        legend_rows = [[
            Paragraph(f"<font color='#27AE60'>●</font>  Looks healthy", s["legend_cell"]),
            Paragraph(f"<font color='#F39C12'>●</font>  Slightly noticeable", s["legend_cell"]),
            Paragraph(f"<font color='#E67E22'>●</font>  Worth monitoring", s["legend_cell"]),
            Paragraph(f"<font color='#C0392B'>●</font>  Noticeably present", s["legend_cell"]),
        ]]
        t = Table(legend_rows, colWidths=[self.COL_W / 4] * 4)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_LGRAY),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return [t, Spacer(1, 4 * mm)]

    # ── Finding helper ────────────────────────────────────────────────────────

    def _finding_para(self, obs, s):
        sev = obs.severity.value.lower()
        dot, dot_color, _ = _dot_color(sev)
        text = _plain(obs.finding)
        hex_col = dot_color.hexval()[2:]

        if sev == "none":
            style = s["finding_good"]
        elif sev == "mild":
            style = s["finding_watch"]
        elif sev == "moderate":
            style = s["finding_notable"]
        else:
            style = s["finding_red"]

        return Paragraph(
            f"<font color='#{hex_col}'>{dot}</font>  {text}",
            style
        )

    # ── Section: Overall Overview ─────────────────────────────────────────────

    def _overview_section(self, p4, s):
        items = [Paragraph("Overall Result", s["section"])]
        label = p4.overall_label
        conf  = p4.observation_score

        # Translate system label to friendly language
        friendly = {
            "No Major Visible Abnormalities": "Your facial features look healthy overall.",
            "Mild Observations Present": "A few minor things were noticed — nothing alarming.",
            "Moderate Observations Present": "Some areas are worth keeping an eye on.",
            "Notable Observations Present": "Several things were flagged that may be worth discussing with a professional.",
            "Insufficient Data — Assessment Incomplete": "We couldn't gather enough clear data from this photo for a reliable assessment. Please retake it in good lighting, facing the camera directly.",
        }.get(label, f"Assessment: {label}.")

        items.append(Paragraph(friendly, s["summary"]))
        items.append(Paragraph(
            f"The system's Observation Score for this analysis: <b>{conf:.0f}%</b>.",
            s["body"]
        ))
        for text in p4.overall:
            items.append(Paragraph(f"• {_plain(text)}", s["body"]))
        return items

    # ── Section: Face ─────────────────────────────────────────────────────────

    def _face_section(self, p1, p3, p4, s):
        items = [Paragraph("Face Shape & Balance", s["section"])]
        items.append(Paragraph(
            "This section looks at the overall balance, symmetry, and shape of your face.",
            s["summary"]
        ))
        for text in [p4.face.symmetry, p4.face.alignment,
                     p4.face.proportion, p4.face.swelling]:
            if text:
                items.append(Paragraph(f"• {_plain(text)}", s["body"]))
        return items

    # ── Section: Eyes ─────────────────────────────────────────────────────────

    def _eye_section(self, p3, p4, s):
        items = [Paragraph("Eyes", s["section"])]
        items.append(Paragraph(
            "We looked at signs of tiredness, puffiness, redness, and how open your eyes appear.",
            s["summary"]
        ))
        all_obs = p4.eyes.left + p4.eyes.right
        if all_obs:
            for obs in all_obs:
                items.append(self._finding_para(obs, s))
        else:
            items.append(Paragraph("● No concerns detected.", s["finding_good"]))
        if p4.eyes.openness_note:
            items.append(Paragraph(f"• {_plain(p4.eyes.openness_note)}", s["body"]))
        return items

    # ── Section: Skin ─────────────────────────────────────────────────────────

    def _skin_section(self, p3, p4, s):
        items = [Paragraph("Skin", s["section"])]
        items.append(Paragraph(
            "We checked for signs of breakouts, redness, and uneven texture across "
            "your forehead, cheeks, and nose.",
            s["summary"]
        ))
        all_obs = (p4.skin.forehead + p4.skin.left_cheek +
                   p4.skin.right_cheek + p4.skin.nose + p4.skin.overall)
        if all_obs:
            for obs in all_obs:
                items.append(self._finding_para(obs, s))
        else:
            items.append(Paragraph("● Skin looks clear and even.", s["finding_good"]))
        return items

    # ── Section: Lips ─────────────────────────────────────────────────────────

    def _lip_section(self, p3, p4, s):
        items = [Paragraph("Lips", s["section"])]
        items.append(Paragraph(
            "We checked your lips for dryness, colour unevenness, and paleness.",
            s["summary"]
        ))
        if not p3.lips.available:
            items.append(Paragraph("Could not analyze lips from this image.", s["body"]))
            return items
        if p4.lips.findings:
            for obs in p4.lips.findings:
                items.append(self._finding_para(obs, s))
        else:
            items.append(Paragraph("● Lips look healthy.", s["finding_good"]))
        return items

    # ── Section: Mouth ────────────────────────────────────────────────────────

    def _mouth_section(self, p3, p4, s):
        items = [Paragraph("Mouth & Expression", s["section"])]
        items.append(Paragraph(
            "We looked at the shape and balance of your mouth at rest.",
            s["summary"]
        ))
        if not p3.smile.available:
            items.append(Paragraph("Could not analyze mouth geometry from this image.", s["body"]))
            return items
        if p4.mouth.findings:
            for obs in p4.mouth.findings:
                items.append(self._finding_para(obs, s))
        else:
            items.append(Paragraph("● Mouth shape looks balanced.", s["finding_good"]))
        return items

    # ── Next steps ────────────────────────────────────────────────────────────

    def _next_steps(self, p4, s):
        items = [
            Spacer(1, 4 * mm),
            HRFlowable(width="100%", thickness=1, color=C_LGRAY),
            Spacer(1, 2 * mm),
            Paragraph("What to do next", s["section"]),
        ]

        # Count non-normal observations
        all_obs = (
            p4.eyes.left + p4.eyes.right +
            p4.skin.forehead + p4.skin.left_cheek +
            p4.skin.right_cheek + p4.skin.nose + p4.skin.overall +
            p4.lips.findings + p4.mouth.findings
        )
        non_normal = [o for o in all_obs if not o.is_normal]
        notable    = [o for o in non_normal
                      if o.severity.value.lower() in ("moderate", "notable")]

        if not non_normal:
            items.append(Paragraph(
                "● Everything looks healthy in this analysis. Keep up your current routine "
                "and feel free to run another analysis in a few days to track changes.",
                s["next"]
            ))
        elif notable:
            items.append(Paragraph(
                "● Some noticeable findings were detected. Consider showing this report "
                "to a healthcare professional if any of the highlighted areas concern you.",
                s["next"]
            ))
            items.append(Paragraph(
                "● This analysis is a starting point, not a diagnosis. A professional "
                "can give you a proper assessment.",
                s["next"]
            ))
        else:
            items.append(Paragraph(
                "● A few minor things were noted. These are often caused by tiredness, "
                "lighting, or day-to-day variation. Run another analysis in a few days "
                "to see if they persist.",
                s["next"]
            ))

        items.append(Paragraph(
            "● For the most accurate results, use good lighting and face the camera "
            "directly with a neutral expression.",
            s["next"]
        ))
        return items

    # ── Disclaimer ────────────────────────────────────────────────────────────

    def _disclaimer(self, s):
        return [
            Spacer(1, 5 * mm),
            HRFlowable(width="100%", thickness=0.5, color=C_LGRAY),
            Spacer(1, 2 * mm),
            Paragraph(
                "Important: This report is produced by an automated image analysis system "
                "and is for personal information only. It is not a medical diagnosis or "
                "clinical assessment. Please consult a qualified healthcare professional "
                "if you have concerns about your health.",
                s["disclaimer"]
            ),
        ]

    # ── Footer ────────────────────────────────────────────────────────────────

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(C_GRAY)
        canvas.drawString(self.MARGIN, 9 * mm, "Mira — Your Facial Health Summary")
        canvas.drawRightString(
            self.PAGE_W - self.MARGIN, 9 * mm,
            f"Page {canvas.getPageNumber()}"
        )
        canvas.restoreState()
