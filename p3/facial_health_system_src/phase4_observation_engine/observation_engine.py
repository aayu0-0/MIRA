"""
Phase 4 — Observation Engine
Converts numerical feature scores from Phase 3 into structured, human-readable observations.
All language is non-diagnostic: it uses "Observation", "Possible indicator", "Visible sign".
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ─── Severity levels ──────────────────────────────────────────────────────────

class Severity(str, Enum):
    NONE     = "None"
    MILD     = "Mild"
    MODERATE = "Moderate"
    NOTABLE  = "Notable"


# ─── Observation item ─────────────────────────────────────────────────────────

@dataclass
class Observation:
    finding: str                        # human-readable text
    severity: Severity = Severity.NONE
    score: Optional[float] = None       # raw score that triggered this
    category: str = ""                  # eye / skin / lips / face
    is_normal: bool = False             # True if "no abnormality found"


# ─── Section reports ─────────────────────────────────────────────────────────

@dataclass
class EyeObservations:
    left: list  = field(default_factory=list)
    right: list = field(default_factory=list)
    openness_note: str = ""


@dataclass
class SkinObservations:
    forehead: list    = field(default_factory=list)
    left_cheek: list  = field(default_factory=list)
    right_cheek: list = field(default_factory=list)
    overall: list     = field(default_factory=list)


@dataclass
class LipObservations:
    findings: list = field(default_factory=list)


@dataclass
class FaceObservations:
    symmetry: str  = ""
    alignment: str = ""
    proportion: str = ""


@dataclass
class Phase4Report:
    face:    FaceObservations  = field(default_factory=FaceObservations)
    eyes:    EyeObservations   = field(default_factory=EyeObservations)
    skin:    SkinObservations  = field(default_factory=SkinObservations)
    lips:    LipObservations   = field(default_factory=LipObservations)
    overall: list              = field(default_factory=list)
    overall_label: str         = ""
    overall_confidence: float  = 0.0   # 0–100


# ─── Threshold helpers ────────────────────────────────────────────────────────

def _severity(score: float,
              mild_thresh: float   = 30,
              mod_thresh: float    = 55,
              notable_thresh: float= 75) -> Severity:
    if score >= notable_thresh:
        return Severity.NOTABLE
    if score >= mod_thresh:
        return Severity.MODERATE
    if score >= mild_thresh:
        return Severity.MILD
    return Severity.NONE


def _obs(finding: str, severity: Severity, score: float, category: str) -> Observation:
    return Observation(finding=finding, severity=severity,
                       score=round(score, 1), category=category)


def _normal(finding: str, category: str) -> Observation:
    return Observation(finding=finding, severity=Severity.NONE,
                       category=category, is_normal=True)


# ─── Observation Engine ───────────────────────────────────────────────────────

class ObservationEngine:
    """
    Rule-based engine: feature scores → structured Phase4Report.
    Rules are explicit and auditable — no black-box ML in this phase.
    """

    def generate(self, phase3) -> Phase4Report:  # phase3: Phase3Result
        report = Phase4Report()

        report.face  = self._face_observations(phase3.face)
        report.eyes  = self._eye_observations(phase3.left_eye, phase3.right_eye)
        report.skin  = self._skin_observations(
            phase3.skin_forehead, phase3.skin_left_cheek, phase3.skin_right_cheek)
        report.lips  = self._lip_observations(phase3.lips)
        report.overall, report.overall_label, report.overall_confidence = \
            self._overall_assessment(report)

        return report

    # ── face ──────────────────────────────────────────────────────────────────

    def _face_observations(self, face) -> FaceObservations:
        fo = FaceObservations()

        if not face.available:
            fo.symmetry = "Face features unavailable."
            return fo

        # Symmetry
        s = face.symmetry_score
        if s >= 90:
            fo.symmetry = f"Face symmetry appears normal ({face.symmetry_label}, score {s:.0f}/100)."
        elif s >= 75:
            fo.symmetry = (f"Observation: Slight facial asymmetry noted "
                           f"({face.symmetry_label}, score {s:.0f}/100). "
                           f"Minor variations are common and typically insignificant.")
        elif s >= 55:
            fo.symmetry = (f"Observation: Mild facial asymmetry detected "
                           f"({face.symmetry_label}, score {s:.0f}/100). "
                           f"Further observation may be warranted.")
        else:
            fo.symmetry = (f"Observation: Notable facial asymmetry observed "
                           f"({face.symmetry_label}, score {s:.0f}/100). "
                           f"This is a visible sign that may merit professional review.")

        # Head tilt / alignment
        angle = abs(face.alignment_angle)
        if angle < 3:
            fo.alignment = "Head alignment appears level."
        elif angle < 8:
            fo.alignment = f"Slight head tilt observed ({face.alignment_angle:.1f}°). May be positional."
        else:
            fo.alignment = f"Notable head tilt observed ({face.alignment_angle:.1f}°)."

        # Proportions (aspect ratio)
        ar = face.aspect_ratio
        if 1.2 <= ar <= 1.6:
            fo.proportion = "Facial proportions appear within typical range."
        elif ar < 1.2:
            fo.proportion = "Observation: Face appears relatively wide for its height."
        else:
            fo.proportion = "Observation: Face appears relatively narrow for its height."

        return fo

    # ── eyes ──────────────────────────────────────────────────────────────────

    def _eye_observations(self, left, right) -> EyeObservations:
        eo = EyeObservations()

        def _eye_obs_list(ef, side: str) -> list:
            obs = []
            if not ef.available:
                obs.append(_normal(f"{side} eye region not available.", "eye"))
                return obs

            # Dark circles
            sev = _severity(ef.dark_circle_score, 35, 55, 75)
            if sev == Severity.NONE:
                obs.append(_normal(f"No significant dark circles observed under {side.lower()} eye.", "eye"))
            else:
                obs.append(_obs(
                    f"Possible indicator: {sev.value} dark circles visible under {side.lower()} eye.",
                    sev, ef.dark_circle_score, "eye"
                ))

            # Redness
            sev = _severity(ef.redness_score, 30, 55, 75)
            if sev == Severity.NONE:
                obs.append(_normal(f"No visible redness in {side.lower()} eye.", "eye"))
            else:
                obs.append(_obs(
                    f"Visible sign: {sev.value} redness observed in {side.lower()} eye region.",
                    sev, ef.redness_score, "eye"
                ))

            # Puffiness
            sev = _severity(ef.puffiness_score, 35, 60, 78)
            if sev != Severity.NONE:
                obs.append(_obs(
                    f"Observation: {sev.value} puffiness around {side.lower()} eye.",
                    sev, ef.puffiness_score, "eye"
                ))

            return obs

        eo.left  = _eye_obs_list(left, "Left")
        eo.right = _eye_obs_list(right, "Right")

        # Openness comparison
        if left.available and right.available:
            diff = abs(left.openness_ratio - right.openness_ratio)
            if diff > 20:
                eo.openness_note = (
                    f"Observation: Noticeable difference in eye openness between left "
                    f"({left.openness_ratio:.0f}/100) and right ({right.openness_ratio:.0f}/100). "
                    f"May indicate asymmetric muscle tone or positional artifact."
                )
            elif diff > 10:
                eo.openness_note = (
                    f"Slight difference in eye openness noted "
                    f"(left {left.openness_ratio:.0f}/100, right {right.openness_ratio:.0f}/100)."
                )

        return eo

    # ── skin ──────────────────────────────────────────────────────────────────

    def _skin_observations(self, forehead, left_cheek, right_cheek) -> SkinObservations:
        so = SkinObservations()

        def _skin_list(sf, location: str) -> list:
            obs = []
            if not sf.available:
                return obs

            # Acne / spots
            sev = _severity(sf.acne_score, 20, 45, 70)
            if sev == Severity.NONE:
                obs.append(_normal(f"No significant skin spots detected on {location}.", "skin"))
            else:
                obs.append(_obs(
                    f"Possible indicator: {sev.value} skin irregularities / blemishes on {location} "
                    f"({sf.spot_count} area(s) detected).",
                    sev, sf.acne_score, "skin"
                ))

            # Redness
            sev = _severity(sf.redness_score, 25, 50, 70)
            if sev != Severity.NONE:
                obs.append(_obs(
                    f"Observation: {sev.value} skin redness on {location}.",
                    sev, sf.redness_score, "skin"
                ))

            # Texture
            sev = _severity(sf.texture_irregularity, 30, 55, 75)
            if sev != Severity.NONE:
                obs.append(_obs(
                    f"Observation: {sev.value} skin texture irregularity on {location}.",
                    sev, sf.texture_irregularity, "skin"
                ))

            return obs

        so.forehead    = _skin_list(forehead,   "forehead")
        so.left_cheek  = _skin_list(left_cheek, "left cheek")
        so.right_cheek = _skin_list(right_cheek,"right cheek")

        # Cross-cheek redness comparison
        if left_cheek.available and right_cheek.available:
            diff = abs(left_cheek.redness_score - right_cheek.redness_score)
            if diff > 25:
                so.overall.append(_obs(
                    "Observation: Asymmetric skin redness between left and right cheeks.",
                    Severity.MILD, diff, "skin"
                ))

        return so

    # ── lips ──────────────────────────────────────────────────────────────────

    def _lip_observations(self, lf) -> LipObservations:
        lo = LipObservations()
        if not lf.available:
            lo.findings.append(_normal("Lip region not available.", "lips"))
            return lo

        # Dryness
        sev = _severity(lf.dryness_score, 30, 55, 75)
        if sev == Severity.NONE:
            lo.findings.append(_normal("Lips appear normally moist with no visible dryness.", "lips"))
        else:
            lo.findings.append(_obs(
                f"Possible indicator: {sev.value} lip dryness observed.",
                sev, lf.dryness_score, "lips"
            ))

        # Color consistency
        sev_incons = _severity(100 - lf.color_consistency, 25, 50, 70)
        if sev_incons != Severity.NONE:
            lo.findings.append(_obs(
                f"Observation: {sev_incons.value} unevenness in lip color distribution.",
                sev_incons, 100 - lf.color_consistency, "lips"
            ))

        # Pallor
        sev = _severity(lf.pallor_score, 35, 60, 80)
        if sev != Severity.NONE:
            lo.findings.append(_obs(
                f"Visible sign: {sev.value} lip pallor (reduced color saturation) observed. "
                f"Possible indicator of reduced circulation in lip area.",
                sev, lf.pallor_score, "lips"
            ))

        return lo

    # ── overall assessment ────────────────────────────────────────────────────

    def _overall_assessment(self, report: Phase4Report):
        all_obs = (
            report.eyes.left + report.eyes.right +
            report.skin.forehead + report.skin.left_cheek +
            report.skin.right_cheek + report.skin.overall +
            report.lips.findings
        )

        abnormal = [o for o in all_obs if not o.is_normal and o.severity != Severity.NONE]

        notable  = [o for o in abnormal if o.severity == Severity.NOTABLE]
        moderate = [o for o in abnormal if o.severity == Severity.MODERATE]
        mild     = [o for o in abnormal if o.severity == Severity.MILD]

        summary_items = []

        if not abnormal:
            summary_items.append(
                "No significant visible abnormalities detected across analyzed facial regions."
            )
            label = "No Major Visible Abnormalities"
            confidence = 85.0
        else:
            if notable:
                summary_items.append(
                    f"{len(notable)} notable observation(s) detected requiring attention."
                )
            if moderate:
                summary_items.append(
                    f"{len(moderate)} moderate observation(s) identified."
                )
            if mild:
                summary_items.append(
                    f"{len(mild)} mild observation(s) noted."
                )

            if notable:
                label = "Notable Observations Present"
            elif moderate:
                label = "Moderate Observations Present"
            else:
                label = "Mild Observations Present"

            confidence = max(60.0, 90.0 - len(abnormal) * 3)

        summary_items.append(
            "Note: This report contains visual observations only and does not constitute "
            "a medical diagnosis. Please consult a qualified healthcare professional for "
            "any health concerns."
        )

        return summary_items, label, round(confidence, 1)
