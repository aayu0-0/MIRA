"""
Phase 4 — Observation Engine
Converts numerical feature scores from Phase 3 into structured, human-readable observations.
All language is non-diagnostic: it uses "Observation", "Possible indicator", "Visible sign".
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from thresholds import (
    EYE_DARK_CIRCLE_SEVERITY, EYE_REDNESS_SEVERITY, EYE_PUFFINESS_SEVERITY,
    SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY,
    SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY, SKIN_NOSE_TEXTURE_SEVERITY,
    LIP_DRYNESS_SEVERITY, LIP_COLOR_INCONSISTENCY_SEVERITY, LIP_PALLOR_SEVERITY,
    FACE_OBSERVATION, MOUTH_OBSERVATION, CROSS_SIDE,
    SeverityBand, CONFIDENCE,
)


# ─── Severity levels ──────────────────────────────────────────────────────────

class Severity(str, Enum):
    NONE     = "None"
    MILD     = "Mild"
    MODERATE = "Moderate"
    NOTABLE  = "Notable"


# ─── Observation item ─────────────────────────────────────────────────────────

@dataclass
class Observation:
    finding: str                        
    severity: Severity = Severity.NONE
    score: Optional[float] = None       
    category: str = ""                 
    is_normal: bool = False             
    condition: str = ""                 
    location: str = ""                  


# ─── Section reports ─────────────────────────────────────────────────────────

@dataclass
class EyeObservations:
    left: list[Observation]  = field(default_factory=list)
    right: list[Observation] = field(default_factory=list)
    openness_note: str = ""


@dataclass
class SkinObservations:
    forehead: list[Observation]    = field(default_factory=list)
    left_cheek: list[Observation]  = field(default_factory=list)
    right_cheek: list[Observation] = field(default_factory=list)
    nose: list[Observation]        = field(default_factory=list)
    overall: list[Observation]     = field(default_factory=list)


@dataclass
class LipObservations:
    findings: list[Observation] = field(default_factory=list)


@dataclass
class MouthObservations:
    """Mouth-shape geometry observations (separate from LipObservations,
    which covers pixel-based dryness/color/pallor on the lips region)."""
    findings: list[Observation] = field(default_factory=list)


@dataclass
class FaceObservations:
    symmetry: str  = ""
    alignment: str = ""
    proportion: str = ""
    swelling: str = ""


@dataclass
class Phase4Report:
    face:    FaceObservations  = field(default_factory=FaceObservations)
    eyes:    EyeObservations   = field(default_factory=EyeObservations)
    skin:    SkinObservations  = field(default_factory=SkinObservations)
    lips:    LipObservations   = field(default_factory=LipObservations)
    mouth:   MouthObservations = field(default_factory=MouthObservations)
    overall: list[Observation] = field(default_factory=list)
    overall_label: str         = ""
    overall_confidence: float  = 0.0   # 0–100 -- data/capture quality ONLY (see below). Internal use (e.g. burst frame selection); not shown to the user as "Confidence" anymore.
    observation_score: float   = 0.0   # 0–100 -- the number actually shown in reports; blends capture quality with how many/how severe the findings were. See _overall_assessment.


# ─── Data quality (capture/analysis confidence) ───────────────────────────────
#
# `overall_confidence` answers "how much of the face could we actually
# analyze?" -- it is about capture/data quality, not about how many
# abnormal findings turned up. A well-lit, fully-captured face with six
# flagged findings is not "less reliable" than the same face with zero
# findings; conflating the two (confidence dropping per finding) was the
# original bug, and it stays fixed here -- this field is used internally
# by burst_selection.py to pick the best-CAPTURED frame out of a burst,
# and letting "fewer findings" masquerade as "better capture" there would
# make the selector prefer blander frames over sharper/better-lit ones.
#
# What's shown to the user/doctor is a *different* number:
# `observation_score` (below, in _overall_assessment), which deliberately
# DOES blend capture quality with the findings count/severity -- that's
# the number people actually want when they ask "how much weight should I
# put on this report", and it is never labelled "Confidence" in any
# report (see phase5_report/*.py). These constants are kept local to this
# module rather than added to thresholds.py, since that file wasn't part
# of this fix pass -- consider moving them there alongside the other
# *_SEVERITY / CONFIDENCE bands if this module's constants are meant to
# live in one place.

FULL_COVERAGE_CONFIDENCE = 100.0   # confidence if every region was usable
MIN_CONFIDENCE = 15.0              # floor -- never claim zero confidence
INSUFFICIENT_DATA_COVERAGE = 0.5   # below this effective coverage -> insufficient-data path
INSUFFICIENT_DATA_CORE_REGIONS = {"face", "left_eye", "right_eye"}  # missing any of these -> insufficient regardless of overall %


@dataclass
class DataQuality:
    """Tracks how much of the face was actually usable for analysis,
    independent of what (if anything) was found in that usable portion."""
    total_regions: int
    unavailable_regions: list        # region names with no usable capture at all
    low_light_regions: list          # region names usable but degraded by lighting

    @property
    def usable_regions(self) -> int:
        return self.total_regions - len(self.unavailable_regions)

    @property
    def effective_coverage(self) -> float:
        """Fraction of the face usefully analyzed, counting a low-light
        region as half-usable rather than fully usable or fully missing."""
        if self.total_regions == 0:
            return 0.0
        degraded = len(self.unavailable_regions) + 0.5 * len(self.low_light_regions)
        return max(0.0, (self.total_regions - degraded) / self.total_regions)

    @property
    def is_insufficient(self) -> bool:
        core_missing = INSUFFICIENT_DATA_CORE_REGIONS & set(self.unavailable_regions)
        return self.effective_coverage < INSUFFICIENT_DATA_COVERAGE or bool(core_missing)


# ─── Threshold helpers ────────────────────────────────────────────────────────

def _severity(score: float, band: SeverityBand) -> Severity:
    if score >= band.notable:
        return Severity.NOTABLE
    if score >= band.moderate:
        return Severity.MODERATE
    if score >= band.mild:
        return Severity.MILD
    return Severity.NONE


def _obs(finding: str, severity: Severity, score: float, category: str,
          condition: str = "", location: str = "") -> Observation:
    return Observation(finding=finding, severity=severity,
                       score=round(score, 1), category=category,
                       condition=condition, location=location)


def _normal(finding: str, category: str,
            condition: str = "", location: str = "") -> Observation:
    return Observation(finding=finding, severity=Severity.NONE,
                       category=category, is_normal=True,
                       condition=condition, location=location)


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
            phase3.skin_forehead, phase3.skin_left_cheek, phase3.skin_right_cheek,
            phase3.skin_nose)
        report.lips  = self._lip_observations(phase3.lips)
        report.mouth = self._mouth_observations(phase3.smile)

        quality = self._assess_data_quality(phase3)
        report.overall, report.overall_label, report.overall_confidence, report.observation_score = \
            self._overall_assessment(report, quality)

        return report

    # ── data quality ──────────────────────────────────────────────────────────

    def _assess_data_quality(self, phase3) -> DataQuality:
        """Independent pass over the raw Phase 3 regions (not the derived
        Observations) to establish how much of the face was actually
        capturable/analyzable, before any severity banding happens."""
        region_available = {
            "face":        phase3.face.available,
            "left_eye":    phase3.left_eye.available,
            "right_eye":   phase3.right_eye.available,
            "forehead":    phase3.skin_forehead.available,
            "left_cheek":  phase3.skin_left_cheek.available,
            "right_cheek": phase3.skin_right_cheek.available,
            "nose":        phase3.skin_nose.available,
            "lips":        phase3.lips.available,
            "mouth":       phase3.smile.available,
        }
        unavailable = [name for name, ok in region_available.items() if not ok]

        # low_light only applies to regions whose scores get suppressed for
        # it upstream (skin regions + lips); only counted if the region was
        # otherwise available, to avoid double-counting against unavailable.
        low_light_candidates = (
            ("forehead", phase3.skin_forehead),
            ("left_cheek", phase3.skin_left_cheek),
            ("right_cheek", phase3.skin_right_cheek),
            ("nose", phase3.skin_nose),
            ("lips", phase3.lips),
        )
        low_light = [
            name for name, region in low_light_candidates
            if region.available and getattr(region, "low_light", False)
        ]

        return DataQuality(
            total_regions=len(region_available),
            unavailable_regions=unavailable,
            low_light_regions=low_light,
        )

    # ── face ──────────────────────────────────────────────────────────────────

    def _face_observations(self, face) -> FaceObservations:
        fo = FaceObservations()

        if not face.available:
            fo.symmetry = "Face features unavailable."
            return fo

        # Symmetry
        s = face.symmetry_score
        if s >= FACE_OBSERVATION.symmetry_normal:
            fo.symmetry = f"Face symmetry appears normal ({face.symmetry_label}, score {s:.0f}/100)."
        elif s >= FACE_OBSERVATION.symmetry_slight:
            fo.symmetry = (f"Observation: Slight facial asymmetry noted "
                           f"({face.symmetry_label}, score {s:.0f}/100). "
                           f"Minor variations are common and typically insignificant.")
        elif s >= FACE_OBSERVATION.symmetry_mild:
            fo.symmetry = (f"Observation: Mild facial asymmetry detected "
                           f"({face.symmetry_label}, score {s:.0f}/100). "
                           f"Further observation may be warranted.")
        else:
            fo.symmetry = (f"Observation: Notable facial asymmetry observed "
                           f"({face.symmetry_label}, score {s:.0f}/100). "
                           f"This is a visible sign that may merit professional review.")

        # Head tilt / alignment
        angle = abs(face.alignment_angle)
        if angle < FACE_OBSERVATION.tilt_level:
            fo.alignment = "Head alignment appears level."
        elif angle < FACE_OBSERVATION.tilt_slight:
            fo.alignment = f"Slight head tilt observed ({face.alignment_angle:.1f}°). May be positional."
        else:
            fo.alignment = f"Notable head tilt observed ({face.alignment_angle:.1f}°)."

        # Proportions (aspect ratio)
        ar = face.aspect_ratio
        if FACE_OBSERVATION.aspect_low <= ar <= FACE_OBSERVATION.aspect_high:
            fo.proportion = "Facial proportions appear within typical range."
        elif ar < FACE_OBSERVATION.aspect_low:
            fo.proportion = "Observation: Face appears relatively wide for its height."
        else:
            fo.proportion = "Observation: Face appears relatively narrow for its height."

        # Swelling (cheek-width to eye-span proportion heuristic)
        sw = face.swelling_score
        if face.swelling_label == "Not Assessed":
            fo.swelling = "Swelling indicator not assessed (landmarks unavailable)."
        elif sw < FACE_OBSERVATION.swelling_none:
            fo.swelling = "No swelling indicators detected based on facial proportions."
        elif sw < FACE_OBSERVATION.swelling_mild:
            fo.swelling = (f"Observation: Mild swelling indicator noted ({face.swelling_label}, "
                            f"score {sw:.0f}/100). This may reflect normal face shape rather than "
                            f"swelling and is not a diagnosis.")
        elif sw < FACE_OBSERVATION.swelling_moderate:
            fo.swelling = (f"Observation: Moderate swelling indicator noted ({face.swelling_label}, "
                            f"score {sw:.0f}/100). This is a visible proportion-based signal, not a "
                            f"diagnosis, and may merit professional review if accompanied by other "
                            f"symptoms.")
        else:
            fo.swelling = (f"Observation: Notable swelling indicator noted ({face.swelling_label}, "
                            f"score {sw:.0f}/100). This is a visible sign that may merit professional "
                            f"review, particularly if it is a recent change rather than your usual "
                            f"face shape.")

        return fo

    # ── eyes ──────────────────────────────────────────────────────────────────

    def _eye_observations(self, left, right) -> EyeObservations:
        eo = EyeObservations()

        def _eye_obs_list(ef, side: str) -> list:
            obs = []
            if not ef.available:
                obs.append(_normal(f"{side} eye region not available.", "eye",
                                   condition="region_unavailable", location=f"{side.lower()}_eye"))
                return obs

            # Dark circles
            sev = _severity(ef.dark_circle_score, EYE_DARK_CIRCLE_SEVERITY)
            if sev == Severity.NONE:
                obs.append(_normal(f"No significant dark circles observed under {side.lower()} eye.", "eye",
                                   condition="dark_circles", location=f"{side.lower()}_eye"))
            else:
                obs.append(_obs(
                    f"Possible indicator: {sev.value} dark circles visible under {side.lower()} eye.",
                    sev, ef.dark_circle_score, "eye",
                    condition="dark_circles", location=f"{side.lower()}_eye"
                ))

            # Redness
            sev = _severity(ef.redness_score, EYE_REDNESS_SEVERITY)
            if sev == Severity.NONE:
                obs.append(_normal(f"No visible redness in {side.lower()} eye.", "eye",
                                   condition="redness", location=f"{side.lower()}_eye"))
            else:
                obs.append(_obs(
                    f"Visible sign: {sev.value} redness observed in {side.lower()} eye region.",
                    sev, ef.redness_score, "eye",
                    condition="redness", location=f"{side.lower()}_eye"
                ))

            # Puffiness
            sev = _severity(ef.puffiness_score, EYE_PUFFINESS_SEVERITY)
            if sev != Severity.NONE:
                obs.append(_obs(
                    f"Observation: {sev.value} puffiness around {side.lower()} eye.",
                    sev, ef.puffiness_score, "eye",
                    condition="puffiness", location=f"{side.lower()}_eye"
                ))

            return obs

        eo.left  = _eye_obs_list(left, "Left")
        eo.right = _eye_obs_list(right, "Right")

        # Openness comparison
        # `available` only means the region/crop was usable -- openness_ratio
        # can still be None if the EAR landmarks specifically were missing
        # (see feature_analyzer._compute_ear), so gate on that too.
        if (left.available and right.available
                and left.openness_ratio is not None
                and right.openness_ratio is not None):
            diff = abs(left.openness_ratio - right.openness_ratio)
            if diff > CROSS_SIDE.eye_openness_notable:
                eo.openness_note = (
                    f"Observation: Noticeable difference in eye openness between left "
                    f"({left.openness_ratio:.0f}/100) and right ({right.openness_ratio:.0f}/100). "
                    f"May indicate asymmetric muscle tone or positional artifact."
                )
            elif diff > CROSS_SIDE.eye_openness_slight:
                eo.openness_note = (
                    f"Slight difference in eye openness noted "
                    f"(left {left.openness_ratio:.0f}/100, right {right.openness_ratio:.0f}/100)."
                )

        return eo

    # ── skin ──────────────────────────────────────────────────────────────────

    def _skin_observations(self, forehead, left_cheek, right_cheek, nose) -> SkinObservations:
        so = SkinObservations()

        def _skin_list(sf, location: str, acne_band: SeverityBand,
                       redness_band: SeverityBand, texture_band: SeverityBand) -> list:
            obs = []
            if not sf.available:
                return obs

            # Acne / spots and texture — suppressed when the region was too
            # dark for these noise-sensitive scores to be reliable (see
            # thresholds.REGION_LIGHTING / FeatureAnalyzer._analyze_skin).
            if sf.low_light:
                obs.append(Observation(
                    finding=(f"Lighting on {location} was too low to reliably assess texture "
                             f"or spot detail in this region; those readings are not included here."),
                    severity=Severity.NONE, category="skin", is_normal=False,
                    condition="lighting_insufficient", location=location.replace(" ", "_")
                ))
            else:
                # Acne / spots
                sev = _severity(sf.acne_score, acne_band)
                if sev == Severity.NONE:
                    obs.append(_normal(f"No significant skin spots detected on {location}.", "skin",
                                       condition="acne", location=location.replace(" ", "_")))
                else:
                    obs.append(_obs(
                        f"Possible indicator: {sev.value} skin irregularities / blemishes on {location} "
                        f"({sf.spot_count} area(s) detected).",
                        sev, sf.acne_score, "skin",
                        condition="acne", location=location.replace(" ", "_")
                    ))

                # Texture
                sev = _severity(sf.texture_irregularity, texture_band)
                if sev != Severity.NONE:
                    obs.append(_obs(
                        f"Observation: {sev.value} skin texture irregularity on {location}.",
                        sev, sf.texture_irregularity, "skin",
                        condition="texture_irregularity", location=location.replace(" ", "_")
                    ))

            # Redness — mean-color based, not gated by the lighting check
            sev = _severity(sf.redness_score, redness_band)
            if sev != Severity.NONE:
                obs.append(_obs(
                    f"Observation: {sev.value} skin redness on {location}.",
                    sev, sf.redness_score, "skin",
                    condition="redness", location=location.replace(" ", "_")
                ))

            return obs

        so.forehead    = _skin_list(forehead,   "forehead",
                                     SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY)
        so.left_cheek  = _skin_list(left_cheek, "left cheek",
                                     SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY)
        so.right_cheek = _skin_list(right_cheek,"right cheek",
                                     SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY)
        so.nose        = _skin_list(nose,       "nose",
                                     SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY,
                                     SKIN_NOSE_TEXTURE_SEVERITY)

        # Cross-cheek redness comparison
        if left_cheek.available and right_cheek.available:
            diff = abs(left_cheek.redness_score - right_cheek.redness_score)
            if diff > CROSS_SIDE.cheek_redness_diff:
                so.overall.append(_obs(
                    "Observation: Asymmetric skin redness between left and right cheeks.",
                    Severity.MILD, diff, "skin",
                    condition="redness_asymmetry", location="cheeks"
                ))

        return so

    # ── lips ──────────────────────────────────────────────────────────────────

    def _lip_observations(self, lf) -> LipObservations:
        lo = LipObservations()
        if not lf.available:
            lo.findings.append(_normal("Lip region not available.", "lips",
                                       condition="region_unavailable", location="lips"))
            return lo

        # Dryness — suppressed when the region was too dark for this
        # noise-sensitive score to be reliable (see thresholds.REGION_LIGHTING).
        if lf.low_light:
            lo.findings.append(Observation(
                finding=("Lighting on the lips was too low to reliably assess dryness; "
                         "that reading is not included here."),
                severity=Severity.NONE, category="lips", is_normal=False,
                condition="lighting_insufficient", location="lips"
            ))
        else:
            sev = _severity(lf.dryness_score, LIP_DRYNESS_SEVERITY)
            if sev == Severity.NONE:
                lo.findings.append(_normal("Lips appear normally moist with no visible dryness.", "lips",
                                           condition="dryness", location="lips"))
            else:
                lo.findings.append(_obs(
                    f"Possible indicator: {sev.value} lip dryness observed.",
                    sev, lf.dryness_score, "lips",
                    condition="dryness", location="lips"
                ))

        # Color consistency
        sev_incons = _severity(100 - lf.color_consistency, LIP_COLOR_INCONSISTENCY_SEVERITY)
        if sev_incons != Severity.NONE:
            lo.findings.append(_obs(
                f"Observation: {sev_incons.value} unevenness in lip color distribution.",
                sev_incons, 100 - lf.color_consistency, "lips",
                condition="color_inconsistency", location="lips"
            ))

        # Pallor
        sev = _severity(lf.pallor_score, LIP_PALLOR_SEVERITY)
        if sev != Severity.NONE:
            lo.findings.append(_obs(
                f"Visible sign: {sev.value} lip pallor (reduced color saturation) observed. "
                f"Possible indicator of reduced circulation in lip area.",
                sev, lf.pallor_score, "lips",
                condition="pallor", location="lips"
            ))

        return lo

    # ── mouth shape (smile geometry) ─────────────────────────────────────────

    def _mouth_observations(self, smile) -> MouthObservations:
        mo = MouthObservations()
        if not smile.available:
            mo.findings.append(_normal("Mouth shape not available.", "mouth",
                                       condition="region_unavailable", location="mouth"))
            return mo

        # Corner asymmetry
        asym = smile.corner_asymmetry
        if asym < MOUTH_OBSERVATION.asymmetry_slight:
            mo.findings.append(_normal(
                "Mouth corners appear level with no notable height difference.", "mouth",
                condition="corner_asymmetry", location="mouth"))
        elif asym < MOUTH_OBSERVATION.asymmetry_mild:
            mo.findings.append(_obs(
                "Slight difference in mouth corner height noted between left and right.",
                Severity.MILD, asym, "mouth",
                condition="corner_asymmetry", location="mouth"
            ))
        elif asym < MOUTH_OBSERVATION.asymmetry_notable:
            mo.findings.append(_obs(
                "Observation: Moderate difference in mouth corner height between left and right. "
                "May reflect natural asymmetry or positional artifact.",
                Severity.MODERATE, asym, "mouth",
                condition="corner_asymmetry", location="mouth"
            ))
        else:
            mo.findings.append(_obs(
                "Observation: Notable difference in mouth corner height between left and right. "
                "This is a visible sign that may merit professional review, particularly if "
                "it is a recent change.",
                Severity.NOTABLE, asym, "mouth",
                condition="corner_asymmetry", location="mouth"
            ))

        # Curvature (descriptive only — not banded as a severity/abnormality,
        # since smiling vs. neutral vs. frowning is an expression, not a
        # health finding by itself).
        curv = smile.curvature_score
        if abs(curv) < MOUTH_OBSERVATION.curvature_neutral_band:
            mo.findings.append(_normal("Mouth appears in a neutral resting position.", "mouth",
                                       condition="curvature", location="mouth"))
        elif curv > 0:
            mo.findings.append(Observation(
                finding=f"Mouth corners appear raised (smile-like curvature, score {curv:.0f}).",
                severity=Severity.NONE, score=round(curv, 1), category="mouth", is_normal=True,
                condition="curvature", location="mouth"
            ))
        else:
            mo.findings.append(Observation(
                finding=f"Mouth corners appear lowered relative to resting position "
                        f"(score {curv:.0f}).",
                severity=Severity.NONE, score=round(curv, 1), category="mouth", is_normal=True,
                condition="curvature", location="mouth"
            ))

        return mo

    # ── overall assessment ────────────────────────────────────────────────────

    def _overall_assessment(self, report: Phase4Report, quality: DataQuality):
        all_obs = (
            report.eyes.left + report.eyes.right +
            report.skin.forehead + report.skin.left_cheek +
            report.skin.right_cheek + report.skin.nose + report.skin.overall +
            report.lips.findings + report.mouth.findings
        )

        abnormal = [o for o in all_obs if not o.is_normal and o.severity != Severity.NONE]

        notable  = [o for o in abnormal if o.severity == Severity.NOTABLE]
        moderate = [o for o in abnormal if o.severity == Severity.MODERATE]
        mild     = [o for o in abnormal if o.severity == Severity.MILD]

        summary_items = []

        # Confidence reflects data/capture quality only -- how much of the
        # face was actually usable -- never the number or severity of
        # findings. See DataQuality / the constants block above. This is
        # NOT shown to the user as "Confidence"; it's kept for internal
        # use (burst frame selection) and as one input to observation_score.
        confidence = max(MIN_CONFIDENCE, quality.effective_coverage * FULL_COVERAGE_CONFIDENCE)

        # observation_score is the number actually surfaced in reports.
        # It clubs together the two signals that used to compete for the
        # same field: how much of the face we could analyze (confidence,
        # above) AND how many/how severe the findings were (findings_component,
        # below) -- a thorough scan with several findings and a rushed scan
        # with none should both land somewhere in the middle, not at the
        # two opposite extremes a single-signal score would put them at.
        if not abnormal:
            findings_component = CONFIDENCE.no_findings_base
        else:
            findings_component = max(
                CONFIDENCE.findings_floor,
                CONFIDENCE.findings_base - len(abnormal) * CONFIDENCE.per_finding_drop,
            )
        observation_score = round((confidence + findings_component) / 2, 1)

        if not abnormal and quality.is_insufficient:
            # "No findings" here is ambiguous: it can mean nothing was
            # detected, or that there wasn't enough usable data to detect
            # anything. With insufficient coverage we must not report this
            # as a clean bill of health -- that would be a false healthy
            # conclusion drawn from missing data rather than an assessment.
            missing_note = ""
            if quality.unavailable_regions:
                missing_note += f" Unavailable: {', '.join(quality.unavailable_regions)}."
            if quality.low_light_regions:
                missing_note += f" Low light affected: {', '.join(quality.low_light_regions)}."
            summary_items.append(
                "Insufficient usable data to reach a reliable assessment "
                f"({quality.usable_regions}/{quality.total_regions} facial regions analyzable)."
                f"{missing_note}"
            )
            label = "Insufficient Data — Assessment Incomplete"

        elif not abnormal:
            summary_items.append(
                "No significant visible abnormalities detected across analyzed facial regions."
            )
            label = "No Major Visible Abnormalities"

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

            if quality.is_insufficient:
                summary_items.append(
                    "Note: Some facial regions could not be fully analyzed "
                    f"({quality.usable_regions}/{quality.total_regions} usable); "
                    "the findings above reflect only the regions that were assessed."
                )

        summary_items.append(
            "Note: This report contains visual observations only and does not constitute "
            "a medical diagnosis. Please consult a qualified healthcare professional for "
            "any health concerns."
        )

        return summary_items, label, round(confidence, 1), observation_score