"""
A5 — test_observation_engine.py

Tests for Phase 4 (ObservationEngine): the rule layer that turns Phase 3
numeric scores into severity-tagged text. Phase 3 scoring is covered
elsewhere (A6); this file is about the *rules*, not the *measurements* --
exactly the layer where a threshold typo can silently turn "Mild" into
"None" and nothing else would catch it.

No camera, no model file, no network. Phase3Result/EyeFeatures/etc. are
plain dataclasses, constructed directly here.

Covers:
  - _severity() band-edge transitions (none/mild/moderate/notable, both
    sides of every boundary, using the real thresholds.py values)
  - Eye dark-circle / redness / puffiness severity wiring
  - Skin acne / redness / texture severity wiring
  - Lip dryness / color-inconsistency / pallor severity wiring
  - Cross-side eye-openness comparison (slight vs notable vs none)
  - Cross-cheek redness asymmetry comparison
  - Face narrative: symmetry / alignment / proportion / swelling text at
    each threshold boundary
  - Overall confidence formula (no-findings base, per-finding drop, floor)
  - Non-diagnostic language guard
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase4_observation_engine.observation_engine import (
    ObservationEngine, Severity, _severity, FULL_COVERAGE_CONFIDENCE,
)
from phase3_feature_analysis.feature_analyzer import (
    Phase3Result, EyeFeatures, SkinFeatures, LipFeatures, FaceFeatures, SmileFeatures,
)
from thresholds import (
    EYE_DARK_CIRCLE_SEVERITY, EYE_REDNESS_SEVERITY, EYE_PUFFINESS_SEVERITY,
    SKIN_ACNE_SEVERITY, SKIN_REDNESS_SEVERITY, SKIN_TEXTURE_SEVERITY,
    SKIN_NOSE_ACNE_SEVERITY, SKIN_NOSE_REDNESS_SEVERITY, SKIN_NOSE_TEXTURE_SEVERITY,
    LIP_DRYNESS_SEVERITY, LIP_COLOR_INCONSISTENCY_SEVERITY, LIP_PALLOR_SEVERITY,
    FACE_OBSERVATION, MOUTH_OBSERVATION, CROSS_SIDE, CONFIDENCE,
)


# ── helpers to build minimal valid Phase3Results ───────────────────────────────

def _eye(available=True, openness=70.0, redness=0.0, dark_circle=0.0, puffiness=0.0):
    return EyeFeatures(openness_ratio=openness, redness_score=redness,
                        dark_circle_score=dark_circle, puffiness_score=puffiness,
                        available=available)


def _skin(available=True, acne=0.0, redness=0.0, texture=0.0, spots=0, low_light=False):
    return SkinFeatures(acne_score=acne, redness_score=redness,
                         texture_irregularity=texture, spot_count=spots,
                         low_light=low_light, available=available)


def _lips(available=True, dryness=0.0, color_consistency=100.0, pallor=0.0, low_light=False):
    return LipFeatures(dryness_score=dryness, color_consistency=color_consistency,
                        pallor_score=pallor, low_light=low_light, available=available)


def _face(available=True, symmetry_score=95.0, symmetry_label="Highly Symmetric",
           alignment_angle=0.0, aspect_ratio=1.4, swelling_score=0.0,
           swelling_label="No Swelling Indicators"):
    return FaceFeatures(symmetry_score=symmetry_score, symmetry_label=symmetry_label,
                         alignment_angle=alignment_angle, face_width=300, face_height=420,
                         aspect_ratio=aspect_ratio, swelling_score=swelling_score,
                         swelling_label=swelling_label, available=available)


def _smile(available=True, mar=50.0, asymmetry=0.0, curvature=0.0):
    return SmileFeatures(mouth_aspect_ratio=mar, corner_asymmetry=asymmetry,
                          curvature_score=curvature, available=available)


def _quiet_phase3(**overrides):
    """A Phase3Result with all-clean (non-triggering) scores, so individual
    tests can override just the field(s) they're probing without other
    sections injecting unrelated findings."""
    base = dict(
        left_eye=_eye(), right_eye=_eye(),
        skin_forehead=_skin(), skin_left_cheek=_skin(), skin_right_cheek=_skin(),
        skin_nose=_skin(), lips=_lips(), face=_face(), smile=_smile(),
    )
    base.update(overrides)
    return Phase3Result(**base)


ENGINE = ObservationEngine()


class TestSeverityBandFunction(unittest.TestCase):
    """_severity() in isolation, against every real band in thresholds.py."""

    def _check_band(self, band, name):
        eps = 0.01
        with self.subTest(band=name, edge="just below mild"):
            self.assertEqual(_severity(band.mild - eps, band), Severity.NONE)
        with self.subTest(band=name, edge="exactly mild"):
            self.assertEqual(_severity(band.mild, band), Severity.MILD)
        with self.subTest(band=name, edge="just below moderate"):
            self.assertEqual(_severity(band.moderate - eps, band), Severity.MILD)
        with self.subTest(band=name, edge="exactly moderate"):
            self.assertEqual(_severity(band.moderate, band), Severity.MODERATE)
        with self.subTest(band=name, edge="just below notable"):
            self.assertEqual(_severity(band.notable - eps, band), Severity.MODERATE)
        with self.subTest(band=name, edge="exactly notable"):
            self.assertEqual(_severity(band.notable, band), Severity.NOTABLE)
        with self.subTest(band=name, edge="far above notable"):
            self.assertEqual(_severity(band.notable + 20, band), Severity.NOTABLE)
        with self.subTest(band=name, edge="zero"):
            self.assertEqual(_severity(0.0, band), Severity.NONE)

    def test_all_real_bands(self):
        bands = {
            "eye_dark_circle": EYE_DARK_CIRCLE_SEVERITY,
            "eye_redness": EYE_REDNESS_SEVERITY,
            "eye_puffiness": EYE_PUFFINESS_SEVERITY,
            "skin_acne": SKIN_ACNE_SEVERITY,
            "skin_redness": SKIN_REDNESS_SEVERITY,
            "skin_texture": SKIN_TEXTURE_SEVERITY,
            "lip_dryness": LIP_DRYNESS_SEVERITY,
            "lip_color_inconsistency": LIP_COLOR_INCONSISTENCY_SEVERITY,
            "lip_pallor": LIP_PALLOR_SEVERITY,
        }
        for name, band in bands.items():
            self._check_band(band, name)

    def test_threshold_values_pinned(self):
        # The boundary tests above read band.mild/moderate/notable dynamically,
        # so they verify _severity()'s logic but would silently pass even if
        # someone changed a threshold *value* in thresholds.py. Pin the actual
        # numbers here so an accidental value change (typo, bad merge) fails
        # loudly instead of just shifting where "Mild" starts.
        self.assertEqual((EYE_DARK_CIRCLE_SEVERITY.mild, EYE_DARK_CIRCLE_SEVERITY.moderate,
                           EYE_DARK_CIRCLE_SEVERITY.notable), (35, 55, 75))
        self.assertEqual((EYE_REDNESS_SEVERITY.mild, EYE_REDNESS_SEVERITY.moderate,
                           EYE_REDNESS_SEVERITY.notable), (30, 55, 75))
        self.assertEqual((EYE_PUFFINESS_SEVERITY.mild, EYE_PUFFINESS_SEVERITY.moderate,
                           EYE_PUFFINESS_SEVERITY.notable), (35, 60, 78))
        self.assertEqual((SKIN_ACNE_SEVERITY.mild, SKIN_ACNE_SEVERITY.moderate,
                           SKIN_ACNE_SEVERITY.notable), (20, 45, 70))
        self.assertEqual((SKIN_REDNESS_SEVERITY.mild, SKIN_REDNESS_SEVERITY.moderate,
                           SKIN_REDNESS_SEVERITY.notable), (25, 50, 70))
        self.assertEqual((SKIN_TEXTURE_SEVERITY.mild, SKIN_TEXTURE_SEVERITY.moderate,
                           SKIN_TEXTURE_SEVERITY.notable), (30, 55, 75))
        self.assertEqual((LIP_DRYNESS_SEVERITY.mild, LIP_DRYNESS_SEVERITY.moderate,
                           LIP_DRYNESS_SEVERITY.notable), (30, 55, 75))
        self.assertEqual((LIP_COLOR_INCONSISTENCY_SEVERITY.mild, LIP_COLOR_INCONSISTENCY_SEVERITY.moderate,
                           LIP_COLOR_INCONSISTENCY_SEVERITY.notable), (25, 50, 70))
        self.assertEqual((LIP_PALLOR_SEVERITY.mild, LIP_PALLOR_SEVERITY.moderate,
                           LIP_PALLOR_SEVERITY.notable), (35, 60, 80))
        self.assertEqual((CROSS_SIDE.cheek_redness_diff, CROSS_SIDE.eye_openness_notable,
                           CROSS_SIDE.eye_openness_slight), (25.0, 20.0, 10.0))
        self.assertEqual((CONFIDENCE.no_findings_base, CONFIDENCE.findings_base,
                           CONFIDENCE.findings_floor, CONFIDENCE.per_finding_drop),
                          (85.0, 90.0, 60.0, 3.0))


class TestEyeObservations(unittest.TestCase):

    def test_dark_circle_below_mild_is_normal(self):
        p3 = _quiet_phase3(left_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.mild - 1))
        report = ENGINE.generate(p3)
        obs = report.eyes.left[0]
        self.assertTrue(obs.is_normal)
        self.assertEqual(obs.severity, Severity.NONE)
        self.assertIn("no significant dark circles", obs.finding.lower())

    def test_dark_circle_at_mild_threshold_flags(self):
        p3 = _quiet_phase3(left_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.mild))
        report = ENGINE.generate(p3)
        obs = report.eyes.left[0]
        self.assertFalse(obs.is_normal)
        self.assertEqual(obs.severity, Severity.MILD)
        self.assertIn("mild", obs.finding.lower())

    def test_dark_circle_at_notable_threshold(self):
        p3 = _quiet_phase3(right_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.notable))
        report = ENGINE.generate(p3)
        obs = report.eyes.right[0]
        self.assertEqual(obs.severity, Severity.NOTABLE)
        self.assertIn("notable", obs.finding.lower())

    def test_redness_none_vs_flagged(self):
        p3 = _quiet_phase3(left_eye=_eye(redness=EYE_REDNESS_SEVERITY.mild - 1))
        report = ENGINE.generate(p3)
        self.assertTrue(report.eyes.left[1].is_normal)

        p3b = _quiet_phase3(left_eye=_eye(redness=EYE_REDNESS_SEVERITY.moderate))
        report_b = ENGINE.generate(p3b)
        self.assertEqual(report_b.eyes.left[1].severity, Severity.MODERATE)

    def test_puffiness_only_appears_when_nonzero_severity(self):
        # Puffiness obs is appended only if sev != NONE (unlike dark circle/
        # redness which always append a normal-or-flagged item).
        p3_clean = _quiet_phase3(left_eye=_eye(puffiness=0.0))
        report_clean = ENGINE.generate(p3_clean)
        self.assertEqual(len(report_clean.eyes.left), 2)  # dark circle + redness only

        p3_puffy = _quiet_phase3(left_eye=_eye(puffiness=EYE_PUFFINESS_SEVERITY.mild))
        report_puffy = ENGINE.generate(p3_puffy)
        self.assertEqual(len(report_puffy.eyes.left), 3)
        self.assertEqual(report_puffy.eyes.left[2].severity, Severity.MILD)

    def test_unavailable_eye_produces_single_normal_note(self):
        p3 = _quiet_phase3(left_eye=_eye(available=False))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.eyes.left), 1)
        self.assertIn("not available", report.eyes.left[0].finding.lower())


class TestCrossSideEyeOpenness(unittest.TestCase):

    def test_no_note_below_slight_threshold(self):
        p3 = _quiet_phase3(
            left_eye=_eye(openness=70.0),
            right_eye=_eye(openness=70.0 + CROSS_SIDE.eye_openness_slight - 1),
        )
        report = ENGINE.generate(p3)
        self.assertEqual(report.eyes.openness_note, "")

    def test_slight_note_between_slight_and_notable(self):
        p3 = _quiet_phase3(
            left_eye=_eye(openness=70.0),
            right_eye=_eye(openness=70.0 + CROSS_SIDE.eye_openness_slight + 1),
        )
        report = ENGINE.generate(p3)
        self.assertIn("slight difference", report.eyes.openness_note.lower())
        self.assertNotIn("noticeable", report.eyes.openness_note.lower())

    def test_notable_note_above_notable_threshold(self):
        p3 = _quiet_phase3(
            left_eye=_eye(openness=50.0),
            right_eye=_eye(openness=50.0 + CROSS_SIDE.eye_openness_notable + 1),
        )
        report = ENGINE.generate(p3)
        self.assertIn("noticeable difference", report.eyes.openness_note.lower())

    def test_no_note_when_either_side_unavailable(self):
        p3 = _quiet_phase3(left_eye=_eye(available=False), right_eye=_eye(openness=99))
        report = ENGINE.generate(p3)
        self.assertEqual(report.eyes.openness_note, "")


class TestSkinObservations(unittest.TestCase):

    def test_acne_normal_vs_flagged_with_spot_count(self):
        p3 = _quiet_phase3(skin_forehead=_skin(acne=SKIN_ACNE_SEVERITY.mild, spots=3))
        report = ENGINE.generate(p3)
        obs = report.skin.forehead[0]
        self.assertEqual(obs.severity, Severity.MILD)
        self.assertIn("3 area", obs.finding)

    def test_redness_zero_appends_nothing(self):
        # Unlike eye redness, skin redness obs is only appended if sev != NONE.
        p3 = _quiet_phase3(skin_left_cheek=_skin(acne=0.0, redness=0.0, texture=0.0))
        report = ENGINE.generate(p3)
        # Only the "no significant skin spots" normal entry, no redness/texture lines
        self.assertEqual(len(report.skin.left_cheek), 1)

    def test_texture_flagged_at_notable(self):
        p3 = _quiet_phase3(skin_right_cheek=_skin(texture=SKIN_TEXTURE_SEVERITY.notable))
        report = ENGINE.generate(p3)
        texture_obs = [o for o in report.skin.right_cheek if "texture" in o.finding.lower()]
        self.assertEqual(len(texture_obs), 1)
        self.assertEqual(texture_obs[0].severity, Severity.NOTABLE)

    def test_unavailable_skin_region_produces_no_observations(self):
        p3 = _quiet_phase3(skin_forehead=_skin(available=False))
        report = ENGINE.generate(p3)
        self.assertEqual(report.skin.forehead, [])

    def test_low_light_replaces_acne_and_texture_with_single_note(self):
        # Even with scores that would normally trigger notable findings,
        # low_light=True should suppress acne/texture entirely and leave
        # one explanatory note instead — not silently report the (zeroed)
        # scores as "no significant" findings.
        p3 = _quiet_phase3(skin_forehead=_skin(
            acne=SKIN_ACNE_SEVERITY.notable, texture=SKIN_TEXTURE_SEVERITY.notable,
            low_light=True))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.skin.forehead), 1)
        self.assertIn("too low", report.skin.forehead[0].finding.lower())
        self.assertEqual(report.skin.forehead[0].severity, Severity.NONE)

    def test_low_light_does_not_suppress_redness(self):
        p3 = _quiet_phase3(skin_forehead=_skin(
            redness=SKIN_REDNESS_SEVERITY.notable, low_light=True))
        report = ENGINE.generate(p3)
        redness_obs = [o for o in report.skin.forehead if "redness" in o.finding.lower()]
        self.assertEqual(len(redness_obs), 1)
        self.assertEqual(redness_obs[0].severity, Severity.NOTABLE)

    def test_low_light_note_not_counted_as_abnormal_finding(self):
        # Severity.NONE notes shouldn't drag down overall_confidence or be
        # counted among "abnormal" findings in the overall assessment.
        p3 = _quiet_phase3(skin_forehead=_skin(low_light=True))
        report = ENGINE.generate(p3)
        self.assertEqual(report.overall_label, "No Major Visible Abnormalities")


class TestCrossCheekRedness(unittest.TestCase):

    def test_no_asymmetry_flag_below_threshold(self):
        p3 = _quiet_phase3(
            skin_left_cheek=_skin(redness=10.0),
            skin_right_cheek=_skin(redness=10.0 + CROSS_SIDE.cheek_redness_diff - 1),
        )
        report = ENGINE.generate(p3)
        self.assertEqual(report.skin.overall, [])

    def test_asymmetry_flag_above_threshold(self):
        p3 = _quiet_phase3(
            skin_left_cheek=_skin(redness=10.0),
            skin_right_cheek=_skin(redness=10.0 + CROSS_SIDE.cheek_redness_diff + 1),
        )
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.skin.overall), 1)
        self.assertEqual(report.skin.overall[0].severity, Severity.MILD)
        self.assertIn("asymmetric skin redness", report.skin.overall[0].finding.lower())


class TestSkinNoseObservations(unittest.TestCase):
    """Nose reuses the same _analyze_skin pipeline as forehead/cheeks, but is
    wired through its own SeverityBand constants (SKIN_NOSE_*) rather than
    the cheek bands — see thresholds.py for why (oil-prone, smaller surface,
    nostril-shadow noise). The acne/redness bands happen to share the same
    numbers as the cheek bands, but texture is deliberately higher
    (35/60/80 vs 30/55/75), so that's the band most worth pinning here."""

    def test_acne_normal_vs_flagged_with_spot_count(self):
        p3 = _quiet_phase3(skin_nose=_skin(acne=SKIN_NOSE_ACNE_SEVERITY.mild, spots=2))
        report = ENGINE.generate(p3)
        obs = report.skin.nose[0]
        self.assertEqual(obs.severity, Severity.MILD)
        self.assertIn("2 area", obs.finding)

    def test_redness_zero_appends_nothing_extra(self):
        p3 = _quiet_phase3(skin_nose=_skin(acne=0.0, redness=0.0, texture=0.0))
        report = ENGINE.generate(p3)
        # Only the "no significant skin spots" normal entry, no redness/texture lines.
        self.assertEqual(len(report.skin.nose), 1)

    def test_texture_flagged_at_nose_notable_threshold(self):
        p3 = _quiet_phase3(skin_nose=_skin(texture=SKIN_NOSE_TEXTURE_SEVERITY.notable))
        report = ENGINE.generate(p3)
        texture_obs = [o for o in report.skin.nose if "texture" in o.finding.lower()]
        self.assertEqual(len(texture_obs), 1)
        self.assertEqual(texture_obs[0].severity, Severity.NOTABLE)

    def test_nose_texture_band_is_independent_of_cheek_band(self):
        """A score of 77 is NOTABLE under the cheek band (notable=75) but
        only MODERATE under the nose band (notable=80) — pins the fact that
        nose texture is wired to SKIN_NOSE_TEXTURE_SEVERITY, not
        SKIN_TEXTURE_SEVERITY, regardless of how the two happen to compare."""
        score = 77.0
        self.assertEqual(_severity(score, SKIN_TEXTURE_SEVERITY), Severity.NOTABLE)
        self.assertEqual(_severity(score, SKIN_NOSE_TEXTURE_SEVERITY), Severity.MODERATE)

        p3 = _quiet_phase3(skin_nose=_skin(texture=score))
        report = ENGINE.generate(p3)
        texture_obs = [o for o in report.skin.nose if "texture" in o.finding.lower()]
        self.assertEqual(texture_obs[0].severity, Severity.MODERATE)

    def test_unavailable_nose_region_produces_no_observations(self):
        p3 = _quiet_phase3(skin_nose=_skin(available=False))
        report = ENGINE.generate(p3)
        self.assertEqual(report.skin.nose, [])

    def test_low_light_replaces_acne_and_texture_with_single_note(self):
        p3 = _quiet_phase3(skin_nose=_skin(
            acne=SKIN_NOSE_ACNE_SEVERITY.notable, texture=SKIN_NOSE_TEXTURE_SEVERITY.notable,
            low_light=True))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.skin.nose), 1)
        self.assertIn("too low", report.skin.nose[0].finding.lower())
        self.assertEqual(report.skin.nose[0].severity, Severity.NONE)

    def test_low_light_does_not_suppress_redness(self):
        p3 = _quiet_phase3(skin_nose=_skin(
            redness=SKIN_NOSE_REDNESS_SEVERITY.notable, low_light=True))
        report = ENGINE.generate(p3)
        redness_obs = [o for o in report.skin.nose if "redness" in o.finding.lower()]
        self.assertEqual(len(redness_obs), 1)
        self.assertEqual(redness_obs[0].severity, Severity.NOTABLE)

    def test_nose_redness_does_not_feed_cross_cheek_check(self):
        """Cross-side redness comparison is cheek-only; an extreme nose
        redness reading must not trigger or alter report.skin.overall."""
        p3 = _quiet_phase3(
            skin_left_cheek=_skin(redness=10.0),
            skin_right_cheek=_skin(redness=12.0),   # below CROSS_SIDE.cheek_redness_diff
            skin_nose=_skin(redness=95.0),
        )
        report = ENGINE.generate(p3)
        self.assertEqual(report.skin.overall, [])


class TestMouthObservations(unittest.TestCase):
    """_mouth_observations() covers smile-geometry findings: corner-asymmetry
    is severity-banded (a real visible-sign candidate), while curvature is
    purely descriptive (smile vs. neutral vs. frown is an expression, not a
    health finding) and must never affect severity/is_normal regardless of
    magnitude."""

    def test_unavailable_mouth_produces_single_normal_note(self):
        p3 = _quiet_phase3(smile=_smile(available=False))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.mouth.findings), 1)
        self.assertTrue(report.mouth.findings[0].is_normal)
        self.assertIn("not available", report.mouth.findings[0].finding.lower())

    def test_asymmetry_below_slight_is_normal(self):
        p3 = _quiet_phase3(smile=_smile(asymmetry=MOUTH_OBSERVATION.asymmetry_slight - 1))
        report = ENGINE.generate(p3)
        asym_obs = report.mouth.findings[0]
        self.assertTrue(asym_obs.is_normal)
        self.assertIn("level", asym_obs.finding.lower())

    def test_asymmetry_at_slight_boundary_flags_mild(self):
        p3 = _quiet_phase3(smile=_smile(asymmetry=MOUTH_OBSERVATION.asymmetry_slight))
        report = ENGINE.generate(p3)
        self.assertEqual(report.mouth.findings[0].severity, Severity.MILD)

    def test_asymmetry_at_mild_boundary_flags_moderate(self):
        p3 = _quiet_phase3(smile=_smile(asymmetry=MOUTH_OBSERVATION.asymmetry_mild))
        report = ENGINE.generate(p3)
        self.assertEqual(report.mouth.findings[0].severity, Severity.MODERATE)

    def test_asymmetry_at_notable_boundary_flags_notable(self):
        p3 = _quiet_phase3(smile=_smile(asymmetry=MOUTH_OBSERVATION.asymmetry_notable))
        report = ENGINE.generate(p3)
        notable_obs = report.mouth.findings[0]
        self.assertEqual(notable_obs.severity, Severity.NOTABLE)
        self.assertIn("professional review", notable_obs.finding.lower())

    def test_curvature_within_neutral_band_is_descriptive_normal(self):
        p3 = _quiet_phase3(smile=_smile(curvature=MOUTH_OBSERVATION.curvature_neutral_band - 1))
        report = ENGINE.generate(p3)
        curv_obs = report.mouth.findings[1]
        self.assertTrue(curv_obs.is_normal)
        self.assertIn("neutral resting", curv_obs.finding.lower())

    def test_positive_curvature_above_band_reads_as_smile_like(self):
        p3 = _quiet_phase3(smile=_smile(curvature=MOUTH_OBSERVATION.curvature_neutral_band + 10))
        report = ENGINE.generate(p3)
        curv_obs = report.mouth.findings[1]
        self.assertIn("raised", curv_obs.finding.lower())
        self.assertIn("smile-like", curv_obs.finding.lower())

    def test_negative_curvature_below_band_reads_as_lowered(self):
        p3 = _quiet_phase3(smile=_smile(curvature=-(MOUTH_OBSERVATION.curvature_neutral_band + 10)))
        report = ENGINE.generate(p3)
        curv_obs = report.mouth.findings[1]
        self.assertIn("lowered", curv_obs.finding.lower())

    def test_curvature_never_escalates_severity_even_at_extreme_value(self):
        """Curvature is descriptive-only: even at its max magnitude (100),
        it must stay Severity.NONE / is_normal=True, never gate a finding
        the way corner-asymmetry does."""
        for extreme in (100.0, -100.0):
            p3 = _quiet_phase3(smile=_smile(curvature=extreme))
            report = ENGINE.generate(p3)
            curv_obs = report.mouth.findings[1]
            self.assertEqual(curv_obs.severity, Severity.NONE)
            self.assertTrue(curv_obs.is_normal)

    def test_mouth_findings_count_two_entries_when_available(self):
        """One asymmetry finding + one curvature finding, no more, no fewer,
        when smile.available is True."""
        p3 = _quiet_phase3(smile=_smile(asymmetry=5.0, curvature=2.0))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.mouth.findings), 2)


class TestLipObservations(unittest.TestCase):

    def test_dryness_normal_vs_flagged(self):
        p3 = _quiet_phase3(lips=_lips(dryness=LIP_DRYNESS_SEVERITY.mild - 1))
        self.assertTrue(ENGINE.generate(p3).lips.findings[0].is_normal)

        p3b = _quiet_phase3(lips=_lips(dryness=LIP_DRYNESS_SEVERITY.moderate))
        obs = ENGINE.generate(p3b).lips.findings[0]
        self.assertEqual(obs.severity, Severity.MODERATE)

    def test_color_inconsistency_uses_inverted_score(self):
        # color_consistency=100 (perfect) -> inconsistency 0 -> no finding.
        p3 = _quiet_phase3(lips=_lips(color_consistency=100.0))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.lips.findings), 1)  # dryness normal only

        # color_consistency=100-mild_threshold -> inconsistency==mild -> flagged.
        target_inconsistency = LIP_COLOR_INCONSISTENCY_SEVERITY.mild
        p3b = _quiet_phase3(lips=_lips(color_consistency=100.0 - target_inconsistency))
        report_b = ENGINE.generate(p3b)
        color_obs = [o for o in report_b.lips.findings if "unevenness" in o.finding.lower()]
        self.assertEqual(len(color_obs), 1)
        self.assertEqual(color_obs[0].severity, Severity.MILD)

    def test_pallor_zero_appends_nothing_extra(self):
        p3 = _quiet_phase3(lips=_lips(pallor=0.0))
        report = ENGINE.generate(p3)
        pallor_obs = [o for o in report.lips.findings if "pallor" in o.finding.lower()]
        self.assertEqual(pallor_obs, [])

    def test_pallor_flagged_mentions_non_diagnostic_language(self):
        p3 = _quiet_phase3(lips=_lips(pallor=LIP_PALLOR_SEVERITY.notable))
        report = ENGINE.generate(p3)
        pallor_obs = [o for o in report.lips.findings if "pallor" in o.finding.lower()]
        self.assertEqual(len(pallor_obs), 1)
        self.assertIn("possible indicator", pallor_obs[0].finding.lower())

    def test_unavailable_lips(self):
        p3 = _quiet_phase3(lips=_lips(available=False))
        report = ENGINE.generate(p3)
        self.assertEqual(len(report.lips.findings), 1)
        self.assertIn("not available", report.lips.findings[0].finding.lower())

    def test_low_light_replaces_dryness_with_single_note(self):
        p3 = _quiet_phase3(lips=_lips(dryness=LIP_DRYNESS_SEVERITY.notable, low_light=True))
        report = ENGINE.generate(p3)
        dryness_related = [o for o in report.lips.findings
                            if "dry" in o.finding.lower() or "too low" in o.finding.lower()]
        self.assertEqual(len(dryness_related), 1)
        self.assertIn("too low", dryness_related[0].finding.lower())
        self.assertEqual(dryness_related[0].severity, Severity.NONE)

    def test_low_light_does_not_suppress_pallor(self):
        p3 = _quiet_phase3(lips=_lips(pallor=LIP_PALLOR_SEVERITY.notable, low_light=True))
        report = ENGINE.generate(p3)
        pallor_obs = [o for o in report.lips.findings if "pallor" in o.finding.lower()]
        self.assertEqual(len(pallor_obs), 1)
        self.assertEqual(pallor_obs[0].severity, Severity.NOTABLE)


class TestFaceNarrativeSymmetry(unittest.TestCase):

    def test_normal_band(self):
        p3 = _quiet_phase3(face=_face(symmetry_score=FACE_OBSERVATION.symmetry_normal))
        text = ENGINE.generate(p3).face.symmetry
        self.assertIn("appears normal", text.lower())
        self.assertNotIn("observation", text.lower())

    def test_slight_band_just_below_normal(self):
        p3 = _quiet_phase3(face=_face(symmetry_score=FACE_OBSERVATION.symmetry_normal - 1))
        text = ENGINE.generate(p3).face.symmetry
        self.assertIn("slight facial asymmetry", text.lower())

    def test_mild_band(self):
        p3 = _quiet_phase3(face=_face(symmetry_score=FACE_OBSERVATION.symmetry_slight - 1))
        text = ENGINE.generate(p3).face.symmetry
        self.assertIn("mild facial asymmetry", text.lower())

    def test_notable_band_below_mild_floor(self):
        p3 = _quiet_phase3(face=_face(symmetry_score=FACE_OBSERVATION.symmetry_mild - 1))
        text = ENGINE.generate(p3).face.symmetry
        self.assertIn("notable facial asymmetry", text.lower())

    def test_unavailable_face(self):
        p3 = _quiet_phase3(face=_face(available=False))
        report = ENGINE.generate(p3)
        self.assertIn("unavailable", report.face.symmetry.lower())
        self.assertEqual(report.face.alignment, "")


class TestFaceNarrativeAlignment(unittest.TestCase):

    def test_level_below_tilt_level_threshold(self):
        p3 = _quiet_phase3(face=_face(alignment_angle=FACE_OBSERVATION.tilt_level - 0.5))
        text = ENGINE.generate(p3).face.alignment
        self.assertIn("level", text.lower())

    def test_slight_tilt_at_boundary(self):
        p3 = _quiet_phase3(face=_face(alignment_angle=FACE_OBSERVATION.tilt_level))
        text = ENGINE.generate(p3).face.alignment
        self.assertIn("slight head tilt", text.lower())

    def test_notable_tilt_at_slight_boundary(self):
        p3 = _quiet_phase3(face=_face(alignment_angle=FACE_OBSERVATION.tilt_slight))
        text = ENGINE.generate(p3).face.alignment
        self.assertIn("notable head tilt", text.lower())

    def test_negative_angle_uses_absolute_value(self):
        p3 = _quiet_phase3(face=_face(alignment_angle=-(FACE_OBSERVATION.tilt_slight + 2)))
        text = ENGINE.generate(p3).face.alignment
        self.assertIn("notable head tilt", text.lower())


class TestFaceNarrativeProportion(unittest.TestCase):

    def test_within_typical_range(self):
        mid = (FACE_OBSERVATION.aspect_low + FACE_OBSERVATION.aspect_high) / 2
        p3 = _quiet_phase3(face=_face(aspect_ratio=mid))
        text = ENGINE.generate(p3).face.proportion
        self.assertIn("typical range", text.lower())

    def test_at_low_boundary_is_still_typical(self):
        p3 = _quiet_phase3(face=_face(aspect_ratio=FACE_OBSERVATION.aspect_low))
        text = ENGINE.generate(p3).face.proportion
        self.assertIn("typical range", text.lower())

    def test_below_low_boundary_is_wide(self):
        p3 = _quiet_phase3(face=_face(aspect_ratio=FACE_OBSERVATION.aspect_low - 0.01))
        text = ENGINE.generate(p3).face.proportion
        self.assertIn("relatively wide", text.lower())

    def test_above_high_boundary_is_narrow(self):
        p3 = _quiet_phase3(face=_face(aspect_ratio=FACE_OBSERVATION.aspect_high + 0.01))
        text = ENGINE.generate(p3).face.proportion
        self.assertIn("relatively narrow", text.lower())


class TestFaceNarrativeSwelling(unittest.TestCase):

    def test_not_assessed_label_short_circuits_score_bands(self):
        p3 = _quiet_phase3(face=_face(swelling_score=99.0, swelling_label="Not Assessed"))
        text = ENGINE.generate(p3).face.swelling
        self.assertIn("not assessed", text.lower())

    def test_none_band(self):
        p3 = _quiet_phase3(face=_face(swelling_score=FACE_OBSERVATION.swelling_none - 1))
        text = ENGINE.generate(p3).face.swelling
        self.assertIn("no swelling indicators", text.lower())

    def test_mild_band_at_boundary(self):
        p3 = _quiet_phase3(face=_face(swelling_score=FACE_OBSERVATION.swelling_none))
        text = ENGINE.generate(p3).face.swelling
        self.assertIn("mild swelling", text.lower())

    def test_moderate_band_at_boundary(self):
        p3 = _quiet_phase3(face=_face(swelling_score=FACE_OBSERVATION.swelling_mild))
        text = ENGINE.generate(p3).face.swelling
        self.assertIn("moderate swelling", text.lower())

    def test_notable_band_above_moderate_boundary(self):
        p3 = _quiet_phase3(face=_face(swelling_score=FACE_OBSERVATION.swelling_moderate))
        text = ENGINE.generate(p3).face.swelling
        self.assertIn("notable swelling", text.lower())


class TestOverallConfidenceIsCoverageOnly(unittest.TestCase):
    """overall_confidence is data/capture quality ONLY -- it must not move
    just because the number of findings changed, with region availability
    held constant. This is the internal field burst_selection.py uses to
    pick the best-CAPTURED frame; findings must never leak into it (see
    the comment block above FULL_COVERAGE_CONFIDENCE in observation_engine.py)."""

    def test_confidence_unaffected_by_findings_count(self):
        quiet = ENGINE.generate(_quiet_phase3())
        one_finding = ENGINE.generate(
            _quiet_phase3(left_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.mild)))
        self.assertEqual(quiet.overall_confidence, one_finding.overall_confidence)
        self.assertEqual(quiet.overall_confidence, FULL_COVERAGE_CONFIDENCE)


class TestObservationScoreFormula(unittest.TestCase):
    """observation_score is the number actually shown in reports (never
    labelled "Confidence" -- see phase5_report/*.py). It deliberately
    clubs together capture-quality confidence AND the findings-based
    signal that used to live in overall_confidence, as a 50/50 blend."""

    def test_no_findings_uses_base_confidence(self):
        p3 = _quiet_phase3()  # everything clean
        report = ENGINE.generate(p3)
        self.assertEqual(report.overall_label, "No Major Visible Abnormalities")
        expected = round((FULL_COVERAGE_CONFIDENCE + CONFIDENCE.no_findings_base) / 2, 1)
        self.assertEqual(report.observation_score, expected)

    def test_score_drops_per_finding(self):
        # One mild dark-circle finding -> exactly one abnormal observation.
        p3 = _quiet_phase3(left_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.mild))
        report = ENGINE.generate(p3)
        findings_component = CONFIDENCE.findings_base - 1 * CONFIDENCE.per_finding_drop
        expected = round((FULL_COVERAGE_CONFIDENCE + findings_component) / 2, 1)
        self.assertEqual(report.observation_score, expected)
        self.assertEqual(report.overall_label, "Mild Observations Present")

    def test_floor_is_respected(self):
        # Pile on enough notable findings to blow past the findings-side
        # floor and confirm the blended score clamps rather than sliding
        # toward zero.
        p3 = _quiet_phase3(
            left_eye=_eye(dark_circle=99, redness=99, puffiness=99),
            right_eye=_eye(dark_circle=99, redness=99, puffiness=99),
            skin_forehead=_skin(acne=99, redness=99, texture=99),
            skin_left_cheek=_skin(acne=99, redness=99, texture=99),
            skin_right_cheek=_skin(acne=99, redness=99, texture=99),
            lips=_lips(dryness=99, color_consistency=0, pallor=99),
        )
        report = ENGINE.generate(p3)
        floor = round((FULL_COVERAGE_CONFIDENCE + CONFIDENCE.findings_floor) / 2, 1)
        self.assertGreaterEqual(report.observation_score, floor)
        self.assertEqual(report.overall_label, "Notable Observations Present")

    def test_label_priority_notable_over_moderate_over_mild(self):
        # One notable + one mild finding -> label should reflect the worst case.
        p3 = _quiet_phase3(
            left_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.notable),
            skin_forehead=_skin(redness=SKIN_REDNESS_SEVERITY.mild),
        )
        report = ENGINE.generate(p3)
        self.assertEqual(report.overall_label, "Notable Observations Present")

    def test_moderate_only_label(self):
        p3 = _quiet_phase3(left_eye=_eye(redness=EYE_REDNESS_SEVERITY.moderate))
        report = ENGINE.generate(p3)
        self.assertEqual(report.overall_label, "Moderate Observations Present")

    def test_summary_always_ends_with_disclaimer(self):
        p3 = _quiet_phase3()
        report = ENGINE.generate(p3)
        self.assertIn("does not constitute", report.overall[-1].lower())

    def test_finding_counts_in_summary_text(self):
        p3 = _quiet_phase3(
            left_eye=_eye(dark_circle=EYE_DARK_CIRCLE_SEVERITY.notable),
            right_eye=_eye(redness=EYE_REDNESS_SEVERITY.moderate),
            skin_forehead=_skin(acne=SKIN_ACNE_SEVERITY.mild),
        )
        report = ENGINE.generate(p3)
        joined = " ".join(report.overall).lower()
        self.assertIn("1 notable observation", joined)
        self.assertIn("1 moderate observation", joined)
        self.assertIn("1 mild observation", joined)


class TestNonDiagnosticLanguageGuard(unittest.TestCase):
    """Cheap regression guard: scan every generated finding string (not the
    intentional disclaimer) for diagnostic/medical-verdict language."""

    BANNED = ("you have ", "disease", "disorder", "condition is", "diagnosed with")

    def _all_finding_text(self, report):
        obs = (report.eyes.left + report.eyes.right +
               report.skin.forehead + report.skin.left_cheek +
               report.skin.right_cheek + report.skin.overall +
               report.lips.findings)
        return " ".join(o.finding.lower() for o in obs)

    def test_worst_case_findings_stay_non_diagnostic(self):
        p3 = _quiet_phase3(
            left_eye=_eye(dark_circle=99, redness=99, puffiness=99),
            right_eye=_eye(dark_circle=99, redness=99, puffiness=99),
            skin_forehead=_skin(acne=99, redness=99, texture=99),
            skin_left_cheek=_skin(acne=99, redness=99, texture=99),
            skin_right_cheek=_skin(acne=99, redness=99, texture=99),
            lips=_lips(dryness=99, color_consistency=0, pallor=99),
            face=_face(symmetry_score=10, alignment_angle=20, aspect_ratio=2.0,
                       swelling_score=99, swelling_label="Notable Swelling Indicator"),
        )
        report = ENGINE.generate(p3)
        text = self._all_finding_text(report) + " ".join(report.overall).lower()
        for term in self.BANNED:
            self.assertNotIn(term, text, f"possible diagnostic language: {term!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
