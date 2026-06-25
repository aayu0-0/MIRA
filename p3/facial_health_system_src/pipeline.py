"""
Facial Health Assessment System — Main Pipeline
Orchestrates all 5 phases for a single image.
"""

import sys
import os
import json
import datetime
import cv2
import numpy as np

# ── make imports work from any working directory ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from phase1_face_understanding.face_detector      import FaceDetector
from phase2_region_extraction.region_extractor    import RegionExtractor
from phase3_feature_analysis.feature_analyzer     import FeatureAnalyzer
from phase4_observation_engine.observation_engine import ObservationEngine
from phase5_report.report_generator               import ReportGenerator


def run_pipeline(image_path: str,
                 output_dir: str = "./output",
                 patient_id: str = "DEMO-001") -> dict:
    """
    Full pipeline: image → PDF report + JSON summary + annotated image.
    Returns a summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Cannot load image: {image_path}"}

    print(f"[Pipeline] Image loaded: {image.shape[1]}×{image.shape[0]}px")

    # ── Phase 1: Face Understanding ───────────────────────────────────────────
    print("[Phase 1] Face detection & landmark analysis...")
    detector = FaceDetector()
    p1 = detector.analyze(image)

    if not p1.success:
        return {"success": False, "error": f"Phase 1 failed: {p1.error}"}

    print(f"  ✓ Face detected | Symmetry: {p1.symmetry_score:.1f}% ({p1.symmetry_label})")
    print(f"  ✓ Alignment angle: {p1.alignment_angle:.1f}°")

    # Save Phase 1 visualization
    p1_viz = detector.draw_landmarks(image, p1)
    cv2.imwrite(f"{output_dir}/phase1_landmarks.jpg", p1_viz)
    detector.close()

    # ── Phase 2: Region Extraction ────────────────────────────────────────────
    print("[Phase 2] Extracting facial regions...")
    extractor = RegionExtractor()
    p2 = extractor.extract(image, p1)

    available_regions = [k for k, v in p2.regions.items() if v.available]
    print(f"  ✓ {len(available_regions)}/{len(p2.regions)} regions extracted: {available_regions}")

    # Save Phase 2 visualization
    if p2.visualization is not None:
        cv2.imwrite(f"{output_dir}/phase2_regions.jpg", p2.visualization)

    # Save individual region crops
    for name, region in p2.regions.items():
        if region.available and region.crop is not None:
            cv2.imwrite(f"{output_dir}/region_{name}.jpg", region.crop)

    # ── Phase 3: Feature Analysis ─────────────────────────────────────────────
    print("[Phase 3] Analyzing facial features...")
    analyzer = FeatureAnalyzer()
    p3 = analyzer.analyze(image, p1, p2)

    print(f"  ✓ Left eye  — dark circles: {p3.left_eye.dark_circle_score:.1f}, "
          f"redness: {p3.left_eye.redness_score:.1f}, puffiness: {p3.left_eye.puffiness_score:.1f}")
    print(f"  ✓ Right eye — dark circles: {p3.right_eye.dark_circle_score:.1f}, "
          f"redness: {p3.right_eye.redness_score:.1f}")
    print(f"  ✓ Skin (L cheek) — acne: {p3.skin_left_cheek.acne_score:.1f}, "
          f"redness: {p3.skin_left_cheek.redness_score:.1f}")
    print(f"  ✓ Lips — dryness: {p3.lips.dryness_score:.1f}, pallor: {p3.lips.pallor_score:.1f}")

    # ── Phase 4: Observation Engine ───────────────────────────────────────────
    print("[Phase 4] Generating observations...")
    engine = ObservationEngine()
    p4 = engine.generate(p3)

    print(f"  ✓ Overall: {p4.overall_label} (confidence {p4.overall_confidence:.0f}%)")

    # ── Phase 5: Report Generation ────────────────────────────────────────────
    print("[Phase 5] Generating PDF report...")
    reporter = ReportGenerator()
    pdf_path = f"{output_dir}/health_report_{patient_id}.pdf"

    # Build combined annotated image (Phase 1 mesh + Phase 2 region overlay)
    p1_mesh = detector if hasattr(detector, '_face_mesh') else FaceDetector()
    annotated = _build_combined_viz(image, p1, p2)

    pdf_bytes = reporter.generate(
        p1, p2, p3, p4,
        annotated_image_bgr=annotated,
        patient_id=patient_id,
        output_path=pdf_path
    )
    print(f"  ✓ PDF saved: {pdf_path} ({len(pdf_bytes)//1024} KB)")

    # Save annotated image
    cv2.imwrite(f"{output_dir}/annotated_face.jpg", annotated)

    # JSON summary
    summary = _build_json_summary(p1, p3, p4)
    json_path = f"{output_dir}/report_summary_{patient_id}.json"
    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return super().default(o)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)
    print(f"  ✓ JSON summary: {json_path}")

    # Print text report to console
    _print_text_report(p1, p3, p4)

    return {
        "success": True,
        "pdf_path": pdf_path,
        "json_path": json_path,
        "annotated_image": f"{output_dir}/annotated_face.jpg",
        "overall_label": p4.overall_label,
        "overall_confidence": p4.overall_confidence,
    }


def _build_combined_viz(image, p1, p2) -> np.ndarray:
    """Blend Phase 2 region overlay onto Phase 1 landmark image."""
    out = image.copy()
    if p2.visualization is not None:
        out = cv2.addWeighted(out, 0.6, p2.visualization, 0.4, 0)

    # Draw face bounding box
    if p1.bbox:
        b = p1.bbox
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 255, 100), 2)

    # Overlay symmetry label
    label = f"Symmetry: {p1.symmetry_score:.0f}% | {p1.symmetry_label}"
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 1)

    return out


def _build_json_summary(p1, p3, p4) -> dict:
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase1": {
            "symmetry_score": p1.symmetry_score,
            "symmetry_label": p1.symmetry_label,
            "alignment_angle": round(p1.alignment_angle, 2),
            "face_width": p1.face_width,
            "face_height": p1.face_height,
        },
        "phase3_features": {
            "left_eye": {
                "dark_circle_score": round(p3.left_eye.dark_circle_score, 1),
                "redness_score": round(p3.left_eye.redness_score, 1),
                "puffiness_score": round(p3.left_eye.puffiness_score, 1),
                "openness_ratio": round(p3.left_eye.openness_ratio, 1),
            } if p3.left_eye.available else None,
            "right_eye": {
                "dark_circle_score": round(p3.right_eye.dark_circle_score, 1),
                "redness_score": round(p3.right_eye.redness_score, 1),
                "puffiness_score": round(p3.right_eye.puffiness_score, 1),
                "openness_ratio": round(p3.right_eye.openness_ratio, 1),
            } if p3.right_eye.available else None,
            "skin": {
                "forehead_acne": round(p3.skin_forehead.acne_score, 1),
                "left_cheek_acne": round(p3.skin_left_cheek.acne_score, 1),
                "right_cheek_acne": round(p3.skin_right_cheek.acne_score, 1),
                "left_cheek_redness": round(p3.skin_left_cheek.redness_score, 1),
                "right_cheek_redness": round(p3.skin_right_cheek.redness_score, 1),
            },
            "lips": {
                "dryness_score": round(p3.lips.dryness_score, 1),
                "color_consistency": round(p3.lips.color_consistency, 1),
                "pallor_score": round(p3.lips.pallor_score, 1),
            } if p3.lips.available else None,
        },
        "phase4_assessment": {
            "overall_label": p4.overall_label,
            "overall_confidence": p4.overall_confidence,
            "overall_findings": p4.overall,
            "face_symmetry": p4.face.symmetry,
            "face_alignment": p4.face.alignment,
        }
    }


def _print_text_report(p1, p3, p4):
    """Print a clean text summary to console."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  FACIAL HEALTH OBSERVATION REPORT")
    print(sep)

    print("\nFACE OVERVIEW")
    print(f"  {p4.face.symmetry}")
    print(f"  {p4.face.alignment}")
    print(f"  {p4.face.proportion}")

    print("\nEYES")
    for obs in p4.eyes.left + p4.eyes.right:
        prefix = "✓" if obs.is_normal else "•"
        print(f"  {prefix} {obs.finding}")
    if p4.eyes.openness_note:
        print(f"  • {p4.eyes.openness_note}")

    print("\nSKIN")
    for obs in (p4.skin.forehead + p4.skin.left_cheek +
                p4.skin.right_cheek + p4.skin.overall):
        prefix = "✓" if obs.is_normal else "•"
        print(f"  {prefix} {obs.finding}")

    print("\nLIPS")
    for obs in p4.lips.findings:
        prefix = "✓" if obs.is_normal else "•"
        print(f"  {prefix} {obs.finding}")

    print(f"\nOVERALL ASSESSMENT: {p4.overall_label} "
          f"(confidence {p4.overall_confidence:.0f}%)")
    for item in p4.overall:
        print(f"  • {item}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Facial Health Assessment System")
    parser.add_argument("image", help="Path to face image")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--patient-id", default="DEMO-001")
    args = parser.parse_args()

    result = run_pipeline(args.image, args.output, args.patient_id)
    if result["success"]:
        print(f"Pipeline complete. PDF: {result['pdf_path']}")
    else:
        print(f"Pipeline failed: {result['error']}")
        sys.exit(1)
