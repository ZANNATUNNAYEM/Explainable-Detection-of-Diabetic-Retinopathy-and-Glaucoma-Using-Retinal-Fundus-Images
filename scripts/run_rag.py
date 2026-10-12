"""
RAG Clinical Explanation demonstration script.
Generates grounded clinical reports from model predictions, confidence scores, and visual Grad-CAM regions.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.report_generator import RetinalRAGReportGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", type=str, default="Diabetic Retinopathy", choices=["Diabetic Retinopathy", "Glaucoma"])
    parser.add_argument("--grade", type=str, default="Severe NPDR (Grade 3)")
    parser.add_argument("--conf", type=float, default=0.94)
    parser.add_argument("--xai_desc", type=str, default="Dense microaneurysms and multiple blot hemorrhages concentrated in superior temporal quadrant.")
    args = parser.parse_args()

    print(f"[*] Initializing Retinal RAG Clinical Explanation Engine...")
    engine = RetinalRAGReportGenerator()
    report = engine.generate_report(
        disease=args.disease,
        predicted_grade=args.grade,
        confidence=args.conf,
        xai_description=args.xai_desc
    )

    print("\n" + "=" * 60)
    print(report["formatted_summary"])
    print("=" * 60)


if __name__ == "__main__":
    main()
