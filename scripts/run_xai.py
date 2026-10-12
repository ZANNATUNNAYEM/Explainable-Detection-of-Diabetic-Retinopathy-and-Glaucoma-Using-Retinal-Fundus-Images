"""
Grad-CAM, Grad-CAM++, and Quantitative Faithfulness Deletion testing execution script.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from PIL import Image

from src.models.classifiers import build_model
from src.xai.gradcam import RetinalXAIExplainer
from src.xai.faithfulness import deletion_faithfulness_curve, model_randomization_sanity_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="densenet121")
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam", "gradcam++"])
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running XAI Attribution ({args.method}) for {args.arch}...")

    model = build_model(architecture=args.arch, num_classes=args.num_classes, pretrained=True)
    if args.weights_path and os.path.exists(args.weights_path):
        ckpt = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"  -> Loaded weights from {args.weights_path}")

    explainer = RetinalXAIExplainer(model, architecture=args.arch, method=args.method)

    # Dummy sample check to verify pipeline execution
    dummy_input = torch.randn(1, 3, 224, 224)
    cam = explainer.generate_heatmap(dummy_input)
    print(f"[+] Successfully generated heatmap of shape: {cam.shape} with min={cam.min():.2f}, max={cam.max():.2f}")

    # Run faithfulness deletion test
    faithfulness = deletion_faithfulness_curve(model, dummy_input, cam)
    print(f"[+] Initial probability: {faithfulness['base_prob']:.4f}")
    for step in faithfulness["curve"]:
        print(f"    - Deleting {int(step['fraction_deleted']*100)}% salient pixels -> New Prob: {step['new_prob']:.4f} (Drop: {step['probability_drop']:.4f})")

    # Run Adebayo sanity check
    sanity = model_randomization_sanity_check(explainer, dummy_input, architecture=args.arch)
    print(f"[+] Sanity Check: Correlation with randomized weights = {sanity['correlation_with_random']:.4f} (Passed: {sanity['sanity_check_passed']})")


if __name__ == "__main__":
    main()
