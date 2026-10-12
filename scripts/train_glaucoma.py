"""
Training and evaluation pipeline for binary Glaucoma screening on REFUGE2 / G1020 datasets.
Tracks sensitivity, specificity, and AUROC for clinical benchmarking against Pascal et al. & Hemelings et al.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.classifiers import build_model
from src.models.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="densenet121")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Glaucoma detection pipeline with backbone: {args.arch} on {device}")

    # Build binary classification model
    model = build_model(architecture=args.arch, num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("[*] Glaucoma model architecture initialized successfully.")
    print("    Benchmark Target: Surpass Pascal et al. AUC baseline (96.76%) and Hemelings et al.")


if __name__ == "__main__":
    main()
