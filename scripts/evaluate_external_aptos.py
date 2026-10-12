"""
External validation script for evaluating trained DR models on APTOS 2019 or DDR datasets.
Member 1: Zannatun Nayem
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.classifiers import build_model
from src.models.metrics import compute_metrics
from src.data.eyepacs_dataset import get_transforms, APTOSDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Path to external images folder")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV labels")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--arch", type=str, default="densenet121")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--referable", action="store_true", help="Evaluate as binary referable DR")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting external dataset validation on: {args.img_dir}")

    num_classes = 2 if args.referable else 5
    model = build_model(architecture=args.arch, num_classes=num_classes, pretrained=False).to(device)

    if os.path.exists(args.weights_path):
        ckpt = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        print(f"  -> Loaded checkpoint from {args.weights_path}")
    else:
        print(f"[!] Warning: weights file not found at {args.weights_path}")

    ds = APTOSDataset(img_dir=args.img_dir, df_or_csv=args.csv_file, transform=get_transforms(is_train=False), binary_referable=args.referable)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="External Eval"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    metrics = compute_metrics(all_targets, all_preds, y_probs=all_probs, num_classes=num_classes)
    print("\n" + "=" * 50)
    print("      EXTERNAL APTOS / DDR BENCHMARK RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:<26}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
