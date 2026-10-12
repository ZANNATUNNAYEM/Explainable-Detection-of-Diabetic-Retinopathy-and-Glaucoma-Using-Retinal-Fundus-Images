"""
Comprehensive Ablation Study Runner for Diabetic Retinopathy Classification.
Lead: Member 1 (Zannatun Nayem)

Compares:
1. Baseline Cross-Entropy (standard imbalanced training)
2. Class-Weighted Cross-Entropy
3. Multi-Class Focal Loss (gamma=2.0)
4. GAN-Augmented Balanced Dataset + Focal Loss
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.eyepacs_dataset import load_eyepacs_splits, EyePACSDataset, get_transforms, GANAugmentedDataset
from src.data.imbalance import compute_class_weights
from src.models.classifiers import build_model
from src.models.losses import FocalLoss
from src.models.metrics import compute_metrics
from src.gan.cgan import ConditionalGANTrainer, generate_synthetic_samples


def train_and_eval(model, train_loader, val_loader, criterion, epochs=5, lr=1e-4, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_qwk = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        m = compute_metrics(all_targets, all_preds, y_probs=all_probs)
        if m["quadratic_weighted_kappa"] > best_qwk:
            best_qwk = m["quadratic_weighted_kappa"]
            best_metrics = m

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="densenet121", choices=["densenet121", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--subset_train", type=int, default=2000)
    parser.add_argument("--subset_val", type=int, default=500)
    parser.add_argument("--output_file", type=str, default="results/ablation_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Member 1 Ablation Study on {args.arch} using device: {device}")

    # Load splits
    train_split, val_split, _ = load_eyepacs_splits(
        subset_train=args.subset_train,
        subset_val=args.subset_val
    )

    train_ds_base = EyePACSDataset(train_split, transform=get_transforms(is_train=True))
    val_ds = EyePACSDataset(val_split, transform=get_transforms(is_train=False))

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    train_labels = [train_ds_base[i][1].item() for i in range(len(train_ds_base))]
    class_weights = compute_class_weights(train_labels, num_classes=5).to(device)

    results = {}

    # 1. Baseline Cross-Entropy
    print("\n[1/4] Running Baseline Standard Cross-Entropy...")
    model_ce = build_model(args.arch, num_classes=5, pretrained=True).to(device)
    train_loader_ce = DataLoader(train_ds_base, batch_size=args.batch_size, shuffle=True)
    results["1_baseline_ce"] = train_and_eval(model_ce, train_loader_ce, val_loader, nn.CrossEntropyLoss(), epochs=args.epochs, device=device)

    # 2. Class-Weighted Cross-Entropy
    print("\n[2/4] Running Class-Weighted Cross-Entropy...")
    model_wce = build_model(args.arch, num_classes=5, pretrained=True).to(device)
    train_loader_wce = DataLoader(train_ds_base, batch_size=args.batch_size, shuffle=True)
    results["2_weighted_ce"] = train_and_eval(model_wce, train_loader_wce, val_loader, nn.CrossEntropyLoss(weight=class_weights), epochs=args.epochs, device=device)

    # 3. Multi-Class Focal Loss
    print("\n[3/4] Running Focal Loss (gamma=2.0)...")
    model_fl = build_model(args.arch, num_classes=5, pretrained=True).to(device)
    train_loader_fl = DataLoader(train_ds_base, batch_size=args.batch_size, shuffle=True)
    results["3_focal_loss"] = train_and_eval(model_fl, train_loader_fl, val_loader, FocalLoss(alpha=class_weights, gamma=2.0), epochs=args.epochs, device=device)

    # 4. GAN-Augmented Dataset + Focal Loss
    print("\n[4/4] Running GAN-Augmented Balanced Training...")
    # Generate synthetic minority samples using cGAN trainer generator
    gan_trainer = ConditionalGANTrainer(latent_dim=100, num_classes=5, device=device)
    synthetic_samples = []
    for c in [1, 2, 3, 4]:
        syn_tensors = generate_synthetic_samples(gan_trainer.netG, class_label=c, num_samples=50, device=device)
        for i in range(syn_tensors.size(0)):
            synthetic_samples.append((syn_tensors[i], c))

    train_ds_gan = GANAugmentedDataset(train_ds_base, synthetic_samples=synthetic_samples, transform=get_transforms(is_train=True))
    train_loader_gan = DataLoader(train_ds_gan, batch_size=args.batch_size, shuffle=True)

    model_gan = build_model(args.arch, num_classes=5, pretrained=True).to(device)
    results["4_gan_augmented_focal"] = train_and_eval(model_gan, train_loader_gan, val_loader, FocalLoss(gamma=2.0), epochs=args.epochs, device=device)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("           MEMBER 1 ABLATION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Configuration':<25} | {'Accuracy':<10} | {'Macro F1':<10} | {'QWK':<10} | {'Sensitivity':<10}")
    print("-" * 70)
    for k, v in results.items():
        print(f"{k:<25} | {v.get('accuracy', 0):<10.4f} | {v.get('macro_f1', 0):<10.4f} | {v.get('quadratic_weighted_kappa', 0):<10.4f} | {v.get('macro_sensitivity', 0):<10.4f}")
    print("=" * 70)
    print(f"[+] Ablation metrics saved to: {args.output_file}")


if __name__ == "__main__":
    main()
