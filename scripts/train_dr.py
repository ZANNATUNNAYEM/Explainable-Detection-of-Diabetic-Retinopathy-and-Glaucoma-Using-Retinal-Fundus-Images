"""
Training script for Diabetic Retinopathy classification on EyePACS.
Supports DenseNet121, EfficientNet-B0, Focal Loss, Class Weights, and Cosine Annealing.
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

from src.data.eyepacs_dataset import load_eyepacs_splits, EyePACSDataset, get_transforms
from src.data.imbalance import compute_class_weights
from src.models.classifiers import build_model
from src.models.losses import get_loss_function
from src.models.metrics import compute_metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = epoch_loss
    return metrics


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [] , []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds, y_probs=all_probs)
    metrics["loss"] = epoch_loss
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="densenet121", choices=["densenet121", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset_train", type=int, default=3000, help="Number of training samples (None for all)")
    parser.add_argument("--subset_val", type=int, default=600, help="Number of validation samples")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "weighted_ce", "ce"])
    parser.add_argument("--referable", action="store_true", help="Binary Referable DR classification mode")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training {args.arch} on device: {device} | Referable Mode: {args.referable}")

    train_split, val_split, _ = load_eyepacs_splits(
        subset_train=args.subset_train,
        subset_val=args.subset_val
    )

    train_ds = EyePACSDataset(train_split, transform=get_transforms(is_train=True), binary_referable=args.referable)
    val_ds = EyePACSDataset(val_split, transform=get_transforms(is_train=False), binary_referable=args.referable)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = 2 if args.referable else 5
    model = build_model(architecture=args.arch, num_classes=num_classes, pretrained=True).to(device)

    # Calculate class weights from training labels
    train_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    class_weights = compute_class_weights(train_labels, num_classes=num_classes)
    criterion = get_loss_function(loss_type=args.loss, class_weights=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_qwk = -1.0
    best_path = os.path.join(args.output_dir, f"best_{args.arch}_{'referable' if args.referable else '5class'}.pt")

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} | "
              f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} QWK: {val_m['quadratic_weighted_kappa']:.4f} Macro-F1: {val_m['macro_f1']:.4f}")

        eval_metric = val_m["quadratic_weighted_kappa"] if not args.referable else val_m["accuracy"]
        if eval_metric > best_qwk:
            best_qwk = eval_metric
            torch.save({"model_state": model.state_dict(), "metrics": val_m, "epoch": epoch}, best_path)
            print(f"  -> Saved best model checkpoint to {best_path}")

    print(f"[+] Training complete. Best score: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
