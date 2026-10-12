"""
Conditional GAN script to train or generate synthetic retinal samples for minority classes.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchvision.utils import save_image

from src.gan.cgan import ConditionalGANTrainer, generate_synthetic_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/gan_samples")
    parser.add_argument("--generate_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Conditional GAN Pipeline on {device}...")

    trainer = ConditionalGANTrainer(latent_dim=100, num_classes=args.num_classes, device=device)

    # Test generation for underrepresented classes (e.g. Class 3: Severe, Class 4: Proliferative)
    for c in [1, 2, 3, 4]:
        syn_samples = generate_synthetic_samples(trainer.netG, class_label=c, num_samples=4, device=device)
        out_path = os.path.join(args.output_dir, f"synthetic_class_{c}_sample.png")
        save_image(syn_samples, out_path, nrow=2)
        print(f"[+] Saved synthetic sample visualization for Class {c} to {out_path}")


if __name__ == "__main__":
    main()
