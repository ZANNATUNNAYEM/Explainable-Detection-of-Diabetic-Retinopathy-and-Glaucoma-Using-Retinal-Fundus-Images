"""
Generates synthetic balanced datasets for Diabetic Retinopathy minority classes.
Member 1: Zannatun Nayem
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from .cgan import generate_synthetic_samples


def balance_training_dataset(
    generator,
    real_train_dataset,
    target_samples_per_class=1000,
    output_dir="data/synthetic_dr",
    device="cpu",
    latent_dim=100
):
    """
    Analyzes class counts of real_train_dataset and synthesizes minority class images
    using the trained cGAN generator to achieve uniform target_samples_per_class.

    Returns:
        synthetic_samples: List of (PIL.Image, label)
        class_counts_before: dict
        class_counts_after: dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # Count real samples
    labels = []
    for i in range(len(real_train_dataset)):
        _, y = real_train_dataset[i]
        labels.append(int(y))

    unique, counts = np.unique(labels, return_counts=True)
    class_counts_before = {int(k): int(v) for k, v in zip(unique, counts)}
    class_counts_after = dict(class_counts_before)

    synthetic_samples = []

    print("[*] Generating synthetic samples to balance minority classes:")
    for c in range(5):
        current_count = class_counts_before.get(c, 0)
        needed = max(0, target_samples_per_class - current_count)

        if needed > 0 and generator is not None:
            print(f"  -> Synthesizing {needed} images for Class {c}...")
            # Generate in batches of 32
            batch_size = 32
            generated_so_far = 0
            while generated_so_far < needed:
                n_gen = min(batch_size, needed - generated_so_far)
                syn_tensor = generate_synthetic_samples(
                    generator,
                    class_label=c,
                    num_samples=n_gen,
                    latent_dim=latent_dim,
                    device=device
                )

                for b in range(n_gen):
                    img_np = syn_tensor[b].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255.0).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)

                    # Save sample image to disk
                    sample_filename = f"syn_c{c}_{generated_so_far + b:05d}.png"
                    sample_path = os.path.join(output_dir, sample_filename)
                    pil_img.save(sample_path)

                    synthetic_samples.append((pil_img, c))

                generated_so_far += n_gen

            class_counts_after[c] = current_count + needed
        else:
            print(f"  -> Class {c} already has {current_count} samples (no synthesis needed).")

    print("[+] Dataset balancing completed.")
    print(f"    Counts before: {class_counts_before}")
    print(f"    Counts after:  {class_counts_after}")

    return synthetic_samples, class_counts_before, class_counts_after
