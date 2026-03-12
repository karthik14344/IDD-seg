"""
Visualize sample images with their binary masks overlaid.

Usage:
    python -m utils.visualize                        (run from idd_seg/)
    python -m utils.visualize --split val --num_samples 5
"""

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def visualize_samples(dataset_dir, split='train', num_samples=3, output_path=None):
    images_dir = os.path.join(dataset_dir, 'images', split)
    masks_dir = os.path.join(dataset_dir, 'masks', split)

    image_files = os.listdir(images_dir)
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, file_name in enumerate(sample_files):
        img_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(masks_dir, file_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img_np = np.array(img)
        mask_np = np.array(mask)

        # Overlay: road pixels shown in red
        overlay = img_np.copy()
        overlay[mask_np == 1] = overlay[mask_np == 1] * 0.5 + np.array([255, 0, 0]) * 0.5

        axes[i][0].imshow(img_np)
        axes[i][0].set_title(f"Image: {file_name}")
        axes[i][0].axis('off')

        axes[i][1].imshow(mask_np, cmap='gray')
        axes[i][1].set_title("Binary Mask (1=Drivable)")
        axes[i][1].axis('off')

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay")
        axes[i][2].axis('off')

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(config.RESULTS_DIR, 'sample_visualization.png')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument('--data_dir', type=str, default=config.DATASET_DIR)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()
    visualize_samples(args.data_dir, args.split, args.num_samples, args.output)
