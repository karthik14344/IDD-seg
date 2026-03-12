"""
Evaluate trained U-Net and save visual results.

Usage:
    python -m pipeline.evaluate                      (run from idd_seg/)
    python -m pipeline.evaluate --num_visualize 10
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data.dataset import IDDSegmentationDataset
from model.unet import UNet
from pipeline.train import calculate_iou, calculate_pixel_accuracy


def evaluate_model(args):
    device = config.DEVICE

    dataset = IDDSegmentationDataset(args.data_dir, split=args.split)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Model weights loaded from {args.model_path}")
    else:
        print(f"Warning: Model weights {args.model_path} not found. Evaluating with random initialization.")

    model.eval()

    val_iou = 0.0
    val_acc = 0.0

    output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluation")):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            val_iou += calculate_iou(outputs, masks)
            val_acc += calculate_pixel_accuracy(outputs, masks)

            if i < args.num_visualize:
                pred_mask = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()[0, 0]
                img_np = images.cpu().numpy()[0].transpose(1, 2, 0)

                # Unnormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                gt_mask = masks.cpu().numpy()[0, 0]

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(img_np)
                axes[0].set_title("Input Image")
                axes[0].axis('off')

                axes[1].imshow(gt_mask, cmap='gray')
                axes[1].set_title("GT Mask")
                axes[1].axis('off')

                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title("Predicted Mask")
                axes[2].axis('off')

                overlay = img_np.copy()
                overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.5 + np.array([1, 0, 0]) * 0.5
                axes[3].imshow(overlay)
                axes[3].set_title("Prediction Overlay")
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"eval_sample_{i}.png"))
                plt.close(fig)

    val_iou /= len(val_loader)
    val_acc /= len(val_loader)

    print(f"Final Evaluation on {args.split} set:")
    print(f"Mean IoU: {val_iou:.4f}")
    print(f"Pixel Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net on IDD")
    parser.add_argument('--data_dir', type=str, default=config.DATASET_DIR)
    parser.add_argument('--model_path', type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument('--output_dir', type=str, default=config.RESULTS_DIR)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--num_visualize', type=int, default=5)

    args = parser.parse_args()
    evaluate_model(args)
