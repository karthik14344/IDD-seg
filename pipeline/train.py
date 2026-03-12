"""
Training loop for U-Net road segmentation.

Usage:
    python -m pipeline.train                         (run from idd_seg/)
    python -m pipeline.train --epochs 20 --batch_size 8
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data.dataset import IDDSegmentationDataset
from model.unet import UNet


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)

        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

        return bce_loss + dice_loss


def calculate_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def calculate_pixel_accuracy(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total


def train(args):
    device = config.DEVICE

    train_dataset = IDDSegmentationDataset(args.data_dir, split='train')
    val_dataset = IDDSegmentationDataset(args.data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_iou = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # ── Training ──
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_acc = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)
            train_acc += calculate_pixel_accuracy(outputs, masks)

            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_acc /= len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
                val_acc += calculate_pixel_accuracy(outputs, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on IDD for road segmentation")
    parser.add_argument('--data_dir', type=str, default=config.DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--workers', type=int, default=config.NUM_WORKERS)
    parser.add_argument('--save_path', type=str, default=config.BEST_MODEL_PATH)

    args = parser.parse_args()
    train(args)
