"""
Step 2: Convert multi-class label PNGs -> binary masks + copy images.

Reads label PNGs (created by prepare_labels.py) and the raw images,
then creates the final dataset/ folder with binary masks (drivable=1, else=0).

Usage:
    python -m data.prepare_data          (run from idd_seg/)
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def prepare_dataset(raw_data_dir, dest_dir):
    """
    For each image in leftImg8bit/, find its corresponding label PNG in gtFine/,
    convert it to a binary mask, and save both to dest_dir.
    """
    img_dir = os.path.join(raw_data_dir, 'leftImg8bit')
    gt_dir = os.path.join(raw_data_dir, 'gtFine')

    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'masks', split), exist_ok=True)

        search_path = os.path.join(img_dir, split, '**', '*_leftImg8bit.png')
        img_paths = glob(search_path, recursive=True)

        if not img_paths:
            print(f"No images found for split '{split}' in {img_dir}")
            continue

        print(f"Processing {len(img_paths)} images for split: {split}")

        for img_path in tqdm(img_paths, desc=f"{split}"):
            # Extract folder and base id
            # e.g. .../leftImg8bit/train/0/005506_leftImg8bit.png
            parts = img_path.replace('\\', '/').split('/')
            filename = parts[-1]                          # 005506_leftImg8bit.png
            folder = parts[-2]                            # 0
            base_id = filename.replace('_leftImg8bit.png', '')  # 005506

            # Find corresponding label PNG
            label_path = os.path.join(gt_dir, split, folder, f"{base_id}_label.png")

            if not os.path.exists(label_path):
                print(f"Warning: Label not found for {img_path}")
                continue

            # Standardized output filename
            new_filename = f"{base_id}.png"

            # Copy image
            dest_img_path = os.path.join(dest_dir, 'images', split, new_filename)
            shutil.copy2(img_path, dest_img_path)

            # Convert label PNG to binary mask
            label_img = Image.open(label_path)
            label_np = np.array(label_img)

            binary_mask = np.zeros_like(label_np, dtype=np.uint8)
            binary_mask[np.isin(label_np, config.DRIVABLE_IDS)] = 1

            # Save binary mask
            dest_mask_path = os.path.join(dest_dir, 'masks', split, new_filename)
            mask_img = Image.fromarray(binary_mask)
            mask_img.save(dest_mask_path)

    print(f"Dataset preparation complete. Saved to {dest_dir}")


if __name__ == "__main__":
    prepare_dataset(config.RAW_DATA_DIR, config.DATASET_DIR)
