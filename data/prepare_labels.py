"""
Step 1: Convert JSON polygon annotations -> label PNG images.

Uses IDD-Dataset helpers (annotation.py, json2labelImg.py) to draw polygons
onto images with 'id' encoding (road=0, parking=1, drivable fallback=2, ...).

Saves *_label.png next to each *_gtFine_polygons.json in gtFine/.

Usage:
    python -m data.prepare_labels        (run from idd_seg/)
"""

import os
import sys
from glob import glob
from tqdm import tqdm

# Add project root to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

# Add IDD-Dataset helpers to path
sys.path.insert(0, os.path.abspath(config.IDD_HELPERS_DIR))
sys.path.insert(0, os.path.abspath(config.IDD_PREP_DIR))

from annotation import Annotation
from json2labelImg import createLabelImage


def convert_json_to_labels(gt_dir):
    """
    Finds all *_gtFine_polygons.json files and converts each to a
    *_label.png using 'id' encoding.
    """
    for split in ['train', 'val']:
        search_path = os.path.join(gt_dir, split, '**', '*_gtFine_polygons.json')
        json_files = glob(search_path, recursive=True)

        if not json_files:
            print(f"No JSON files found for split '{split}' in {gt_dir}")
            continue

        print(f"Converting {len(json_files)} JSON files for split: {split}")

        for json_path in tqdm(json_files, desc=f"{split}"):
            # Output: same folder, e.g. 005506_gtFine_polygons.json -> 005506_label.png
            label_path = json_path.replace('_gtFine_polygons.json', '_label.png')

            if os.path.exists(label_path):
                continue  # skip already converted

            try:
                annotation = Annotation()
                annotation.fromJsonFile(json_path)
                label_img = createLabelImage(json_path, annotation, 'id')
                label_img.save(label_path)
            except Exception as e:
                print(f"Failed: {json_path} -> {e}")

    print("Done! Label PNGs saved next to JSON files in gtFine/")


if __name__ == "__main__":
    convert_json_to_labels(config.RAW_GT_DIR)
