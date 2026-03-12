import os
import torch

# ── Device (GPU if available) ──
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    _gpu_name = torch.cuda.get_device_name(0)
    _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {_gpu_name} ({_gpu_mem:.1f} GB)")
else:
    DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

# ── Project root ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

# ── Raw dataset (untouched) ──
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'idd_segmentation', 'IDD_Segmentation')
RAW_IMAGES_DIR = os.path.join(RAW_DATA_DIR, 'leftImg8bit')
RAW_GT_DIR = os.path.join(RAW_DATA_DIR, 'gtFine')

# ── IDD-Dataset helpers (from idd_lite) ──
IDD_HELPERS_DIR = os.path.join(REPO_ROOT, 'idd_lite', 'IDD-Dataset', 'helpers')
IDD_PREP_DIR = os.path.join(REPO_ROOT, 'idd_lite', 'IDD-Dataset', 'preperation')

# ── Prepared dataset (generated) ──
DATASET_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'dataset')

# ── Outputs ──
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'results')
BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'best_unet.pth')

# ── Drivable class IDs (using 'id' encoding from anue_labels.py) ──
# 0: road, 1: parking, 2: drivable fallback
DRIVABLE_IDS = [0, 1, 2]

# ── Training hyperparameters ──
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_WORKERS = 2
