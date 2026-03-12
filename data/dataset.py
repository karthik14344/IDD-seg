import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class IDDSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', size=None):
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.masks_dir = os.path.join(root_dir, 'masks', split)
        self.image_files = os.listdir(self.images_dir)
        self.size = size or config.IMAGE_SIZE

        self.transform_img = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, file_name)
        mask_path = os.path.join(self.masks_dir, file_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        # ensure mask is strictly 0 or 1
        mask = (mask > 0).float()

        return image, mask
