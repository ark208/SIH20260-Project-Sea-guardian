"""
Project DRISHTI - PyTorch SAR Dataset Loader
Loads Sentinel-1 SAR tiles and 3-Class Segmentation Masks:
  Class 0: Background Ocean
  Class 1: Oil Spill Slick
  Class 2: Look-Alike False Alarm
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any, Optional

class SAROilSpillDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", patch_size: int = 512, augment: bool = True):
        self.data_dir = data_dir
        self.split = split
        self.patch_size = patch_size
        self.augment = augment

        self.img_dir = os.path.join(data_dir, split, "images")
        self.mask_dir = os.path.join(data_dir, split, "masks")

        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")) + glob.glob(os.path.join(self.img_dir, "*.tif*")))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, base_name)

        # Load SAR image (grayscale uint8 or float)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Load ground truth mask (values 0, 1, 2)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros_like(img, dtype=np.uint8)

        # Resize to fixed patch size if needed
        if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
            img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        # Data augmentation
        if self.augment and self.split == "train":
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            # Random 90 deg rotation
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, rot_k)
                mask = np.rot90(mask, rot_k)

        # Normalize image to [0, 1] and format as [C, H, W]
        img_tensor = torch.from_numpy(img.copy()).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask.copy()).long()

        return img_tensor, mask_tensor
