"""Low-only adapter for the official DCTTA PromptIR training tuple.

The third field is a low-image surrogate for an unused clean-image slot in
upstream ``compute_fisher``. No reference directory is accepted or inspected.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class LowOnlyPromptTrainDataset(Dataset):
    def __init__(self, lq_dir, patch_size=128, seed=42):
        self.lq_dir = lq_dir
        self.patch_size = patch_size
        self.names = sorted(
            name for name in os.listdir(lq_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        low = np.array(Image.open(os.path.join(self.lq_dir, name)).convert("RGB"))
        height, width = low.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(f"image too small: {height}x{width}, patch={self.patch_size}")
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        patch = low[top:top + self.patch_size, left:left + self.patch_size]
        # The paired upstream dataset independently crops GT, consuming two
        # RNG draws. Mirror those draws using only low-image dimensions.
        random.randint(0, height - self.patch_size)
        random.randint(0, width - self.patch_size)
        degraded = self.to_tensor(patch)
        return name, degraded, degraded
