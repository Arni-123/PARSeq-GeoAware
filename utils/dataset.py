# utils/dataset.py
"""
Dataset utilities – simple line-based dataset + loader factory
"""

import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF


class LineSTRDataset(Dataset):
    """
    Simple line-based dataset: path\tlabel format
    Supports charset filtering & length clipping
    """
    def __init__(self, txt_path: str, transform=None, max_samples=None,
                 charset_filter=None, min_len=1, max_len=25):
        self.transform = transform or TF.Compose([
            TF.Resize((64, 256)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.charset_filter = charset_filter
        self.min_len = min_len
        self.max_len = max_len

        self.samples = []
        with open(txt_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                path, label = line.split('\t', 1)
                label = label.strip().lower()

                if self.charset_filter:
                    label = ''.join(c for c in label if c in self.charset_filter)
                if not label or not (self.min_len <= len(label) <= self.max_len):
                    continue

                if os.path.exists(path):
                    self.samples.append((path, label))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        random.shuffle(self.samples)
        print(f"[Dataset] Loaded {len(self.samples):,} valid samples from {txt_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('L')
        except Exception as e:
            print(f"Warning: failed to load {path} → {e}")
            img = Image.new('L', (256, 64), 180)
            label = "error"

        img = self.transform(img)
        return img, label


def create_dataloader(txt_path: str, batch_size: int, shuffle: bool = True,
                      num_workers: int = 2, max_samples: int = None,
                      charset_filter: set = None, min_len: int = 1, max_len: int = 25) -> DataLoader:
    dataset = LineSTRDataset(
        txt_path=txt_path,
        max_samples=max_samples,
        charset_filter=charset_filter,
        min_len=min_len,
        max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )