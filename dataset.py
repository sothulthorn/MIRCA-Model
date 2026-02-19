"""
PyTorch Dataset and DataLoader creation.

Loads precomputed CWT images from disk (produced by preprocessing.py).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class BearingDataset(Dataset):
    """
    Dataset that loads precomputed CWT images from numpy arrays.

    Each sample returns:
        vib_img: (1, H, W) float32 tensor  — vibration CWT image
        cur_img: (1, H, W) float32 tensor  — current CWT image
        label:   scalar long tensor
    """

    def __init__(self, vib_images, cur_images, labels, train=False):
        self.vib_images = vib_images  # (N, H, W) float32
        self.cur_images = cur_images
        self.labels = labels          # (N,) int64
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # .astype(np.float32) handles float16 -> float32 conversion
        vib = torch.from_numpy(self.vib_images[idx].astype(np.float32)).unsqueeze(0)
        cur = torch.from_numpy(self.cur_images[idx].astype(np.float32)).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.train:
            # Synchronized random flips — both images get the same transform
            # (they represent the same physical event)
            paired = torch.cat([vib, cur], dim=0)  # (2, H, W)
            if torch.rand(1).item() < 0.5:
                paired = torch.flip(paired, dims=[2])  # horizontal flip
            if torch.rand(1).item() < 0.5:
                paired = torch.flip(paired, dims=[1])  # vertical flip
            vib, cur = paired[0:1], paired[1:2]

        return vib, cur, label


def load_processed_data(cfg):
    """
    Load precomputed arrays and split indices from cfg.processed_dir.

    Returns:
        vib_images: (N, H, W) float32
        cur_images: (N, H, W) float32
        labels:     (N,) int64
        split:      dict with 'train_idx', 'val_idx', 'test_idx'
    """
    d = cfg.processed_dir
    vib = np.load(os.path.join(d, "vib_images.npy"), mmap_mode="r")
    cur = np.load(os.path.join(d, "cur_images.npy"), mmap_mode="r")
    labels = np.load(os.path.join(d, "labels.npy"))
    split = np.load(os.path.join(d, "split_indices.npz"))

    print(f"Loaded {len(labels)} precomputed samples from {os.path.abspath(d)}")
    return vib, cur, labels, split


def create_dataloaders(cfg=None):
    """
    Create train/val/test DataLoaders from precomputed data on disk.

    Returns:
        train_loader, val_loader, test_loader
    """
    if cfg is None:
        cfg = Config()

    vib, cur, labels, split = load_processed_data(cfg)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    # Index into memory-mapped arrays — copies only the needed slices
    train_ds = BearingDataset(np.array(vib[train_idx]),
                              np.array(cur[train_idx]),
                              labels[train_idx],
                              train=True)
    val_ds = BearingDataset(np.array(vib[val_idx]),
                            np.array(cur[val_idx]),
                            labels[val_idx])
    test_ds = BearingDataset(np.array(vib[test_idx]),
                             np.array(cur[test_idx]),
                             labels[test_idx])

    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
