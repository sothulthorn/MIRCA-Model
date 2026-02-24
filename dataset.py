"""
<<<<<<< HEAD
PyTorch Dataset and DataLoader creation.

Loads precomputed CWT images from disk (produced by preprocessing.py).
"""

=======
PyTorch Dataset for preprocessed MIRCA data.

Loads precomputed CWT grayscale images (vibration + current) and labels.
Handles train/val/test splitting per paper ratios (70/20/10).
"""
>>>>>>> mirca-v2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

<<<<<<< HEAD
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
=======
import config


class PaderbornDataset(Dataset):
    """
    Dataset for Paderborn bearing fault diagnosis with multi-source CWT images.

    Args:
        vibration_images: (N, H, W) uint8 array of vibration CWT images
        current_images: (N, H, W) uint8 array of current CWT images
        labels: (N,) int64 array of fault labels
    """

    def __init__(self, vibration_images, current_images, labels):
        self.vibration_images = vibration_images
        self.current_images = current_images
        self.labels = labels
>>>>>>> mirca-v2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
<<<<<<< HEAD
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
=======
        # Load and normalize to [0, 1] float32, add channel dimension
        vib_img = self.vibration_images[idx].astype(np.float32) / 255.0
        cur_img = self.current_images[idx].astype(np.float32) / 255.0

        vib_tensor = torch.from_numpy(vib_img).unsqueeze(0)  # (1, H, W)
        cur_tensor = torch.from_numpy(cur_img).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return vib_tensor, cur_tensor, label


def load_and_split_data(processed_dir=None, seed=None):
    """
    Load preprocessed data and split into train/val/test sets.

    Split ratios from paper: 70% train, 20% validation, 10% test.
    Stratified splitting to maintain class balance.

    Returns:
        train_dataset, val_dataset, test_dataset: PaderbornDataset instances
    """
    if processed_dir is None:
        processed_dir = config.PROCESSED_DIR
    if seed is None:
        seed = config.SEED

    # Load preprocessed data
    vib_images = np.load(os.path.join(processed_dir, "vibration_images.npy"))
    cur_images = np.load(os.path.join(processed_dir, "current_images.npy"))
    labels = np.load(os.path.join(processed_dir, "labels.npy"))

    print(f"Loaded data: {len(labels)} samples")

    # Stratified split
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        rng.shuffle(class_indices)
        n = len(class_indices)

        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])

    # Shuffle each split
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    print(f"Split: train={len(train_indices)}, "
          f"val={len(val_indices)}, test={len(test_indices)}")

    # Create datasets
    train_dataset = PaderbornDataset(
        vib_images[train_indices], cur_images[train_indices],
        labels[train_indices]
    )
    val_dataset = PaderbornDataset(
        vib_images[val_indices], cur_images[val_indices],
        labels[val_indices]
    )
    test_dataset = PaderbornDataset(
        vib_images[test_indices], cur_images[test_indices],
        labels[test_indices]
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size=None, num_workers=None):
    """Create DataLoaders for train/val/test datasets."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_and_split_data()
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Test a sample
    vib, cur, label = train_ds[0]
    print(f"Sample shapes: vib={vib.shape}, cur={cur.shape}, label={label}")
>>>>>>> mirca-v2
