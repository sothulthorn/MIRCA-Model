"""
PyTorch Dataset for preprocessed MIRCA data.

Loads precomputed CWT grayscale images (vibration + current) and labels.
Handles train/val/test splitting per paper ratios (70/20/10).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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
