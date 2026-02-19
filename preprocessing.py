"""
Data preprocessing: CWT computation and disk caching.

Run this script ONCE before training to precompute all CWT images:
    python preprocessing.py

Produces:
    data/processed/
        vib_images.npy   (N, 224, 224) float32
        cur_images.npy   (N, 224, 224) float32
        labels.npy       (N,)          int64
        split_indices.npz  {train_idx, val_idx, test_idx}
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config


# ============================================================
# Signal processing utilities
# ============================================================

def zscore_normalize(signal_data):
    """Z-score normalization: (x - mean) / std."""
    mu = np.mean(signal_data)
    sigma = np.std(signal_data)
    if sigma < 1e-10:
        return signal_data - mu
    return (signal_data - mu) / sigma


def sliding_window_segment(signal_data, window_length, stride):
    """Segment a 1-D signal into overlapping windows."""
    n = len(signal_data)
    starts = range(0, n - window_length + 1, stride)
    return [signal_data[s: s + window_length] for s in starts]


def morlet_cwt(signal_data, scales, omega0=6.0):
    """
    Continuous Wavelet Transform using Morlet wavelet (FFT-based).

    Args:
        signal_data: 1-D numpy array.
        scales: 1-D array of wavelet scales.
        omega0: Morlet center frequency.

    Returns:
        2-D array of shape (len(scales), len(signal_data)).
    """
    N = len(signal_data)
    signal_fft = np.fft.fft(signal_data)
    angular_freqs = 2.0 * np.pi * np.fft.fftfreq(N)

    cwt_matrix = np.zeros((len(scales), N), dtype=np.float64)
    norm_const = np.pi ** (-0.25)

    for i, s in enumerate(scales):
        psi_hat = norm_const * np.exp(-0.5 * (s * angular_freqs - omega0) ** 2)
        psi_hat *= np.sqrt(2.0 * np.pi * s)
        cwt_matrix[i] = np.abs(np.fft.ifft(signal_fft * np.conj(psi_hat)))

    return cwt_matrix


def signal_to_cwt_image(signal_1d, cfg):
    """
    Convert a 1-D signal segment to a (image_size, image_size) grayscale
    CWT image normalized to [0, 1].
    """
    scales = np.geomspace(cfg.cwt_scale_min, cfg.cwt_scale_max, num=cfg.cwt_num_scales)
    cwt_mat = morlet_cwt(signal_1d, scales, omega0=cfg.cwt_omega0)

    vmin, vmax = cwt_mat.min(), cwt_mat.max()
    if vmax - vmin > 1e-10:
        cwt_mat = (cwt_mat - vmin) / (vmax - vmin)
    else:
        cwt_mat = np.zeros_like(cwt_mat)

    tensor = torch.from_numpy(cwt_mat).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(cfg.image_size, cfg.image_size),
                            mode="bilinear", align_corners=False)
    return resized.squeeze().numpy().astype(np.float32)


# ============================================================
# Paderborn .mat file loader
# ============================================================

def load_paderborn_mat(file_path, vib_ch, cur_ch):
    """
    Load vibration and current signals from a Paderborn .mat file.

    Paderborn .mat structure:
        data[key]['Y'][0,0][0]  ->  array of channels, each channel:
            [0] = name,  [1] = unit,  [2][0] = signal data
        Channel indices (for standard Paderborn files):
            0: force            (16 001 samples)
            1: phase_current_1  (256 001 samples)
            2: phase_current_2  (256 001 samples)
            3: speed            (16 001 samples)
            4: temp_2_bearing   (5 samples)
            5: torque           (16 001 samples)
            6: vibration_1      (256 001 samples)

    Args:
        file_path: Path to the .mat file.
        vib_ch:    Channel index for vibration (default 6).
        cur_ch:    Channel index for current   (default 1).

    Returns:
        vibration: 1-D numpy float64 array
        current:   1-D numpy float64 array
    """
    import scipy.io as sio

    data = sio.loadmat(file_path, squeeze_me=False)
    keys = [k for k in data.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No data keys found in {file_path}")
    struct = data[keys[0]]

    y_array = struct["Y"][0, 0][0]
    vibration = y_array[vib_ch][2][0].flatten().astype(np.float64)
    current = y_array[cur_ch][2][0].flatten().astype(np.float64)
    return vibration, current


# ============================================================
# Precomputation pipeline
# ============================================================

def collect_mat_files(cfg):
    """
    Walk the Paderborn dataset and collect .mat file paths that match:
      1. A bearing code listed in cfg.bearing_label_map.
      2. The target operating condition (cfg.operating_condition).

    Paderborn directory layout:
        data_dir/
            K001/
                N09_M07_F10_K001_1.mat   <- different operating condition, skipped
                N15_M07_F10_K001_1.mat   <- matches, kept
                ...
            KA01/
                ...

    Returns:
        List of (file_path, bearing_code, label) tuples.
    """
    results = []

    for bearing_code, label in sorted(cfg.bearing_label_map.items()):
        bearing_dir = os.path.join(cfg.data_dir, bearing_code)
        if not os.path.isdir(bearing_dir):
            print(f"  [WARN] Directory not found: {bearing_dir}")
            continue

        count = 0
        for fname in sorted(os.listdir(bearing_dir)):
            if not fname.endswith(".mat"):
                continue
            # Filter by operating condition (e.g. "N15_M07_F10")
            if cfg.operating_condition and cfg.operating_condition not in fname:
                continue
            results.append((os.path.join(bearing_dir, fname), bearing_code, label))
            count += 1

        print(f"  {bearing_code} -> label {label} ({cfg.label_names[label]:12s}): "
              f"{count} files")

    return results


def _compute_segments_per_file(file_list, cfg):
    """
    Compute per-class segments/file so every class has the same total samples.

    If cfg.target_samples_per_class is set, segments/file is calculated as:
        target_samples_per_class / num_files_in_that_class

    Falls back to cfg.max_segments_per_file if target is None.
    """
    if not cfg.target_samples_per_class:
        # Uniform cap for all classes
        return {label: cfg.max_segments_per_file
                for _, _, label in file_list}

    # Count files per label
    from collections import Counter
    files_per_label = Counter(label for _, _, label in file_list)

    segs_per_label = {}
    for label, n_files in sorted(files_per_label.items()):
        segs = cfg.target_samples_per_class // n_files
        segs_per_label[label] = segs
        print(f"  Class {label} ({cfg.label_names[label]:12s}): "
              f"{n_files} files × {segs} segs/file = "
              f"{n_files * segs} samples")

    return segs_per_label


def precompute_from_paderborn(cfg):
    """
    Load the Paderborn dataset, apply CWT to every segment, and return
    numpy arrays ready to save.

    Only files matching the target operating condition and the bearings
    in cfg.bearing_label_map are processed.
    """
    file_list = collect_mat_files(cfg)
    if not file_list:
        return None, None, None

    total_files = len(file_list)

    # Compute per-class segment caps for balanced classes
    segs_per_label = _compute_segments_per_file(file_list, cfg)

    print(f"\n{total_files} .mat files to process "
          f"(condition={cfg.operating_condition})\n")

    vib_images, cur_images, labels = [], [], []
    t0 = time.time()

    file_pbar = tqdm(file_list, desc="Files", unit="file", dynamic_ncols=True)
    for fpath, bearing_code, label in file_pbar:
        fname = os.path.basename(fpath)
        cap = segs_per_label[label]
        file_pbar.set_postfix_str(f"{fname}")

        try:
            vibration, current = load_paderborn_mat(
                fpath, cfg.vibration_channel, cfg.current_channel)
        except Exception as e:
            tqdm.write(f"  SKIP {fname}: {e}")
            continue

        # Z-score normalization
        vibration = zscore_normalize(vibration)
        current = zscore_normalize(current)

        # Sliding window segmentation
        vib_segs = sliding_window_segment(vibration, cfg.window_length,
                                          cfg.window_stride)
        cur_segs = sliding_window_segment(current, cfg.window_length,
                                          cfg.window_stride)
        n_segs = min(len(vib_segs), len(cur_segs))

        # Cap segments per file to control dataset size
        if cap and n_segs > cap:
            # Evenly spaced selection to cover the full recording
            indices = np.linspace(0, n_segs - 1, cap, dtype=int)
            vib_segs = [vib_segs[i] for i in indices]
            cur_segs = [cur_segs[i] for i in indices]
            n_segs = cap

        seg_pbar = tqdm(range(n_segs), desc=f"  CWT {bearing_code}",
                        unit="seg", leave=False, dynamic_ncols=True)
        for i in seg_pbar:
            vib_images.append(signal_to_cwt_image(vib_segs[i], cfg))
            cur_images.append(signal_to_cwt_image(cur_segs[i], cfg))
            labels.append(label)

        elapsed = time.time() - t0
        rate = len(labels) / elapsed if elapsed > 0 else 0
        file_pbar.set_postfix_str(
            f"{fname} | total={len(labels)} | {rate:.0f} img/s")

    if not labels:
        return None, None, None

    return (np.array(vib_images, dtype=np.float32),
            np.array(cur_images, dtype=np.float32),
            np.array(labels, dtype=np.int64))


def generate_synthetic_data(cfg, samples_per_class=100):
    """Generate synthetic demo data when the real dataset is unavailable."""
    print(f"Generating synthetic data: {samples_per_class} x {cfg.num_classes} classes")
    t = np.linspace(0, cfg.window_length / cfg.sampling_rate, cfg.window_length)

    class_freqs = [
        (50, 200),  (80, 300),  (120, 150), (60, 400),
        (90, 250),  (110, 350), (70, 180),  (40, 100),
    ]

    vib_list, cur_list, lab_list = [], [], []
    total = cfg.num_classes * samples_per_class

    pbar = tqdm(total=total, desc="Synthetic CWT", unit="img", dynamic_ncols=True)
    for cls_idx in range(cfg.num_classes):
        f1, f2 = class_freqs[cls_idx]
        for _ in range(samples_per_class):
            phase1 = np.random.rand() * 2 * np.pi
            phase2 = np.random.rand() * 2 * np.pi

            vib = (np.sin(2 * np.pi * f1 * t + phase1)
                   + 0.5 * np.sin(2 * np.pi * f2 * t + phase2)
                   + np.random.randn(cfg.window_length) * 0.3)
            cur = (0.7 * np.sin(2 * np.pi * f1 * 0.8 * t + phase1)
                   + 0.3 * np.sin(2 * np.pi * f2 * 1.2 * t + phase2)
                   + np.random.randn(cfg.window_length) * 0.3)

            vib_list.append(signal_to_cwt_image(zscore_normalize(vib), cfg))
            cur_list.append(signal_to_cwt_image(zscore_normalize(cur), cfg))
            lab_list.append(cls_idx)
            pbar.update(1)
    pbar.close()

    return (np.array(vib_list, dtype=np.float32),
            np.array(cur_list, dtype=np.float32),
            np.array(lab_list, dtype=np.int64))


def save_split_indices(n_samples, cfg):
    """Generate and save train/val/test split indices."""
    indices = list(range(n_samples))
    random.seed(cfg.seed)
    random.shuffle(indices)

    n_train = int(n_samples * cfg.train_ratio)
    n_val = int(n_samples * cfg.val_ratio)

    train_idx = np.array(indices[:n_train])
    val_idx = np.array(indices[n_train: n_train + n_val])
    test_idx = np.array(indices[n_train + n_val:])

    path = os.path.join(cfg.processed_dir, "split_indices.npz")
    np.savez(path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"Saved split indices: train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")


def run_preprocessing(cfg=None):
    """
    Main preprocessing entry point.

    1. Load raw signals from .mat files (or generate synthetic data).
    2. Compute CWT images for every segment.
    3. Save vib_images.npy, cur_images.npy, labels.npy, split_indices.npz.
    """
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.processed_dir, exist_ok=True)
    start = time.time()

    # Try real dataset first
    vib, cur, labels = None, None, None
    if os.path.isdir(cfg.data_dir):
        vib, cur, labels = precompute_from_paderborn(cfg)

    if vib is None or len(vib) == 0:
        print(f"[WARNING] No data in '{cfg.data_dir}'. Using synthetic data.")
        vib, cur, labels = generate_synthetic_data(cfg)

    # Save to disk as float16 to halve storage (no precision loss for [0,1] images)
    np.save(os.path.join(cfg.processed_dir, "vib_images.npy"), vib.astype(np.float16))
    np.save(os.path.join(cfg.processed_dir, "cur_images.npy"), cur.astype(np.float16))
    np.save(os.path.join(cfg.processed_dir, "labels.npy"), labels)
    save_split_indices(len(labels), cfg)

    elapsed = time.time() - start
    print(f"\nPreprocessing complete in {elapsed:.1f}s")
    print(f"  Samples:  {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u} ({cfg.label_names[u]}): {c}")
    print(f"  Saved to: {os.path.abspath(cfg.processed_dir)}")
    vib_size = vib.size * 2 / 1e6   # float16 = 2 bytes
    cur_size = cur.size * 2 / 1e6
    print(f"  Files:    vib_images.npy ({vib_size:.0f} MB), "
          f"cur_images.npy ({cur_size:.0f} MB), "
          f"labels.npy, split_indices.npz  [saved as float16]")


if __name__ == "__main__":
    run_preprocessing()
