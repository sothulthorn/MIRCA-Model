"""
<<<<<<< HEAD
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
=======
Data preprocessing pipeline for MIRCA model.

Steps (from paper):
1. Load .mat files from Paderborn bearing dataset
2. Extract vibration and current signals
3. Z-score normalization
4. Sliding window segmentation (window=1024)
5. CWT transformation to time-frequency images
6. Convert to grayscale images and save

Usage:
    python preprocessing.py
"""
import os
import glob
import numpy as np
from scipy.io import loadmat
from PIL import Image
import pywt
from tqdm import tqdm

import config


def explore_mat_structure(filepath, max_depth=6):
    """Recursively explore .mat file structure to find signal arrays."""
    mat = loadmat(filepath, struct_as_record=True)
    var_name = [k for k in mat.keys() if not k.startswith("__")][0]
    print(f"Variable name: {var_name}")
    print(f"Type: {type(mat[var_name])}")
    print(f"Shape: {mat[var_name].shape}")
    print(f"Dtype: {mat[var_name].dtype}")

    if mat[var_name].dtype.names:
        print(f"Fields: {mat[var_name].dtype.names}")

    def _explore(obj, depth=0, prefix=""):
        if depth > max_depth:
            return
        indent = "  " * depth
        if hasattr(obj, "dtype") and obj.dtype.names:
            for name in obj.dtype.names:
                child = obj[name]
                if hasattr(child, "shape"):
                    print(f"{indent}{prefix}{name}: shape={child.shape}, "
                          f"dtype={child.dtype}")
                _explore(child.flat[0] if child.size == 1 else child,
                         depth + 1, f"{name}.")
        elif isinstance(obj, np.ndarray):
            if obj.dtype == object and obj.size > 0:
                for i in range(min(obj.shape[-1], 8)):
                    try:
                        elem = obj.flat[i] if obj.ndim == 1 else obj[0, i]
                        if isinstance(elem, np.ndarray):
                            print(f"{indent}{prefix}[{i}]: shape={elem.shape}, "
                                  f"dtype={elem.dtype}")
                            _explore(elem, depth + 1, f"[{i}].")
                    except (IndexError, TypeError):
                        pass

    _explore(mat[var_name])
    return mat, var_name


def load_paderborn_signals(filepath):
    """
    Load vibration and current signals from a Paderborn dataset .mat file.

    The Paderborn .mat files have a nested struct:
        variable -> Y -> channels[0..N] -> (description, unit, signal_data)

    Channel layout (typical):
        0: force, 1: phase_current_1, 2: phase_current_2,
        3-5: other sensors, 6: vibration_1

    Returns:
        vibration: 1D numpy array of vibration signal
        current: 1D numpy array of current signal (phase 1)
    """
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    var_name = [k for k in mat.keys() if not k.startswith("__")][0]
    data = mat[var_name]

    # Navigate the nested structure to get Y measurement channels
    # Try multiple common access patterns
    vibration = None
    current = None

    try:
        # Pattern 1: data.Y is an array of channel objects
        Y = data.Y
        if hasattr(Y, "__len__") and not isinstance(Y, str):
            channels = Y
            # Try to extract vibration (usually last or index 6)
            # and current (usually index 1)
            for i, ch in enumerate(channels):
                ch_data = None
                ch_name = ""

                # Each channel may be a struct with fields
                if hasattr(ch, "Y"):
                    ch_data = np.asarray(ch.Y).flatten()
                elif isinstance(ch, np.ndarray) and ch.ndim >= 1:
                    ch_data = ch.flatten()

                # Try to get channel description
                if hasattr(ch, "Name"):
                    ch_name = str(ch.Name).lower()
                elif hasattr(ch, "name"):
                    ch_name = str(ch.name).lower()

                if ch_data is not None and len(ch_data) > 1000:
                    if "vibration" in ch_name or i == 6:
                        vibration = ch_data.astype(np.float64)
                    elif ("current" in ch_name and "1" in ch_name) or i == 1:
                        current = ch_data.astype(np.float64)
    except (AttributeError, TypeError, IndexError):
        pass

    # Pattern 2: direct struct array access
    if vibration is None or current is None:
        try:
            mat2 = loadmat(filepath, struct_as_record=True)
            var_name2 = [k for k in mat2.keys() if not k.startswith("__")][0]
            struct = mat2[var_name2]
            Y_data = struct["Y"][0, 0]

            if Y_data.dtype == object:
                # Y_data is array of channel structs
                for i in range(Y_data.shape[-1]):
                    try:
                        ch = Y_data[0, i] if Y_data.ndim == 2 else Y_data[i]
                        # Channel struct: [description, unit, signal_data]
                        if isinstance(ch, np.ndarray) and ch.dtype == object:
                            signal = ch[2] if ch.size >= 3 else ch[-1]
                        elif hasattr(ch, "dtype") and ch.dtype.names:
                            signal = ch["Y"][0, 0] if "Y" in ch.dtype.names else None
                        else:
                            signal = ch

                        if signal is not None:
                            signal = np.asarray(signal).flatten().astype(np.float64)
                            if len(signal) > 1000:
                                if i == 6 and vibration is None:
                                    vibration = signal
                                elif i == 1 and current is None:
                                    current = signal
                    except (IndexError, TypeError, ValueError):
                        continue
        except (KeyError, IndexError, TypeError):
            pass

    # Pattern 3: flat signal arrays
    if vibration is None or current is None:
        try:
            mat3 = loadmat(filepath, struct_as_record=True)
            var_name3 = [k for k in mat3.keys() if not k.startswith("__")][0]
            struct = mat3[var_name3]

            # Try nested indexing: struct[0][0][2][0][channel_idx][2]
            Y_outer = struct[0][0][2][0]
            for idx, target in [(6, "vibration"), (1, "current")]:
                try:
                    ch_struct = Y_outer[idx]
                    signal = ch_struct[2].flatten().astype(np.float64)
                    if len(signal) > 1000:
                        if target == "vibration" and vibration is None:
                            vibration = signal
                        elif target == "current" and current is None:
                            current = signal
                except (IndexError, TypeError):
                    continue
        except (IndexError, TypeError, KeyError):
            pass

    if vibration is None:
        raise ValueError(f"Could not extract vibration signal from {filepath}")
    if current is None:
        raise ValueError(f"Could not extract current signal from {filepath}")

    return vibration, current


def zscore_normalize(signal):
    """Z-score normalization: (x - mean) / std."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-10:
        return signal - mean
    return (signal - mean) / std


def sliding_window_segment(signal, window_length, stride):
    """Segment signal into fixed-length windows."""
    segments = []
    num_segments = (len(signal) - window_length) // stride + 1
    for i in range(num_segments):
        start = i * stride
        segment = signal[start:start + window_length]
        segments.append(segment)
    return np.array(segments)


def apply_cwt(segment, wavelet="morl", num_scales=100, sampling_period=1/64000):
    """
    Apply Continuous Wavelet Transform to a 1D signal segment.

    Returns:
        coefficients: 2D array of CWT coefficients (num_scales, segment_length)
    """
    scales = np.arange(1, num_scales + 1)
    coefficients, _ = pywt.cwt(segment, scales, wavelet,
                               sampling_period=sampling_period)
    return np.abs(coefficients)


def cwt_to_grayscale(coefficients, target_size=224):
    """
    Convert CWT coefficient matrix to a grayscale image.

    Steps:
    1. Take absolute value (already done in apply_cwt)
    2. Normalize to [0, 255]
    3. Resize to target_size x target_size
    """
    # Normalize to [0, 255]
    c_min, c_max = coefficients.min(), coefficients.max()
    if c_max - c_min < 1e-10:
        normalized = np.zeros_like(coefficients, dtype=np.uint8)
    else:
        normalized = ((coefficients - c_min) / (c_max - c_min) * 255).astype(
            np.uint8
        )

    # Resize to target size
    img = Image.fromarray(normalized, mode="L")
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def get_mat_files(bearing_code, dataset_root, operating_condition):
    """Get all .mat files for a bearing under the specified operating condition."""
    bearing_dir = os.path.join(dataset_root, bearing_code)
    pattern = os.path.join(bearing_dir, f"{operating_condition}_{bearing_code}_*.mat")
    files = sorted(glob.glob(pattern))
    return files


def preprocess_dataset(dataset_root=None, output_dir=None,
                       operating_condition=None, max_files_per_bearing=20):
    """
    Full preprocessing pipeline:
    1. Load .mat files for each bearing
    2. Extract & normalize signals
    3. Segment with sliding window
    4. Apply CWT and convert to grayscale
    5. Save as numpy arrays

    Output structure:
        output_dir/
            vibration_images.npy   (N, H, W) uint8
            current_images.npy     (N, H, W) uint8
            labels.npy             (N,) int64
    """
    if dataset_root is None:
        dataset_root = config.DATASET_ROOT
    if output_dir is None:
        output_dir = config.PROCESSED_DIR
    if operating_condition is None:
        operating_condition = config.OPERATING_CONDITION

    os.makedirs(output_dir, exist_ok=True)

    all_vib_images = []
    all_cur_images = []
    all_labels = []

    bearing_codes = sorted(config.BEARING_LABEL_MAP.keys())

    bearing_pbar = tqdm(bearing_codes, desc="Bearings", unit="bearing")
    for bearing_code in bearing_pbar:
        label = config.BEARING_LABEL_MAP[bearing_code]
        mat_files = get_mat_files(bearing_code, dataset_root, operating_condition)

        if not mat_files:
            bearing_pbar.set_postfix_str(
                f"{bearing_code}: no files for {operating_condition}")
            continue

        mat_files = mat_files[:max_files_per_bearing]
        bearing_pbar.set_postfix_str(
            f"{bearing_code} (label={label}, {config.LABEL_NAMES[label]})")

        file_pbar = tqdm(mat_files, desc=f"  {bearing_code} files",
                         unit="file", leave=False)
        for fpath in file_pbar:
            try:
                vibration, current = load_paderborn_signals(fpath)
            except (ValueError, KeyError) as e:
                file_pbar.set_postfix_str(f"SKIP: {e}")
                continue

            # Z-score normalization
            vibration = zscore_normalize(vibration)
            current = zscore_normalize(current)

            # Sliding window segmentation
            vib_segments = sliding_window_segment(
                vibration, config.WINDOW_LENGTH, config.WINDOW_STRIDE
            )
            cur_segments = sliding_window_segment(
                current, config.WINDOW_LENGTH, config.WINDOW_STRIDE
            )

            # Use minimum number of segments from both signals
            n_segments = min(len(vib_segments), len(cur_segments))

            for seg_idx in range(n_segments):
                # CWT transform
                vib_cwt = apply_cwt(
                    vib_segments[seg_idx],
                    wavelet=config.CWT_WAVELET,
                    num_scales=config.CWT_NUM_SCALES,
                    sampling_period=1.0 / config.SAMPLING_RATE,
                )
                cur_cwt = apply_cwt(
                    cur_segments[seg_idx],
                    wavelet=config.CWT_WAVELET,
                    num_scales=config.CWT_NUM_SCALES,
                    sampling_period=1.0 / config.SAMPLING_RATE,
                )

                # Convert to grayscale images
                vib_img = cwt_to_grayscale(vib_cwt, config.IMAGE_SIZE)
                cur_img = cwt_to_grayscale(cur_cwt, config.IMAGE_SIZE)

                all_vib_images.append(vib_img)
                all_cur_images.append(cur_img)
                all_labels.append(label)

            file_pbar.set_postfix_str(f"{n_segments} segments")

    # Convert to numpy arrays
    all_vib_images = np.array(all_vib_images, dtype=np.uint8)
    all_cur_images = np.array(all_cur_images, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Class distribution:")
    for label in sorted(np.unique(all_labels)):
        count = np.sum(all_labels == label)
        print(f"  Label {label} ({config.LABEL_NAMES[label]}): {count}")

    # Save
    np.save(os.path.join(output_dir, "vibration_images.npy"), all_vib_images)
    np.save(os.path.join(output_dir, "current_images.npy"), all_cur_images)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)

    print(f"\nSaved to {output_dir}/")
    print(f"  vibration_images.npy: {all_vib_images.shape}")
    print(f"  current_images.npy:   {all_cur_images.shape}")
    print(f"  labels.npy:           {all_labels.shape}")

    return all_vib_images, all_cur_images, all_labels


if __name__ == "__main__":
    print("=" * 60)
    print("MIRCA Data Preprocessing")
    print("=" * 60)
    print(f"Dataset root:        {config.DATASET_ROOT}")
    print(f"Operating condition: {config.OPERATING_CONDITION}")
    print(f"Window length:       {config.WINDOW_LENGTH}")
    print(f"CWT wavelet:         {config.CWT_WAVELET}")
    print(f"Image size:          {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Output dir:          {config.PROCESSED_DIR}")
    print()

    # Optional: explore .mat structure first
    sample_file = get_mat_files(
        "K001", config.DATASET_ROOT, config.OPERATING_CONDITION
    )
    if sample_file:
        print("Exploring .mat file structure:")
        explore_mat_structure(sample_file[0])
        print()

    print("Starting preprocessing...")
    preprocess_dataset()
>>>>>>> mirca-v2
