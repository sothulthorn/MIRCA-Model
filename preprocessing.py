"""
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