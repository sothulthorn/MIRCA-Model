"""Configuration for the MIRCA fault diagnosis framework."""

import os
import random
import numpy as np
import torch


class Config:
    # --- Dataset ---
    data_dir = "./data/paderborn"                   # Root of Paderborn bearing dirs
    processed_dir = "./data/processed"              # Precomputed CWT images saved here
    sampling_rate = 64000                           # 64 kHz
    window_length = 1024                            # Sliding window size
    window_stride = 512                             # Sliding window stride
    image_size = 224                                # CWT image resized to 224x224
    num_classes = 8                                 # 8 bearing health states
    train_ratio = 0.70
    val_ratio = 0.20
    test_ratio = 0.10

    # --- Operating condition filter (paper: 1500 rpm, 0.7 Nm, 1000 N) ---
    # Paderborn file naming: N{rpm/100}_M{torque*10}_F{force/100}_{bearing}_{run}.mat
    # N15 = 1500 rpm,  M07 = 0.7 Nm,  F10 = 1000 N
    operating_condition = "N15_M07_F10"

    # --- Signal channels inside .mat files ---
    vibration_channel = 6      # "vibration_1"       (256001 samples)
    current_channel = 1        # "phase_current_1"   (256001 samples)

    # --- Segment budget ---
    # Each .mat file has ~499 segments (256k samples, window=1024, stride=512).
    # target_samples_per_class equalizes all classes by adjusting segments/file:
    #   1-bearing class: 180 segs/file × 20 files = 3600
    #   2-bearing class:  90 segs/file × 20 files = 3600
    #   3-bearing class:  60 segs/file × 20 files = 3600
    # Set to None to use max_segments_per_file uniformly instead.
    target_samples_per_class = 3600
    max_segments_per_file = 60              # Fallback when target is None

    # --- CWT ---
    cwt_num_scales = 224                    # Number of wavelet scales
    cwt_omega0 = 6.0                        # Morlet wavelet center frequency
    cwt_scale_min = 1.0
    cwt_scale_max = 128.0

    # --- Model ---
    in_channels = 1                         # Grayscale fused image
    stage_channels = [64, 128, 256, 512]
    stage_blocks = [3, 4, 6, 3]             # Matches ResNet-34 layout
    ca_reduction = 32                       # CA channel reduction ratio
    ema_groups = 8                          # EMA channel groups

    # --- Training ---
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    num_workers = 0                         # Set >0 for multi-process data loading

    # --- Output ---
    output_dir = "./results"

    # --- Paderborn bearing-to-label mapping (Table 1 of the paper) ---
    #
    # The dataset has 32 bearing directories, but the paper only uses the
    # 20 bearings below.  The rest (KA04, KA06, KA15, KA16, KA22, KA30,
    # KI14, KI16, KI17, KI18, KI21) are ignored.
    #
    # Label 0: Electrical discharge trenches — Inner Race
    # Label 1: Electrical discharge trenches — Outer Race
    # Label 2: Fatigue pitting              — Inner Race
    # Label 3: Fatigue pitting              — Outer Race
    # Label 4: Drilling holes               — Outer Race
    # Label 5: Electric engraver pitting    — Inner Race
    # Label 6: Electric engraver pitting    — Outer Race
    # Label 7: Normal
    # bearing_label_map = {
    #     "KI01": 0, "KI04": 0,                        # EDM inner race
    #     "KA01": 1,                                   # EDM outer race
    #     "KB23": 2, "KB27": 2,                        # Fatigue pitting inner race
    #     "KB24": 3,                                   # Fatigue pitting outer race
    #     "KA07": 4, "KA08": 4, "KA09": 4,             # Drilling outer race
    #     "KI03": 5, "KI07": 5, "KI08": 5,             # Electric engraver inner race
    #     "KA03": 6, "KA05": 6, "KI05": 6,             # Electric engraver outer race
    #     "K001": 7, "K002": 7, "K003": 7,             # Normal / healthy
    # }

    bearing_label_map = {
        "KI01": 0,          # EDM inner race
        "KA01": 1,          # EDM outer race
        "KI04": 2,          # Fatigue pitting inner race
        "KA04": 3,          # Fatigue pitting outer race
        "KA07": 4,          # Drilling outer race
        "KI03": 5,          # Electric engraver inner race
        "KA03": 6,          # Electric engraver outer race
        "K001": 7,          # Normal / healthy
    }

    label_names = [
        "EDM-IR", "EDM-OR", "Fatigue-IR", "Fatigue-OR",
        "Drill-OR", "Engrave-IR", "Engrave-OR", "Normal",
    ]


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
