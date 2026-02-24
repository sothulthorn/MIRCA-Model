"""
Configuration for MIRCA fault diagnosis model.
Based on: "Intelligent fault diagnosis based on multi-source information fusion
           and attention-enhanced networks" (Scientific Reports, 2025)
"""
import os

# ==============================================================================
# Dataset Paths
# ==============================================================================
DATASET_ROOT = "./data/paderborn"
PROCESSED_DIR = "./data/processed"
OUTPUT_DIR = "./results"

# Operating condition filter (paper uses 1500 rpm, 0.7 Nm, 1000 N)
OPERATING_CONDITION = "N15_M07_F10"

# ==============================================================================
# Bearing-to-Label Mapping (Table 1 from paper)
# ==============================================================================
# Label 0: Electrical discharge trenches - Inner Race
# Label 1: Electrical discharge trenches - Outer Race
# Label 2: Fatigue pitting - Inner Race
# Label 3: Fatigue pitting - Outer Race
# Label 4: Drilling holes - Outer Race
# Label 5: Electric engraver pitting - Inner Race
# Label 6: Electric engraver pitting - Outer Race
# Label 7: Normal

BEARING_LABEL_MAP = {
    # Electrical discharge trenches (EDM) - Inner Race -> Label 0
    "KI01": 0,
    # Electrical discharge trenches (EDM) - Outer Race -> Label 1
    "KA01": 1,
    "KA06": 1,
    # Fatigue pitting - Inner Race -> Label 2
    "KI04": 2,
    "KI14": 2,
    # "KI16": 2,
    # "KI17": 2,
    # "KI18": 2,
    # "KI21": 2,
    # Fatigue pitting - Outer Race -> Label 3
    "KA04": 3,
    "KA15": 3,
    # "KA16": 3,
    # "KA22": 3,
    # "KA30": 3,
    # Drilling holes - Outer Race -> Label 4
    "KA07": 4,
    "KA08": 4,
    # "KA09": 4,
    # Electric engraver pitting - Inner Race -> Label 5
    "KI03": 5,
    "KI05": 5,
    # "KI07": 5,
    # "KI08": 5,
    # Electric engraver pitting - Outer Race -> Label 6
    "KA03": 6,
    "KA05": 6,
    # Normal -> Label 7
    "K001": 7,
    "K002": 7,
    # "K003": 7,
    # "K004": 7,
    # "K005": 7,
    # "K006": 7,
}

LABEL_NAMES = {
    0: "EDM_Inner",
    1: "EDM_Outer",
    2: "Fatigue_Inner",
    3: "Fatigue_Outer",
    4: "Drilling_Outer",
    5: "Engraver_Inner",
    6: "Engraver_Outer",
    7: "Normal",
}

NUM_CLASSES = 8

# ==============================================================================
# Signal Processing Parameters
# ==============================================================================
SAMPLING_RATE = 64000          # 64 kHz
WINDOW_LENGTH = 1024           # Sliding window length
WINDOW_STRIDE = 1024           # Non-overlapping windows
CWT_WAVELET = "morl"           # Morlet wavelet for CWT
CWT_NUM_SCALES = 100           # Number of CWT scales
IMAGE_SIZE = 224               # Target image size for CNN input

# ==============================================================================
# Dataset Split Ratios (from paper)
# ==============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ==============================================================================
# Training Hyperparameters (from paper)
# ==============================================================================
LEARNING_RATE = 0.001
OPTIMIZER = "SGD"
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 4

# ==============================================================================
# Model Configuration
# ==============================================================================
# ResNet-34 style: layers [3, 4, 6, 3], channels [64, 128, 256, 512]
STAGE_LAYERS = [3, 4, 6, 3]
STAGE_CHANNELS = [64, 128, 256, 512]
EMA_GROUPS = 8                 # EMA channel group size

# Ablation study configurations
ABLATION_CONFIGS = {
    "Baseline":     {"use_ca": False, "use_ema": False},
    "Baseline+EMA": {"use_ca": False, "use_ema": True},
    "Baseline+CA":  {"use_ca": True,  "use_ema": False},
    "MIRCA":        {"use_ca": True,  "use_ema": True},
}

# Random seed for reproducibility
SEED = 42
