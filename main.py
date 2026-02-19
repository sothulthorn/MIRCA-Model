"""
MIRCA — Full pipeline entry point.

Runs all three stages in sequence:
    1. Preprocess  — CWT computation, saved to disk (skipped if already done).
    2. Train       — Model training with validation monitoring.
    3. Analyze     — Metrics, confusion matrix, t-SNE, training curves.

Usage:
    python main.py

Or run each step individually:
    python preprocessing.py      # step 1 (run once)
    python train.py              # step 2
    python analysis.py           # step 3
"""

import os
from config import Config, set_seed
from preprocessing import run_preprocessing
from train import train_model
from analysis import run_analysis


def main():
    cfg = Config()
    set_seed(cfg.seed)

    # ---- Step 1: Preprocessing ----
    processed_flag = os.path.join(cfg.processed_dir, "labels.npy")
    if os.path.exists(processed_flag):
        print("=" * 60)
        print("STEP 1: PREPROCESSING — already done, skipping.")
        print(f"  (delete '{cfg.processed_dir}' to force re-run)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("STEP 1: PREPROCESSING")
        print("=" * 60)
        run_preprocessing(cfg)

    # ---- Step 2: Training ----
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    model, history, test_loader = train_model(cfg)

    # ---- Step 3: Analysis ----
    print("\n" + "=" * 60)
    print("STEP 3: RESULTS ANALYSIS")
    print("=" * 60)
    run_analysis(model=model, test_loader=test_loader,
                 history=history, cfg=cfg)

    print("\n" + "=" * 60)
    print("DONE — all results saved to:", os.path.abspath(cfg.output_dir))
    print("=" * 60)


if __name__ == "__main__":
    main()
