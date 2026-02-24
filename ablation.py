"""
Ablation Study Runner for MIRCA Model.

Runs all 4 ablation variants (Table 3 from paper):
  1. Baseline       - No attention modules
  2. Baseline+EMA   - Only EMA attention
  3. Baseline+CA    - Only CA attention
  4. MIRCA          - Both EMA and CA (full model)

Usage:
    Step 1: python preprocessing.py   (preprocess raw data)
    Step 2: python ablation.py         (run ablation study)
"""
import os
import json
import torch
import numpy as np

import config
from dataset import load_and_split_data, create_dataloaders
from train import train_model
from utils import (
    plot_confusion_matrix,
    plot_tsne,
    plot_training_curves,
    plot_ablation_comparison,
    print_classification_report,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_ablation_study():
    """Run full ablation study comparing all 4 model variants."""
    print("=" * 70)
    print("MIRCA Ablation Study")
    print("=" * 70)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load and split data
    print("\nLoading preprocessed data...")
    train_dataset, val_dataset, test_dataset = load_and_split_data()

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # Run each ablation variant
    ablation_variants = ["Baseline", "Baseline+EMA", "Baseline+CA", "MIRCA"]
    all_results = {}

    for variant in ablation_variants:
        print(f"\n{'=' * 70}")
        print(f"Training: {variant}")
        print(f"{'=' * 70}")

        set_seed(config.SEED)

        variant_output_dir = os.path.join(config.OUTPUT_DIR, variant)
        results = train_model(
            variant_name=variant,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=variant_output_dir,
            num_epochs=config.NUM_EPOCHS,
        )
        all_results[variant] = results

        # Generate per-variant plots
        # Confusion matrix
        plot_confusion_matrix(
            results["test_preds"], results["test_labels"],
            title=f"Confusion Matrix - {variant}",
            save_path=os.path.join(variant_output_dir, "confusion_matrix.png"),
        )

        # t-SNE visualization
        plot_tsne(
            results["features"], results["feature_labels"],
            title=f"t-SNE Feature Visualization - {variant}",
            save_path=os.path.join(variant_output_dir, "tsne.png"),
        )

        # Training curves
        plot_training_curves(
            results["train_losses"], results["val_losses"],
            results["train_accs"], results["val_accs"],
            title=variant,
            save_path=os.path.join(variant_output_dir, "training_curves.png"),
        )

        # Detailed classification report
        print(f"\nClassification Report - {variant}:")
        print_classification_report(results["test_preds"], results["test_labels"])

    # =========================================================================
    # Summary: Ablation comparison (Table 3 from paper)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS (Table 3)")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'F1-Score':>10} "
          f"{'Params':>12} {'Inference (ms)':>15}")
    print("-" * 70)

    comparison_data = {}
    for variant in ablation_variants:
        r = all_results[variant]
        print(f"{variant:<20} {r['accuracy']:>9.2f}% {r['f1']:>9.2f}% "
              f"{r['params']/1e6:>10.2f}M {r['inference_time_ms']:>13.2f}")
        comparison_data[variant] = {
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            "params_M": r["params"] / 1e6,
            "inference_ms": r["inference_time_ms"],
        }

    # Save comparison plot
    plot_ablation_comparison(
        {v: {"accuracy": all_results[v]["accuracy"],
             "f1": all_results[v]["f1"]}
         for v in ablation_variants},
        save_path=os.path.join(config.OUTPUT_DIR, "ablation_comparison.png"),
    )

    # Save summary JSON
    summary_path = os.path.join(config.OUTPUT_DIR, "ablation_results.json")
    with open(summary_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(f"Plots saved to {config.OUTPUT_DIR}/")

    return all_results


if __name__ == "__main__":
    run_ablation_study()
