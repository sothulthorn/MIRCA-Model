"""
Results analysis: metrics, confusion matrix, t-SNE, training curves.

Usage:
    # After training:
    python analysis.py

    # Or call from main.py which chains everything together.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from config import Config, set_seed
from dataset import create_dataloaders
from model import MIRCA
from train import evaluate, extract_all_features


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, label_names):
    """Print accuracy, F1, and per-class classification report."""
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:            {acc * 100:.2f}%")
    print(f"  F1 Score (macro):    {f1_macro * 100:.2f}%")
    print(f"  F1 Score (weighted): {f1_weighted * 100:.2f}%")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=label_names, digits=4))

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# ============================================================
# Confusion matrix
# ============================================================

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Normalized
    im1 = axes[1].imshow(cm_norm, interpolation="nearest",
                         cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center",
                         va="center",
                         color="white" if cm_norm[i, j] > 0.5 else "black")

    for ax in axes:
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(label_names, fontsize=9)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


# ============================================================
# t-SNE visualization
# ============================================================

def plot_tsne(features, labels, label_names, save_path, perplexity=30):
    from sklearn.manifold import TSNE

    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))

    for cls_idx, name in enumerate(label_names):
        mask = labels == cls_idx
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[colors[cls_idx]], label=name,
                   alpha=0.7, s=20, edgecolors="none")

    ax.legend(loc="best", fontsize=9, markerscale=2)
    ax.set_title("t-SNE Visualization of Extracted Features", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


# ============================================================
# Training curves
# ============================================================

def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-", label="Train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {save_path}")


# ============================================================
# Run full analysis
# ============================================================

def run_analysis(model=None, test_loader=None, history=None, cfg=None):
    """
    Load the best model and evaluate on the test set.

    Can be called standalone (loads model + data from disk) or
    with a model/loader already in memory from train.py.
    """
    if cfg is None:
        cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # Load model from checkpoint if not provided
    if model is None:
        model = MIRCA(cfg)
        ckpt = os.path.join(cfg.output_dir, "best_model.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        print(f"Loaded model from {ckpt}")
    model = model.to(device)

    # Load data if not provided
    if test_loader is None:
        _, _, test_loader = create_dataloaders(cfg)

    # Load history if not provided
    if history is None:
        hist_path = os.path.join(cfg.output_dir, "history.npz")
        if os.path.exists(hist_path):
            h = np.load(hist_path)
            history = {k: h[k].tolist() for k in h.files}

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f},  Test Accuracy: {test_acc:.4f}")

    # Metrics
    metrics = compute_metrics(y_true, y_pred, cfg.label_names)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, cfg.label_names,
                          os.path.join(cfg.output_dir, "confusion_matrix.png"))

    # t-SNE
    features, feat_labels = extract_all_features(model, test_loader, device)
    plot_tsne(features, feat_labels, cfg.label_names,
              os.path.join(cfg.output_dir, "tsne_visualization.png"))

    # Training curves
    if history is not None:
        plot_training_curves(history,
                             os.path.join(cfg.output_dir, "training_curves.png"))

    return metrics


if __name__ == "__main__":
    run_analysis()
