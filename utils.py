"""
Evaluation utilities for MIRCA model.

- Accuracy and F1 score computation
- Confusion matrix plotting
- t-SNE feature visualization
- Training curve plotting
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE

import config


def compute_metrics(all_preds, all_labels):
    """Compute accuracy and macro F1 score."""
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average="macro") * 100
    return acc, f1


def plot_confusion_matrix(all_preds, all_labels, title="Confusion Matrix",
                          save_path=None):
    """Plot and save confusion matrix (Figure 9 from paper)."""
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize to percentages
    cm_normalized = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm_normalized / row_sums * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues,
                   vmin=0, vmax=100)
    plt.colorbar(im, ax=ax)

    num_classes = cm.shape[0]
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )

    # Add text annotations
    thresh = 50.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_tsne(features, labels, title="t-SNE Visualization", save_path=None):
    """
    t-SNE visualization of extracted features (Figure 8 from paper).

    Args:
        features: (N, D) numpy array of features from model's last hidden layer
        labels: (N,) numpy array of class labels
    """
    tsne = TSNE(n_components=2, random_state=config.SEED, perplexity=30)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    for label in range(config.NUM_CLASSES):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            s=15, alpha=0.7,
            label=f"{label}: {config.LABEL_NAMES.get(label, str(label))}"
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         title="Training Curves", save_path=None):
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.close()


def plot_ablation_comparison(results_dict, save_path=None):
    """
    Plot ablation study results comparison (Table 3 from paper).

    Args:
        results_dict: {variant_name: {"accuracy": float, "f1": float}}
    """
    variants = list(results_dict.keys())
    accuracies = [results_dict[v]["accuracy"] for v in variants]
    f1_scores = [results_dict[v]["f1"] for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)",
                   color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score (%)",
                   color="#DD8452")

    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation Study Results")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(80, 101)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Ablation comparison saved to {save_path}")
    plt.close()


def print_classification_report(all_preds, all_labels):
    """Print detailed classification report."""
    target_names = [
        f"{i}: {config.LABEL_NAMES[i]}" for i in range(config.NUM_CLASSES)
    ]
    print(classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    ))
