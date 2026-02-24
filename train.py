"""
<<<<<<< HEAD
Training and evaluation routines for the MIRCA model.

Usage:
    # Make sure preprocessing has been run first:
    #   python preprocessing.py
    python train.py
"""

import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config, set_seed
from dataset import create_dataloaders
from model import MIRCA


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{num_epochs} [train]",
                leave=False, dynamic_ncols=True)
    for vib, cur, labels in pbar:
        vib, cur, labels = vib.to(device), cur.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(vib, cur)
        loss = criterion(logits, labels)
=======
Training and evaluation functions for MIRCA model.

Implements:
- Training loop with SGD optimizer and cross-entropy loss
- Validation loop
- Test evaluation with metrics extraction
- Feature extraction for t-SNE visualization
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

import config
from models import build_model
from utils import compute_metrics


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    epoch=None, num_epochs=None):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    desc = "Train"
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch}/{num_epochs} [Train]"

    pbar = tqdm(train_loader, desc=desc, leave=False, unit="batch")
    for vib_img, cur_img, labels in pbar:
        vib_img = vib_img.to(device)
        cur_img = cur_img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(vib_img, cur_img)
        loss = criterion(outputs, labels)
>>>>>>> mirca-v2
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
<<<<<<< HEAD
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"           [val]  ", leave=False, dynamic_ncols=True)
    for vib, cur, labels in pbar:
        vib, cur, labels = vib.to(device), cur.to(device), labels.to(device)

        logits = model(vib, cur)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def extract_all_features(model, loader, device):
    """Extract penultimate-layer features for t-SNE."""
    model.eval()
    feats, labs = [], []
    for vib, cur, labels in loader:
        vib, cur = vib.to(device), cur.to(device)
        feats.append(model.extract_features(vib, cur).cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def train_model(cfg=None):
    """Full training loop: build model, train, save best weights."""
    if cfg is None:
        cfg = Config()
    set_seed(cfg.seed)

    # Create a unique timestamped subfolder for each run
    run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    cfg.output_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # Model
    model = MIRCA(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total params, {trainable_params:,} trainable")
    print(f"Device: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(cfg.output_dir, "best_model.pth")

    print(f"Results will be saved to: {cfg.output_dir}")
    print(f"Training for {cfg.num_epochs} epochs...\n")

    for epoch in range(1, cfg.num_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device, epoch, cfg.num_epochs)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:3d}/{cfg.num_epochs}]  "
              f"Train: {t_loss:.4f} / {t_acc:.4f}  |  "
              f"Val: {v_loss:.4f} / {v_acc:.4f}  |  LR: {lr:.6f}", end="")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  *best*", end="")
        print()

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Fusion weight (vibration): {model.fusion.fusion_weight:.4f}")

    # Reload best weights
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))

    # Save history
    np.savez(os.path.join(cfg.output_dir, "history.npz"), **history)

    return model, history, test_loader


if __name__ == "__main__":
    model, history, test_loader = train_model()
    print("\nTraining complete. Run `python analysis.py` for results.")
=======
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar with running metrics
        cur_loss = running_loss / len(all_labels)
        cur_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.1f}%")

    avg_loss = running_loss / len(all_labels)
    acc, f1 = compute_metrics(np.array(all_preds), np.array(all_labels))
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, desc="Eval"):
    """Evaluate model on a dataset. Returns loss, accuracy, F1, preds, labels."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(data_loader, desc=desc, leave=False, unit="batch")
    for vib_img, cur_img, labels in pbar:
        vib_img = vib_img.to(device)
        cur_img = cur_img.to(device)
        labels = labels.to(device)

        outputs = model(vib_img, cur_img)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc, f1 = compute_metrics(all_preds, all_labels)
    return avg_loss, acc, f1, all_preds, all_labels


@torch.no_grad()
def extract_features(model, data_loader, device):
    """Extract features from the last hidden layer for t-SNE visualization."""
    model.eval()
    all_features = []
    all_labels = []

    pbar = tqdm(data_loader, desc="Extracting features", leave=False, unit="batch")
    for vib_img, cur_img, labels in pbar:
        vib_img = vib_img.to(device)
        cur_img = cur_img.to(device)

        features = model.extract_features(vib_img, cur_img)
        all_features.append(features.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels)


def measure_inference_time(model, device, input_size=(1, 1, 224, 224),
                           num_runs=100, warmup=10):
    """Measure average inference time in milliseconds."""
    model.eval()
    vib = torch.randn(*input_size).to(device)
    cur = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(vib, cur)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(vib, cur)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms

    return elapsed


def train_model(variant_name, train_loader, val_loader, test_loader,
                device, output_dir=None, num_epochs=None):
    """
    Full training pipeline for one model variant.

    Args:
        variant_name: "Baseline", "Baseline+EMA", "Baseline+CA", or "MIRCA"
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs

    Returns:
        results: dict with accuracy, f1, training history, etc.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, variant_name)
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS

    os.makedirs(output_dir, exist_ok=True)

    # Build model
    model = build_model(
        variant=variant_name,
        num_classes=config.NUM_CLASSES,
        ema_groups=config.EMA_GROUPS,
        use_fusion=True,
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.2f}M")

    # Optimizer and loss (from paper: SGD, lr=0.001, cross-entropy)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_epochs=num_epochs,
        )
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device, desc="Val",
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pth"))

        epoch_pbar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            tr_acc=f"{train_acc:.1f}%",
            val_acc=f"{val_acc:.1f}%",
            best=f"{best_val_acc:.1f}%",
        )

    # Load best model for test evaluation
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pth"),
                    weights_only=True)
    )

    # Test evaluation
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"  Test Results: Accuracy={test_acc:.2f}%, F1={test_f1:.2f}%")

    # Measure inference time
    inference_time = measure_inference_time(model, device)
    print(f"  Inference time: {inference_time:.2f} ms")

    # Extract features for t-SNE
    features, feat_labels = extract_features(model, test_loader, device)

    results = {
        "variant": variant_name,
        "accuracy": test_acc,
        "f1": test_f1,
        "params": num_params,
        "inference_time_ms": inference_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "features": features,
        "feature_labels": feat_labels,
        "model_path": os.path.join(output_dir, "best_model.pth"),
    }

    return results
>>>>>>> mirca-v2
