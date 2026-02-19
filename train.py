"""
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
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
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
