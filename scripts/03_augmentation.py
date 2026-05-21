#!/usr/bin/env python3
"""Augmentation ablation study for skin disease classification.

A 2^3 = 8 configuration ablation. Horizontal and vertical flips are fixed
ON in every configuration; the three remaining transformations are toggled
independently:

  crop      RandomResizedCrop(224, scale=(0.8, 1.0))   (replaces Resize+CenterCrop)
  rotation  RandomRotation(15)
  colour    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

Two subcommands:

  run       Train ResNet18 under one configuration  (GPU required; sbatch this).
  compare   Read all eight JSONs and write the summary CSV + bar chart
            (CPU only; no torch needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Project paths and constants used by both subcommands ---
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = PROJECT / "output" / "transfer_learning"

# 2^3 ablation grid: flips always on; crop/rotation/colour toggled.
CONFIGS = [
    {"id": 1, "crop": False, "rotation": False, "colour": False},
    {"id": 2, "crop": True,  "rotation": False, "colour": False},
    {"id": 3, "crop": False, "rotation": True,  "colour": False},
    {"id": 4, "crop": False, "rotation": False, "colour": True},
    {"id": 5, "crop": True,  "rotation": True,  "colour": False},
    {"id": 6, "crop": True,  "rotation": False, "colour": True},
    {"id": 7, "crop": False, "rotation": True,  "colour": True},
    {"id": 8, "crop": True,  "rotation": True,  "colour": True},
]
CONFIG_BY_ID = {c["id"]: c for c in CONFIGS}

ARCH = "resnet18"
IMG_SIZE = 224
RESIZE_SHORT = 256          # shorter-side resize for the no-crop case and test set
BATCH_SIZE = 64
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
DROP_CLASSES = {"Unknown_Normal", "Lupus", "Sun_Sunlight_Damage", "Moles"}


def config_code(cfg):
    """Short label of which transformations are on, e.g. '-', 'C', 'C+R+J'."""
    parts = []
    if cfg["crop"]:
        parts.append("C")
    if cfg["rotation"]:
        parts.append("R")
    if cfg["colour"]:
        parts.append("J")
    return "+".join(parts) if parts else "—"   # em dash for "none"


# =============================================================================
# Subcommand: run
# =============================================================================
def cmd_run(args):
    """Train ResNet18 under one ablation configuration."""
    # Lazy imports so the compare path doesn't need torch
    import os
    import random
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms

    cfg = CONFIG_BY_ID[args.config_id]
    # Distinct reproducible seed per config: cfg1 -> 42, ..., cfg8 -> 49
    SEED = 42 + args.config_id - 1
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Config {cfg['id']}: crop={cfg['crop']} rotation={cfg['rotation']} "
          f"colour={cfg['colour']}  ({config_code(cfg)})")
    print(f"Seed: {SEED}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def build_train_transform(cfg):
        steps = []
        # Geometric base: random resized crop when crop is on, else deterministic.
        if cfg["crop"]:
            steps.append(transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)))
        else:
            steps.append(transforms.Resize(RESIZE_SHORT))
            steps.append(transforms.CenterCrop(IMG_SIZE))
        # Flips are fixed on in every configuration.
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
        steps.append(transforms.RandomVerticalFlip(p=0.5))
        # Toggled transformations.
        if cfg["rotation"]:
            steps.append(transforms.RandomRotation(15))
        if cfg["colour"]:
            steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                saturation=0.2, hue=0.1))
        steps.append(transforms.ToTensor())
        steps.append(NORMALIZE)
        return transforms.Compose(steps)

    train_transform = build_train_transform(cfg)
    test_transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    # --- Datasets (filtered to 18 classes) ---
    print("\nLoading datasets...")
    full_train = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    full_test = datasets.ImageFolder(str(TEST_DIR), transform=test_transform)

    keep_idxs = {i for i, c in enumerate(full_train.classes) if c not in DROP_CLASSES}
    classes = [c for i, c in enumerate(full_train.classes) if i in keep_idxs]
    old_to_new = {old: new for new, old in enumerate(sorted(keep_idxs))}
    num_classes = len(classes)

    train_indices = [i for i, (_, lbl) in enumerate(full_train.samples) if lbl in keep_idxs]
    test_indices = [i for i, (_, lbl) in enumerate(full_test.samples) if lbl in keep_idxs]
    train_dataset = Subset(full_train, train_indices)
    test_dataset = Subset(full_test, test_indices)

    print(f"  Classes: {num_classes}")
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Class weights for loss
    all_train_labels = [full_train.targets[i] for i in train_indices]
    new_train_labels = [old_to_new[l] for l in all_train_labels]
    class_counts = np.bincount(new_train_labels, minlength=num_classes)
    class_weights = 1.0 / class_counts

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    loss_weights = torch.tensor(class_weights / class_weights.sum() * num_classes,
                                dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    label_remap = torch.tensor([old_to_new.get(i, -1) for i in range(len(full_train.classes))],
                               dtype=torch.long, device=DEVICE)

    # --- Model ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    model = model.to(DEVICE)

    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = label_remap[labels.to(DEVICE)]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = label_remap[labels.to(DEVICE)]
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return np.array(all_preds), np.array(all_labels)

    # --- Phase 1: head only ---
    print(f"\nPhase 1: head only ({PHASE1_EPOCHS} epochs)")
    for p in backbone_params:
        p.requires_grad = False
    optimizer = optim.Adam(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        y_pred, y_true = evaluate(model, test_loader)
        bal = balanced_accuracy_score(y_true, y_pred)
        print(f"  Epoch {epoch}/{PHASE1_EPOCHS} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | test_bal_acc={bal:.4f} | {time.time()-t0:.1f}s")

    # --- Phase 2: full fine-tune ---
    print(f"\nPhase 2: full fine-tune ({PHASE2_EPOCHS} epochs)")
    for p in backbone_params:
        p.requires_grad = True
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR_FINETUNE},
        {"params": head_params, "lr": LR_FINETUNE * 0.1},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    best_bal_acc = 0.0
    best_state = None
    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        y_pred, y_true = evaluate(model, test_loader)
        bal = balanced_accuracy_score(y_true, y_pred)
        scheduler.step()
        print(f"  Epoch {epoch}/{PHASE2_EPOCHS} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | test_bal_acc={bal:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | {time.time()-t0:.1f}s")
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --- Final eval with best checkpoint ---
    print(f"\nLoading best checkpoint (bal_acc={best_bal_acc:.4f}) for final evaluation")
    model.load_state_dict(best_state)
    y_pred, y_true = evaluate(model, test_loader)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # --- Bootstrap CIs ---
    print(f"\nBootstrap CIs ({args.bootstrap_n} resamples)...")
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    acc_boot, bal_boot, f1_boot = [], [], []
    for _ in range(args.bootstrap_n):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        acc_boot.append(accuracy_score(yt, yp))
        bal_boot.append(balanced_accuracy_score(yt, yp))
        f1_boot.append(f1_score(yt, yp, average="macro", zero_division=0))

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    acc_lo, acc_hi = ci(acc_boot)
    bal_lo, bal_hi = ci(bal_boot)
    f1_lo, f1_hi = ci(f1_boot)

    print(f"  Accuracy:          {acc:.4f} ({acc_lo:.4f}-{acc_hi:.4f})")
    print(f"  Balanced Accuracy: {bal_acc:.4f} ({bal_lo:.4f}-{bal_hi:.4f})")
    print(f"  Macro F1:          {f1_macro:.4f} ({f1_lo:.4f}-{f1_hi:.4f})")

    result = {
        "config_id": cfg["id"],
        "crop": cfg["crop"],
        "rotation": cfg["rotation"],
        "colour": cfg["colour"],
        "config_code": config_code(cfg),
        "arch": ARCH,
        "seed": SEED,
        "n_test": int(n),
        "num_classes": int(num_classes),
        "accuracy": round(float(acc), 4),
        "accuracy_ci_low": round(acc_lo, 4),
        "accuracy_ci_high": round(acc_hi, 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "balanced_accuracy_ci_low": round(bal_lo, 4),
        "balanced_accuracy_ci_high": round(bal_hi, 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_macro_ci_low": round(f1_lo, 4),
        "f1_macro_ci_high": round(f1_hi, 4),
        "bootstrap_n": args.bootstrap_n,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
    }
    out_path = OUT_DIR / f"aug_cfg{cfg['id']}_{ARCH}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


# =============================================================================
# Subcommand: compare
# =============================================================================
def cmd_compare(args):
    """Read all eight JSONs and write the summary CSV + comparison bar chart."""
    # Lazy imports (no torch needed here)
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _chicklet import apply_atlas_theme, atlas_bars, ATLAS_GREEN

    rows = []
    for cfg in CONFIGS:
        path = OUT_DIR / f"aug_cfg{cfg['id']}_{ARCH}.json"
        with open(path) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)

    cols = [
        "config_id", "crop", "rotation", "colour", "config_code",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high",
        "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high",
    ]
    csv_path = OUT_DIR / "augmentation_summary.csv"
    df[cols].to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(df[cols].to_string(index=False))

    apply_atlas_theme()
    mm2in = 1 / 25.4

    bal_acc = df["balanced_accuracy"].to_numpy()
    lo_err = bal_acc - df["balanced_accuracy_ci_low"].to_numpy()
    hi_err = df["balanced_accuracy_ci_high"].to_numpy() - bal_acc
    yerr = np.vstack([lo_err, hi_err])
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(150 * mm2in, 75 * mm2in))
    atlas_bars(ax, x, bal_acc, width=0.72, facecolor=ATLAS_GREEN, linewidth=0.3,
               yerr=yerr, capsize=3,
               error_kw={"elinewidth": 0.5, "ecolor": "black"})
    for xi, val, hi in zip(x, bal_acc, df["balanced_accuracy_ci_high"]):
        ax.text(xi, hi + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\ncfg{i}" for c, i in
                        zip(df["config_code"], df["config_id"])])
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Augmentation configuration (C = crop, R = rotation, J = colour jitter)")
    ax.set_ylim(0, 1.0)
    ax.set_title("ResNet18 augmentation ablation")
    ax.margins(x=0.04)
    ax.tick_params(axis="both", length=2, width=0.4)
    ax.axhline(0, color="black", linewidth=0.5, zorder=4, clip_on=False)

    pdf_path = OUT_DIR / "augmentation_comparison.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Augmentation ablation: train ResNet18 over a 2^3 grid of "
                    "crop/rotation/colour toggles, then compare results.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Train one ablation configuration")
    p_run.add_argument("--config_id", type=int, choices=range(1, 9), required=True,
                       metavar="{1..8}")
    p_run.add_argument("--bootstrap_n", type=int, default=1000)
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Compile summary CSV + comparison chart")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
