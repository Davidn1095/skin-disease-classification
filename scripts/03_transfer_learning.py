#!/usr/bin/env python3
"""Transfer learning with pretrained CNN for Skin Disease Classification.

Strategy:
  1. Fine-tune a pretrained ResNet18 (or EfficientNet-B0) on 224x224 images
  2. Replace final FC layer for 18 classes (4 noisy/overlapping classes dropped)
  3. Use weighted CrossEntropyLoss for class imbalance
  4. Two-phase training: frozen backbone (5 epochs) -> full fine-tune (15 epochs)
  5. Data augmentation: random crop, flip, rotation, color jitter
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix, f1_score
)
balanced_accuracy_score_fn = balanced_accuracy_score  # alias for training loop

# --- Config ---
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = PROJECT / "output" / "transfer_learning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCH = os.environ.get("ARCH", "resnet18")  # resnet18 or efficientnet_b0
IMG_SIZE = 224
BATCH_SIZE = 64
PHASE1_EPOCHS = 5   # frozen backbone
PHASE2_EPOCHS = 15  # full fine-tune
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROP_CLASSES = {"Unknown_Normal", "Lupus", "Sun_Sunlight_Damage", "Moles"}

print(f"Device: {DEVICE}")
print(f"Architecture: {ARCH}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Data transforms ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),  # 256
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Datasets (filtered) ---
print("\nLoading datasets...")
full_train = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
full_test = datasets.ImageFolder(str(TEST_DIR), transform=test_transform)

# Filter out dropped classes
keep_idxs = {i for i, c in enumerate(full_train.classes) if c not in DROP_CLASSES}
classes = [c for i, c in enumerate(full_train.classes) if i in keep_idxs]
old_to_new = {old: new for new, old in enumerate(sorted(keep_idxs))}
num_classes = len(classes)

train_indices = [i for i, (_, lbl) in enumerate(full_train.samples) if lbl in keep_idxs]
test_indices = [i for i, (_, lbl) in enumerate(full_test.samples) if lbl in keep_idxs]
train_dataset = Subset(full_train, train_indices)
test_dataset = Subset(full_test, test_indices)

print(f"  Classes: {num_classes} (dropped {len(DROP_CLASSES)}: {sorted(DROP_CLASSES)})")
print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# --- Class weights for loss (inverse frequency on kept classes) ---
all_train_labels = [full_train.targets[i] for i in train_indices]
new_train_labels = [old_to_new[l] for l in all_train_labels]
class_counts = np.bincount(new_train_labels, minlength=num_classes)
class_weights = 1.0 / class_counts

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# --- Loss weights ---
loss_weights = torch.tensor(class_weights / class_weights.sum() * num_classes, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=loss_weights)

# --- Label remapping (Subset returns original ImageFolder indices) ---
label_remap = torch.tensor([old_to_new.get(i, -1) for i in range(len(full_train.classes))],
                           dtype=torch.long, device=DEVICE)

# --- Model ---
def build_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_params = list(model.classifier.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model, head_params, backbone_params

model, head_params, backbone_params = build_model(ARCH, num_classes)
model = model.to(DEVICE)

# --- Training loop ---
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
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = label_remap[labels.to(DEVICE)]
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

history = {"epoch": [], "phase": [], "train_loss": [], "train_acc": [],
           "test_loss": [], "test_acc": [], "lr": []}

# Phase 1: Train only head (backbone frozen)
print(f"\n{'='*60}")
print(f"PHASE 1: Train head only ({PHASE1_EPOCHS} epochs)")
print(f"{'='*60}")
for p in backbone_params:
    p.requires_grad = False
optimizer = optim.Adam(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

for epoch in range(1, PHASE1_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
    elapsed = time.time() - t0
    print(f"  Epoch {epoch}/{PHASE1_EPOCHS} | "
          f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"Test: loss={test_loss:.4f} acc={test_acc:.4f} | {elapsed:.1f}s")
    history["epoch"].append(epoch)
    history["phase"].append("head")
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["lr"].append(LR_HEAD)

# Phase 2: Fine-tune full model
print(f"\n{'='*60}")
print(f"PHASE 2: Full fine-tune ({PHASE2_EPOCHS} epochs)")
print(f"{'='*60}")
for p in backbone_params:
    p.requires_grad = True
optimizer = optim.Adam([
    {"params": backbone_params, "lr": LR_FINETUNE},
    {"params": head_params, "lr": LR_FINETUNE * 0.1},
], weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

best_bal_acc = 0.0
for epoch in range(1, PHASE2_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
    bal_acc_epoch = balanced_accuracy_score_fn(y_true, y_pred)
    scheduler.step()
    elapsed = time.time() - t0
    cur_lr = scheduler.get_last_lr()[0]
    print(f"  Epoch {epoch}/{PHASE2_EPOCHS} | "
          f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"Test: loss={test_loss:.4f} acc={test_acc:.4f} bal_acc={bal_acc_epoch:.4f} | "
          f"lr={cur_lr:.6f} | {elapsed:.1f}s")

    total_epoch = PHASE1_EPOCHS + epoch
    history["epoch"].append(total_epoch)
    history["phase"].append("finetune")
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["lr"].append(cur_lr)

    if bal_acc_epoch > best_bal_acc:
        best_bal_acc = bal_acc_epoch
        torch.save(model.state_dict(), OUT_DIR / f"best_{ARCH}.pt")
        print(f"    -> New best model saved (bal_acc={best_bal_acc:.4f})")

# --- Final evaluation with best model ---
print(f"\n{'='*60}")
print("FINAL EVALUATION (best checkpoint)")
print(f"{'='*60}")
model.load_state_dict(torch.load(OUT_DIR / f"best_{ARCH}.pt", weights_only=True))
test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)

bal_acc = balanced_accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

print(f"  Accuracy:          {test_acc:.4f}")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  F1 (macro):        {f1_macro:.4f}")
print(f"  F1 (weighted):     {f1_weighted:.4f}")

# Classification report
report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
pd.DataFrame(report).T.to_csv(OUT_DIR / f"report_{ARCH}.csv")

# --- Style (FIGURE_STYLE.md) ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.which": "major",
    "grid.color": "#E5E5E5", "grid.linewidth": 0.3,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "savefig.bbox": "tight", "savefig.facecolor": "white", "savefig.pad_inches": 0.05,
})
TRAIN_COLOUR = "#0072B2"
TEST_COLOUR = "#D55E00"
mm2in = 1 / 25.4

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(140 * mm2in, 130 * mm2in))
n_cls = len(classes)
# Draw with pcolormesh for white cell borders
im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="equal")
fig.colorbar(im, ax=ax, shrink=0.75)
ax.set_xticks(np.arange(n_cls))
ax.set_yticks(np.arange(n_cls))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
ax.set_yticklabels(classes, rotation=0, fontsize=6)
# White grid lines between cells
for edge in range(n_cls + 1):
    ax.axhline(edge - 0.5, color="white", linewidth=0.3)
    ax.axvline(edge - 0.5, color="white", linewidth=0.3)
# Annotate cells
thresh = cm.max() / 2.0
for i in range(n_cls):
    for j in range(n_cls):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=6)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {ARCH} (Acc={test_acc:.3f}, BalAcc={bal_acc:.3f})")
fig.savefig(OUT_DIR / f"confusion_{ARCH}.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(155 * mm2in, 60 * mm2in))
epochs = history["epoch"]

ax = axes[0]
ax.plot(epochs, history["train_loss"], "o-", color=TRAIN_COLOUR, markersize=3, label="Train")
ax.plot(epochs, history["test_loss"], "o--", color=TEST_COLOUR, markersize=3, label="Test")
ax.axvline(PHASE1_EPOCHS + 0.5, color="grey", linestyle="--", alpha=0.5, label="Unfreeze")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.legend(fontsize=6)

ax = axes[1]
ax.plot(epochs, history["train_acc"], "o-", color=TRAIN_COLOUR, markersize=3, label="Train")
ax.plot(epochs, history["test_acc"], "o--", color=TEST_COLOUR, markersize=3, label="Test")
ax.axvline(PHASE1_EPOCHS + 0.5, color="grey", linestyle="--", alpha=0.5, label="Unfreeze")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training Accuracy")
ax.legend(fontsize=6)

plt.suptitle(f"{ARCH} Transfer Learning", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / f"training_curves_{ARCH}.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# Save history for offline regeneration
with open(OUT_DIR / f"history_{ARCH}.json", "w") as f:
    json.dump(history, f)

# Save summary
summary = {
    "arch": ARCH, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
    "phase1_epochs": PHASE1_EPOCHS, "phase2_epochs": PHASE2_EPOCHS,
    "accuracy": round(test_acc, 4), "balanced_accuracy": round(bal_acc, 4),
    "f1_macro": round(f1_macro, 4), "f1_weighted": round(f1_weighted, 4),
    "best_epoch": int(history["epoch"][np.argmax(history["test_acc"])]),
    "device": str(DEVICE),
}
with open(OUT_DIR / f"summary_{ARCH}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to: {OUT_DIR}")
print(f"Random baseline ({num_classes} classes): {1/num_classes:.4f}")
