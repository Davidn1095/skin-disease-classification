#!/usr/bin/env python3
"""Generate PDF figures from saved model checkpoints (no training).

Loads best_resnet18.pt and best_efficientnet_b0.pt, runs inference on the
test set, and produces:
  - confusion_resnet18.pdf
  - confusion_efficientnet_b0.pdf
  - training_curves_resnet18.pdf   (only if history JSON exists)
  - training_curves_efficientnet_b0.pdf (only if history JSON exists)
  - comparison_bal_acc_bar.pdf

Also computes 95% bootstrap CIs (1000 resamples) for accuracy,
balanced accuracy, and macro F1.

Follows FIGURE_STYLE.md conventions throughout.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

# --- Style (FIGURE_STYLE.md) ---
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.which": "major",
        "grid.color": "#E5E5E5", "grid.linewidth": 0.3,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.fontsize": 7, "legend.framealpha": 0.8, "legend.edgecolor": "none",
        "figure.facecolor": "white", "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.facecolor": "white", "savefig.pad_inches": 0.05,
    })

set_style()

TRAIN_COLOUR = "#0072B2"
TEST_COLOUR = "#D55E00"
RESNET_COLOUR = "#0072B2"
EFFNET_COLOUR = "#56B4E9"
mm2in = 1 / 25.4

# --- Config ---
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = PROJECT / "output" / "transfer_learning"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROP_CLASSES = {"Unknown_Normal", "Lupus", "Sun_Sunlight_Damage", "Moles"}
PHASE1_EPOCHS = 5
N_BOOTSTRAP = 1000
RNG_SEED = 42

print(f"Device: {DEVICE}")

# --- Test transform (same as training script) ---
test_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),  # 256
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Class list (from directory structure, matching ImageFolder alphabetical order) ---
TRAIN_DIR = DATA_DIR / "train"
all_classes = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
keep_idxs = {i for i, c in enumerate(all_classes) if c not in DROP_CLASSES}
classes = [c for i, c in enumerate(all_classes) if i in keep_idxs]
old_to_new = {old: new for new, old in enumerate(sorted(keep_idxs))}
num_classes = len(classes)

print(f"Classes: {num_classes}")

# --- Test dataset (only needed for inference fallback) ---
test_loader = None
label_remap = None

def get_test_loader():
    """Lazily build the test loader (only needed when .npz predictions are missing)."""
    global test_loader, label_remap
    if test_loader is not None:
        return test_loader
    full_test = datasets.ImageFolder(str(TEST_DIR), transform=test_transform)
    test_indices = [i for i, (_, lbl) in enumerate(full_test.samples) if lbl in keep_idxs]
    test_dataset = Subset(full_test, test_indices)
    print(f"Test images: {len(test_dataset)}")
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    label_remap = torch.tensor(
        [old_to_new.get(i, -1) for i in range(len(all_classes))],
        dtype=torch.long, device=DEVICE
    )
    return test_loader

# --- Model builder ---
def build_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

# --- Evaluate ---
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

# --- Bootstrap CIs ---
def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Compute 95% bootstrap CIs for accuracy, balanced accuracy, macro F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_acc, boot_bal, boot_f1 = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        boot_acc.append((yt == yp).mean())
        boot_bal.append(balanced_accuracy_score(yt, yp))
        boot_f1.append(f1_score(yt, yp, average="macro", zero_division=0))
    results = {}
    for name, vals in [("accuracy", boot_acc), ("balanced_accuracy", boot_bal),
                        ("f1_macro", boot_f1)]:
        vals = np.array(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        results[name] = (lo, hi)
    return results

# --- Plot confusion matrix ---
def plot_confusion_matrix(y_true, y_pred, arch, acc, bal_acc):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(140 * mm2in, 130 * mm2in))
    n_cls = len(classes)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_xticks(np.arange(n_cls))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(classes, rotation=0, fontsize=6)
    for edge in range(n_cls + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=0.3)
        ax.axvline(edge - 0.5, color="white", linewidth=0.3)
    thresh = cm.max() / 2.0
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {arch} (Acc={acc:.3f}, BalAcc={bal_acc:.3f})")
    ax.grid(False)
    out_path = OUT_DIR / f"confusion_{arch}.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

# --- Plot training curves (from history JSON) ---
def plot_training_curves(arch):
    hist_path = OUT_DIR / f"history_{arch}.json"
    if not hist_path.exists():
        print(f"SKIP: {hist_path} not found — training curves require a training run")
        return
    with open(hist_path) as f:
        history = json.load(f)

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

    plt.suptitle(f"{arch} Transfer Learning", fontsize=10)
    plt.tight_layout()
    out_path = OUT_DIR / f"training_curves_{arch}.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

# --- Per-class balanced accuracy (one-vs-rest) ---
RF_COLOUR = "#D55E00"   # Okabe-Ito vermillion
BASELINE_DIR = PROJECT / "output" / "baseline"

def per_class_bal_acc_ovr(y_true, y_pred, class_list):
    """Compute one-vs-rest balanced accuracy for each class.

    For class c:
      TP = correctly predicted as c
      FN = truly c but predicted as other
      TN = truly other and predicted as other
      FP = truly other but predicted as c
      sensitivity = TP / (TP + FN)
      specificity = TN / (TN + FP)
      balanced_accuracy = (sensitivity + specificity) / 2
    """
    result = {}
    for idx, c in enumerate(class_list):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        tn = int(((y_true != idx) & (y_pred != idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        result[c] = (sens + spec) / 2.0
    return result

def rf_per_class_bal_acc():
    """Compute per-class OVR balanced accuracy for RF from saved report.

    RF report has precision, recall (=sensitivity), and support per class.
    We need specificity = TN / (TN + FP).
    From the report: FP = support_c * precision_c^{-1} * recall_c * ...
    Easier: reconstruct from confusion matrix or use precision + support.

    Actually, recall = TP/(TP+FN) = sensitivity. We need FP.
    precision = TP/(TP+FP) => FP = TP/precision - TP = TP*(1-precision)/precision
    support = TP + FN, recall = TP/(TP+FN) => TP = recall * support
    total_test = sum of all supports
    TN = total_test - TP - FN - FP
    """
    rf_report = pd.read_csv(BASELINE_DIR / "report_RF.csv", index_col=0)
    summary_rows = {"accuracy", "macro avg", "weighted avg"}
    all_classes = [c for c in rf_report.index if c not in summary_rows]
    total = sum(rf_report.loc[c, "support"] for c in all_classes)

    result = {}
    for c in classes:
        if c not in rf_report.index:
            result[c] = 0.0
            continue
        recall_c = rf_report.loc[c, "recall"]
        prec_c = rf_report.loc[c, "precision"]
        support_c = rf_report.loc[c, "support"]
        tp = recall_c * support_c
        fn = support_c - tp
        fp = (tp / prec_c - tp) if prec_c > 0 else 0.0
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        result[c] = (sens + spec) / 2.0
    return result

def plot_comparison_bal_acc(per_class_ba):
    """Vertical grouped bar chart: per-class OVR balanced accuracy for 3 models."""
    rf_ba = rf_per_class_bal_acc()

    comp = pd.DataFrame({
        "class": classes,
        "rf_ba": [rf_ba.get(c, 0.0) for c in classes],
        "resnet18_ba": [per_class_ba["resnet18"][c] for c in classes],
        "effnet_ba": [per_class_ba["efficientnet_b0"][c] for c in classes],
    }).sort_values("resnet18_ba", ascending=False)

    fig, ax = plt.subplots(figsize=(155 * mm2in, 70 * mm2in))
    x = np.arange(len(comp))
    w = 0.25
    cls_sorted = comp["class"].tolist()

    ax.bar(x - w, comp["rf_ba"], w, label="RF Baseline",
           color=RF_COLOUR, edgecolor="black", linewidth=0.5)
    ax.bar(x, comp["resnet18_ba"], w, label="ResNet18",
           color=RESNET_COLOUR, edgecolor="black", linewidth=0.5)
    ax.bar(x + w, comp["effnet_ba"], w, label="EfficientNet-B0",
           color=EFFNET_COLOUR, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_sorted, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Per-class Balanced Accuracy: RF vs ResNet18 vs EfficientNet-B0")
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.grid(False)

    out_path = OUT_DIR / "comparison_bal_acc_bar.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

# =====================================================================
# Main: process both architectures
# =====================================================================
results = {}           # arch -> {acc, bal_acc, f1_m, cis}
per_class_ba = {}      # arch -> {class_name: ovr_balanced_accuracy}

for arch in ["resnet18", "efficientnet_b0"]:
    pred_path = OUT_DIR / f"preds_{arch}.npz"

    # Try to load saved predictions first (no GPU needed)
    if pred_path.exists():
        print(f"\n{'='*60}")
        print(f"Loading saved predictions: {arch}")
        print(f"{'='*60}")
        saved = np.load(pred_path)
        y_pred, y_true = saved["y_pred"], saved["y_true"]
    else:
        ckpt = OUT_DIR / f"best_{arch}.pt"
        if not ckpt.exists():
            print(f"SKIP: {ckpt} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Running inference: {arch}")
        print(f"{'='*60}")

        model = build_model(arch, num_classes)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)

        y_pred, y_true = evaluate(model, get_test_loader())

        # Save predictions for future runs without GPU
        np.savez(pred_path, y_pred=y_pred, y_true=y_true)
        print(f"Saved predictions: {pred_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc = float((y_pred == y_true).mean())
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")

    # Bootstrap CIs
    cis = bootstrap_ci(y_true, y_pred)

    results[arch] = {"acc": acc, "bal_acc": bal_acc, "f1_m": f1_m, "f1_w": f1_w, "cis": cis}

    # Per-class OVR balanced accuracy for comparison chart
    per_class_ba[arch] = per_class_bal_acc_ovr(y_true, y_pred, classes)

    # Print metrics with CIs
    print(f"  Accuracy:          {acc:.3f} ({cis['accuracy'][0]:.3f}-{cis['accuracy'][1]:.3f})")
    print(f"  Balanced Accuracy: {bal_acc:.3f} ({cis['balanced_accuracy'][0]:.3f}-{cis['balanced_accuracy'][1]:.3f})")
    print(f"  F1 (macro):        {f1_m:.3f} ({cis['f1_macro'][0]:.3f}-{cis['f1_macro'][1]:.3f})")
    print(f"  F1 (weighted):     {f1_w:.3f}")

    # Confusion matrix PDF
    plot_confusion_matrix(y_true, y_pred, arch, acc, bal_acc)

    # Training curves PDF (if history exists)
    plot_training_curves(arch)

# --- Per-class balanced accuracy comparison bar chart (needs both models) ---
if len(per_class_ba) == 2:
    plot_comparison_bal_acc(per_class_ba)

# --- Summary table for LaTeX ---
print(f"\n{'='*70}")
print("METRICS TABLE FOR LATEX (with 95% bootstrap CIs)")
print(f"{'='*70}")
print(f"{'Metric':<25} {'ResNet18':>25} {'EfficientNet-B0':>25}")
print("-" * 75)
for metric, key, ci_key in [
    ("Accuracy", "acc", "accuracy"),
    ("Balanced Accuracy", "bal_acc", "balanced_accuracy"),
    ("F1 (macro)", "f1_m", "f1_macro"),
]:
    r = results.get("resnet18", {})
    e = results.get("efficientnet_b0", {})
    if r and e:
        r_ci = r["cis"][ci_key]
        e_ci = e["cis"][ci_key]
        r_str = f"{r[key]:.3f} ({r_ci[0]:.3f}-{r_ci[1]:.3f})"
        e_str = f"{e[key]:.3f} ({e_ci[0]:.3f}-{e_ci[1]:.3f})"
        print(f"{metric:<25} {r_str:>25} {e_str:>25}")

print("\nDone.")
