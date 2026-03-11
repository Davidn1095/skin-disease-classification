#!/usr/bin/env python3
"""Compare ResNet18 vs EfficientNet-B0 results on 18-class skin disease dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

RESNET_COLOUR = "#0072B2"
EFFNET_COLOUR = "#56B4E9"
mm2in = 1 / 25.4

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "output" / "transfer_learning"

# --- Load summaries ---
with open(OUT_DIR / "summary_resnet18.json") as f:
    res18 = json.load(f)
with open(OUT_DIR / "summary_efficientnet_b0.json") as f:
    effb0 = json.load(f)

# --- Side-by-side metrics ---
print("=" * 70)
print("MODEL COMPARISON: ResNet18 vs EfficientNet-B0 (18 classes)")
print("=" * 70)
metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
print(f"{'Metric':<25} {'ResNet18':>12} {'EfficientNet-B0':>16} {'Delta':>10}")
print("-" * 70)
for m in metrics:
    r = res18[m]
    e = effb0[m]
    d = e - r
    print(f"{m:<25} {r:>12.4f} {e:>16.4f} {d:>+10.4f}")

# --- Per-class comparison ---
print("\n" + "=" * 70)
print("PER-CLASS RECALL COMPARISON")
print("=" * 70)

rep_r = pd.read_csv(OUT_DIR / "report_resnet18.csv", index_col=0)
rep_e = pd.read_csv(OUT_DIR / "report_efficientnet_b0.csv", index_col=0)

summary_rows = {"accuracy", "macro avg", "weighted avg"}
classes = [c for c in rep_r.index if c not in summary_rows]

print(f"\n{'Class':<25} {'ResNet18':>10} {'EffNet-B0':>10} {'Delta':>8} {'Winner':>10}")
print("-" * 70)
for cls in classes:
    rr = rep_r.loc[cls, "recall"]
    er = rep_e.loc[cls, "recall"]
    d = er - rr
    winner = "EffNet" if d > 0.01 else "ResNet" if d < -0.01 else "~Tie"
    print(f"{cls:<25} {rr:>10.3f} {er:>10.3f} {d:>+8.3f} {winner:>10}")

# --- Worst-performing classes ---
print("\n" + "=" * 70)
print("WORST-PERFORMING CLASSES (by F1, ResNet18)")
print("=" * 70)
f1_r = rep_r.loc[classes, "f1-score"].sort_values()
for cls in f1_r.index[:5]:
    print(f"  {cls:<25} F1={f1_r[cls]:.3f}  (recall={rep_r.loc[cls, 'recall']:.3f}, prec={rep_r.loc[cls, 'precision']:.3f})")

print("\n" + "=" * 70)
print("WORST-PERFORMING CLASSES (by F1, EfficientNet-B0)")
print("=" * 70)
f1_e = rep_e.loc[classes, "f1-score"].sort_values()
for cls in f1_e.index[:5]:
    print(f"  {cls:<25} F1={f1_e[cls]:.3f}  (recall={rep_e.loc[cls, 'recall']:.3f}, prec={rep_e.loc[cls, 'precision']:.3f})")

# --- Save comparison table ---
comp = pd.DataFrame({
    "class": classes,
    "resnet18_precision": [rep_r.loc[c, "precision"] for c in classes],
    "resnet18_recall": [rep_r.loc[c, "recall"] for c in classes],
    "resnet18_f1": [rep_r.loc[c, "f1-score"] for c in classes],
    "effnet_precision": [rep_e.loc[c, "precision"] for c in classes],
    "effnet_recall": [rep_e.loc[c, "recall"] for c in classes],
    "effnet_f1": [rep_e.loc[c, "f1-score"] for c in classes],
})
comp["f1_delta"] = comp["effnet_f1"] - comp["resnet18_f1"]
comp = comp.sort_values("resnet18_f1")
comp.to_csv(OUT_DIR / "comparison_18class.csv", index=False)

# --- Fig 3: Per-class F1 comparison bar chart ---
fig, ax = plt.subplots(figsize=(155 * mm2in, 80 * mm2in))
x = np.arange(len(classes))
w = 0.35
cls_sorted = comp["class"].tolist()

ax.barh(x - w/2, comp["resnet18_f1"], w, label="ResNet18",
        color=RESNET_COLOUR, edgecolor="black", linewidth=0.5)
ax.barh(x + w/2, comp["effnet_f1"], w, label="EfficientNet-B0",
        color=EFFNET_COLOUR, edgecolor="black", linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels(cls_sorted, fontsize=7)
ax.set_xlabel("F1 Score")
ax.set_title("Per-class F1: ResNet18 vs EfficientNet-B0 (18 classes)")
ax.legend(fontsize=7)
ax.yaxis.grid(False)
ax.invert_yaxis()

fig.savefig(OUT_DIR / "comparison_f1_bar.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nSaved: {OUT_DIR / 'comparison_18class.csv'}")
print(f"Saved: {OUT_DIR / 'comparison_f1_bar.pdf'}")
