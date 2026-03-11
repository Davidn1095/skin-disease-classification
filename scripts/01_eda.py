#!/usr/bin/env python3
"""Exploratory Data Analysis for Skin Disease Classification Dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# --- Style (FIGURE_STYLE.md) ---
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8,
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

BAR_FILL = "#0072B2"  # Okabe-Ito blue
BAR_EDGE = "black"
BAR_LW = 0.5
DROP_CLASSES = {"Unknown_Normal", "Lupus", "Sun_Sunlight_Damage", "Moles"}

# --- Paths ---
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = PROJECT / "output" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Class distribution ---
print("=" * 60)
print("1. CLASS DISTRIBUTION")
print("=" * 60)

def count_images(root):
    counts = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            n = len([f for f in cls_dir.iterdir() if f.is_file()])
            counts[cls_dir.name] = n
    return counts

train_counts = count_images(TRAIN_DIR)
test_counts = count_images(TEST_DIR)

df_dist = pd.DataFrame({
    "class": list(train_counts.keys()),
    "train": list(train_counts.values()),
    "test": [test_counts.get(c, 0) for c in train_counts.keys()],
})
df_dist["total"] = df_dist["train"] + df_dist["test"]
df_dist["train_pct"] = (df_dist["train"] / df_dist["train"].sum() * 100).round(1)
df_dist = df_dist.sort_values("train", ascending=False).reset_index(drop=True)

print(df_dist.to_string(index=False))
print(f"\nTotal: {df_dist['train'].sum()} train, {df_dist['test'].sum()} test, {df_dist['total'].sum()} overall")
print(f"Classes: {len(df_dist)}")
print(f"Imbalance ratio (max/min): {df_dist['train'].max() / df_dist['train'].min():.1f}x")

# Save full table (all 22 classes)
df_dist.to_csv(OUT_DIR / "class_distribution.csv", index=False)

# --- Filter to 18 classes ---
df_18 = df_dist[~df_dist["class"].isin(DROP_CLASSES)].reset_index(drop=True)
df_18["train_pct"] = (df_18["train"] / df_18["train"].sum() * 100).round(1)
df_18 = df_18.sort_values("train", ascending=False).reset_index(drop=True)

print(f"\n18-class set: {df_18['train'].sum()} train, {df_18['test'].sum()} test")
print(f"Imbalance ratio (18-class): {df_18['train'].max() / df_18['train'].min():.1f}x")

# --- Fig 1: Class distribution bar chart (18 classes, vertical) ---
mm2in = 1 / 25.4
fig, ax = plt.subplots(figsize=(155 * mm2in, 70 * mm2in))

ax.bar(range(len(df_18)), df_18["train"], color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=BAR_LW)
for i, tr in enumerate(df_18["train"]):
    ax.text(i, tr + 15, f"{tr}", ha="center", fontsize=6)
ax.set_xticks(range(len(df_18)))
ax.set_xticklabels(df_18["class"], rotation=45, ha="right", fontsize=6)
ax.set_ylabel("Number of training images")
ax.xaxis.grid(False)

fig.savefig(OUT_DIR / "class_distribution.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {OUT_DIR / 'class_distribution.pdf'}")

# --- 2. Image properties (data collection only, no figures) ---
print("\n" + "=" * 60)
print("2. IMAGE PROPERTIES")
print("=" * 60)

widths, heights, aspects, filesizes, classes_col, splits, channels = [], [], [], [], [], [], []

for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if not img_path.is_file():
                continue
            try:
                fsize = img_path.stat().st_size
                with Image.open(img_path) as img:
                    w, h = img.size
                    mode = img.mode
                widths.append(w)
                heights.append(h)
                aspects.append(w / h if h > 0 else 0)
                filesizes.append(fsize / 1024)
                classes_col.append(cls_dir.name)
                splits.append(split_name)
                channels.append(mode)
            except Exception as e:
                print(f"  WARNING: Failed to read {img_path}: {e}")

df_img = pd.DataFrame({
    "width": widths, "height": heights, "aspect_ratio": aspects,
    "filesize_kb": filesizes, "class": classes_col, "split": splits, "mode": channels
})

print(f"\nTotal images analysed: {len(df_img)}")
print(f"\nImage modes: {df_img['mode'].value_counts().to_dict()}")
print(f"\nWidth:  min={df_img['width'].min()}, max={df_img['width'].max()}, "
      f"mean={df_img['width'].mean():.0f}, median={df_img['width'].median():.0f}")
print(f"Height: min={df_img['height'].min()}, max={df_img['height'].max()}, "
      f"mean={df_img['height'].mean():.0f}, median={df_img['height'].median():.0f}")
print(f"Aspect: min={df_img['aspect_ratio'].min():.2f}, max={df_img['aspect_ratio'].max():.2f}, "
      f"mean={df_img['aspect_ratio'].mean():.2f}")
print(f"Size:   min={df_img['filesize_kb'].min():.0f}KB, max={df_img['filesize_kb'].max():.0f}KB, "
      f"mean={df_img['filesize_kb'].mean():.0f}KB")

non_rgb = df_img[df_img["mode"] != "RGB"]
if len(non_rgb) > 0:
    print(f"\nNon-RGB images: {len(non_rgb)}")
    print(non_rgb["mode"].value_counts())

df_img.to_csv(OUT_DIR / "image_metadata.csv", index=False)

# --- 3. Per-class dimension stats ---
print("\n" + "=" * 60)
print("3. PER-CLASS IMAGE SIZE STATS")
print("=" * 60)

cls_stats = df_img.groupby("class").agg(
    n=("width", "count"),
    w_mean=("width", "mean"), h_mean=("height", "mean"),
    w_std=("width", "std"), h_std=("height", "std"),
    aspect_mean=("aspect_ratio", "mean"),
    size_kb_mean=("filesize_kb", "mean"),
).round(1)
print(cls_stats.to_string())

# --- Fig 2: Sample images grid (18 classes only) ---
print("\n" + "=" * 60)
print("4. SAMPLE IMAGES PER CLASS (18 classes)")
print("=" * 60)

order_18 = df_18["class"].tolist()
n_samples = 4
fig, axes = plt.subplots(len(order_18), n_samples,
                         figsize=(180 * mm2in, 280 * mm2in))

for i, cls in enumerate(order_18):
    cls_dir = TRAIN_DIR / cls
    imgs = sorted(cls_dir.iterdir())[:n_samples]
    for j in range(n_samples):
        ax = axes[i, j]
        if j < len(imgs):
            try:
                img = Image.open(imgs[j]).convert("RGB")
                ax.imshow(img)
                if j == 0:
                    ax.set_ylabel(cls.replace("_", "\n"), fontsize=7,
                                  rotation=0, labelpad=80, va="center")
            except Exception:
                ax.text(0.5, 0.5, "Error", ha="center", va="center",
                        transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Sample images per class (first 4 from train)", fontsize=10, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "sample_images.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'sample_images.pdf'}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Classes (18):     {len(order_18)}")
print(f"  Train images:     {df_18['train'].sum()}")
print(f"  Test images:      {df_18['test'].sum()}")
print(f"  Imbalance ratio:  {df_18['train'].max() / df_18['train'].min():.1f}x")
print(f"  Image modes:      {df_img['mode'].value_counts().to_dict()}")
print(f"  Median size:      {df_img['width'].median():.0f} x {df_img['height'].median():.0f}")
print(f"  Non-RGB images:   {len(non_rgb)}")
print(f"\nAll outputs saved to: {OUT_DIR}")
