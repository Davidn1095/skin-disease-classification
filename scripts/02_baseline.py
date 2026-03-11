#!/usr/bin/env python3
"""Baseline classifiers for Skin Disease Classification.

Strategy:
  1. Resize all images to 64x64, flatten to feature vectors (64*64*3 = 12288)
  2. Train/evaluate: Logistic Regression, Random Forest, SVM, KNN
  3. Also try HOG features + Logistic Regression as a hand-crafted baseline
  4. Report accuracy, balanced accuracy, per-class F1, confusion matrix
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

# --- Paths ---
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUT_DIR = PROJECT / "output" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 32  # small to fit in login-node memory (~15k images * 32*32*3 = ~45MB)

# --- Load images ---
def load_dataset(root, img_size):
    images, labels = [], []
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    for cls in classes:
        cls_dir = root / cls
        for img_path in cls_dir.iterdir():
            if not img_path.is_file():
                continue
            try:
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                images.append(np.array(img, dtype=np.uint8).flatten())
                labels.append(cls)
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")
    # Convert to float32 only at the end to save memory during loading
    return np.array(images, dtype=np.float32) / 255.0, np.array(labels), classes

print("Loading train set...")
t0 = time.time()
X_train, y_train, classes = load_dataset(TRAIN_DIR, IMG_SIZE)
print(f"  {X_train.shape[0]} images, {X_train.shape[1]} features, {len(classes)} classes ({time.time()-t0:.1f}s)")

print("Loading test set...")
t0 = time.time()
X_test, y_test, _ = load_dataset(TEST_DIR, IMG_SIZE)
print(f"  {X_test.shape[0]} images ({time.time()-t0:.1f}s)")

# Encode labels
le = LabelEncoder()
le.fit(classes)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

# --- Compute class weights ---
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train_enc)
class_weight_dict = dict(zip(range(len(classes)), cw))

# --- Feature variants ---
# 1. Raw pixels (scaled)
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. PCA reduction (keep 95% variance)
print("Running PCA...")
t0 = time.time()
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"  PCA: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} components ({time.time()-t0:.1f}s)")

# --- Models ---
# Free unscaled data
del X_train, X_test

models = {
    "LogReg_PCA": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1),
        X_train_pca, X_test_pca, "PCA features"
    ),
    "LogReg_raw": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1),
        X_train_scaled, X_test_scaled, "Raw pixels (scaled)"
    ),
    "RF": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        X_train_pca, X_test_pca, "PCA features"
    ),
    "LinearSVC": (
        LinearSVC(max_iter=2000, class_weight="balanced", random_state=42, dual="auto"),
        X_train_pca, X_test_pca, "PCA features"
    ),
    "KNN": (
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        X_train_pca, X_test_pca, "PCA features"
    ),
}

# --- Train and evaluate ---
results = []

for name, (model, X_tr, X_te, feat_desc) in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name} ({feat_desc})")
    print(f"{'='*60}")
    t0 = time.time()
    model.fit(X_tr, y_train_enc)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - t0

    acc = accuracy_score(y_test_enc, y_pred)
    bal_acc = balanced_accuracy_score(y_test_enc, y_pred)
    f1_macro = f1_score(y_test_enc, y_pred, average="macro")
    f1_weighted = f1_score(y_test_enc, y_pred, average="weighted")

    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  Train time:        {train_time:.1f}s")
    print(f"  Predict time:      {pred_time:.1f}s")

    results.append({
        "model": name, "features": feat_desc,
        "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_macro, 4), "f1_weighted": round(f1_weighted, 4),
        "train_time_s": round(train_time, 1), "predict_time_s": round(pred_time, 1),
    })

    # Per-class report
    report = classification_report(y_test_enc, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).T
    df_report.to_csv(OUT_DIR / f"report_{name}.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name} (Acc={acc:.3f}, BalAcc={bal_acc:.3f})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_{name}.png", dpi=120, bbox_inches="tight")
    plt.close()

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
df_results = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
print(df_results.to_string(index=False))
df_results.to_csv(OUT_DIR / "baseline_results.csv", index=False)

# Bar plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_results))
w = 0.25
ax.bar(x - w, df_results["accuracy"], w, label="Accuracy", color="steelblue")
ax.bar(x, df_results["balanced_accuracy"], w, label="Balanced Acc", color="coral")
ax.bar(x + w, df_results["f1_macro"], w, label="F1 (macro)", color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels(df_results["model"], rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_title("Baseline Model Comparison")
ax.legend()
ax.set_ylim(0, 1)
for i, (a, b, f) in enumerate(zip(df_results["accuracy"], df_results["balanced_accuracy"], df_results["f1_macro"])):
    ax.text(i - w, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
    ax.text(i, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
    ax.text(i + w, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAll outputs saved to: {OUT_DIR}")
print(f"Random baseline (22 classes): {1/22:.4f}")
