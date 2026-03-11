# Skin Disease Classification

Transfer learning for automated classification of 18 dermatological conditions using ResNet18 and EfficientNet-B0.

Group 3 — University of Graz, Summer Semester 2026

## Pipeline

| Script | Description |
|--------|-------------|
| 01_eda.py | Exploratory data analysis |
| 02_baseline.py | Traditional ML baselines (LogReg, RF, SVM, KNN) |
| 03_transfer_learning.py | ResNet18 / EfficientNet-B0 fine-tuning (GPU) |
| 04_comparison.py | Model comparison table |
| 05_generate_figures.py | Figure generation from saved predictions |

## Results

| Metric | RF Baseline | ResNet18 | EfficientNet-B0 |
|--------|------------|----------|-----------------|
| Balanced Acc. | 0.284 | **0.699 (0.673–0.725)** | 0.694 (0.668–0.721) |

95% bootstrap CIs (1000 resamples).

## Dataset

https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
