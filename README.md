# Skin Disease Classification

Transfer learning for automated classification of 18 dermatological conditions using ResNet18 and EfficientNet-B0.

Group 1 — University of Graz, Summer Semester 2026

## Pipeline

| Script | Description |
|--------|-------------|
| 01_eda.py | Exploratory data analysis |
| 02_baseline.py | Traditional ML baselines |
| 03_transfer_learning.py | ResNet18 / EfficientNet-B0 fine-tuning |
| 04_comparison.py | Model comparison |
| 05_generate_figures.py | Figure generation from saved checkpoints |

## Results

| Metric | RF Baseline | ResNet18 | EfficientNet-B0 |
|--------|------------|----------|-----------------|
| Balanced Acc. | 0.284 | **0.699 (0.673-0.725)** | 0.694 (0.668-0.721) |

Dataset: https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
