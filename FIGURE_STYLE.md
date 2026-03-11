# Figure Style Guide — Skin Disease Classification

Current aesthetics for all report figures.
Scripts: `scripts/01_eda.py`, `scripts/03_transfer_learning.py`, `scripts/04_comparison.py`.
Target: A4 LaTeX report (`report/main.tex`), article class, 11pt, 2.5 cm margins.

---

## Report Specifications

| Parameter | Value |
|-----------|-------|
| Page width (A4, 2.5 cm margins) | 160 mm |
| Full-width figure | 155 mm |
| Font family | Helvetica / Arial (sans-serif) |
| Base text size | 8 pt |
| Panel label size | 9 pt bold |
| Output format | PDF only (vector) |
| Background | white |

---

## Core Style: matplotlib rcParams

Defined as a `set_style()` function to be called at the top of every script:

```python
import matplotlib.pyplot as plt

def set_style():
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        # Axes
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.which": "major",
        # Grid
        "grid.color": "#E5E5E5",       # grey90
        "grid.linewidth": 0.3,
        # Ticks
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        # Legend
        "legend.fontsize": 7,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "none",
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.05,
    })
```

Key points:
- Top and right spines removed on all axes
- Minor grid and minor ticks disabled
- Major grid very faint (grey90, 0.3 lwd)
- Tight save padding (0.05 in)

---

## Colour Palette

All palettes are colourblind-safe (Okabe-Ito family or sequential blues).

### 18-disease categorical palette

Extended Okabe-Ito set. Do NOT use `tab20`.

```python
DISEASE_COLOURS = {
    "Acne":               "#D55E00",   # vermillion
    "Actinic_Keratosis":  "#E69F00",   # orange
    "Benign_tumors":      "#CC79A7",   # reddish purple
    "Bullous":            "#009E73",   # bluish green
    "Candidiasis":        "#0072B2",   # blue
    "DrugEruption":       "#56B4E9",   # sky blue
    "Eczema":             "#F0E442",   # yellow
    "Infestations_Bites": "#999999",   # grey
    "Lichen":             "#882255",   # wine
    "Psoriasis":          "#44AA99",   # teal
    "Rosacea":            "#332288",   # indigo
    "Seborrh_Keratoses":  "#DDCC77",   # sand
    "SkinCancer":         "#AA4499",   # purple
    "Tinea":              "#88CCEE",   # light cyan
    "Vascular_Tumors":    "#661100",   # dark red
    "Vasculitis":         "#117733",   # green
    "Vitiligo":           "#6699CC",   # steel blue
    "Warts":              "#CC6677",   # rose
}
```

### Default fills

```python
# Bar chart default (Figs 1)
BAR_FILL = "#404040"        # grey25
BAR_EDGE = "black"
BAR_LW = 0.5

# Confusion matrix
CM_CMAP = "Blues"           # plt.cm.Blues

# Training curves (train vs test)
TRAIN_COLOUR = "#0072B2"   # Okabe-Ito blue
TEST_COLOUR  = "#D55E00"   # Okabe-Ito vermillion

# Model comparison (ResNet18 vs EfficientNet-B0)
RESNET_COLOUR  = "#0072B2" # blue
EFFNET_COLOUR  = "#56B4E9" # sky blue
```

---

## Figure Inventory

| Figure | Content | Width (mm) | Height (mm) |
|--------|---------|-----------|-------------|
| Fig 1 | Class distribution bar chart (horizontal, 18 classes only) | 155 | 80 |
| Fig 2 | Sample image grid (18 classes x 4 images) | 155 | 200 |
| Fig 3 | Per-class F1 comparison bar (ResNet18 vs EfficientNet-B0) | 155 | 80 |
| Fig 4 | Confusion matrix ResNet18 | 140 | 130 |
| Fig 5 | Confusion matrix EfficientNet-B0 | 140 | 130 |
| Fig 6 | Training curves (side-by-side, both models) | 155 | 60 |

No pie charts. Fig 1 is bar chart only.

---

## Plot Type Conventions

### Bar charts (Figs 1, 3)

- Horizontal orientation (`ax.barh`)
- `edgecolor="black"`, `linewidth=0.5`
- Fill: single colour `grey25` for Fig 1; model colours for Fig 3
- Value labels at bar end: `fontsize=6`
- No gridlines on bar axis (y-axis): `ax.yaxis.grid(False)`
- Light grid on value axis (x-axis)
- Sort by value descending (largest at top)
- `ax.invert_yaxis()` so largest class is at top

```python
ax.barh(classes, values, color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=BAR_LW)
for i, v in enumerate(values):
    ax.text(v + offset, i, f"{v}", va="center", fontsize=6)
ax.yaxis.grid(False)
ax.invert_yaxis()
```

### Confusion matrices (Figs 4, 5)

- `sns.heatmap` with `cmap="Blues"`, `annot=True`, `fmt="d"`
- Annotation fontsize: 6 (`annot_kws={"size": 6}`)
- Square cells: `square=True`
- White cell borders: `linewidths=0.3`, `linecolor="white"`
- Class labels: fontsize 6
- Y-axis labels: no rotation (`rotation=0`)
- X-axis labels: 45-degree rotation (`rotation=45, ha="right"`)
- Title includes accuracy and balanced accuracy

```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
            linewidths=0.3, linecolor="white",
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 6}, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
ax.set_title(f"Confusion Matrix — {arch} (Acc={acc:.3f}, BalAcc={bal_acc:.3f})")
```

### Training curves (Fig 6)

- Two subplots side by side: loss (left), accuracy (right)
- Train line: solid, colour `#0072B2`
- Test line: dashed, colour `#D55E00`
- Vertical dashed grey line at phase 1 / phase 2 transition
- Legend inside plot, fontsize 6
- Marker: small circle (`"o"`, `markersize=3`)

```python
ax.plot(epochs, train_vals, "o-", color=TRAIN_COLOUR, markersize=3, label="Train")
ax.plot(epochs, test_vals, "o--", color=TEST_COLOUR, markersize=3, label="Test")
ax.axvline(phase1_epochs + 0.5, color="grey", linestyle="--", alpha=0.5, label="Unfreeze")
ax.legend(fontsize=6)
```

### Image grid (Fig 2)

- 18 rows x 4 columns (only retained classes, not all 22)
- No axes, no ticks: `ax.set_xticks([]); ax.set_yticks([])`
- Class name as row label: `ax.set_ylabel(cls, fontsize=7, rotation=0, labelpad=80, va="center")`
- `plt.tight_layout()` with minimal padding
- Suptitle: fontsize 10

---

## Save Convention

All figures saved as PDF only (vector format for LaTeX inclusion):

```python
def save_fig(fig, name, output_dir="output"):
    """Save figure as PDF. Name should include subdirectory, e.g. 'eda/class_distribution'."""
    path = f"{output_dir}/{name}.pdf"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
```

Output directory structure:
```
output/
├── eda/                  # Figures from 01_eda.py
├── baseline/             # Figures from 02_baseline.py
└── transfer_learning/    # Figures from 03_transfer_learning.py + 04_comparison.py
```

---

## Checklist

1. Width <= 155 mm?
2. Font = Helvetica/Arial, base size 8 pt?
3. All colours colourblind-safe (Okabe-Ito)?
4. No red/green combinations?
5. Spines: only left and bottom?
6. Grid: major only, grey90, 0.3 lwd?
7. Saved as PDF only (vector)?
8. Bar chart labels: fontsize 6?
9. Confusion matrix: square cells, white borders?
10. No unnecessary bold in labels or titles?
11. No pie charts anywhere?
