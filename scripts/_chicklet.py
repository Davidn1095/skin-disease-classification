"""Atlas-style bar plotting helpers for matplotlib.

Approximates the look of ggchicklet::geom_chicklet used in the
autoimmune-atlas Nature figures (theme_nature + theme_minimal).

NOTE on rounded corners:
ggchicklet renders bars with a 1.5 pt corner radius. matplotlib has no
clean equivalent — FancyBboxPatch's "round" boxstyle adds rounding
*outside* the bounding box and produces rendering artefacts at bar tops
in data-coordinate axes. Implementing a true rounded path with Path +
PathPatch is possible but the per-bar rounding-in-points-vs-data-units
conversion is non-trivial when x and y scales differ wildly (e.g., 0–18
categorical positions vs 0–1100 counts). We therefore render
**rectangular** bars and match every other atlas spec faithfully. The
rounding is the only visual gap.

Usage:
    from _chicklet import atlas_bars, apply_atlas_theme

    apply_atlas_theme()
    fig, ax = plt.subplots(...)
    atlas_bars(ax, x, heights, facecolor=ATLAS_GREEN, ...)
"""
from __future__ import annotations

import matplotlib.pyplot as plt

ATLAS_GREEN = "#009E73"
ATLAS_BLUE = "#0072B2"
ATLAS_ORANGE = "#D55E00"
ATLAS_GREY = "#999999"


def apply_atlas_theme():
    """rcParams matching theme_nature() + theme_minimal() from the atlas.

    Key choices (vs ggplot equivalents):
      - All four spines hidden (theme_minimal)
      - Major y-grid only, grey90, lwd 0.3
      - Helvetica, base size 7, axis text x size 5, y size 6
      - Plot titles bold and centred
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        # theme_minimal: hide all spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.titlesize": 7,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "axes.labelsize": 7,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.grid.axis": "y",                # gridlines on value axis only
        "grid.color": "#E5E5E5",              # grey90
        "grid.linewidth": 0.3,
        "xtick.labelsize": 5,
        "ytick.labelsize": 6,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "legend.fontsize": 6,
        "legend.framealpha": 0,
        "legend.edgecolor": "none",
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.05,
    })


def atlas_bars(ax, x, heights, *, width=0.9, bottom=0,
               facecolor=ATLAS_GREEN, edgecolor="black",
               linewidth=0.3, yerr=None, capsize=0,
               error_kw=None, zorder=3, label=None):
    """Rectangular bars styled like ggchicklet (minus the 1.5 pt corner radius)."""
    return ax.bar(
        x, heights, width=width, bottom=bottom,
        color=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        yerr=yerr, capsize=capsize, error_kw=error_kw or {},
        zorder=zorder, label=label,
    )
