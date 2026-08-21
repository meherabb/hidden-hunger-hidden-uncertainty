"""
Shared matplotlib style for publication-quality (NeurIPS-style) figures.
Vector PDF output, consistent sans-serif type, minimal chartjunk.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "face": "#3B6FA0",        # muted blue  -- FACE evidence
    "geonutrition": "#D9713C", # muted orange -- GeoNutrition validation sites
    "query": "#5FA777",       # muted green -- other query regions
    "highlight": "#B23A48",   # muted red   -- mismatch / risk highlight
    "neutral": "#6B6B6B",
    "grid": "#E4E4E4",
}

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,   # embed TrueType, editable text in PDF (NeurIPS-friendly)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })

def panel_label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
