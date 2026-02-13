"""Shared plotting utilities for LWCP experiments.

Publication-quality style targeting NeurIPS formatting.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- Color palette: high-contrast, colorblind-friendly ---
COLORS = {
    "Vanilla CP": "#7f7f7f",       # neutral gray
    "LWCP": "#d62728",             # strong red — stands out against gray
    "CQR": "#ff7f0e",             # orange
    "Studentized CP": "#2ca02c",  # green
    "LWCP+": "#1f77b4",           # blue
    "CQR-GBR": "#ff7f0e",        # orange (same as CQR)
    "Localized CP": "#9467bd",    # purple
}

MARKERS = {
    "Vanilla CP": "s",
    "LWCP": "o",
    "CQR": "^",
    "Studentized CP": "D",
    "LWCP+": "P",
    "CQR-GBR": "^",
    "Localized CP": "v",
}

LINESTYLES = {
    "Vanilla CP": "--",
    "LWCP": "-",
    "CQR": "-.",
    "Studentized CP": ":",
    "LWCP+": "-",
    "CQR-GBR": "-.",
    "Localized CP": "-.",
}

FIGURES_DIR = Path(__file__).parent / "figures"


def setup_style():
    """Set publication-quality matplotlib defaults for NeurIPS."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "lines.markeredgewidth": 0.8,
        "lines.markeredgecolor": "white",
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        # Figure
        "figure.figsize": (8, 5),
        "figure.facecolor": "white",
    })


def get_color(method_name: str) -> str:
    return COLORS.get(method_name, "#000000")


def get_marker(method_name: str) -> str:
    return MARKERS.get(method_name, "o")


def get_linestyle(method_name: str) -> str:
    return LINESTYLES.get(method_name, "-")


def method_plot_kwargs(method_name: str) -> dict:
    """Return consistent kwargs for plotting a method."""
    return {
        "color": get_color(method_name),
        "marker": get_marker(method_name),
        "linestyle": get_linestyle(method_name),
        "label": method_name,
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    }


def ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def savefig(fig, name: str):
    """Save figure to the figures directory."""
    fdir = ensure_figures_dir()
    path = fdir / f"{name}.pdf"
    fig.savefig(path)
    path_png = fdir / f"{name}.png"
    fig.savefig(path_png)
    print(f"  Saved: {path}")
    plt.close(fig)


def nominal_line(ax, alpha, label=True):
    """Draw the nominal coverage line."""
    lbl = f"Nominal {1-alpha:.0%}" if label else None
    ax.axhline(
        1 - alpha, color="#333333", linestyle=":",
        linewidth=1.2, alpha=0.8, label=lbl, zorder=0,
    )


def shade_nominal_band(ax, alpha, n_cal=500):
    """Shade a +/- 1 binomial SE band around nominal coverage."""
    nom = 1 - alpha
    se = np.sqrt(nom * alpha / n_cal) * 1.96
    ax.axhspan(nom - se, nom + se, color="#333333", alpha=0.07, zorder=0)


def plot_coverage_by_bin(
    ax, h_values, covered, method_name, alpha, n_bins=10, **kwargs
):
    """Plot empirical coverage as a function of leverage decile."""
    bin_edges = np.percentile(h_values, np.linspace(0, 100, n_bins + 1))
    bin_coverages = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (h_values >= lo) & (h_values <= hi)
        else:
            mask = (h_values >= lo) & (h_values < hi)
        if mask.sum() > 0:
            bin_coverages.append(np.mean(covered[mask]))
        else:
            bin_coverages.append(np.nan)

    kw = method_plot_kwargs(method_name)
    kw.update(kwargs)
    ax.plot(range(1, n_bins + 1), bin_coverages, **kw)
    ax.set_xlabel("Leverage decile (1=lowest, 10=highest)")
    ax.set_ylabel("Conditional coverage")


def plot_width_vs_leverage(ax, h_values, widths, method_name, **kwargs):
    """Scatter plot of interval width vs leverage."""
    color = get_color(method_name)
    ax.scatter(
        h_values, widths, alpha=0.35, s=12,
        color=color, label=method_name,
        edgecolors="white", linewidths=0.3,
        **kwargs,
    )
    ax.set_xlabel("Leverage score $h(x)$")
    ax.set_ylabel("Interval width")
