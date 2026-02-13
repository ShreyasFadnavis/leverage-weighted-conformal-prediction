"""Regenerate Figures 7 and 8 with improved designs.

Figure 7: Coverage with approximate leverage → dot plot with truncation fractions
Figure 8: Approximate vs exact leverage scatter → 2×2 grid with Spearman ρ
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from lwcp.leverage import LeverageComputer
from experiments.plotting import setup_style

RESULTS_DIR = Path(__file__).parent / "results"
PAPER_FIGURES = Path(__file__).parent.parent / "paper-uai" / "figures"
EXP_FIGURES = Path(__file__).parent / "figures"


def load_results():
    with open(RESULTS_DIR / "exp8_approximate.json") as f:
        return json.load(f)


def figure7_coverage_dot_plot(data):
    """Redesigned Figure 7: Coverage preservation under approximation.

    Dot plot with truncation fraction on x-axis, one marker per p.
    Shows that coverage is indistinguishable from nominal at all truncation levels.
    """
    setup_style()

    p_values = [10, 30, 50, 100]
    fracs = [1.0, 0.5, 0.25]
    frac_labels = ["$k = p$\n(exact)", "$k = p/2$", "$k = p/4$"]

    # Extract coverage values in consistent truncation-fraction form
    cov_matrix = np.zeros((len(p_values), len(fracs)))
    for i, p in enumerate(p_values):
        pdata = data[str(p)]
        keys = list(pdata.keys())  # Exact, Approx(k=p), Approx(k=p/2), Approx(k=p/4)
        for j, frac in enumerate(fracs):
            k = max(int(p * frac), 1)
            if j == 0:
                cov_matrix[i, j] = pdata["Exact"]
            else:
                key = f"Approx (k={k})"
                cov_matrix[i, j] = pdata.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Colors: distinct, colorblind-friendly
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "D", "^"]

    x_positions = np.arange(len(fracs))
    offsets = np.linspace(-0.15, 0.15, len(p_values))

    for i, p in enumerate(p_values):
        ax.plot(
            x_positions + offsets[i], cov_matrix[i],
            marker=markers[i], color=colors[i],
            markersize=9, markeredgecolor="white", markeredgewidth=1.0,
            linestyle="-", linewidth=1.5, alpha=0.85,
            label=f"$p = {p}$",
        )

    # Nominal level + SE band
    nom = 0.90
    n_cal = 200  # calibration set size used
    se = np.sqrt(nom * (1 - nom) / n_cal) * 1.96
    ax.axhline(nom, color="#333333", linestyle=":", linewidth=1.2, alpha=0.7, zorder=0)
    ax.axhspan(nom - se, nom + se, color="#333333", alpha=0.06, zorder=0)
    ax.text(2.35, nom + 0.0003, "Nominal 90%", fontsize=9, color="#333333", ha="right", va="bottom")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(frac_labels, fontsize=11)
    ax.set_xlabel("Truncation level", fontsize=12)
    ax.set_ylabel("Marginal coverage", fontsize=12)
    ax.set_ylim(0.893, 0.908)
    ax.legend(fontsize=10, loc="upper right", ncol=2, framealpha=0.9)

    # Remove title (caption handles it in the paper)
    ax.set_title("")

    fig.tight_layout()
    return fig


def figure8_scatter_2x2(n_train=200):
    """Redesigned Figure 8: Approximate vs exact leverage scatter.

    2×2 grid with Spearman ρ annotations, better colors, and cleaner layout.
    """
    setup_style()

    p_values = [10, 30, 50, 100]
    approx_fracs = [1.0, 0.5, 0.25]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    axes = axes.ravel()

    # Colors for truncation levels: exact=blue, half=orange, quarter=green
    colors = ["#1f77b4", "#e08214", "#1b7837"]
    labels_template = ["$k = p$ (exact)", "$k = p/2$", "$k = p/4$"]

    for idx, p in enumerate(p_values):
        ax = axes[idx]
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_train, p))
        h_exact = LeverageComputer(method="exact").fit(X).leverage_scores(X)

        for i, frac in enumerate(approx_fracs):
            k = max(int(p * frac), 1)
            h_approx = LeverageComputer(
                method="approximate", n_components=k, random_state=42
            ).fit(X).leverage_scores(X)

            # Compute Spearman correlation
            rho, _ = stats.spearmanr(h_exact, h_approx)

            label = labels_template[i].replace("p", str(p)) if idx == 0 else None
            # For exact (k=p), correlation is 1.0; only show annotation for k<p
            if frac == 1.0:
                ax.scatter(
                    h_exact, h_approx, alpha=0.4, s=12,
                    color=colors[i], label=label,
                    edgecolors="none", zorder=3,
                )
            else:
                ax.scatter(
                    h_exact, h_approx, alpha=0.35, s=12,
                    color=colors[i], label=label,
                    edgecolors="none", zorder=2,
                )
                # Annotate with Spearman ρ
                k_val = max(int(p * frac), 1)
                ax.annotate(
                    f"$k={k_val}$: $\\rho_s={rho:.3f}$",
                    xy=(0.97, 0.18 - 0.08 * (i - 1)),
                    xycoords="axes fraction",
                    fontsize=8, ha="right",
                    color=colors[i], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
                )

        # Identity line
        h_max = max(h_exact.max(), 0.01)
        ax.plot([0, h_max * 1.05], [0, h_max * 1.05], "k--", alpha=0.3, linewidth=1, zorder=1)

        ax.set_xlabel("Exact leverage $h(x)$", fontsize=10)
        ax.set_ylabel("Approximate leverage $\\tilde{h}(x)$", fontsize=10)
        ax.set_title(f"$p = {p}$", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Shared legend from first panel
    handles, labels_list = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, ["Exact ($k = p$)", "Half ($k = p/2$)", "Quarter ($k = p/4$)"],
        loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9, edgecolor="0.7",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def main():
    data = load_results()

    # Figure 7: Coverage dot plot
    print("Generating Figure 7 (coverage dot plot)...")
    fig7 = figure7_coverage_dot_plot(data)
    for outdir in [EXP_FIGURES, PAPER_FIGURES]:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "exp8_approx_coverage.pdf"
        fig7.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        path_png = outdir / "exp8_approx_coverage.png"
        fig7.savefig(path_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"  Saved: {path}")
    plt.close(fig7)

    # Figure 8: Scatter 2×2
    print("Generating Figure 8 (leverage scatter 2×2)...")
    fig8 = figure8_scatter_2x2()
    for outdir in [EXP_FIGURES, PAPER_FIGURES]:
        path = outdir / "exp8_approx_leverage_scatter.pdf"
        fig8.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        path_png = outdir / "exp8_approx_leverage_scatter.png"
        fig8.savefig(path_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
        print(f"  Saved: {path}")
    plt.close(fig8)

    print("\nDone! Both figures regenerated.")


if __name__ == "__main__":
    main()
