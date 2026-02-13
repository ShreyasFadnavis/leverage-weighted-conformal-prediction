"""Experiment 3: Interval Width Comparison (Theorem 2, Part 2).

Shows that LWCP produces shorter average intervals than vanilla CP
under heteroscedastic DGPs, with width that adapts to leverage.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.baselines import LWCPMethod, VanillaCP
from experiments.dgps import ALL_DGPS
from experiments.plotting import get_color, savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_width_experiment(n_reps: int = 200, alpha: float = 0.1):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 3: Interval Width Comparison (Theorem 2, Part 2)")
    print("=" * 60)

    dgp_names = list(ALL_DGPS.keys())
    methods = ["Vanilla CP", "LWCP"]
    mean_widths = {dgp: {m: [] for m in methods} for dgp in dgp_names}
    scatter_data = {}

    for dgp_name in dgp_names:
        dgp_func = ALL_DGPS[dgp_name]
        print(f"\n  DGP: {dgp_name}")

        for rep in range(n_reps):
            X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(
                sigma=1.0, random_state=rep,
            )
            for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
                method = method_cls(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                widths = upper - lower
                mean_widths[dgp_name][method_name].append(np.mean(widths))
                if rep == 0:
                    scatter_data.setdefault(dgp_name, {})[method_name] = {
                        "h": meta["h_test"], "widths": widths,
                    }

        for m in methods:
            arr = np.array(mean_widths[dgp_name][m])
            print(f"    {m}: mean width = {arr.mean():.4f} +/- {arr.std():.4f}")

    summary = {}
    print("\n  Width Ratios (LWCP / Vanilla):")
    for dgp_name in dgp_names:
        v = np.mean(mean_widths[dgp_name]["Vanilla CP"])
        l = np.mean(mean_widths[dgp_name]["LWCP"])
        ratio = l / v
        print(f"    {dgp_name}: {ratio:.4f}")
        summary[dgp_name] = {
            "Vanilla CP": {"mean": float(v), "std": float(np.std(mean_widths[dgp_name]["Vanilla CP"]))},
            "LWCP": {"mean": float(l), "std": float(np.std(mean_widths[dgp_name]["LWCP"]))},
            "ratio": float(ratio),
        }

    with open(RESULTS_DIR / "exp3_width.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp3_width.json'}")

    # --- Figure 1: Bar chart ---
    fig1, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dgp_names))
    width_bar = 0.35
    for i, m in enumerate(methods):
        vals = [np.mean(mean_widths[d][m]) for d in dgp_names]
        errs = [np.std(mean_widths[d][m]) for d in dgp_names]
        ax.bar(x + i * width_bar, vals, width_bar, yerr=errs, label=m,
               color=get_color(m), alpha=0.8, capsize=3)
    ax.set_xticks(x + width_bar / 2)
    ax.set_xticklabels(dgp_names, rotation=15, ha="right")
    ax.set_ylabel("Mean interval width")
    ax.set_title("Experiment 3: Mean Interval Width by DGP")
    ax.legend()
    fig1.tight_layout()
    savefig(fig1, "exp3_width_barplot")

    # --- Figure 2: Width vs leverage scatter ---
    n_dgps = len(dgp_names)
    fig2, axes2 = plt.subplots(1, n_dgps, figsize=(3.2 * n_dgps, 3.0), sharey=True)
    axes_flat = axes2.flatten()

    for i, dgp_name in enumerate(dgp_names):
        ax = axes_flat[i]
        for m in methods:
            d = scatter_data[dgp_name][m]
            color = get_color(m)

            ax.scatter(
                d["h"], d["widths"], alpha=0.4, s=12,
                color=color, label=m,
                edgecolors="white", linewidths=0.3,
                zorder=3,
            )

            # Add LOESS-like trend line via binned means
            n_trend = 20
            bin_edges_t = np.percentile(d["h"], np.linspace(0, 100, n_trend + 1))
            trend_x, trend_y = [], []
            for b in range(n_trend):
                if b == n_trend - 1:
                    mask = (d["h"] >= bin_edges_t[b]) & (d["h"] <= bin_edges_t[b + 1])
                else:
                    mask = (d["h"] >= bin_edges_t[b]) & (d["h"] < bin_edges_t[b + 1])
                if mask.sum() > 0:
                    trend_x.append(np.mean(d["h"][mask]))
                    trend_y.append(np.mean(d["widths"][mask]))
            ax.plot(trend_x, trend_y, color=color, linewidth=2.0,
                    alpha=0.8, zorder=4)

        ax.set_title(dgp_name, fontsize=10, fontweight="bold")
        ax.set_xlabel(r"Leverage $h(x)$", fontsize=9)
        if i == 0:
            ax.set_ylabel("Interval width", fontsize=9)
        ax.tick_params(labelsize=8)

    # Shared legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig2.legend(
        handles, labels, loc="upper center",
        ncol=2, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=2.0, markerscale=1.3, columnspacing=2.0,
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.91])
    fig2.subplots_adjust(wspace=0.08)
    savefig(fig2, "exp3_width_vs_leverage")


if __name__ == "__main__":
    run_width_experiment()
