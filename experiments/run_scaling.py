"""Experiment 9: Dimensionality and Sample Size Scaling.

Shows that the conditional coverage gap shrinks as
O(1/sqrt(n_2)) + O(p/n_1), confirming Proposition 1.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.baselines import LWCPMethod
from experiments.dgps import dgp_textbook
from experiments.plotting import savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_scaling_experiment(n_reps: int = 100, alpha: float = 0.1, n_bins: int = 10):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 9: Dimensionality and Sample Size Scaling")
    print("=" * 60)

    n_values = [200, 500, 1000, 2000, 5000]
    p_values = [5, 20, 50, 100]

    gap_matrix = np.full((len(p_values), len(n_values)), np.nan)

    for i, p in enumerate(p_values):
        for j, n in enumerate(n_values):
            n_train = n // 3
            n_cal = n // 3
            n_test = n - n_train - n_cal

            if n_train < 2 * p:
                print(f"  Skipping n={n}, p={p} (insufficient samples)")
                continue

            print(f"  n={n}, p={p}...", end=" ", flush=True)
            max_gaps = []

            for rep in range(n_reps):
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_textbook(
                    n_train=n_train, n_cal=n_cal, n_test=n_test,
                    p=p, sigma=1.0, random_state=rep,
                )
                h_test = meta["h_test"]

                method = LWCPMethod(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                covered = (y_te >= lower) & (y_te <= upper)

                bin_edges = np.percentile(h_test, np.linspace(0, 100, n_bins + 1))
                bin_gaps = []
                for b in range(n_bins):
                    if b == n_bins - 1:
                        mask = (h_test >= bin_edges[b]) & (h_test <= bin_edges[b + 1])
                    else:
                        mask = (h_test >= bin_edges[b]) & (h_test < bin_edges[b + 1])
                    if mask.sum() >= 5:
                        bin_gaps.append(abs(np.mean(covered[mask]) - (1 - alpha)))

                if bin_gaps:
                    max_gaps.append(max(bin_gaps))

            if max_gaps:
                gap = np.mean(max_gaps)
                gap_matrix[i, j] = gap
                print(f"max gap = {gap:.4f}")
            else:
                print("no data")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(gap_matrix, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels(p_values)
    ax.set_xlabel("Total sample size n")
    ax.set_ylabel("Number of features p")
    ax.set_title("Experiment 9: Max Conditional Coverage Gap (LWCP)")
    for ii in range(len(p_values)):
        for jj in range(len(n_values)):
            val = gap_matrix[ii, jj]
            if not np.isnan(val):
                ax.text(jj, ii, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if val > 0.05 else "black")
    fig.colorbar(im, ax=ax, label="Max coverage gap")
    fig.tight_layout()
    savefig(fig, "exp9_scaling_heatmap")

    # Gap vs n
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = ["#2166ac", "#e08214", "#1b7837", "#d73027"]
    for i, p in enumerate(p_values):
        valid = ~np.isnan(gap_matrix[i])
        if valid.sum() > 1:
            ax2.plot(np.array(n_values)[valid], gap_matrix[i, valid], "o-",
                     color=colors[i % len(colors)], label=f"p={p}", markersize=6)
    ax2.set_xlabel("Total sample size n")
    ax2.set_ylabel("Max conditional coverage gap")
    ax2.set_xscale("log")
    ax2.set_title("Experiment 9: Coverage Gap vs Sample Size")
    ax2.legend()
    fig2.tight_layout()
    savefig(fig2, "exp9_gap_vs_n")

    # Save
    save_data = {"n_values": n_values, "p_values": p_values,
                 "gap_matrix": gap_matrix.tolist()}
    with open(RESULTS_DIR / "exp9_scaling.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp9_scaling.json'}")


if __name__ == "__main__":
    run_scaling_experiment()
