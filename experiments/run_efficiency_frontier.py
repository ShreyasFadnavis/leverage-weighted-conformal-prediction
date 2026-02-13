"""Experiment 4: Efficiency Frontier — Coverage vs Width across alpha.

LWCP should Pareto-dominate vanilla CP under heteroscedastic settings
(same coverage, shorter intervals). Under homoscedastic settings,
curves should overlap.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.baselines import LWCPMethod, VanillaCP
from experiments.dgps import dgp_homoscedastic, dgp_textbook
from experiments.plotting import get_color, savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_efficiency_frontier(n_reps: int = 200):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 4: Efficiency Frontier (Coverage vs Width)")
    print("=" * 60)

    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    dgps = [
        ("Textbook (heteroscedastic)", dgp_textbook),
        ("Homoscedastic (null)", dgp_homoscedastic),
    ]
    methods = ["Vanilla CP", "LWCP"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    all_results = {}

    for col, (dgp_label, dgp_func) in enumerate(dgps):
        print(f"\n  DGP: {dgp_label}")
        frontier = {m: {"coverage": [], "width": []} for m in methods}

        for alpha in alphas:
            for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
                coverages, widths_list = [], []
                for rep in range(n_reps):
                    X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(
                        sigma=1.0, random_state=rep,
                    )
                    method = method_cls(predictor=LinearRegression(), alpha=alpha)
                    method.fit(X_tr, y_tr, X_cal, y_cal)
                    _, lower, upper = method.predict(X_te)
                    coverages.append(np.mean((y_te >= lower) & (y_te <= upper)))
                    widths_list.append(np.mean(upper - lower))

                frontier[method_name]["coverage"].append(float(np.mean(coverages)))
                frontier[method_name]["width"].append(float(np.mean(widths_list)))

            print(f"    alpha={alpha:.2f}: "
                  f"Vanilla width={frontier['Vanilla CP']['width'][-1]:.3f}, "
                  f"LWCP width={frontier['LWCP']['width'][-1]:.3f}")

        ax = axes[col]
        for m in methods:
            ax.plot(frontier[m]["coverage"], frontier[m]["width"], "o-",
                    color=get_color(m), label=m, markersize=7)
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Mean interval width")
        ax.set_title(dgp_label)
        ax.legend()
        ax.invert_xaxis()
        all_results[dgp_label] = frontier

    fig.suptitle("Experiment 4: Efficiency Frontier", fontsize=14)
    fig.tight_layout()
    savefig(fig, "exp4_efficiency_frontier")

    with open(RESULTS_DIR / "exp4_efficiency.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp4_efficiency.json'}")


if __name__ == "__main__":
    run_efficiency_frontier()
