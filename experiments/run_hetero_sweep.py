"""Experiment 6: Heteroscedasticity Structure Sweep.

Shows when LWCP helps and when it doesn't:
- g(h) = 1+h: LWCP wins (optimal match with InverseRootLeverageWeight)
- g(h) = 1+5h: LWCP wins (partial match)
- g(h) = 1 (homoscedastic): tie
- Var ~ ||X||^2 (adversarial): LWCP ~ vanilla
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.baselines import LWCPMethod, VanillaCP
from experiments.dgps import dgp_adversarial, dgp_custom_g
from experiments.plotting import get_color, savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_hetero_sweep(n_reps: int = 200, alpha: float = 0.1, n_bins: int = 10):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 6: Heteroscedasticity Structure Sweep")
    print("=" * 60)

    configs = [
        ("g(h) = 1+h", lambda h: 1.0 + h, "leverage"),
        ("g(h) = 1+5h", lambda h: 1.0 + 5.0 * h, "leverage"),
        ("g(h) = 1", lambda h: np.ones_like(h), "leverage"),
        ("Var ~ ||X||^2", None, "adversarial"),
    ]

    methods = ["Vanilla CP", "LWCP"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    all_results = {}

    for col, (label, g_func, kind) in enumerate(configs):
        print(f"\n  Config: {label}")

        all_h = {m: [] for m in methods}
        all_covered = {m: [] for m in methods}
        all_widths = {m: [] for m in methods}

        for rep in range(n_reps):
            if kind == "leverage":
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_custom_g(
                    g_func=g_func, g_name=label, sigma=1.0, random_state=rep,
                )
            else:
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_adversarial(
                    sigma=1.0, random_state=rep,
                )
            h_test = meta["h_test"]

            for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
                method = method_cls(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                covered = (y_te >= lower) & (y_te <= upper)
                widths = upper - lower
                all_h[method_name].append(h_test)
                all_covered[method_name].append(covered)
                all_widths[method_name].append(widths)

        config_results = {}
        # Top row: conditional coverage
        ax_cov = axes[0, col]
        for method_name in methods:
            h_pooled = np.concatenate(all_h[method_name])
            cov_pooled = np.concatenate(all_covered[method_name])
            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            bin_coverages = []
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled <= bin_edges[i + 1])
                else:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled < bin_edges[i + 1])
                bin_coverages.append(float(np.mean(cov_pooled[mask])) if mask.sum() > 0 else None)
            ax_cov.plot(range(1, n_bins + 1), bin_coverages, "o-",
                        color=get_color(method_name), label=method_name, markersize=5)
        ax_cov.axhline(1 - alpha, color="red", linestyle="--", alpha=0.7)
        ax_cov.set_title(label)
        ax_cov.set_ylim(0.55, 1.05)
        if col == 0:
            ax_cov.set_ylabel("Conditional coverage")
        ax_cov.set_xlabel("Leverage decile")
        ax_cov.legend(fontsize=7)

        # Bottom row: width vs leverage
        ax_w = axes[1, col]
        for method_name in methods:
            ax_w.scatter(all_h[method_name][0], all_widths[method_name][0],
                         alpha=0.3, s=8, color=get_color(method_name), label=method_name)
        if col == 0:
            ax_w.set_ylabel("Interval width")
        ax_w.set_xlabel("Leverage")
        ax_w.legend(fontsize=7)

        for m in methods:
            mean_w = np.mean([np.mean(w) for w in all_widths[m]])
            mean_c = np.mean([np.mean(c) for c in all_covered[m]])
            print(f"    {m}: mean coverage = {mean_c:.4f}, mean width = {mean_w:.4f}")
            config_results[m] = {"coverage": float(mean_c), "width": float(mean_w)}
        all_results[label] = config_results

    fig.suptitle("Experiment 6: When Does LWCP Help?", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "exp6_hetero_sweep")

    with open(RESULTS_DIR / "exp6_hetero.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp6_hetero.json'}")


if __name__ == "__main__":
    run_hetero_sweep()
