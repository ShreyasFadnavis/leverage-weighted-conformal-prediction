"""Experiment 1: Marginal Coverage Validation (Theorem 1).

Confirms that both Vanilla CP and LWCP achieve the nominal 1-alpha
coverage across many repetitions on all 5 DGPs.
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


def run_coverage_experiment(n_reps: int = 1000, alpha: float = 0.1):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 1: Marginal Coverage Validation (Theorem 1)")
    print("=" * 60)

    dgp_names = list(ALL_DGPS.keys())
    methods = ["Vanilla CP", "LWCP"]
    results = {dgp: {m: [] for m in methods} for dgp in dgp_names}

    for dgp_name in dgp_names:
        dgp_func = ALL_DGPS[dgp_name]
        print(f"\n  DGP: {dgp_name}")

        for rep in range(n_reps):
            # Use DGP defaults for sizes, only override random_state
            X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(
                sigma=1.0, random_state=rep,
            )

            for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
                method = method_cls(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                cov = np.mean((y_te >= lower) & (y_te <= upper))
                results[dgp_name][method_name].append(cov)

        for m in methods:
            arr = np.array(results[dgp_name][m])
            print(f"    {m}: coverage = {arr.mean():.4f} +/- {arr.std():.4f}")

    # --- Save results ---
    summary = {}
    print("\n" + "=" * 60)
    print(f"{'DGP':<25} {'Vanilla CP':>20} {'LWCP':>20}")
    print("-" * 65)
    for dgp_name in dgp_names:
        v = np.array(results[dgp_name]["Vanilla CP"])
        l = np.array(results[dgp_name]["LWCP"])
        print(f"{dgp_name:<25} {v.mean():.4f} +/- {v.std():.4f}   {l.mean():.4f} +/- {l.std():.4f}")
        summary[dgp_name] = {
            "Vanilla CP": {"mean": float(v.mean()), "std": float(v.std())},
            "LWCP": {"mean": float(l.mean()), "std": float(l.std())},
        }

    with open(RESULTS_DIR / "exp1_coverage.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp1_coverage.json'}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, dgp_name in enumerate(dgp_names):
        ax = axes[i]
        for m in methods:
            ax.hist(
                results[dgp_name][m], bins=30, alpha=0.6,
                color=get_color(m), label=m, density=True,
            )
        ax.axvline(1 - alpha, color="red", linestyle="--", linewidth=2, label=f"Nominal {1-alpha:.0%}")
        ax.set_title(dgp_name)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for j in range(len(dgp_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Experiment 1: Marginal Coverage Distribution", fontsize=14)
    fig.tight_layout()
    savefig(fig, "exp1_marginal_coverage")

    return results


if __name__ == "__main__":
    run_coverage_experiment()
