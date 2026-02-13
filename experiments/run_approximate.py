"""Experiment 8: Approximate Leverage Scores (Theorem 4).

Shows that coverage is preserved exactly with approximate leverage
scores, while adaptivity degrades gracefully.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from lwcp import LWCP, InverseRootLeverageWeight
from lwcp.leverage import LeverageComputer
from experiments.dgps import dgp_textbook
from experiments.plotting import savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_approximate_experiment(n_reps: int = 200, alpha: float = 0.1):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 8: Approximate Leverage Scores (Theorem 4)")
    print("=" * 60)

    p_values = [10, 30, 50, 100]
    approx_fracs = [1.0, 0.5, 0.25]
    n_train = 200

    # Part 1: Coverage
    print("\n  Part 1: Marginal Coverage")
    coverage_results = {}

    for p in p_values:
        coverage_results[p] = {}
        configs = [("Exact", "exact", None)]
        for frac in approx_fracs:
            k = max(int(p * frac), 1)
            configs.append((f"Approx (k={k})", "approximate", k))

        for label, method, n_comp in configs:
            coverages = []
            for rep in range(n_reps):
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_textbook(
                    n_train=n_train, n_cal=n_train, n_test=500,
                    p=p, sigma=1.0, random_state=rep,
                )
                model = LWCP(
                    predictor=LinearRegression(),
                    weight_fn=InverseRootLeverageWeight(),
                    alpha=alpha,
                    leverage_method=method,
                    n_components=n_comp,
                    random_state=42,
                )
                model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = model.predict(X_te)
                coverages.append(np.mean((y_te >= lower) & (y_te <= upper)))

            coverage_results[p][label] = float(np.mean(coverages))
            print(f"    p={p:>3d}, {label:<20s}: coverage = {np.mean(coverages):.4f}")

    # Part 2: Leverage accuracy scatter
    print("\n  Part 2: Leverage Score Accuracy")
    fig2, axes = plt.subplots(1, len(p_values), figsize=(4 * len(p_values), 4))
    if len(p_values) == 1:
        axes = [axes]

    for idx, p in enumerate(p_values):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_train, p))
        h_exact = LeverageComputer(method="exact").fit(X).leverage_scores(X)

        ax = axes[idx]
        colors = ["#2166ac", "#e08214", "#1b7837"]
        for i, frac in enumerate(approx_fracs):
            k = max(int(p * frac), 1)
            h_approx = LeverageComputer(method="approximate", n_components=k, random_state=42).fit(X).leverage_scores(X)
            ax.scatter(h_exact, h_approx, alpha=0.3, s=8, color=colors[i], label=f"k={k}")
        ax.plot([0, h_exact.max()], [0, h_exact.max()], "k--", alpha=0.5)
        ax.set_xlabel("Exact leverage")
        ax.set_ylabel("Approximate leverage")
        ax.set_title(f"p = {p}")
        ax.legend(fontsize=8)

    fig2.suptitle("Experiment 8: Approximate vs Exact Leverage Scores", fontsize=13)
    fig2.tight_layout()
    savefig(fig2, "exp8_approx_leverage_scatter")

    # Part 3: Runtime
    print("\n  Part 3: Runtime")
    exact_times, approx_times = [], []
    for p in p_values:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_train, p))

        t0 = time.perf_counter()
        for _ in range(10):
            LeverageComputer(method="exact").fit(X)
        exact_t = (time.perf_counter() - t0) / 10

        k = max(p // 2, 1)
        t0 = time.perf_counter()
        for _ in range(10):
            LeverageComputer(method="approximate", n_components=k, random_state=42).fit(X)
        approx_t = (time.perf_counter() - t0) / 10

        exact_times.append(exact_t)
        approx_times.append(approx_t)
        print(f"    p={p:>3d}: exact={exact_t:.4f}s, approx(k={k})={approx_t:.4f}s")

    fig3, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, exact_times, "o-", color="#d73027", label="Exact SVD", markersize=7)
    ax.plot(p_values, approx_times, "s-", color="#2166ac", label="Randomized SVD (k=p/2)", markersize=7)
    ax.set_xlabel("Number of features p")
    ax.set_ylabel("Fit time (seconds)")
    ax.set_title("Experiment 8: SVD Runtime Comparison")
    ax.legend()
    fig3.tight_layout()
    savefig(fig3, "exp8_runtime")

    # Coverage bar chart
    fig1, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(p_values))
    all_labels = list(next(iter(coverage_results.values())).keys())
    bar_width = 0.2
    colors_bar = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0"]
    for i, label in enumerate(all_labels):
        vals = [coverage_results[p].get(label, float("nan")) for p in p_values]
        ax.bar(x + i * bar_width, vals, bar_width, label=label, color=colors_bar[i], alpha=0.9)
    ax.axhline(1 - alpha, color="red", linestyle="--", alpha=0.7, label=f"Nominal {1-alpha:.0%}")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([f"p={p}" for p in p_values])
    ax.set_ylabel("Marginal coverage")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Experiment 8: Coverage with Approximate Leverage")
    ax.legend(fontsize=8)
    fig1.tight_layout()
    savefig(fig1, "exp8_approx_coverage")

    # Part 4: Concrete epsilon values and conditional coverage gap
    print("\n  Part 4: Concrete ε Values and Conditional Gap")
    epsilon_results = {}

    for p in p_values:
        epsilon_results[p] = {}
        for frac in approx_fracs:
            k = max(int(p * frac), 1)
            epsilons, cond_gaps_exact, cond_gaps_approx = [], [], []

            for rep in range(min(n_reps, 100)):
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_textbook(
                    n_train=n_train, n_cal=n_train, n_test=500,
                    p=p, sigma=1.0, random_state=rep,
                )
                # Exact leverage
                lev_exact = LeverageComputer(method="exact").fit(X_tr)
                h_exact = lev_exact.leverage_scores(X_te)

                # Approximate leverage
                lev_approx = LeverageComputer(method="approximate", n_components=k, random_state=42).fit(X_tr)
                h_approx = lev_approx.leverage_scores(X_te)

                # Epsilon: max relative error
                h_max = np.max(h_exact)
                if h_max > 1e-15:
                    eps = float(np.max(np.abs(h_approx - h_exact)) / h_max)
                else:
                    eps = 0.0
                epsilons.append(eps)

                # Conditional gap for exact
                model_exact = LWCP(
                    predictor=LinearRegression(),
                    weight_fn=InverseRootLeverageWeight(),
                    alpha=alpha,
                    leverage_method="exact",
                )
                model_exact.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
                _, lower_e, upper_e = model_exact.predict(X_te)
                covered_e = (y_te >= lower_e) & (y_te <= upper_e)

                # Conditional gap for approximate
                model_approx = LWCP(
                    predictor=LinearRegression(),
                    weight_fn=InverseRootLeverageWeight(),
                    alpha=alpha,
                    leverage_method="approximate",
                    n_components=k,
                    random_state=42,
                )
                model_approx.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
                _, lower_a, upper_a = model_approx.predict(X_te)
                covered_a = (y_te >= lower_a) & (y_te <= upper_a)

                h_lo = np.percentile(h_exact, 20)
                h_hi = np.percentile(h_exact, 80)
                mask_lo = h_exact <= h_lo
                mask_hi = h_exact >= h_hi
                if mask_lo.sum() > 0 and mask_hi.sum() > 0:
                    gap_e = abs(np.mean(covered_e[mask_lo]) - np.mean(covered_e[mask_hi]))
                    gap_a = abs(np.mean(covered_a[mask_lo]) - np.mean(covered_a[mask_hi]))
                else:
                    gap_e = gap_a = 0.0
                cond_gaps_exact.append(gap_e)
                cond_gaps_approx.append(gap_a)

            epsilon_results[p][f"k={k}"] = {
                "epsilon_mean": float(np.mean(epsilons)),
                "epsilon_max": float(np.max(epsilons)),
                "gap_exact": float(np.mean(cond_gaps_exact)),
                "gap_approx": float(np.mean(cond_gaps_approx)),
                "gap_diff": float(abs(np.mean(cond_gaps_approx) - np.mean(cond_gaps_exact))),
            }
            r = epsilon_results[p][f"k={k}"]
            print(f"    p={p:>3d}, k={k:>3d}: ε_mean={r['epsilon_mean']:.4f}, "
                  f"ε_max={r['epsilon_max']:.4f}, "
                  f"gap_exact={r['gap_exact']:.4f}, gap_approx={r['gap_approx']:.4f}")

    # Save all results
    save_data = {str(p): cov for p, cov in coverage_results.items()}
    save_data["epsilon"] = {str(p): v for p, v in epsilon_results.items()}
    with open(RESULTS_DIR / "exp8_approximate.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp8_approximate.json'}")


if __name__ == "__main__":
    run_approximate_experiment()
