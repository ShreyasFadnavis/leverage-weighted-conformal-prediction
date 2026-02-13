"""Experiment 5: Recovery of Classical Gaussian Prediction Intervals (Theorem 3).

Under Gaussian linear model with homoscedastic errors, LWCP intervals
should converge to the classical parametric intervals as n grows.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from lwcp import LWCP, InverseRootLeverageWeight
from lwcp.leverage import LeverageComputer
from experiments.plotting import savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_gaussian_recovery(n_reps: int = 200, alpha: float = 0.1, p: int = 5, sigma: float = 1.0):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 5: Recovery of Classical Intervals (Theorem 3)")
    print("=" * 60)

    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    ratios = {n: [] for n in sample_sizes}

    for n in sample_sizes:
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            X = rng.standard_normal((n, p))
            beta = np.ones(p)
            y = X @ beta + sigma * rng.standard_normal(n)

            n_train = n // 2
            X_train, X_cal = X[:n_train], X[n_train:]
            y_train, y_cal = y[:n_train], y[n_train:]
            x_test = np.array([[0.5] * p])

            # LWCP
            model = LWCP(
                predictor=LinearRegression(),
                weight_fn=InverseRootLeverageWeight(),
                alpha=alpha,
            )
            model.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)
            _, lwcp_lo, lwcp_hi = model.predict(x_test)
            lwcp_width = (lwcp_hi - lwcp_lo)[0]

            # Classical parametric
            ols = LinearRegression().fit(X_train, y_train)
            resid_train = y_train - ols.predict(X_train)
            sigma_hat = np.sqrt(np.sum(resid_train**2) / (n_train - p))
            h_test = x_test @ np.linalg.inv(X_train.T @ X_train) @ x_test.T
            t_crit = stats.t.ppf(1 - alpha / 2, df=n_train - p)
            classical_width = 2 * t_crit * sigma_hat * np.sqrt(1 + h_test[0, 0])

            ratios[n].append(lwcp_width / classical_width)

        mean_r = np.mean(ratios[n])
        std_r = np.std(ratios[n])
        print(f"  n={n:>5d}: LWCP/classical width ratio = {mean_r:.4f} +/- {std_r:.4f}")

    # Save results
    summary = {str(n): {"mean": float(np.mean(ratios[n])), "std": float(np.std(ratios[n]))}
               for n in sample_sizes}
    with open(RESULTS_DIR / "exp5_gaussian.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Figure 1: Ratio vs n ---
    fig1, ax = plt.subplots(figsize=(8, 5))
    means = [np.mean(ratios[n]) for n in sample_sizes]
    stds = [np.std(ratios[n]) for n in sample_sizes]
    ax.errorbar(sample_sizes, means, yerr=stds, fmt="o-", color="#2166ac", capsize=4, markersize=7)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Ratio = 1 (exact recovery)")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel("LWCP width / Classical width")
    ax.set_xscale("log")
    ax.set_title("Experiment 5: Convergence to Classical Intervals")
    ax.legend()
    fig1.tight_layout()
    savefig(fig1, "exp5_gaussian_recovery")

    # --- Figure 2: Overlay intervals ---
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, n_show in enumerate([200, 2000]):
        ax = axes[idx]
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_show, p))
        beta = np.ones(p)
        y = X @ beta + sigma * rng.standard_normal(n_show)

        n_train = n_show // 2
        X_train, X_cal = X[:n_train], X[n_train:]
        y_train, y_cal = y[:n_train], y[n_train:]

        X_test = rng.standard_normal((100, p))
        lc = LeverageComputer().fit(X_train)
        h_tests = lc.leverage_scores(X_test)
        order = np.argsort(h_tests)
        X_test, h_tests = X_test[order], h_tests[order]

        model = LWCP(predictor=LinearRegression(), weight_fn=InverseRootLeverageWeight(), alpha=alpha)
        model.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)
        _, lwcp_lo, lwcp_hi = model.predict(X_test)

        ols = LinearRegression().fit(X_train, y_train)
        resid = y_train - ols.predict(X_train)
        sigma_hat = np.sqrt(np.sum(resid**2) / (n_train - p))
        t_crit = stats.t.ppf(1 - alpha / 2, df=n_train - p)
        classical_half = t_crit * sigma_hat * np.sqrt(1 + h_tests)

        ax.plot(h_tests, lwcp_hi - lwcp_lo, "o", color="#2166ac", alpha=0.5, markersize=4, label="LWCP width")
        ax.plot(h_tests, 2 * classical_half, "x", color="#d73027", alpha=0.5, markersize=4, label="Classical width")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Interval width")
        ax.set_title(f"n = {n_show}")
        ax.legend(fontsize=9)

    fig2.suptitle("Experiment 5: LWCP vs Classical Intervals", fontsize=14)
    fig2.tight_layout()
    savefig(fig2, "exp5_interval_overlay")

    print(f"\n  Results saved to {RESULTS_DIR / 'exp5_gaussian.json'}")


if __name__ == "__main__":
    run_gaussian_recovery()
