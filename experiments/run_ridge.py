"""Experiment: Ridge Leverage LWCP for p > n settings.

Tests LWCP with ridge leverage scores when the number of features
exceeds the training sample size, demonstrating the extension
described in Appendix B.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from lwcp import LWCP, ConstantWeight, InverseRootLeverageWeight
from lwcp._utils import conformal_quantile
from experiments.plotting import savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_ridge_experiment(n_reps: int = 200, alpha: float = 0.1):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Ridge Leverage Experiment: LWCP for p > n")
    print("=" * 60)

    n_train = 100
    n_cal = 200
    n_test = 200
    sigma = 1.0
    p_values = [50, 100, 200, 500]
    ridge_lambda = 1.0  # fixed ridge parameter

    results = {}

    for p in p_values:
        print(f"\n  p = {p}, n_train = {n_train}, p/n = {p/n_train:.1f}")

        coverages_lwcp = []
        coverages_vanilla = []
        widths_lwcp = []
        widths_vanilla = []
        cond_gaps_lwcp = []
        cond_gaps_vanilla = []

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)

            # Generate data with decaying eigenvalues
            Sigma_diag = 1.0 / np.arange(1, p + 1)
            L = np.diag(np.sqrt(Sigma_diag))

            n_total = n_train + n_cal + n_test
            X_all = rng.standard_normal((n_total, p)) @ L.T
            beta = rng.standard_normal(p) / np.sqrt(p)  # scale beta with p

            X_tr = X_all[:n_train]
            X_cal = X_all[n_train:n_train + n_cal]
            X_te = X_all[n_train + n_cal:]

            # Homoscedastic errors (g=1)
            y_tr = X_tr @ beta + sigma * rng.standard_normal(n_train)
            y_cal = X_cal @ beta + sigma * rng.standard_normal(n_cal)
            y_te = X_te @ beta + sigma * rng.standard_normal(n_test)

            # LWCP with ridge
            lwcp_model = LWCP(
                predictor=Ridge(alpha=ridge_lambda),
                weight_fn=InverseRootLeverageWeight(),
                alpha=alpha,
                ridge=ridge_lambda,
            )
            lwcp_model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
            _, lower_l, upper_l = lwcp_model.predict(X_te)

            # Vanilla CP with ridge (no leverage weighting)
            from lwcp import ConstantWeight
            vanilla_model = LWCP(
                predictor=Ridge(alpha=ridge_lambda),
                weight_fn=ConstantWeight(),
                alpha=alpha,
                ridge=ridge_lambda,
            )
            vanilla_model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
            _, lower_v, upper_v = vanilla_model.predict(X_te)

            # Coverage
            cov_l = np.mean((y_te >= lower_l) & (y_te <= upper_l))
            cov_v = np.mean((y_te >= lower_v) & (y_te <= upper_v))
            coverages_lwcp.append(cov_l)
            coverages_vanilla.append(cov_v)

            # Width
            widths_lwcp.append(np.mean(upper_l - lower_l))
            widths_vanilla.append(np.mean(upper_v - lower_v))

            # Conditional gap (using ridge leverage)
            h_te = lwcp_model.leverage_computer_.leverage_scores(X_te)
            h20 = np.percentile(h_te, 20)
            h80 = np.percentile(h_te, 80)
            mask_lo = h_te <= h20
            mask_hi = h_te >= h80

            if mask_lo.sum() > 0 and mask_hi.sum() > 0:
                covered_l = (y_te >= lower_l) & (y_te <= upper_l)
                covered_v = (y_te >= lower_v) & (y_te <= upper_v)
                gap_l = np.mean(covered_l[mask_lo]) - np.mean(covered_l[mask_hi])
                gap_v = np.mean(covered_v[mask_lo]) - np.mean(covered_v[mask_hi])
            else:
                gap_l = gap_v = 0.0
            cond_gaps_lwcp.append(abs(gap_l))
            cond_gaps_vanilla.append(abs(gap_v))

        results[str(p)] = {
            "LWCP": {
                "coverage": float(np.mean(coverages_lwcp)),
                "coverage_std": float(np.std(coverages_lwcp)),
                "width": float(np.mean(widths_lwcp)),
                "width_std": float(np.std(widths_lwcp)),
                "cond_gap": float(np.mean(cond_gaps_lwcp)),
            },
            "Vanilla": {
                "coverage": float(np.mean(coverages_vanilla)),
                "coverage_std": float(np.std(coverages_vanilla)),
                "width": float(np.mean(widths_vanilla)),
                "width_std": float(np.std(widths_vanilla)),
                "cond_gap": float(np.mean(cond_gaps_vanilla)),
            },
            "width_ratio": float(np.mean(widths_lwcp) / np.mean(widths_vanilla)),
        }

        print(f"    Vanilla: cov={np.mean(coverages_vanilla):.3f}, "
              f"width={np.mean(widths_vanilla):.3f}, gap={np.mean(cond_gaps_vanilla):.3f}")
        print(f"    LWCP:    cov={np.mean(coverages_lwcp):.3f}, "
              f"width={np.mean(widths_lwcp):.3f}, gap={np.mean(cond_gaps_lwcp):.3f}")
        print(f"    Width ratio: {results[str(p)]['width_ratio']:.4f}")

    with open(RESULTS_DIR / "exp_ridge.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp_ridge.json'}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    ps = [int(k) for k in results.keys()]

    # (a) Coverage
    ax = axes[0]
    ax.plot(ps, [results[str(p_)]["Vanilla"]["coverage"] for p_ in ps],
            "s--", color="#7f7f7f", label="Vanilla CP", markersize=7)
    ax.plot(ps, [results[str(p_)]["LWCP"]["coverage"] for p_ in ps],
            "o-", color="#d62728", label="LWCP", markersize=7)
    ax.axhline(0.9, color="#333", linestyle=":", alpha=0.5)
    ax.set_xlabel("$p$ (features)")
    ax.set_ylabel("Marginal coverage")
    ax.set_title("(a) Coverage")
    ax.legend(fontsize=9)

    # (b) Width ratio
    ax = axes[1]
    ax.plot(ps, [results[str(p_)]["width_ratio"] for p_ in ps],
            "D-", color="#1f77b4", markersize=7)
    ax.axhline(1.0, color="#333", linestyle=":", alpha=0.5)
    ax.set_xlabel("$p$ (features)")
    ax.set_ylabel("LWCP / Vanilla width ratio")
    ax.set_title("(b) Width ratio")

    # (c) Conditional gap
    ax = axes[2]
    ax.plot(ps, [results[str(p_)]["Vanilla"]["cond_gap"] for p_ in ps],
            "s--", color="#7f7f7f", label="Vanilla CP", markersize=7)
    ax.plot(ps, [results[str(p_)]["LWCP"]["cond_gap"] for p_ in ps],
            "o-", color="#d62728", label="LWCP", markersize=7)
    ax.set_xlabel("$p$ (features)")
    ax.set_ylabel("Conditional coverage gap")
    ax.set_title("(c) Coverage gap")
    ax.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, "exp_ridge_leverage")

    return results


def run_ridge_extended(n_reps: int = 100, alpha: float = 0.1):
    """Extended ridge experiment: LWCP+, heteroscedastic variant, lambda sweep."""
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Ridge Extended: LWCP+, Heteroscedastic, Lambda Sweep")
    print("=" * 60)

    from experiments.baselines import LWCPMethod, LWCPPlus, VanillaCP
    from lwcp.leverage import LeverageComputer

    n_train, n_cal, n_test = 100, 200, 200
    p = 200
    sigma = 1.0

    # --- Part A: LWCP+ with ridge leverage ---
    print("\n  Part A: LWCP+ with ridge leverage (p=200, n=100)")
    ridge_lambda = 1.0
    results_plus = {"Vanilla": [], "LWCP": [], "LWCP+": []}

    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        Sigma_diag = 1.0 / np.arange(1, p + 1)
        L = np.diag(np.sqrt(Sigma_diag))
        n_total = n_train + n_cal + n_test
        X_all = rng.standard_normal((n_total, p)) @ L.T
        beta = rng.standard_normal(p) / np.sqrt(p)

        X_tr = X_all[:n_train]
        X_cal = X_all[n_train:n_train + n_cal]
        X_te = X_all[n_train + n_cal:]

        y_tr = X_tr @ beta + sigma * rng.standard_normal(n_train)
        y_cal = X_cal @ beta + sigma * rng.standard_normal(n_cal)
        y_te = X_te @ beta + sigma * rng.standard_normal(n_test)

        lev = LeverageComputer(ridge=ridge_lambda).fit(X_tr)
        h_te = lev.leverage_scores(X_te)
        h_lo = np.percentile(h_te, 20)
        h_hi = np.percentile(h_te, 80)
        mask_lo = h_te <= h_lo
        mask_hi = h_te >= h_hi

        methods = {
            "Vanilla": LWCP(predictor=Ridge(alpha=ridge_lambda),
                            weight_fn=ConstantWeight(), alpha=alpha, ridge=ridge_lambda),
            "LWCP": LWCP(predictor=Ridge(alpha=ridge_lambda),
                         weight_fn=InverseRootLeverageWeight(), alpha=alpha, ridge=ridge_lambda),
        }

        for name, model in methods.items():
            model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
            _, lower, upper = model.predict(X_te)
            covered = (y_te >= lower) & (y_te <= upper)
            gap = abs(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi])) if mask_lo.sum() > 0 and mask_hi.sum() > 0 else 0.0
            results_plus[name].append({"cov": float(np.mean(covered)), "gap": gap,
                                        "width": float(np.mean(upper - lower))})

        # LWCP+ with ridge
        lwcp_plus = LWCPPlus(predictor=Ridge(alpha=ridge_lambda), alpha=alpha)
        lwcp_plus.fit(X_tr, y_tr, X_cal, y_cal)
        _, lower, upper = lwcp_plus.predict(X_te)
        covered = (y_te >= lower) & (y_te <= upper)
        gap = abs(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi])) if mask_lo.sum() > 0 and mask_hi.sum() > 0 else 0.0
        results_plus["LWCP+"].append({"cov": float(np.mean(covered)), "gap": gap,
                                       "width": float(np.mean(upper - lower))})

    for name in results_plus:
        cov = np.mean([r["cov"] for r in results_plus[name]])
        gap = np.mean([r["gap"] for r in results_plus[name]])
        width = np.mean([r["width"] for r in results_plus[name]])
        print(f"    {name:<10s}: cov={cov:.3f}, gap={gap:.4f}, width={width:.3f}")

    # --- Part B: Lambda sensitivity sweep ---
    print("\n  Part B: Lambda sensitivity sweep (p=200, n=100)")
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    lambda_results = {}

    for lam in lambdas:
        gaps_v, gaps_l = [], []
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            Sigma_diag = 1.0 / np.arange(1, p + 1)
            L = np.diag(np.sqrt(Sigma_diag))
            n_total = n_train + n_cal + n_test
            X_all = rng.standard_normal((n_total, p)) @ L.T
            beta = rng.standard_normal(p) / np.sqrt(p)

            X_tr = X_all[:n_train]
            X_cal = X_all[n_train:n_train + n_cal]
            X_te = X_all[n_train + n_cal:]

            y_tr = X_tr @ beta + sigma * rng.standard_normal(n_train)
            y_cal = X_cal @ beta + sigma * rng.standard_normal(n_cal)
            y_te = X_te @ beta + sigma * rng.standard_normal(n_test)

            for weight_fn, gap_list in [
                (ConstantWeight(), gaps_v),
                (InverseRootLeverageWeight(), gaps_l),
            ]:
                model = LWCP(predictor=Ridge(alpha=lam), weight_fn=weight_fn,
                             alpha=alpha, ridge=lam)
                model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = model.predict(X_te)
                covered = (y_te >= lower) & (y_te <= upper)
                h_te = model.leverage_computer_.leverage_scores(X_te)
                h_lo = np.percentile(h_te, 20)
                h_hi = np.percentile(h_te, 80)
                mask_lo = h_te <= h_lo
                mask_hi = h_te >= h_hi
                gap = abs(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi])) if mask_lo.sum() > 0 and mask_hi.sum() > 0 else 0.0
                gap_list.append(gap)

        lambda_results[str(lam)] = {
            "vanilla_gap": float(np.mean(gaps_v)),
            "lwcp_gap": float(np.mean(gaps_l)),
        }
        print(f"    λ={lam:<6.2f}: Vanilla gap={np.mean(gaps_v):.4f}, LWCP gap={np.mean(gaps_l):.4f}")

    # --- Part C: Heteroscedastic ridge ---
    print("\n  Part C: Heteroscedastic ridge (g(h^λ)=1+h^λ)")
    hetero_results = {"Vanilla": [], "LWCP": []}
    ridge_lambda = 1.0

    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        Sigma_diag = 1.0 / np.arange(1, p + 1)
        L = np.diag(np.sqrt(Sigma_diag))
        n_total = n_train + n_cal + n_test
        X_all = rng.standard_normal((n_total, p)) @ L.T
        beta = rng.standard_normal(p) / np.sqrt(p)

        X_tr = X_all[:n_train]
        X_cal = X_all[n_train:n_train + n_cal]
        X_te = X_all[n_train + n_cal:]

        # Heteroscedastic errors: g(h^λ) = 1 + h^λ
        lev = LeverageComputer(ridge=ridge_lambda).fit(X_tr)
        h_tr = lev.leverage_scores(X_tr)
        h_cal = lev.leverage_scores(X_cal)
        h_te = lev.leverage_scores(X_te)

        g = lambda h: 1.0 + h
        y_tr = X_tr @ beta + sigma * np.sqrt(g(h_tr)) * rng.standard_normal(n_train)
        y_cal = X_cal @ beta + sigma * np.sqrt(g(h_cal)) * rng.standard_normal(n_cal)
        y_te = X_te @ beta + sigma * np.sqrt(g(h_te)) * rng.standard_normal(n_test)

        h_lo = np.percentile(h_te, 20)
        h_hi = np.percentile(h_te, 80)
        mask_lo = h_te <= h_lo
        mask_hi = h_te >= h_hi

        for name, weight_fn in [("Vanilla", ConstantWeight()),
                                 ("LWCP", InverseRootLeverageWeight())]:
            model = LWCP(predictor=Ridge(alpha=ridge_lambda), weight_fn=weight_fn,
                         alpha=alpha, ridge=ridge_lambda)
            model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
            _, lower, upper = model.predict(X_te)
            covered = (y_te >= lower) & (y_te <= upper)
            gap = abs(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi])) if mask_lo.sum() > 0 and mask_hi.sum() > 0 else 0.0
            hetero_results[name].append(gap)

    for name in hetero_results:
        print(f"    {name:<10s}: gap={np.mean(hetero_results[name]):.4f}")

    all_extended = {
        "lwcp_plus": {n: {"cov": np.mean([r["cov"] for r in results_plus[n]]),
                           "gap": np.mean([r["gap"] for r in results_plus[n]]),
                           "width": np.mean([r["width"] for r in results_plus[n]])}
                       for n in results_plus},
        "lambda_sweep": lambda_results,
        "heteroscedastic": {n: float(np.mean(hetero_results[n])) for n in hetero_results},
    }
    with open(RESULTS_DIR / "exp_ridge_extended.json", "w") as f:
        json.dump(all_extended, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp_ridge_extended.json'}")


if __name__ == "__main__":
    run_ridge_experiment()
    run_ridge_extended()
