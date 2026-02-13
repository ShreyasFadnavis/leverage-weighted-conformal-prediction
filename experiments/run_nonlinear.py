"""Experiment: LWCP with Non-Linear Predictors.

Demonstrates that LWCP works beyond OLS, using Random Forest and MLP
predictors with both raw-X leverage and feature-space leverage.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from experiments.baselines import (
    LWCPMethod,
    LWCPMethod_NonLinear,
    LocalizedCP,
    StudentizedCP,
    VanillaCP,
    run_method_timed,
)
from experiments.dgps import dgp_nonlinear, dgp_textbook
from experiments.plotting import get_color, nominal_line, savefig, setup_style
from lwcp.leverage import mlp_feature_extractor

RESULTS_DIR = Path(__file__).parent / "results"

# Colors for non-linear experiment methods
NL_COLORS = {
    "Vanilla CP (OLS)": "#7f7f7f",
    "LWCP (OLS)": "#d62728",
    "Vanilla CP (RF)": "#bcbd22",
    "LWCP raw-X (RF)": "#e377c2",
    "Vanilla CP (MLP)": "#8c564b",
    "LWCP last-layer (MLP)": "#17becf",
    "Studentized CP (RF)": "#2ca02c",
}
NL_MARKERS = {
    "Vanilla CP (OLS)": "s",
    "LWCP (OLS)": "o",
    "Vanilla CP (RF)": "s",
    "LWCP raw-X (RF)": "D",
    "Vanilla CP (MLP)": "s",
    "LWCP last-layer (MLP)": "P",
    "Studentized CP (RF)": "^",
}


def _make_rf_predictor(random_state=42):
    return RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1,
    )


def _make_mlp_predictor(random_state=42):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )


def run_nonlinear_experiment(n_reps: int = 100, alpha: float = 0.1, n_bins: int = 10):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment: LWCP with Non-Linear Predictors")
    print("=" * 60)

    dgps = [
        ("Textbook (linear)", dgp_textbook),
        ("Non-linear (Σsin)", dgp_nonlinear),
    ]

    results = {}

    for dgp_label, dgp_func in dgps:
        print(f"\n  DGP: {dgp_label}")
        results[dgp_label] = {}

        for rep in range(n_reps):
            X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(
                sigma=1.0, random_state=rep,
            )

            # Build methods for this rep (MLP needs to be trained to extract features)
            rf = _make_rf_predictor(random_state=rep)
            mlp = _make_mlp_predictor(random_state=rep)

            # Train MLP to get feature extractor
            mlp_for_features = _make_mlp_predictor(random_state=rep)
            mlp_for_features.fit(X_tr, y_tr)
            feat_extractor = mlp_feature_extractor(mlp_for_features)

            methods = {
                "Vanilla CP (OLS)": VanillaCP(predictor=LinearRegression(), alpha=alpha),
                "LWCP (OLS)": LWCPMethod(predictor=LinearRegression(), alpha=alpha),
                "Vanilla CP (RF)": VanillaCP(predictor=_make_rf_predictor(rep), alpha=alpha),
                "LWCP raw-X (RF)": LWCPMethod_NonLinear(
                    predictor=_make_rf_predictor(rep),
                    feature_extractor=None,  # raw-X leverage
                    alpha=alpha,
                ),
                "Vanilla CP (MLP)": VanillaCP(predictor=_make_mlp_predictor(rep), alpha=alpha),
                "LWCP last-layer (MLP)": LWCPMethod_NonLinear(
                    predictor=_make_mlp_predictor(rep),
                    feature_extractor=feat_extractor,
                    alpha=alpha,
                    ridge=1e-3,  # regularize feature-space gram
                ),
                "Studentized CP (RF)": StudentizedCP(
                    predictor=_make_rf_predictor(rep), alpha=alpha,
                ),
            }

            for method_name, method in methods.items():
                if method_name not in results[dgp_label]:
                    results[dgp_label][method_name] = {
                        "coverages": [], "widths": [], "times": [],
                        "all_h": [], "all_covered": [],
                    }
                try:
                    y_pred, lower, upper, fit_t, pred_t = run_method_timed(
                        method, X_tr, y_tr, X_cal, y_cal, X_te,
                    )
                    covered = (y_te >= lower) & (y_te <= upper)
                    r = results[dgp_label][method_name]
                    r["coverages"].append(np.mean(covered))
                    r["widths"].append(np.mean(upper - lower))
                    r["times"].append(fit_t + pred_t)
                    r["all_h"].append(meta["h_test"])
                    r["all_covered"].append(covered)
                except Exception as e:
                    if rep == 0:
                        print(f"    [WARN] {method_name} failed: {e}")

            if (rep + 1) % 25 == 0:
                print(f"    rep {rep + 1}/{n_reps} done")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'DGP':<22s} {'Method':<26s} {'Cov':>6s} {'Width':>8s} "
          f"{'Gap':>8s} {'Time':>8s}")
    print("-" * 78)

    for dgp_label in results:
        for method_name in results[dgp_label]:
            r = results[dgp_label][method_name]
            if not r["coverages"]:
                continue
            cov = np.mean(r["coverages"])
            width = np.mean(r["widths"])
            time_s = np.mean(r["times"])

            # Conditional gap
            h_pooled = np.concatenate(r["all_h"])
            cov_pooled = np.concatenate(r["all_covered"])
            h_lo = np.percentile(h_pooled, 20)
            h_hi = np.percentile(h_pooled, 80)
            mask_lo = h_pooled <= h_lo
            mask_hi = h_pooled >= h_hi
            gap = abs(np.mean(cov_pooled[mask_lo]) - np.mean(cov_pooled[mask_hi]))

            print(f"{dgp_label:<22s} {method_name:<26s} {cov:6.3f} "
                  f"{width:8.3f} {gap:8.4f} {time_s:8.4f}s")

    # Figure: conditional coverage by leverage decile
    fig, axes = plt.subplots(1, len(dgps), figsize=(8 * len(dgps), 5))
    if len(dgps) == 1:
        axes = [axes]

    for col, (dgp_label, _) in enumerate(dgps):
        ax = axes[col]
        for method_name in results[dgp_label]:
            r = results[dgp_label][method_name]
            if not r["all_h"]:
                continue
            h_pooled = np.concatenate(r["all_h"])
            cov_pooled = np.concatenate(r["all_covered"])
            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            bin_coverages = []
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled <= bin_edges[i + 1])
                else:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled < bin_edges[i + 1])
                bin_coverages.append(np.mean(cov_pooled[mask]) if mask.sum() > 0 else np.nan)

            color = NL_COLORS.get(method_name, "#333333")
            marker = NL_MARKERS.get(method_name, "o")
            ax.plot(range(1, n_bins + 1), bin_coverages, marker=marker,
                    color=color, label=method_name, markersize=5, linewidth=1.5)

        nominal_line(ax, alpha, label=(col == 0))
        ax.set_title(dgp_label, fontsize=12)
        ax.set_xlabel("Leverage decile")
        ax.set_ylabel("Conditional coverage")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("LWCP with Non-Linear Predictors", fontsize=14)
    fig.tight_layout()
    savefig(fig, "exp_nonlinear")

    # Save results (exclude non-serializable arrays)
    save_results = {}
    for dgp_label in results:
        save_results[dgp_label] = {}
        for m in results[dgp_label]:
            r = results[dgp_label][m]
            if not r["coverages"]:
                continue
            h_pooled = np.concatenate(r["all_h"])
            cov_pooled = np.concatenate(r["all_covered"])
            h_lo = np.percentile(h_pooled, 20)
            h_hi = np.percentile(h_pooled, 80)
            mask_lo = h_pooled <= h_lo
            mask_hi = h_pooled >= h_hi
            gap = abs(float(np.mean(cov_pooled[mask_lo]) - np.mean(cov_pooled[mask_hi])))
            save_results[dgp_label][m] = {
                "coverage": float(np.mean(r["coverages"])),
                "coverage_std": float(np.std(r["coverages"])),
                "width": float(np.mean(r["widths"])),
                "width_std": float(np.std(r["widths"])),
                "cond_gap": gap,
                "time": float(np.mean(r["times"])),
            }

    with open(RESULTS_DIR / "exp_nonlinear.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp_nonlinear.json'}")


if __name__ == "__main__":
    run_nonlinear_experiment()
