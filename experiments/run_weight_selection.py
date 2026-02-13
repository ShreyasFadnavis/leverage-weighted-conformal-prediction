"""Experiment: Weight Selection Sensitivity Study.

Evaluates data-driven weight selection via 3-way split validation
across DGPs and real datasets, comparing selected weight vs. default
vs. constant (vanilla CP).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as t_dist
from scipy import linalg

from lwcp import LWCP, ConstantWeight, InverseRootLeverageWeight
from lwcp.weights import PowerLawWeight, WeightSelector
from experiments.dgps import (
    dgp_textbook,
    dgp_heavy_tailed,
    dgp_polynomial,
    dgp_adversarial,
    dgp_homoscedastic,
)
from experiments.metrics import compute_msce, compute_wsc

RESULTS_DIR = Path(__file__).parent / "results"


def _compute_leverage(X_ref, X_query):
    _, s, Vt = linalg.svd(X_ref, full_matrices=False)
    inv_d = np.where(s > 1e-15, 1.0 / s**2, 0.0)
    gram_inv = (Vt.T * inv_d) @ Vt
    XG = X_query @ gram_inv
    return np.sum(XG * X_query, axis=1)


def _eval_weight(X_tr, y_tr, X_cal, y_cal, X_te, y_te, h_te, weight_fn, alpha):
    """Evaluate a specific weight function and return coverage metrics."""
    model = LWCP(predictor=LinearRegression(), weight_fn=weight_fn, alpha=alpha)
    model.fit_with_precomputed_split(X_tr, y_tr, X_cal, y_cal)
    _, lower, upper = model.predict(X_te)
    covered = (y_te >= lower) & (y_te <= upper)
    cov = float(np.mean(covered))
    gap_lo = np.percentile(h_te, 20)
    gap_hi = np.percentile(h_te, 80)
    mask_lo = h_te <= gap_lo
    mask_hi = h_te >= gap_hi
    gap = abs(float(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi]))) if mask_lo.sum() > 0 and mask_hi.sum() > 0 else 0.0
    msce = compute_msce(h_te, covered, alpha=alpha)
    wsc = compute_wsc(h_te, covered)
    return {"cov": cov, "gap": gap, "msce": msce, "wsc": wsc}


def run_weight_selection_experiment(n_reps: int = 50, alpha: float = 0.1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Weight Selection Sensitivity Experiment")
    print("=" * 60)

    candidates = [
        ("Constant", ConstantWeight()),
        ("InvRoot", InverseRootLeverageWeight()),
        ("Power(0.3)", PowerLawWeight(gamma=0.3)),
        ("Power(0.7)", PowerLawWeight(gamma=0.7)),
    ]
    candidate_weights = [w for _, w in candidates]
    candidate_names = [n for n, _ in candidates]

    # Part 1: Synthetic DGPs
    dgps = [
        ("Textbook", dgp_textbook),
        ("Heavy-tail", dgp_heavy_tailed),
        ("Polynomial", dgp_polynomial),
        ("Adversarial", dgp_adversarial),
        ("Homoscedastic", dgp_homoscedastic),
    ]

    results = {"synthetic": {}, "real": {}}

    for dgp_name, dgp_func in dgps:
        print(f"\n  DGP: {dgp_name}")
        selection_counts = {n: 0 for n in candidate_names}
        per_weight_gaps = {n: [] for n in candidate_names}
        auto_gaps, auto_msces = [], []

        for rep in range(n_reps):
            X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(
                sigma=1.0, random_state=rep,
            )
            h_cal = meta["h_cal"]
            h_te = meta["h_test"]

            # Evaluate each weight individually
            for wname, wfn in candidates:
                r = _eval_weight(X_tr, y_tr, X_cal, y_cal, X_te, y_te, h_te, wfn, alpha)
                per_weight_gaps[wname].append(r["gap"])

            # Data-driven selection
            model_lr = LinearRegression().fit(X_tr, y_tr)
            cal_residuals = np.abs(y_cal - model_lr.predict(X_cal))
            selector = WeightSelector(
                candidates=candidate_weights,
                val_fraction=0.3,
                random_state=rep,
            )
            selected = selector.select(cal_residuals, h_cal, alpha=alpha)
            sel_name = candidate_names[candidate_weights.index(selected)]
            selection_counts[sel_name] += 1

            # Evaluate selected weight
            r_auto = _eval_weight(X_tr, y_tr, X_cal, y_cal, X_te, y_te, h_te, selected, alpha)
            auto_gaps.append(r_auto["gap"])
            auto_msces.append(r_auto["msce"])

        results["synthetic"][dgp_name] = {
            "selection_counts": selection_counts,
            "selection_fractions": {n: c / n_reps for n, c in selection_counts.items()},
            "per_weight_gap": {n: float(np.mean(per_weight_gaps[n])) for n in candidate_names},
            "auto_gap": float(np.mean(auto_gaps)),
            "auto_msce": float(np.mean(auto_msces)),
            "default_gap": float(np.mean(per_weight_gaps["InvRoot"])),
        }
        print(f"    Selection: {selection_counts}")
        print(f"    Gap — default: {np.mean(per_weight_gaps['InvRoot']):.4f}, "
              f"auto: {np.mean(auto_gaps):.4f}, "
              f"constant: {np.mean(per_weight_gaps['Constant']):.4f}")

    # Part 2: Real datasets
    print("\n  --- Real Datasets ---")
    real_datasets = {}
    diab = load_diabetes()
    real_datasets["Diabetes"] = (diab.data, diab.target)

    data = fetch_openml(name="cpu_act", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(float), data.target.astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    real_datasets["CPU Activity"] = (X[mask], y[mask])

    rng_ht = np.random.default_rng(42)
    n_ht, p_ht = 500, 100
    X_ht = t_dist.rvs(3, size=(n_ht, p_ht), random_state=42)
    beta_ht = np.zeros(p_ht)
    beta_ht[:10] = rng_ht.standard_normal(10) * 5
    y_ht = X_ht @ beta_ht + rng_ht.standard_normal(n_ht) * 10.0
    real_datasets["Heavy-tailed"] = (X_ht, y_ht)

    n_real_reps = min(n_reps, 20)

    for ds_name, (X_full, y_full) in real_datasets.items():
        print(f"\n  Dataset: {ds_name}")
        n_total = X_full.shape[0]
        selection_counts = {n: 0 for n in candidate_names}
        per_weight_gaps = {n: [] for n in candidate_names}
        auto_gaps = []

        for rep in range(n_real_reps):
            rng = np.random.default_rng(rep)
            perm = rng.permutation(n_total)
            n_train = int(0.6 * n_total)
            n_cal = int(0.2 * n_total)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_full[perm[:n_train]])
            X_cal = scaler.transform(X_full[perm[n_train:n_train + n_cal]])
            X_te = scaler.transform(X_full[perm[n_train + n_cal:]])
            y_tr = y_full[perm[:n_train]]
            y_cal = y_full[perm[n_train:n_train + n_cal]]
            y_te = y_full[perm[n_train + n_cal:]]

            h_cal = _compute_leverage(X_tr, X_cal)
            h_te = _compute_leverage(X_tr, X_te)

            for wname, wfn in candidates:
                r = _eval_weight(X_tr, y_tr, X_cal, y_cal, X_te, y_te, h_te, wfn, alpha)
                per_weight_gaps[wname].append(r["gap"])

            model_lr = LinearRegression().fit(X_tr, y_tr)
            cal_residuals = np.abs(y_cal - model_lr.predict(X_cal))
            selector = WeightSelector(
                candidates=candidate_weights,
                val_fraction=0.3,
                random_state=rep,
            )
            selected = selector.select(cal_residuals, h_cal, alpha=alpha)
            sel_name = candidate_names[candidate_weights.index(selected)]
            selection_counts[sel_name] += 1

            r_auto = _eval_weight(X_tr, y_tr, X_cal, y_cal, X_te, y_te, h_te, selected, alpha)
            auto_gaps.append(r_auto["gap"])

        results["real"][ds_name] = {
            "selection_counts": selection_counts,
            "selection_fractions": {n: c / n_real_reps for n, c in selection_counts.items()},
            "per_weight_gap": {n: float(np.mean(per_weight_gaps[n])) for n in candidate_names},
            "auto_gap": float(np.mean(auto_gaps)),
            "default_gap": float(np.mean(per_weight_gaps["InvRoot"])),
        }
        print(f"    Selection: {selection_counts}")
        print(f"    Gap — default: {np.mean(per_weight_gaps['InvRoot']):.4f}, "
              f"auto: {np.mean(auto_gaps):.4f}")

    # Save
    with open(RESULTS_DIR / "exp_weight_selection.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp_weight_selection.json'}")

    return results


if __name__ == "__main__":
    run_weight_selection_experiment()
