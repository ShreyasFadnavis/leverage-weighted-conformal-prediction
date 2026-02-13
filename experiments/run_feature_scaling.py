"""Experiment: Feature Scaling Sensitivity for LWCP.

Tests LWCP performance under different feature preprocessing regimes:
no scaling, StandardScaler, MinMaxScaler, RobustScaler.
"""

import json
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from experiments.baselines import LWCPMethod, VanillaCP
from experiments.dgps import dgp_textbook
from experiments.plotting import setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def _compute_leverage(X_ref, X_query):
    _, s, Vt = linalg.svd(X_ref, full_matrices=False)
    inv_d = np.where(s > 1e-15, 1.0 / s**2, 0.0)
    gram_inv = (Vt.T * inv_d) @ Vt
    XG = X_query @ gram_inv
    return np.sum(XG * X_query, axis=1)


def run_feature_scaling_experiment(n_reps: int = 100, alpha: float = 0.1):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment: Feature Scaling Sensitivity")
    print("=" * 60)

    scalers = {
        "No scaling": None,
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
    }

    # --- Part 1: Synthetic DGP ---
    print("\n--- Part 1: Textbook DGP ---")
    synthetic_results = {}

    for scaler_name, scaler_cls in scalers.items():
        print(f"  Scaler: {scaler_name}")
        coverages_v, coverages_l = [], []
        gaps_v, gaps_l = [], []
        eta_hats = []

        for rep in range(n_reps):
            X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_textbook(
                sigma=1.0, random_state=rep,
            )

            if scaler_cls is not None:
                scaler = scaler_cls()
                X_tr = scaler.fit_transform(X_tr)
                X_cal = scaler.transform(X_cal)
                X_te = scaler.transform(X_te)

            h_te = _compute_leverage(X_tr, X_te)
            eta_hat = np.std(h_te) / np.mean(h_te) if np.mean(h_te) > 0 else 0.0
            eta_hats.append(eta_hat)

            for method_cls, cov_list, gap_list in [
                (VanillaCP, coverages_v, gaps_v),
                (LWCPMethod, coverages_l, gaps_l),
            ]:
                method = method_cls(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                covered = (y_te >= lower) & (y_te <= upper)
                cov_list.append(np.mean(covered))

                h_lo = np.percentile(h_te, 20)
                h_hi = np.percentile(h_te, 80)
                mask_lo = h_te <= h_lo
                mask_hi = h_te >= h_hi
                if mask_lo.sum() > 0 and mask_hi.sum() > 0:
                    gap = abs(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi]))
                else:
                    gap = 0.0
                gap_list.append(gap)

        synthetic_results[scaler_name] = {
            "vanilla_cov": float(np.mean(coverages_v)),
            "lwcp_cov": float(np.mean(coverages_l)),
            "vanilla_gap": float(np.mean(gaps_v)),
            "lwcp_gap": float(np.mean(gaps_l)),
            "eta_hat_mean": float(np.mean(eta_hats)),
            "eta_hat_std": float(np.std(eta_hats)),
        }
        r = synthetic_results[scaler_name]
        print(f"    Vanilla gap={r['vanilla_gap']:.4f}, LWCP gap={r['lwcp_gap']:.4f}, "
              f"η̂={r['eta_hat_mean']:.3f}±{r['eta_hat_std']:.3f}")

    # --- Part 2: Real datasets ---
    print("\n--- Part 2: Real Datasets ---")
    real_datasets = {}

    diab = load_diabetes()
    real_datasets["Diabetes"] = (diab.data, diab.target)

    data = fetch_openml(name="cpu_act", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(float), data.target.astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    real_datasets["CPU Activity"] = (X[mask], y[mask])

    real_results = {}

    for ds_name, (X_full, y_full) in real_datasets.items():
        print(f"\n  Dataset: {ds_name}")
        real_results[ds_name] = {}
        n_total = X_full.shape[0]

        for scaler_name, scaler_cls in scalers.items():
            gaps_v, gaps_l = [], []
            eta_hats = []

            for rep in range(min(n_reps, 30)):
                rng = np.random.default_rng(rep)
                perm = rng.permutation(n_total)
                n_train = int(0.6 * n_total)
                n_cal = int(0.2 * n_total)

                idx_tr = perm[:n_train]
                idx_cal = perm[n_train:n_train + n_cal]
                idx_te = perm[n_train + n_cal:]

                X_tr = X_full[idx_tr].copy()
                X_cal = X_full[idx_cal].copy()
                X_te = X_full[idx_te].copy()
                y_tr, y_cal, y_te = y_full[idx_tr], y_full[idx_cal], y_full[idx_te]

                if scaler_cls is not None:
                    scaler = scaler_cls()
                    X_tr = scaler.fit_transform(X_tr)
                    X_cal = scaler.transform(X_cal)
                    X_te = scaler.transform(X_te)

                h_te = _compute_leverage(X_tr, X_te)
                eta_hat = np.std(h_te) / np.mean(h_te) if np.mean(h_te) > 0 else 0.0
                eta_hats.append(eta_hat)

                h_lo = np.percentile(h_te, 20)
                h_hi = np.percentile(h_te, 80)
                mask_lo = h_te <= h_lo
                mask_hi = h_te >= h_hi

                for method_cls, gap_list in [
                    (VanillaCP, gaps_v),
                    (LWCPMethod, gaps_l),
                ]:
                    method = method_cls(predictor=LinearRegression(), alpha=alpha)
                    method.fit(X_tr, y_tr, X_cal, y_cal)
                    _, lower, upper = method.predict(X_te)
                    covered = (y_te >= lower) & (y_te <= upper)
                    if mask_lo.sum() > 0 and mask_hi.sum() > 0:
                        gap = abs(np.mean(covered[mask_lo]) - np.mean(covered[mask_hi]))
                    else:
                        gap = 0.0
                    gap_list.append(gap)

            real_results[ds_name][scaler_name] = {
                "vanilla_gap": float(np.mean(gaps_v)),
                "lwcp_gap": float(np.mean(gaps_l)),
                "eta_hat_mean": float(np.mean(eta_hats)),
            }
            r = real_results[ds_name][scaler_name]
            print(f"    {scaler_name:<18s}: V gap={r['vanilla_gap']:.4f}, "
                  f"L gap={r['lwcp_gap']:.4f}, η̂={r['eta_hat_mean']:.3f}")

    all_results = {"synthetic": synthetic_results, "real": real_results}
    with open(RESULTS_DIR / "exp_feature_scaling.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp_feature_scaling.json'}")

    return all_results


if __name__ == "__main__":
    run_feature_scaling_experiment()
