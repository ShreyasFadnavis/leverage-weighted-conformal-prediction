"""Experiment: Real-Data Evaluation of LWCP.

Evaluates LWCP on real-world regression datasets from sklearn,
showing conditional coverage by leverage decile (like Figure 1).
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from experiments.baselines import (
    CQR_GBR,
    LWCPMethod,
    LWCPPlus,
    LocalizedCP,
    StudentizedCP,
    VanillaCP,
)
from experiments.metrics import compute_msce, compute_wsc
from experiments.plotting import nominal_line, savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"

COLORS = {
    "Vanilla CP": "#7f7f7f",
    "LWCP": "#d62728",
    "LWCP+": "#1f77b4",
    "CQR-GBR": "#ff7f0e",
    "Studentized CP": "#2ca02c",
    "Localized CP": "#9467bd",
}
MARKERS = {
    "Vanilla CP": "s",
    "LWCP": "o",
    "LWCP+": "P",
    "CQR-GBR": "^",
    "Studentized CP": "D",
    "Localized CP": "v",
}
LINESTYLES = {
    "Vanilla CP": "--",
    "LWCP": "-",
    "LWCP+": "-",
    "CQR-GBR": "-.",
    "Studentized CP": ":",
    "Localized CP": "-.",
}

DS_LABELS = {
    "Diabetes": "Diabetes\n($n$=442, $p$=10)",
    "CPU Activity": "CPU Activity\n($n$=8192, $p$=21)",
    "Superconductor": "Superconductor\n($n$=21263, $p$=81)",
    "Heavy-tailed": "Heavy-tailed\n($n$=500, $p$=100, $t_3$)",
}


def _load_datasets():
    """Load real-world regression datasets from sklearn."""
    datasets = {}

    # 1. Diabetes (n=442, p=10)
    diab = load_diabetes()
    datasets["Diabetes"] = (diab.data, diab.target)

    # 2. CPU Activity (n=8192, p=21)
    data = fetch_openml(name="cpu_act", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(float), data.target.astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    datasets["CPU Activity"] = (X[mask], y[mask])

    # 3. Superconductor (n=21263, p=81)
    data = fetch_openml(name="superconduct", version=1, as_frame=False, parser="auto")
    X, y = data.data.astype(float), data.target.astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    datasets["Superconductor"] = (X[mask], y[mask])

    # 4. Heavy-tailed linear benchmark (n=500, p=100, t_3 features)
    # Linear DGP with t_3 features: high leverage variation, OLS is correct model.
    rng = np.random.default_rng(42)
    n_ht, p_ht = 500, 100
    X_ht = t_dist.rvs(3, size=(n_ht, p_ht), random_state=42)
    beta = np.zeros(p_ht)
    beta[:10] = rng.standard_normal(10) * 5
    y_ht = X_ht @ beta + rng.standard_normal(n_ht) * 10.0
    datasets["Heavy-tailed"] = (X_ht, y_ht)

    return datasets


def _compute_leverage(X_ref, X_query):
    """Compute exact leverage scores of X_query w.r.t. X_ref."""
    from scipy import linalg

    _, s, Vt = linalg.svd(X_ref, full_matrices=False)
    inv_d = np.where(s > 1e-15, 1.0 / s**2, 0.0)
    gram_inv = (Vt.T * inv_d) @ Vt
    XG = X_query @ gram_inv
    return np.sum(XG * X_query, axis=1)


def run_real_data_experiment(n_reps: int = 30, alpha: float = 0.1):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Real-Data Experiment: LWCP on sklearn Datasets")
    print("=" * 60)

    datasets = _load_datasets()
    ds_keys = list(datasets.keys())
    print(f"  Loaded {len(datasets)} datasets: {ds_keys}")

    methods_spec = [
        ("Vanilla CP", VanillaCP),
        ("LWCP", LWCPMethod),
        ("LWCP+", LWCPPlus),
        ("CQR-GBR", CQR_GBR),
        ("Studentized CP", StudentizedCP),
        ("Localized CP", LocalizedCP),
    ]
    method_names = [m for m, _ in methods_spec]
    n_bins = 10

    all_results = {}
    decile_data = {}  # pooled test data for conditional coverage plots

    for ds_name in ds_keys:
        X_full, y_full = datasets[ds_name]
        n_total = X_full.shape[0]
        p = X_full.shape[1]
        print(f"\n  Dataset: {ds_name} (n={n_total}, p={p})")

        # Diagnostic: leverage variation coefficient η̂
        scaler_diag = StandardScaler()
        X_std = scaler_diag.fit_transform(X_full)
        h_diag = _compute_leverage(X_std, X_std)
        eta_hat = np.std(h_diag) / np.mean(h_diag) if np.mean(h_diag) > 0 else 0.0
        print(f"    Leverage diagnostic η̂ = {eta_hat:.3f} "
              f"(mean h = {np.mean(h_diag):.4f}, std h = {np.std(h_diag):.4f})")

        ds_results = {m: {"coverage": [], "width": [], "cond_gap": [],
                          "msce": [], "wsc": [], "time": []}
                      for m in method_names}
        pooled = {m: {"h": [], "covered": [], "widths": []} for m in method_names}

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            perm = rng.permutation(n_total)

            n_train = int(0.6 * n_total)
            n_cal = int(0.2 * n_total)

            idx_tr = perm[:n_train]
            idx_cal = perm[n_train:n_train + n_cal]
            idx_te = perm[n_train + n_cal:]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_full[idx_tr])
            X_cal = scaler.transform(X_full[idx_cal])
            X_te = scaler.transform(X_full[idx_te])
            y_tr, y_cal, y_te = y_full[idx_tr], y_full[idx_cal], y_full[idx_te]

            h_te = _compute_leverage(X_tr, X_te)
            h_lo = np.percentile(h_te, 20)
            h_hi = np.percentile(h_te, 80)
            mask_lo = h_te <= h_lo
            mask_hi = h_te >= h_hi

            for method_name, method_cls in methods_spec:
                try:
                    method = method_cls(predictor=LinearRegression(), alpha=alpha)
                    t0 = time.perf_counter()
                    method.fit(X_tr, y_tr, X_cal, y_cal)
                    _, lower, upper = method.predict(X_te)
                    elapsed = time.perf_counter() - t0

                    covered = (y_te >= lower) & (y_te <= upper)
                    widths = upper - lower

                    cov = np.mean(covered)
                    width = np.mean(widths)
                    gap = (np.mean(covered[mask_lo]) - np.mean(covered[mask_hi])
                           if mask_lo.sum() > 0 and mask_hi.sum() > 0 else 0.0)

                    msce = compute_msce(h_te, covered, alpha=alpha)
                    wsc = compute_wsc(h_te, covered)

                    ds_results[method_name]["coverage"].append(cov)
                    ds_results[method_name]["width"].append(width)
                    ds_results[method_name]["cond_gap"].append(gap)
                    ds_results[method_name]["msce"].append(msce)
                    ds_results[method_name]["wsc"].append(wsc)
                    ds_results[method_name]["time"].append(elapsed)

                    pooled[method_name]["h"].append(h_te)
                    pooled[method_name]["covered"].append(covered.astype(float))
                    pooled[method_name]["widths"].append(widths)
                except Exception as e:
                    if rep == 0:
                        print(f"    [WARN] {method_name} failed: {e}")

            if (rep + 1) % 10 == 0:
                print(f"    rep {rep + 1}/{n_reps} done")

        # Summarize
        print(f"    {'Method':<16s} {'Cov':>6s} {'Width':>8s} {'Gap':>8s} "
              f"{'MSCE':>8s} {'WSC':>6s} {'Time':>8s}")
        print(f"    {'-'*62}")
        ds_summary = {}
        for m in method_names:
            r = ds_results[m]
            if len(r["coverage"]) == 0:
                continue
            cov_m = np.mean(r["coverage"])
            wid_m = np.mean(r["width"])
            gap_m = np.mean(np.abs(r["cond_gap"]))
            msce_m = np.mean(r["msce"])
            wsc_m = np.mean(r["wsc"])
            time_m = np.mean(r["time"])
            print(f"    {m:<16s} {cov_m:6.3f} {wid_m:8.3f} {gap_m:8.3f} "
                  f"{msce_m:8.5f} {wsc_m:6.3f} {time_m:8.4f}s")
            ds_summary[m] = {
                "coverage": float(cov_m),
                "coverage_std": float(np.std(r["coverage"])),
                "width": float(wid_m),
                "width_std": float(np.std(r["width"])),
                "cond_gap": float(gap_m),
                "cond_gap_std": float(np.std(np.abs(r["cond_gap"]))),
                "msce": float(msce_m),
                "msce_std": float(np.std(r["msce"])),
                "wsc": float(wsc_m),
                "wsc_std": float(np.std(r["wsc"])),
                "time": float(time_m),
            }
        ds_summary["_eta_hat"] = float(eta_hat)
        all_results[ds_name] = ds_summary
        decile_data[ds_name] = pooled

    # Save JSON
    with open(RESULTS_DIR / "exp_real_data.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp_real_data.json'}")

    # =====================================================================
    # Figure: Conditional coverage by leverage decile (1×3)
    # =====================================================================
    n_ds = len(ds_keys)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 3.5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for col, ds_name in enumerate(ds_keys):
        ax = axes[col]
        pooled = decile_data[ds_name]

        for m in method_names:
            if len(pooled[m]["h"]) == 0:
                continue
            h_all = np.concatenate(pooled[m]["h"])
            cov_all = np.concatenate(pooled[m]["covered"])

            bin_edges = np.percentile(h_all, np.linspace(0, 100, n_bins + 1))
            bin_covs, bin_ses = [], []
            for b in range(n_bins):
                if b == n_bins - 1:
                    mask = (h_all >= bin_edges[b]) & (h_all <= bin_edges[b + 1])
                else:
                    mask = (h_all >= bin_edges[b]) & (h_all < bin_edges[b + 1])
                if mask.sum() > 0:
                    c = np.mean(cov_all[mask])
                    se = np.sqrt(c * (1 - c) / mask.sum()) * 1.96
                    bin_covs.append(c)
                    bin_ses.append(se)
                else:
                    bin_covs.append(np.nan)
                    bin_ses.append(0)

            x_pos = np.arange(1, n_bins + 1)
            ax.plot(
                x_pos, bin_covs,
                color=COLORS[m], marker=MARKERS[m], linestyle=LINESTYLES[m],
                label=m, markeredgecolor="white", markeredgewidth=0.8,
                markersize=6, linewidth=1.8,
            )
            ax.fill_between(
                x_pos,
                np.array(bin_covs) - np.array(bin_ses),
                np.array(bin_covs) + np.array(bin_ses),
                alpha=0.12, color=COLORS[m],
            )

        nominal_line(ax, alpha, label=(col == 0))
        ax.set_title(DS_LABELS.get(ds_name, ds_name), fontsize=11)
        ax.set_xlabel("Leverage decile", fontsize=10)
        if col == 0:
            ax.set_ylabel("Conditional coverage", fontsize=10)
        ax.set_xlim(0.5, n_bins + 0.5)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.set_ylim(0.80, 1.0)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=len(method_names) + 1, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 1.05),
        handlelength=2.0, columnspacing=1.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.subplots_adjust(wspace=0.08)
    savefig(fig, "exp_real_data")

    return all_results


if __name__ == "__main__":
    run_real_data_experiment()
