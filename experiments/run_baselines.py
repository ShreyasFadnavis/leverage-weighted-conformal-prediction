"""Experiment 7: Comparison with CQR and Studentized CP.

Compares LWCP against CQR and Studentized CP on coverage, width,
conditional coverage, and wall-clock time.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.baselines import CQR, LWCPMethod, LocalizedCP, StudentizedCP, VanillaCP, run_method_timed
from experiments.dgps import dgp_adversarial, dgp_polynomial, dgp_textbook
from experiments.metrics import compute_msce, compute_wsc
from experiments.plotting import get_color, savefig, setup_style

RESULTS_DIR = Path(__file__).parent / "results"


def run_baselines_comparison(n_reps: int = 100, alpha: float = 0.1, n_bins: int = 10):
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Experiment 7: Comparison with CQR and Studentized CP")
    print("=" * 60)

    dgps = [
        ("Textbook", dgp_textbook),
        ("Polynomial", dgp_polynomial),
        ("Adversarial", dgp_adversarial),
    ]

    method_factories = [
        ("Vanilla CP", lambda a: VanillaCP(predictor=LinearRegression(), alpha=a)),
        ("LWCP", lambda a: LWCPMethod(predictor=LinearRegression(), alpha=a)),
        ("CQR", lambda a: CQR(predictor=LinearRegression(), alpha=a)),
        ("Studentized CP", lambda a: StudentizedCP(predictor=LinearRegression(), alpha=a)),
        ("Localized CP", lambda a: LocalizedCP(predictor=LinearRegression(), alpha=a)),
    ]
    method_names = [name for name, _ in method_factories]

    results = {}

    for dgp_label, dgp_func in dgps:
        print(f"\n  DGP: {dgp_label}")
        results[dgp_label] = {}

        for method_name, method_factory in method_factories:
            coverages, widths_list, times = [], [], []
            msce_list, wsc_list = [], []
            all_h, all_covered = [], []

            for rep in range(n_reps):
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(
                    sigma=1.0, random_state=rep,
                )
                method = method_factory(alpha)
                y_pred, lower, upper, fit_t, pred_t = run_method_timed(
                    method, X_tr, y_tr, X_cal, y_cal, X_te
                )
                covered = (y_te >= lower) & (y_te <= upper)
                coverages.append(np.mean(covered))
                widths_list.append(np.mean(upper - lower))
                times.append(fit_t + pred_t)
                msce_list.append(compute_msce(meta["h_test"], covered, alpha=alpha))
                wsc_list.append(compute_wsc(meta["h_test"], covered))
                all_h.append(meta["h_test"])
                all_covered.append(covered)

            results[dgp_label][method_name] = {
                "coverage": float(np.mean(coverages)),
                "coverage_std": float(np.std(coverages)),
                "width": float(np.mean(widths_list)),
                "width_std": float(np.std(widths_list)),
                "msce": float(np.mean(msce_list)),
                "msce_std": float(np.std(msce_list)),
                "wsc": float(np.mean(wsc_list)),
                "wsc_std": float(np.std(wsc_list)),
                "time": float(np.mean(times)),
                "all_h": all_h,
                "all_covered": all_covered,
            }
            print(f"    {method_name:<16s}: cov={np.mean(coverages):.4f}, "
                  f"width={np.mean(widths_list):.4f}, "
                  f"msce={np.mean(msce_list):.5f}, "
                  f"wsc={np.mean(wsc_list):.3f}, "
                  f"time={np.mean(times):.4f}s")

    # Table
    print("\n" + "=" * 100)
    print(f"{'DGP':<14s} {'Method':<18s} {'Coverage':>10s} {'Width':>10s} {'Time (s)':>10s}")
    print("-" * 62)
    for dgp_label in [d[0] for d in dgps]:
        for m in method_names:
            r = results[dgp_label][m]
            print(f"{dgp_label:<14s} {m:<18s} {r['coverage']:>10.4f} {r['width']:>10.4f} {r['time']:>10.4f}")

    # Figure: conditional coverage
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for col, (dgp_label, _) in enumerate(dgps):
        ax = axes[col]
        for method_name in method_names:
            r = results[dgp_label][method_name]
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
            ax.plot(range(1, n_bins + 1), bin_coverages, "o-",
                    color=get_color(method_name), label=method_name, markersize=5)
        ax.axhline(1 - alpha, color="red", linestyle="--", alpha=0.7)
        ax.set_title(dgp_label)
        ax.set_xlabel("Leverage decile")
        ax.set_ylabel("Conditional coverage")
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Experiment 7: Conditional Coverage — All Methods", fontsize=14)
    fig.tight_layout()
    savefig(fig, "exp7_baselines_conditional")

    # Save (exclude non-serializable arrays)
    save_results = {}
    for dgp_label in results:
        save_results[dgp_label] = {}
        for m in results[dgp_label]:
            r = results[dgp_label][m]
            save_results[dgp_label][m] = {
                k: v for k, v in r.items() if k not in ("all_h", "all_covered")
            }
    with open(RESULTS_DIR / "exp7_baselines.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR / 'exp7_baselines.json'}")


if __name__ == "__main__":
    run_baselines_comparison()
