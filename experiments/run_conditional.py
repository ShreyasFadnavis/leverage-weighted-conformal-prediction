"""Experiment 2: Conditional Coverage by Leverage Bin (Theorem 2 / Proposition 1).

The most important experiment. Shows that vanilla CP has leverage-dependent
conditional coverage (overcoverage at low h, undercoverage at high h),
while LWCP achieves approximately flat conditional coverage.

The homoscedastic DGP is the clearest showcase: under g(h)=1, the total
OLS prediction variance is exactly sigma^2*(1+h), so InverseRootLeverageWeight
achieves EXACT variance stabilization at any sample size.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.baselines import LWCPMethod, VanillaCP
from experiments.dgps import (
    dgp_heavy_tailed,
    dgp_homoscedastic,
    dgp_polynomial,
    dgp_textbook,
)
from experiments.plotting import get_color, savefig, setup_style


def run_conditional_experiment(
    n_reps: int = 200,
    alpha: float = 0.1,
    n_bins: int = 10,
):
    setup_style()
    print("=" * 60)
    print("Experiment 2: Conditional Coverage by Leverage Bin (Theorem 2)")
    print("=" * 60)

    dgps = [
        ("Homoscedastic", dgp_homoscedastic, {"n_train": 100, "n_cal": 200}),
        ("Textbook", dgp_textbook, {}),
        ("Heavy-tailed", dgp_heavy_tailed, {}),
        ("Polynomial", dgp_polynomial, {}),
    ]

    dgp_subtitles = {
        "Homoscedastic": r"$g(h)=1,\; p/n{=}0.3$",
        "Textbook": r"$g(h)=1{+}h$",
        "Heavy-tailed": r"$g(h)=1{+}h,\; t_3$",
        "Polynomial": r"$g(h)=1{+}h,\; \deg 8$",
    }

    # --- 1x4 single row: all subplots side by side ---
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 3.2),
        sharey=True, sharex=True,
    )
    axes_flat = axes.flatten()

    for col, (dgp_label, dgp_func, extra_kwargs) in enumerate(dgps):
        print(f"\n  DGP: {dgp_label}")

        all_h = {"Vanilla CP": [], "LWCP": []}
        all_covered = {"Vanilla CP": [], "LWCP": []}

        for rep in range(n_reps):
            kwargs = dict(sigma=1.0, random_state=rep, **extra_kwargs)
            X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(**kwargs)
            h_test = meta["h_test"]

            for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
                method = method_cls(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                covered = (y_te >= lower) & (y_te <= upper)
                all_h[method_name].append(h_test)
                all_covered[method_name].append(covered)

        ax = axes_flat[col]

        for method_name in ["Vanilla CP", "LWCP"]:
            h_pooled = np.concatenate(all_h[method_name])
            cov_pooled = np.concatenate(all_covered[method_name]).astype(float)

            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            bin_coverages = []
            bin_ses = []
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled <= bin_edges[i + 1])
                else:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled < bin_edges[i + 1])
                if mask.sum() > 0:
                    cov = np.mean(cov_pooled[mask])
                    se = np.sqrt(cov * (1 - cov) / mask.sum()) * 1.96
                    bin_coverages.append(cov)
                    bin_ses.append(se)
                else:
                    bin_coverages.append(np.nan)
                    bin_ses.append(0)

            xs = np.arange(1, n_bins + 1)
            covs = np.array(bin_coverages)
            ses = np.array(bin_ses)
            color = get_color(method_name)

            # Filled confidence band
            ax.fill_between(
                xs, covs - ses, covs + ses,
                alpha=0.15, color=color, linewidth=0,
            )
            # Bold line with prominent markers
            ax.plot(
                xs, covs, "o-",
                color=color, label=method_name,
                linewidth=2.5, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0,
                zorder=5,
            )

        # Nominal line and band
        ax.axhline(1 - alpha, color="#333333", linestyle="--",
                    linewidth=1.2, alpha=0.6, zorder=1)
        nom = 1 - alpha
        n_eff = n_reps * 50
        se_nom = np.sqrt(nom * alpha / n_eff) * 1.96
        ax.axhspan(nom - se_nom, nom + se_nom, color="#333333",
                    alpha=0.06, zorder=0)

        # Title with subtitle
        ax.set_title(
            f"{dgp_label}\n",
            fontsize=12, fontweight="bold", pad=2,
        )
        ax.text(
            0.5, 1.0, dgp_subtitles.get(dgp_label, ""),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="0.35", style="italic",
        )
        ax.set_ylim(0.82, 0.98)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.tick_params(labelsize=10)

        # Axis labels: x on all panels, y only on leftmost
        ax.set_xlabel("Leverage decile", fontsize=10)
        if col == 0:
            ax.set_ylabel("Conditional coverage", fontsize=10)

        # Print summary
        for method_name in ["Vanilla CP", "LWCP"]:
            h_pooled = np.concatenate(all_h[method_name])
            cov_pooled = np.concatenate(all_covered[method_name])
            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            low_mask = h_pooled <= bin_edges[2]
            high_mask = h_pooled >= bin_edges[-3]
            cov_low = np.mean(cov_pooled[low_mask])
            cov_high = np.mean(cov_pooled[high_mask])
            print(f"    {method_name}: cov(low-h) = {cov_low:.3f}, cov(high-h) = {cov_high:.3f}")

    # Shared legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=2, fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=2.5, columnspacing=3.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(wspace=0.08)
    savefig(fig, "exp2_conditional_coverage")


if __name__ == "__main__":
    run_conditional_experiment()
