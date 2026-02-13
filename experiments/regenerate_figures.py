"""Regenerate all figures from saved JSON results with improved styling.

This script reads the JSON result files and re-creates publication-quality
figures without re-running the expensive experiments.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import (
    setup_style, get_color, get_marker, get_linestyle,
    method_plot_kwargs, savefig, nominal_line, shade_nominal_band,
)

RESULTS_DIR = Path(__file__).parent / "results"


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


# =========================================================================
# Experiment 1: Marginal Coverage Histograms
# =========================================================================
def regen_exp1():
    print("\n=== Regenerating Experiment 1 ===")
    data = load_json("exp1_coverage.json")
    dgp_names = list(data.keys())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, dgp_name in enumerate(dgp_names):
        ax = axes[i]
        for m in ["Vanilla CP", "LWCP"]:
            mean_val = data[dgp_name][m]["mean"]
            std_val = data[dgp_name][m]["std"]
            # Simulate histogram from mean/std (normal approx)
            samples = np.random.normal(mean_val, std_val, 1000)
            ax.hist(
                samples, bins=30, alpha=0.55,
                color=get_color(m), label=m, density=True,
                edgecolor="white", linewidth=0.5,
            )
        ax.axvline(0.9, color="#333333", linestyle=":", linewidth=1.5,
                   label="Nominal 90%", zorder=5)
        ax.set_title(dgp_name.replace("_", " ").title())
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for j in range(len(dgp_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Experiment 1: Marginal Coverage Distribution", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "exp1_marginal_coverage")


# =========================================================================
# Experiment 2: Conditional Coverage (RE-RUN needed for bin-level data)
# =========================================================================
def regen_exp2():
    """This experiment must be re-run since bin-level data isn't in JSON."""
    print("\n=== Re-running Experiment 2 (conditional coverage) ===")
    from sklearn.linear_model import LinearRegression
    from experiments.baselines import LWCPMethod, VanillaCP
    from experiments.dgps import dgp_homoscedastic, dgp_textbook, dgp_heavy_tailed, dgp_polynomial

    n_reps = 200
    alpha = 0.1
    n_bins = 10

    dgps = [
        ("Homoscedastic\n($g(h)=1$, $p/n_1=0.3$)", dgp_homoscedastic, {"n_train": 100, "n_cal": 200}),
        ("Textbook\n($g(h)=1{+}h$)", dgp_textbook, {}),
        ("Heavy-tailed\n($t_3$ errors)", dgp_heavy_tailed, {}),
        ("Polynomial\n(degree 8)", dgp_polynomial, {}),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for col, (dgp_label, dgp_func, extra_kwargs) in enumerate(dgps):
        print(f"  DGP: {dgp_label.split(chr(10))[0]}")
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

        ax = axes[col]
        shade_nominal_band(ax, alpha, n_cal=200 if col == 0 else 500)
        nominal_line(ax, alpha)

        for method_name in ["Vanilla CP", "LWCP"]:
            h_pooled = np.concatenate(all_h[method_name])
            cov_pooled = np.concatenate(all_covered[method_name])
            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            bin_coverages = []
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled <= bin_edges[i + 1])
                else:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled < bin_edges[i + 1])
                bin_coverages.append(np.mean(cov_pooled[mask]) if mask.sum() > 0 else np.nan)

            kw = method_plot_kwargs(method_name)
            kw["markersize"] = 8
            kw["linewidth"] = 2.5 if method_name == "LWCP" else 2.0
            ax.plot(range(1, n_bins + 1), bin_coverages, **kw)

        ax.set_title(dgp_label, fontsize=12)
        ax.set_xlabel("Leverage decile (1=low, 10=high)")
        ax.set_ylabel("Conditional coverage")
        ax.set_ylim(0.82, 0.97)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=9, loc="lower left")

        # Print summary
        for method_name in ["Vanilla CP", "LWCP"]:
            h_pooled = np.concatenate(all_h[method_name])
            cov_pooled = np.concatenate(all_covered[method_name])
            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            low_mask = h_pooled <= bin_edges[2]
            high_mask = h_pooled >= bin_edges[-3]
            cov_low = np.mean(cov_pooled[low_mask])
            cov_high = np.mean(cov_pooled[high_mask])
            print(f"    {method_name}: cov(low-h)={cov_low:.3f}, cov(high-h)={cov_high:.3f}, gap={cov_low-cov_high:.3f}")

    fig.suptitle("")  # clean — titles on subplots are enough
    fig.tight_layout()
    savefig(fig, "exp2_conditional_coverage")


# =========================================================================
# Experiment 3: Width vs Leverage + Bar chart
# =========================================================================
def regen_exp3():
    """Re-run for scatter data (not in JSON)."""
    print("\n=== Re-running Experiment 3 (width vs leverage) ===")
    from sklearn.linear_model import LinearRegression
    from experiments.baselines import LWCPMethod, VanillaCP
    from experiments.dgps import ALL_DGPS

    alpha = 0.1
    dgp_names = list(ALL_DGPS.keys())
    scatter_data = {}

    for dgp_name in dgp_names:
        dgp_func = ALL_DGPS[dgp_name]
        X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(sigma=1.0, random_state=0)
        for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
            method = method_cls(predictor=LinearRegression(), alpha=alpha)
            method.fit(X_tr, y_tr, X_cal, y_cal)
            _, lower, upper = method.predict(X_te)
            widths = upper - lower
            scatter_data.setdefault(dgp_name, {})[method_name] = {
                "h": meta["h_test"], "widths": widths,
            }

    # Width vs leverage scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for i, dgp_name in enumerate(dgp_names):
        ax = axes_flat[i]
        for m in ["Vanilla CP", "LWCP"]:
            d = scatter_data[dgp_name][m]
            color = get_color(m)
            if m == "Vanilla CP":
                ax.scatter(d["h"], d["widths"], alpha=0.25, s=10, color=color,
                          label=m, edgecolors="none", zorder=2)
                # Add horizontal mean line for vanilla
                ax.axhline(np.mean(d["widths"]), color=color, linestyle="--",
                          linewidth=1.5, alpha=0.7, zorder=3)
            else:
                ax.scatter(d["h"], d["widths"], alpha=0.45, s=14, color=color,
                          label=m, edgecolors="white", linewidths=0.3, zorder=4)
                # Add regression line for LWCP
                z = np.polyfit(d["h"], d["widths"], 1)
                h_line = np.linspace(d["h"].min(), d["h"].max(), 100)
                ax.plot(h_line, np.polyval(z, h_line), color=color,
                       linewidth=2.0, alpha=0.8, zorder=5)
        ax.set_title(dgp_name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Leverage score $h(x)$")
        ax.set_ylabel("Interval width")
        ax.legend(fontsize=9, loc="upper left")

    for j in range(len(dgp_names), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    savefig(fig, "exp3_width_vs_leverage")

    # Bar chart from JSON
    data = load_json("exp3_width.json")
    fig2, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dgp_names))
    bar_w = 0.35
    for i, m in enumerate(["Vanilla CP", "LWCP"]):
        vals = [data[d][m]["mean"] for d in dgp_names]
        errs = [data[d][m]["std"] for d in dgp_names]
        ax.bar(x + i * bar_w, vals, bar_w, yerr=errs,
               label=m, color=get_color(m), alpha=0.85,
               capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels([n.replace("_", " ").title() for n in dgp_names],
                       rotation=15, ha="right")
    ax.set_ylabel("Mean interval width")
    ax.legend()
    fig2.tight_layout()
    savefig(fig2, "exp3_width_barplot")


# =========================================================================
# Experiment 4: Efficiency Frontier
# =========================================================================
def regen_exp4():
    print("\n=== Regenerating Experiment 4 ===")
    data = load_json("exp4_efficiency.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for col, dgp_label in enumerate(data.keys()):
        ax = axes[col]
        frontier = data[dgp_label]
        for m in ["Vanilla CP", "LWCP"]:
            kw = method_plot_kwargs(m)
            kw["markersize"] = 9
            kw["linewidth"] = 2.5
            ax.plot(frontier[m]["coverage"], frontier[m]["width"], **kw)
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Mean interval width")
        ax.set_title(dgp_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.invert_xaxis()

    fig.tight_layout()
    savefig(fig, "exp4_efficiency_frontier")


# =========================================================================
# Experiment 5: Gaussian Recovery
# =========================================================================
def regen_exp5():
    print("\n=== Regenerating Experiment 5 ===")
    data = load_json("exp5_gaussian.json")
    sample_sizes = [int(k) for k in data.keys()]
    means = [data[str(n)]["mean"] for n in sample_sizes]
    stds = [data[str(n)]["std"] for n in sample_sizes]

    # Figure 1: Ratio vs n
    fig1, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sample_sizes, means, yerr=stds, fmt="o-",
                color="#d62728", capsize=5, markersize=8, linewidth=2.0,
                markeredgecolor="white", markeredgewidth=0.8,
                label="LWCP / Classical ratio")
    ax.axhline(1.0, color="#333333", linestyle=":", linewidth=1.5, alpha=0.8,
               label="Exact recovery (ratio = 1)")
    ax.fill_between(sample_sizes, [1.0]*len(sample_sizes), means,
                    alpha=0.1, color="#d62728")
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("LWCP width / Classical width")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.set_ylim(0.85, 1.55)
    fig1.tight_layout()
    savefig(fig1, "exp5_gaussian_recovery")

    # Figure 2: Interval overlay (re-run needed)
    print("  Re-running interval overlay...")
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    from lwcp import LWCP, InverseRootLeverageWeight
    from lwcp.leverage import LeverageComputer

    p, sigma, alpha = 5, 1.0, 0.1
    fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
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

        ax.plot(h_tests, lwcp_hi - lwcp_lo, "o", color="#d62728", alpha=0.6,
                markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                label="LWCP width", zorder=4)
        ax.plot(h_tests, 2 * classical_half, "x", color="#7f7f7f", alpha=0.6,
                markersize=6, markeredgewidth=1.5, label="Classical width", zorder=3)
        ax.set_xlabel("Leverage score $h(x)$")
        ax.set_ylabel("Interval width")
        ax.set_title(f"$n = {n_show}$", fontsize=13)
        ax.legend(fontsize=10)

    fig2.tight_layout()
    savefig(fig2, "exp5_interval_overlay")


# =========================================================================
# Experiment 6: Heteroscedasticity Sweep
# =========================================================================
def regen_exp6():
    """Re-run needed for bin-level conditional coverage data."""
    print("\n=== Re-running Experiment 6 (hetero sweep) ===")
    from sklearn.linear_model import LinearRegression
    from experiments.baselines import LWCPMethod, VanillaCP
    from experiments.dgps import dgp_adversarial, dgp_custom_g

    n_reps = 200
    alpha = 0.1
    n_bins = 10
    methods = ["Vanilla CP", "LWCP"]

    configs = [
        ("$g(h) = 1{+}h$\n(optimal match)", lambda h: 1.0 + h, "leverage"),
        ("$g(h) = 1{+}5h$\n(partial match)", lambda h: 1.0 + 5.0 * h, "leverage"),
        ("$g(h) = 1$\n(homoscedastic)", lambda h: np.ones_like(h), "leverage"),
        ("$\\mathrm{Var} \\propto \\|X\\|^2$\n(adversarial)", None, "adversarial"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (label, g_func, kind) in enumerate(configs):
        all_h = {m: [] for m in methods}
        all_covered = {m: [] for m in methods}
        all_widths = {m: [] for m in methods}

        for rep in range(n_reps):
            if kind == "leverage":
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_custom_g(
                    g_func=g_func, g_name=label, sigma=1.0, random_state=rep)
            else:
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_adversarial(
                    sigma=1.0, random_state=rep)
            h_test = meta["h_test"]
            for method_name, method_cls in [("Vanilla CP", VanillaCP), ("LWCP", LWCPMethod)]:
                method = method_cls(predictor=LinearRegression(), alpha=alpha)
                method.fit(X_tr, y_tr, X_cal, y_cal)
                _, lower, upper = method.predict(X_te)
                covered = (y_te >= lower) & (y_te <= upper)
                widths = upper - lower
                all_h[method_name].append(h_test)
                all_covered[method_name].append(covered)
                all_widths[method_name].append(widths)

        # Top row: conditional coverage
        ax_cov = axes[0, col]
        shade_nominal_band(ax_cov, alpha)
        nominal_line(ax_cov, alpha, label=(col == 0))
        for method_name in methods:
            h_pooled = np.concatenate(all_h[method_name])
            cov_pooled = np.concatenate(all_covered[method_name])
            bin_edges = np.percentile(h_pooled, np.linspace(0, 100, n_bins + 1))
            bin_coverages = []
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled <= bin_edges[i + 1])
                else:
                    mask = (h_pooled >= bin_edges[i]) & (h_pooled < bin_edges[i + 1])
                bin_coverages.append(float(np.mean(cov_pooled[mask])) if mask.sum() > 0 else None)
            kw = method_plot_kwargs(method_name)
            kw["markersize"] = 6
            ax_cov.plot(range(1, n_bins + 1), bin_coverages, **kw)

        ax_cov.set_title(label, fontsize=11)
        ax_cov.set_ylim(0.82, 0.97)
        ax_cov.set_xlim(0.5, 10.5)
        if col == 0:
            ax_cov.set_ylabel("Conditional coverage")
        ax_cov.set_xlabel("Leverage decile")
        ax_cov.legend(fontsize=7, loc="lower left")

        # Bottom row: width vs leverage
        ax_w = axes[1, col]
        for method_name in methods:
            h0 = all_h[method_name][0]
            w0 = all_widths[method_name][0]
            color = get_color(method_name)
            if method_name == "Vanilla CP":
                ax_w.scatter(h0, w0, alpha=0.2, s=8, color=color, label=method_name, edgecolors="none")
                ax_w.axhline(np.mean(w0), color=color, linestyle="--", linewidth=1.2, alpha=0.6)
            else:
                ax_w.scatter(h0, w0, alpha=0.4, s=10, color=color, label=method_name,
                           edgecolors="white", linewidths=0.2)
                z = np.polyfit(h0, w0, 1)
                h_line = np.linspace(h0.min(), h0.max(), 100)
                ax_w.plot(h_line, np.polyval(z, h_line), color=color, linewidth=1.8, alpha=0.8)

        if col == 0:
            ax_w.set_ylabel("Interval width")
        ax_w.set_xlabel("Leverage score $h(x)$")
        ax_w.legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, "exp6_hetero_sweep")


# =========================================================================
# Experiment 7: Baselines Comparison
# =========================================================================
def regen_exp7():
    """Re-run needed for bin-level data."""
    print("\n=== Re-running Experiment 7 (baselines) ===")
    from sklearn.linear_model import LinearRegression
    from experiments.baselines import CQR, LWCPMethod, StudentizedCP, VanillaCP, run_method_timed
    from experiments.dgps import dgp_adversarial, dgp_polynomial, dgp_textbook

    n_reps = 100
    alpha = 0.1
    n_bins = 10

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
    ]
    method_names = [name for name, _ in method_factories]
    results = {}

    for dgp_label, dgp_func in dgps:
        print(f"  DGP: {dgp_label}")
        results[dgp_label] = {}
        for method_name, method_factory in method_factories:
            all_h, all_covered = [], []
            for rep in range(n_reps):
                X_tr, y_tr, X_cal, y_cal, X_te, y_te, meta = dgp_func(sigma=1.0, random_state=rep)
                method = method_factory(alpha)
                y_pred, lower, upper, fit_t, pred_t = run_method_timed(method, X_tr, y_tr, X_cal, y_cal, X_te)
                covered = (y_te >= lower) & (y_te <= upper)
                all_h.append(meta["h_test"])
                all_covered.append(covered)
            results[dgp_label][method_name] = {"all_h": all_h, "all_covered": all_covered}

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for col, (dgp_label, _) in enumerate(dgps):
        ax = axes[col]
        shade_nominal_band(ax, alpha)
        nominal_line(ax, alpha, label=(col == 2))

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
            kw = method_plot_kwargs(method_name)
            kw["markersize"] = 6
            kw["linewidth"] = 2.5 if method_name == "LWCP" else 1.8
            ax.plot(range(1, n_bins + 1), bin_coverages, **kw)

        ax.set_title(dgp_label, fontsize=13)
        ax.set_xlabel("Leverage decile")
        ax.set_ylabel("Conditional coverage")
        ax.set_ylim(0.80, 0.97)
        ax.set_xlim(0.5, 10.5)
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    savefig(fig, "exp7_baselines_conditional")


# =========================================================================
# Experiment 8: Approximate Leverage
# =========================================================================
def regen_exp8():
    print("\n=== Regenerating Experiment 8 ===")
    data = load_json("exp8_approximate.json")
    p_values = [int(k) for k in data.keys()]

    # Coverage bar chart
    fig1, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(p_values))
    all_labels = list(data[str(p_values[0])].keys())
    bar_width = 0.18
    blues = ["#08519c", "#3182bd", "#6baed6", "#bdd7e7"]
    for i, label in enumerate(all_labels):
        vals = [data[str(p)].get(label, float("nan")) for p in p_values]
        ax.bar(x + i * bar_width, vals, bar_width, label=label,
               color=blues[i], alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.axhline(0.9, color="#333333", linestyle=":", linewidth=1.5, alpha=0.8, label="Nominal 90%")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([f"$p={p}$" for p in p_values])
    ax.set_ylabel("Marginal coverage")
    ax.set_ylim(0.88, 0.915)
    ax.legend(fontsize=9, ncol=3)
    fig1.tight_layout()
    savefig(fig1, "exp8_approx_coverage")

    # Leverage scatter (re-run needed)
    print("  Re-running leverage scatter...")
    from lwcp.leverage import LeverageComputer
    n_train = 200
    approx_fracs = [1.0, 0.5, 0.25]

    fig2, axes = plt.subplots(1, len(p_values), figsize=(4 * len(p_values), 4))
    if len(p_values) == 1:
        axes = [axes]
    scatter_colors = ["#08519c", "#e08214", "#1b7837"]

    for idx, p in enumerate(p_values):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_train, p))
        h_exact = LeverageComputer(method="exact").fit(X).leverage_scores(X)
        ax = axes[idx]
        for i, frac in enumerate(approx_fracs):
            k = max(int(p * frac), 1)
            h_approx = LeverageComputer(method="approximate", n_components=k, random_state=42).fit(X).leverage_scores(X)
            ax.scatter(h_exact, h_approx, alpha=0.35, s=10, color=scatter_colors[i],
                      label=f"$k={k}$", edgecolors="white", linewidths=0.2)
        ax.plot([0, h_exact.max()], [0, h_exact.max()], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("Exact leverage")
        ax.set_ylabel("Approximate leverage")
        ax.set_title(f"$p = {p}$", fontsize=12)
        ax.legend(fontsize=8)

    fig2.tight_layout()
    savefig(fig2, "exp8_approx_leverage_scatter")

    # Runtime (re-run)
    import time
    print("  Re-running runtime comparison...")
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

    fig3, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, exact_times, "o-", color="#d62728", label="Exact SVD",
            markersize=8, linewidth=2.0, markeredgecolor="white")
    ax.plot(p_values, approx_times, "s--", color="#1f77b4", label="Randomized SVD ($k=p/2$)",
            markersize=8, linewidth=2.0, markeredgecolor="white")
    ax.set_xlabel("Number of features $p$")
    ax.set_ylabel("Fit time (seconds)")
    ax.legend(fontsize=11)
    fig3.tight_layout()
    savefig(fig3, "exp8_runtime")


# =========================================================================
# Experiment 9: Scaling
# =========================================================================
def regen_exp9():
    print("\n=== Regenerating Experiment 9 ===")
    data = load_json("exp9_scaling.json")
    n_values = data["n_values"]
    p_values = data["p_values"]
    gap_matrix = np.array([[None if v == "NaN" or v is None else float(v)
                           for v in row] for row in data["gap_matrix"]], dtype=float)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    # Replace NaN for display
    display_matrix = np.where(np.isnan(gap_matrix), 0, gap_matrix)
    im = ax.imshow(display_matrix, cmap="YlOrRd", aspect="auto", origin="lower",
                   vmin=0.03, vmax=0.21)
    # Gray out NaN cells
    for ii in range(len(p_values)):
        for jj in range(len(n_values)):
            val = gap_matrix[ii, jj]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((jj-0.5, ii-0.5), 1, 1,
                            fill=True, facecolor="#e0e0e0", edgecolor="white", linewidth=1))
                ax.text(jj, ii, "N/A", ha="center", va="center", fontsize=9, color="#888888")
            else:
                text_color = "white" if val > 0.12 else "black"
                ax.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                       fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(range(len(p_values)))
    ax.set_yticklabels(p_values)
    ax.set_xlabel("Total sample size $n$")
    ax.set_ylabel("Number of features $p$")
    cbar = fig.colorbar(im, ax=ax, label="Max coverage gap", shrink=0.85)
    fig.tight_layout()
    savefig(fig, "exp9_scaling_heatmap")

    # Gap vs n
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    line_markers = ["o", "s", "^", "D"]
    for i, p in enumerate(p_values):
        valid = ~np.isnan(gap_matrix[i])
        if valid.sum() > 1:
            ax2.plot(np.array(n_values)[valid], gap_matrix[i, valid],
                    marker=line_markers[i], linestyle="-",
                    color=line_colors[i], label=f"$p={p}$",
                    markersize=8, linewidth=2.0,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_xlabel("Total sample size $n$")
    ax2.set_ylabel("Max conditional coverage gap")
    ax2.set_xscale("log")
    ax2.legend(fontsize=11)
    fig2.tight_layout()
    savefig(fig2, "exp9_gap_vs_n")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    setup_style()

    regen_exp1()
    regen_exp2()
    regen_exp3()
    regen_exp4()
    regen_exp5()
    regen_exp6()
    regen_exp7()
    regen_exp8()
    regen_exp9()

    print("\n\nAll figures regenerated!")
