"""Regenerate Figures 4 and 5 for the NeurIPS paper with improved design.

Figure 4: Gaussian Recovery (Theorem 3)
  - Left: convergence of width ratio with 1/sqrt(n) reference line
  - Right: overlay of LWCP vs classical intervals at n=2000

Figure 5: Scaling Behavior (Proposition 1)
  - Left: heatmap with cleaner styling
  - Right: gap vs n with 1/sqrt(n) reference curve
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from lwcp import LWCP, InverseRootLeverageWeight
from lwcp.leverage import LeverageComputer

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def setup_neurips_style():
    """Minimal, clean NeurIPS publication style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
    })


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Gaussian Recovery
# ─────────────────────────────────────────────────────────────────────────────
def make_figure4():
    setup_neurips_style()

    with open(RESULTS_DIR / "exp5_gaussian.json") as f:
        data = json.load(f)

    sample_sizes = [int(k) for k in data.keys()]
    means = [data[str(n)]["mean"] for n in sample_sizes]
    stds = [data[str(n)]["std"] for n in sample_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # --- Left panel: Width ratio vs n with confidence band ---
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ns = np.array(sample_sizes)

    # Plot ±1 std band
    ax1.fill_between(ns, means_arr - stds_arr, means_arr + stds_arr,
                      color="#2166ac", alpha=0.12, label=r"$\pm 1$ std")
    # Plot mean line with markers
    ax1.plot(ns, means_arr, "o-", color="#2166ac", markersize=6,
             markeredgecolor="white", markeredgewidth=0.8, linewidth=2,
             label="LWCP / Classical ratio", zorder=5)
    # Reference: exact recovery
    ax1.axhline(1.0, color="#b2182b", linestyle="-", linewidth=1.0, alpha=0.8,
                label="Exact recovery", zorder=3)
    # Reference: 1/sqrt(n) decay toward 1
    n_ref = np.linspace(50, 5500, 200)
    # Fit: ratio ≈ 1 + c/sqrt(n)
    c_fit = (means_arr[0] - 1) * np.sqrt(ns[0])
    ax1.plot(n_ref, 1 + c_fit / np.sqrt(n_ref), ":", color="#999999",
             linewidth=1.2, label=r"$1 + c/\sqrt{n}$ reference", zorder=2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Sample size $n$")
    ax1.set_ylabel("Width ratio (LWCP / Classical)")
    ax1.set_ylim(0.85, 1.55)
    ax1.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
    ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_title("(a) Convergence to classical interval")

    # Annotate key data points
    for n_annot, m, s in [(50, means_arr[0], stds_arr[0]),
                           (1000, means_arr[4], stds_arr[4])]:
        idx = list(ns).index(n_annot)
        ax1.annotate(f"{m:.3f}", (n_annot, m),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=8, color="#2166ac")

    # --- Right panel: Interval overlay at n=2000 ---
    rng = np.random.default_rng(0)
    p = 5
    n_show = 2000
    alpha = 0.1
    sigma = 1.0

    X = rng.standard_normal((n_show, p))
    beta = np.ones(p)
    y = X @ beta + sigma * rng.standard_normal(n_show)

    n_train = n_show // 2
    X_train, X_cal = X[:n_train], X[n_train:]
    y_train, y_cal = y[:n_train], y[n_train:]

    X_test = rng.standard_normal((80, p))
    lc = LeverageComputer().fit(X_train)
    h_tests = lc.leverage_scores(X_test)
    order = np.argsort(h_tests)
    X_test, h_tests = X_test[order], h_tests[order]

    model = LWCP(predictor=LinearRegression(),
                 weight_fn=InverseRootLeverageWeight(), alpha=alpha)
    model.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)
    _, lwcp_lo, lwcp_hi = model.predict(X_test)
    lwcp_widths = lwcp_hi - lwcp_lo

    ols = LinearRegression().fit(X_train, y_train)
    resid = y_train - ols.predict(X_train)
    sigma_hat = np.sqrt(np.sum(resid**2) / (n_train - p))
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_train - p)
    classical_widths = 2 * t_crit * sigma_hat * np.sqrt(1 + h_tests)

    # Plot as connected lines showing width vs leverage
    ax2.plot(h_tests, classical_widths, "-", color="#b2182b", linewidth=2.0,
             alpha=0.8, label="Classical ($t$-interval)", zorder=3)
    ax2.plot(h_tests, lwcp_widths, "--", color="#2166ac", linewidth=2.0,
             alpha=0.8, label="LWCP", zorder=4)

    # Shade the gap
    ax2.fill_between(h_tests, lwcp_widths, classical_widths,
                      color="#999999", alpha=0.08)

    ax2.set_xlabel("Leverage score $h(x)$")
    ax2.set_ylabel("Interval width")
    ax2.legend(loc="upper left", frameon=True)
    ax2.set_title(f"(b) LWCP vs. classical ($n = {n_show}$)")

    fig.tight_layout(w_pad=3)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"exp5_gaussian_recovery.{fmt}")
    # Also save the overlay
    print(f"  Saved: {FIGURES_DIR / 'exp5_gaussian_recovery.pdf'}")
    plt.close(fig)

    # Save a separate overlay figure (for appendix compatibility)
    fig2, ax_ov = plt.subplots(1, 2, figsize=(10, 3.8))
    for idx, n_s in enumerate([200, 2000]):
        ax = ax_ov[idx]
        rng2 = np.random.default_rng(0)
        X2 = rng2.standard_normal((n_s, p))
        y2 = X2 @ beta + sigma * rng2.standard_normal(n_s)
        n_tr2 = n_s // 2
        X_tr2, X_cal2 = X2[:n_tr2], X2[n_tr2:]
        y_tr2, y_cal2 = y2[:n_tr2], y2[n_tr2:]
        X_te2 = rng2.standard_normal((80, p))
        lc2 = LeverageComputer().fit(X_tr2)
        h2 = lc2.leverage_scores(X_te2)
        ord2 = np.argsort(h2)
        X_te2, h2 = X_te2[ord2], h2[ord2]

        m2 = LWCP(predictor=LinearRegression(),
                   weight_fn=InverseRootLeverageWeight(), alpha=alpha)
        m2.fit_with_precomputed_split(X_tr2, y_tr2, X_cal2, y_cal2)
        _, lo2, hi2 = m2.predict(X_te2)

        ols2 = LinearRegression().fit(X_tr2, y_tr2)
        res2 = y_tr2 - ols2.predict(X_tr2)
        sh2 = np.sqrt(np.sum(res2**2) / (n_tr2 - p))
        tc2 = stats.t.ppf(1 - alpha / 2, df=n_tr2 - p)
        cw2 = 2 * tc2 * sh2 * np.sqrt(1 + h2)

        ax.plot(h2, cw2, "-", color="#b2182b", linewidth=1.8, alpha=0.8,
                label="Classical")
        ax.plot(h2, hi2 - lo2, "--", color="#2166ac", linewidth=1.8, alpha=0.8,
                label="LWCP")
        ax.fill_between(h2, hi2 - lo2, cw2, color="#999999", alpha=0.08)
        ax.set_xlabel("Leverage $h(x)$")
        ax.set_ylabel("Interval width")
        ax.set_title(f"$n = {n_s}$")
        ax.legend(fontsize=8, frameon=True)

    fig2.tight_layout(w_pad=3)
    for fmt in ["pdf", "png"]:
        fig2.savefig(FIGURES_DIR / f"exp5_interval_overlay.{fmt}")
    print(f"  Saved: {FIGURES_DIR / 'exp5_interval_overlay.pdf'}")
    plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Scaling Behavior
# ─────────────────────────────────────────────────────────────────────────────
def make_figure5():
    setup_neurips_style()

    with open(RESULTS_DIR / "exp9_scaling.json") as f:
        data = json.load(f)

    n_values = data["n_values"]
    p_values = data["p_values"]
    gap_matrix = np.array(data["gap_matrix"])  # shape: (len(p_values), len(n_values))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    # --- Left: Heatmap ---
    # Mask NaN for cleaner display
    masked = np.ma.masked_invalid(gap_matrix)
    cmap = mpl.cm.YlOrRd.copy()
    cmap.set_bad(color="#f0f0f0")

    im = ax1.imshow(masked, cmap=cmap, aspect="auto", origin="lower",
                     vmin=0.04, vmax=0.21)
    ax1.set_xticks(range(len(n_values)))
    ax1.set_xticklabels([f"{n:,}" for n in n_values], fontsize=8)
    ax1.set_yticks(range(len(p_values)))
    ax1.set_yticklabels(p_values)
    ax1.set_xlabel("Total sample size $n$")
    ax1.set_ylabel("Dimension $p$")

    # Annotate cells
    for ii in range(len(p_values)):
        for jj in range(len(n_values)):
            val = gap_matrix[ii][jj]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                textcol = "white" if val > 0.12 else "black"
                ax1.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                         fontsize=8, fontweight="bold", color=textcol)
            else:
                ax1.text(jj, ii, "---", ha="center", va="center",
                         fontsize=8, color="#999999")

    cb = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cb.set_label("Max coverage gap", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax1.set_title("(a) Coverage gap heatmap")

    # --- Right: Gap vs n with 1/sqrt(n) reference ---
    colors = ["#2166ac", "#e08214", "#1b7837", "#d73027"]
    markers = ["o", "s", "^", "D"]

    for i, p in enumerate(p_values):
        gaps = np.array([gap_matrix[i][j] for j in range(len(n_values))])
        valid = ~np.isnan(gaps)
        if valid.sum() > 1:
            ns_valid = np.array(n_values)[valid]
            gaps_valid = gaps[valid]
            ax2.plot(ns_valid, gaps_valid, marker=markers[i], linestyle="-",
                     color=colors[i], markersize=5, markeredgecolor="white",
                     markeredgewidth=0.6, linewidth=1.8,
                     label=f"$p = {p}$", zorder=5)

    # 1/sqrt(n) reference line
    n_ref = np.linspace(180, 5500, 200)
    c_ref = 0.045 * np.sqrt(5000)  # calibrate to match n=5000 value
    ax2.plot(n_ref, c_ref / np.sqrt(n_ref), ":", color="#999999",
             linewidth=1.5, label=r"$O(1/\sqrt{n})$ reference", zorder=2)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Total sample size $n$")
    ax2.set_ylabel("Max conditional coverage gap")
    ax2.set_xticks([200, 500, 1000, 2000, 5000])
    ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax2.set_ylim(0.03, 0.25)
    ax2.legend(loc="upper right", frameon=True, ncol=2, fontsize=8)
    ax2.set_title(r"(b) Gap $\to 0$ at rate $O(1/\sqrt{n})$")

    fig.tight_layout(w_pad=3)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig5_scaling_combined.{fmt}")
    print(f"  Saved: {FIGURES_DIR / 'fig5_scaling_combined.pdf'}")

    # Also save individual files for backward compatibility
    # Heatmap only
    fig_h, ax_h = plt.subplots(figsize=(5, 4))
    im_h = ax_h.imshow(masked, cmap=cmap, aspect="auto", origin="lower",
                        vmin=0.04, vmax=0.21)
    ax_h.set_xticks(range(len(n_values)))
    ax_h.set_xticklabels([f"{n:,}" for n in n_values], fontsize=8)
    ax_h.set_yticks(range(len(p_values)))
    ax_h.set_yticklabels(p_values)
    ax_h.set_xlabel("Total sample size $n$")
    ax_h.set_ylabel("Dimension $p$")
    for ii in range(len(p_values)):
        for jj in range(len(n_values)):
            val = gap_matrix[ii][jj]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                tc = "white" if val > 0.12 else "black"
                ax_h.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                          fontsize=9, fontweight="bold", color=tc)
            else:
                ax_h.text(jj, ii, "---", ha="center", va="center",
                          fontsize=9, color="#999999")
    fig_h.colorbar(im_h, ax=ax_h, shrink=0.85).set_label("Max coverage gap")
    fig_h.tight_layout()
    for fmt in ["pdf", "png"]:
        fig_h.savefig(FIGURES_DIR / f"exp9_scaling_heatmap.{fmt}")
    plt.close(fig_h)

    # Gap vs n only
    fig_g, ax_g = plt.subplots(figsize=(5, 4))
    for i, p in enumerate(p_values):
        gaps = np.array([gap_matrix[i][j] for j in range(len(n_values))])
        valid = ~np.isnan(gaps)
        if valid.sum() > 1:
            ax_g.plot(np.array(n_values)[valid], gaps[valid],
                      marker=markers[i], linestyle="-", color=colors[i],
                      markersize=6, markeredgecolor="white",
                      markeredgewidth=0.6, linewidth=1.8,
                      label=f"$p = {p}$")
    ax_g.plot(n_ref, c_ref / np.sqrt(n_ref), ":", color="#999999",
              linewidth=1.5, label=r"$O(1/\sqrt{n})$")
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    ax_g.set_xlabel("Total sample size $n$")
    ax_g.set_ylabel("Max conditional coverage gap")
    ax_g.set_xticks([200, 500, 1000, 2000, 5000])
    ax_g.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_g.legend(frameon=True, ncol=2, fontsize=9)
    fig_g.tight_layout()
    for fmt in ["pdf", "png"]:
        fig_g.savefig(FIGURES_DIR / f"exp9_gap_vs_n.{fmt}")
    plt.close(fig_g)

    plt.close(fig)
    print("  Saved individual heatmap and gap-vs-n figures too")


if __name__ == "__main__":
    print("Regenerating Figure 4 (Gaussian Recovery)...")
    make_figure4()
    print("\nRegenerating Figure 5 (Scaling Behavior)...")
    make_figure5()
    print("\nDone!")
