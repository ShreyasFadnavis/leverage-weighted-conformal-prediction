"""Generate the README hero image for LWCP."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import Ridge
from scipy.optimize import curve_fit
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lwcp import LWCP, ConstantWeight

# ── Data (same as README) ──────────────────────────────────────────────
rng = np.random.default_rng(5)
p, n = 30, 200
X = rng.standard_normal((n, p))
X[-20:] *= 3
beta = rng.standard_normal(p) * 0.5
y = X @ beta + rng.normal(0, 2.0, size=n)

# ── Fit ────────────────────────────────────────────────────────────────
model = LWCP(predictor=Ridge(alpha=1.0), alpha=0.1, random_state=0)
model.fit(X, y)
vanilla = LWCP(predictor=Ridge(alpha=1.0), alpha=0.1, random_state=0,
               weight_fn=ConstantWeight())
vanilla.fit(X, y)

# ── Predict ────────────────────────────────────────────────────────────
X_test = np.vstack([rng.standard_normal((150, p)),
                    rng.standard_normal((150, p)) * 3])
y_pred, lo, hi = model.predict(X_test)
_, lo_v, hi_v = vanilla.predict(X_test)

h = model.leverage_computer_.leverage_scores(X_test)
w_lwcp = hi - lo
w_van = hi_v - lo_v
hw_lwcp = w_lwcp / 2
hw_van = w_van / 2

# Sort by leverage
order = np.argsort(h)
h_s = h[order]
hw_lwcp_s = hw_lwcp[order]
hw_van_s = hw_van[order]
w_lwcp_s = w_lwcp[order]
w_van_s = w_van[order]
idx = np.arange(len(h_s))
n_pts = len(idx)

# Stats
pct_narrower = 100 * np.mean(w_lwcp < w_van)
van_w = np.mean(w_van)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#d0d0d0",
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelcolor": "#555555",
    "text.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "#fafbfc",
    "axes.grid": False,
})

GREEN = "#34c759"
GREEN_DARK = "#248a3d"
CORAL = "#ff453a"
CORAL_DARK = "#d70015"
LWCP_COL = "#ff453a"
VAN_COL = "#8e8e93"
DARK = "#1d1d1f"

# ── Figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.26,
                       left=0.05, right=0.97, top=0.85, bottom=0.13)

# Shared y-limits for centered intervals
ymax_hw = max(hw_van_s.max(), hw_lwcp_s.max()) * 1.15

# ═══════════════════════════════════════════════════════════════════════
# PANEL 1: Vanilla CP — flat centered intervals
# ═══════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0])

# Centered: prediction at y=0, interval = ±half_width
ax1.axhline(0, color="#e0e0e0", linewidth=0.5, zorder=0)
ax1.fill_between(idx, -hw_van_s, hw_van_s, color=VAN_COL, alpha=0.15, zorder=1)
ax1.plot(idx, hw_van_s, color=VAN_COL, linewidth=1.8, zorder=2)
ax1.plot(idx, -hw_van_s, color=VAN_COL, linewidth=1.8, zorder=2)

# Width double-arrow in the middle
mid = n_pts // 2
ax1.annotate("", xy=(mid, hw_van_s[mid]), xytext=(mid, -hw_van_s[mid]),
             arrowprops=dict(arrowstyle="<->", color="#636366", lw=1.5))
ax1.text(mid + 12, 0.4, f"width = {van_w:.1f}\n(constant everywhere)",
         fontsize=9, color="#636366", ha="left", fontstyle="italic",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                   edgecolor="#dddddd", alpha=0.9, linewidth=0.5))

ax1.set_ylim(-ymax_hw, ymax_hw)
ax1.set_xlim(0, n_pts - 1)
ax1.set_xticks([0, n_pts - 1])
ax1.set_xticklabels(["low $h$", "high $h$"], fontsize=9)
ax1.set_ylabel("Interval half-width", fontsize=10.5)
ax1.set_xlabel("Test points  (sorted by leverage $\\longrightarrow$)",
               fontsize=10, labelpad=5)
ax1.set_title("Vanilla Conformal Prediction", fontsize=12.5,
              fontweight="bold", color=VAN_COL, pad=8)

# ═══════════════════════════════════════════════════════════════════════
# PANEL 2: LWCP — adaptive centered intervals
# ═══════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1], sharey=ax1)

ax2.axhline(0, color="#e0e0e0", linewidth=0.5, zorder=0)
ax2.fill_between(idx, -hw_lwcp_s, hw_lwcp_s, color=LWCP_COL, alpha=0.10, zorder=1)
ax2.plot(idx, hw_lwcp_s, color=LWCP_COL, linewidth=1.8, zorder=2)
ax2.plot(idx, -hw_lwcp_s, color=LWCP_COL, linewidth=1.8, zorder=2)

# Overlay vanilla bounds as ghost dashed for comparison
ax2.plot(idx, hw_van_s, color=VAN_COL, linewidth=1.2, linestyle="--",
         alpha=0.5, zorder=3)
ax2.plot(idx, -hw_van_s, color=VAN_COL, linewidth=1.2, linestyle="--",
         alpha=0.5, zorder=3)

# Green/red difference fills (the key creative element)
savings_mask = hw_lwcp_s < hw_van_s
cost_mask = hw_lwcp_s > hw_van_s

# Green where LWCP is narrower
ax2.fill_between(idx, hw_lwcp_s, hw_van_s, where=savings_mask,
                 color=GREEN, alpha=0.35, zorder=2)
ax2.fill_between(idx, -hw_van_s, -hw_lwcp_s, where=savings_mask,
                 color=GREEN, alpha=0.35, zorder=2)

# Coral where LWCP is wider
ax2.fill_between(idx, hw_van_s, hw_lwcp_s, where=cost_mask,
                 color=CORAL, alpha=0.22, zorder=2)
ax2.fill_between(idx, -hw_lwcp_s, -hw_van_s, where=cost_mask,
                 color=CORAL, alpha=0.22, zorder=2)

# Crossover line
cross_pts = np.where(np.diff(savings_mask.astype(int)))[0]
if len(cross_pts) > 0:
    cx = cross_pts[0]
    ax2.axvline(cx, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=5)

# Width double-arrows — narrow end
narrow_i = int(n_pts * 0.12)
ax2.annotate("", xy=(narrow_i, hw_lwcp_s[narrow_i]),
             xytext=(narrow_i, -hw_lwcp_s[narrow_i]),
             arrowprops=dict(arrowstyle="<->", color=GREEN_DARK, lw=1.5))
ax2.text(narrow_i + 6, 0.4, f"{w_lwcp_s[narrow_i]:.1f}",
         fontsize=10, color=GREEN_DARK, fontweight="bold")

# Width double-arrows — wide end
wide_i = int(n_pts * 0.90)
ax2.annotate("", xy=(wide_i, hw_lwcp_s[wide_i]),
             xytext=(wide_i, -hw_lwcp_s[wide_i]),
             arrowprops=dict(arrowstyle="<->", color=CORAL_DARK, lw=1.5))
ax2.text(wide_i - 40, hw_lwcp_s[wide_i] * 0.25, f"{w_lwcp_s[wide_i]:.1f}",
         fontsize=10, color=CORAL_DARK, fontweight="bold")

# Small legend in upper-left
leg_el = [
    Line2D([0], [0], color=VAN_COL, linewidth=1.2, linestyle="--",
           label="Vanilla CP"),
    Patch(facecolor=GREEN, alpha=0.4, label="Narrower"),
    Patch(facecolor=CORAL, alpha=0.28, label="Wider"),
]
leg2 = ax2.legend(handles=leg_el, loc="upper left", fontsize=8,
                  frameon=True, fancybox=True, framealpha=0.95,
                  edgecolor="#dddddd", borderpad=0.4, handletextpad=0.4)
leg2.get_frame().set_linewidth(0.4)

ax2.set_xlim(0, n_pts - 1)
ax2.set_xticks([0, n_pts - 1])
ax2.set_xticklabels(["low $h$", "high $h$"], fontsize=9)
ax2.set_xlabel("Test points  (sorted by leverage $\\longrightarrow$)",
               fontsize=10, labelpad=5)
ax2.set_title("LWCP  (Ours)", fontsize=12.5, fontweight="bold",
              color=LWCP_COL, pad=8)
plt.setp(ax2.get_yticklabels(), visible=False)

# ═══════════════════════════════════════════════════════════════════════
# PANEL 3: Width vs leverage scatter
# ═══════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2])

# Smooth LWCP trend
def width_model(hh, a):
    return a * np.sqrt(1 + hh)

popt, _ = curve_fit(width_model, h_s, w_lwcp_s, p0=[w_lwcp_s[0]])
h_smooth = np.linspace(h_s.min(), h_s.max(), 300)
w_smooth = width_model(h_smooth, popt[0])

# Vanilla: dashed line + ghost dots
ax3.axhline(van_w, color=VAN_COL, linewidth=2.0, linestyle="--", zorder=3)
ax3.scatter(h, w_van, alpha=0.10, s=10, color=VAN_COL, edgecolors="none",
            zorder=2)

# LWCP: scatter + trend
ax3.scatter(h, w_lwcp, alpha=0.30, s=16, color=LWCP_COL, edgecolors="none",
            zorder=3)
ax3.plot(h_smooth, w_smooth, color=LWCP_COL, linewidth=2.2, alpha=0.85,
         zorder=4)

# Green/coral fills
ax3.fill_between(h_smooth, van_w, w_smooth, where=w_smooth <= van_w,
                 color=GREEN, alpha=0.30, zorder=1, interpolate=True)
ax3.fill_between(h_smooth, van_w, w_smooth, where=w_smooth >= van_w,
                 color=CORAL, alpha=0.20, zorder=1, interpolate=True)

# "Saves width" annotation
save_h = h_smooth[35]
save_mid = (van_w + width_model(save_h, popt[0])) / 2
ax3.annotate(f"Saves width\n({pct_narrower:.0f}% of points)",
             xy=(save_h, save_mid),
             xytext=(2.2, van_w * 0.22),
             fontsize=9.5, color=GREEN_DARK, fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="-|>", color=GREEN_DARK, lw=1.4,
                             connectionstyle="arc3,rad=0.12"),
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor=GREEN, alpha=0.95, linewidth=0.7))

# "Extra protection" annotation
prot_h = h_smooth[220]
prot_mid = (van_w + width_model(prot_h, popt[0])) / 2
ax3.annotate("Extra protection\nfor risky points",
             xy=(prot_h, prot_mid),
             xytext=(3.2, w_smooth[-1] * 0.92),
             fontsize=9.5, color=CORAL_DARK, fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="-|>", color=CORAL_DARK, lw=1.4,
                             connectionstyle="arc3,rad=-0.15"),
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor=CORAL, alpha=0.95, linewidth=0.7))

# Legend
legend_elements = [
    Line2D([0], [0], color=VAN_COL, linewidth=2, linestyle="--",
           label="Vanilla CP"),
    Line2D([0], [0], color=LWCP_COL, linewidth=2, label="LWCP"),
    Patch(facecolor=GREEN, alpha=0.35, label="Width saved"),
    Patch(facecolor=CORAL, alpha=0.25, label="Extra protection"),
]
leg3 = ax3.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
                  frameon=True, fancybox=True, framealpha=0.95,
                  edgecolor="#dddddd", borderpad=0.5)
leg3.get_frame().set_linewidth(0.4)

ax3.set_xlabel("Leverage score  $h(\\mathbf{x})$", fontsize=10.5)
ax3.set_ylabel("Interval width", fontsize=10.5)
ax3.set_title("Width Adapts to Leverage", fontsize=12.5, fontweight="bold",
              color=DARK, pad=8)
ax3.set_ylim(bottom=0)

# ── Save ───────────────────────────────────────────────────────────────
fig.savefig(os.path.join(os.path.dirname(__file__), "lwcp_example.png"),
            dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.12)
plt.close(fig)
print("Saved examples/lwcp_example.png")
