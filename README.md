# LWCP: Leverage-Weighted Conformal Prediction

Prediction intervals that adapt to statistical leverage. High-leverage points
(far from the training centroid) receive wider intervals; low-leverage points
receive narrower intervals. The result is tighter intervals on average while
preserving both marginal and approximate conditional coverage.

**Paper:** *Leverage-Weighted Conformal Prediction*, Fadnavis (2026). UAI 2026.

## Installation

```bash
pip install -e .
```

To run the experiments and reproduce paper figures:

```bash
pip install -e ".[experiments]"
```

### Requirements

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- scikit-learn >= 1.3
- matplotlib >= 3.7 (experiments only)
- quantile-forest >= 1.3 (experiments only)

## Example

A complete example on the Diabetes dataset with visualization:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from lwcp import LWCP, LeverageComputer

# Load data and split into train+calibration vs test
X, y = load_diabetes(return_X_y=True)
X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit LWCP with 90% coverage
model = LWCP(predictor=LinearRegression(), alpha=0.1, random_state=42)
model.fit(X_fit, y_fit)
y_pred, lower, upper = model.predict(X_test)

# Compute test-point leverage scores for coloring
h_test = model.leverage_computer_.leverage_scores(X_test)

# Plot prediction intervals sorted by leverage
order = np.argsort(h_test)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: intervals colored by leverage
ax = axes[0]
for i, idx in enumerate(order):
    color = plt.cm.viridis(h_test[idx] / h_test.max())
    ax.plot([i, i], [lower[idx], upper[idx]], color=color, linewidth=1.5, alpha=0.7)
ax.scatter(range(len(order)), y_test[order], color="black", s=10, zorder=5, label="True y")
ax.set_xlabel("Test points (sorted by leverage)")
ax.set_ylabel("Response")
ax.set_title("LWCP Prediction Intervals")
ax.legend()

# Right: interval width vs leverage
ax = axes[1]
widths = upper - lower
ax.scatter(h_test, widths, alpha=0.6, edgecolors="white", linewidths=0.5)
ax.set_xlabel("Leverage score h(x)")
ax.set_ylabel("Interval width")
ax.set_title("Width Adapts to Leverage")

plt.tight_layout()
plt.savefig("lwcp_example.png", dpi=150)
plt.show()

# Print summary
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"Coverage: {coverage:.1%}")
print(f"Mean width: {np.mean(widths):.2f}")
print(f"Width range: [{np.min(widths):.2f}, {np.max(widths):.2f}]")
```

## Quick Start

```python
from sklearn.linear_model import LinearRegression
from lwcp import LWCP

model = LWCP(
    predictor=LinearRegression(),
    alpha=0.1,  # 90% coverage
)
model.fit(X, y)
y_pred, lower, upper = model.predict(X_test)
```

### With a pre-computed train/calibration split

```python
model = LWCP(predictor=LinearRegression(), alpha=0.1)
model.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)
y_pred, lower, upper = model.predict(X_test)
```

### Automatic weight selection

```python
model = LWCP(predictor=LinearRegression(), weight_fn="auto", alpha=0.1)
model.fit(X, y)
```

### Approximate leverage (for large p)

```python
model = LWCP(
    predictor=LinearRegression(),
    leverage_method="approximate",
    n_components=15,  # rank-k truncated SVD
    alpha=0.1,
)
model.fit(X, y)
```

### Diagnostics

```python
from lwcp import diagnose_weight_alignment

result = diagnose_weight_alignment(
    model.calibration_scores_,
    model.calibration_leverages_,
)
print(result.recommendation)  # "use LWCP", "use vanilla", or "inconclusive"
print(f"eta_hat = {result.eta_hat:.3f}")  # leverage variation coefficient
```

## API Reference

### Core

| Class | Description |
|---|---|
| `LWCP` | Main class. scikit-learn `BaseEstimator` API with `fit` / `predict`. |
| `LeverageComputer` | Computes leverage scores via SVD (exact or randomized). |
| `FeatureSpaceLeverageComputer` | Leverage in a learned feature space (e.g., neural net penultimate layer). |

### Weight Functions

| Class | Formula | Use case |
|---|---|---|
| `InverseRootLeverageWeight` | `(1 + h)^{-1/2}` | Default. Optimal under homoscedastic noise. |
| `PowerLawWeight` | `(1 + h)^{-gamma}` | Tunable exponent for heteroscedastic noise. |
| `ConstantWeight` | `1` | Equivalent to vanilla (unweighted) conformal prediction. |
| `WeightSelector` | Validation-based | Selects the best weight from a candidate set. |

### Diagnostics

| Function | Description |
|---|---|
| `diagnose_weight_alignment` | Tests whether LWCP weights are well-aligned with heteroscedasticity. Returns recommendation, leverage variation coefficient (eta_hat), and regression statistics. |

## Reproducing Paper Figures

All 13 figures from the paper can be reproduced with:

```bash
python -m experiments.run_all
```

This runs all 14 experiments and saves figures to `experiments/figures/`. For a
quick sanity check with fewer Monte Carlo repetitions:

```bash
python -m experiments.run_all --quick
```

### Individual experiments

| Figure | Script | Description |
|---|---|---|
| Fig 1 | `python -m experiments.run_conditional` | Conditional coverage by leverage decile |
| Fig 2 | `python -m experiments.run_baselines` | Baselines comparison (CQR, Studentized CP) |
| Fig 3 | `python -m experiments.run_gaussian_recovery` | Gaussian oracle width recovery |
| Fig 4 | `python -m experiments.run_hetero_sweep` | Heteroscedasticity sweep |
| Fig 5 | `python -m experiments.run_width` | Width vs leverage scatter |
| Fig 6 | `python -m experiments.run_scaling` | Scaling heatmap and gap vs n |
| Fig 7-8 | `python -m experiments.regenerate_fig7_fig8` | Approximate leverage coverage and scatter |
| Fig 9 | `python -m experiments.run_real_data` | Real datasets (Diabetes, CPU, Superconductor) |
| Fig 10 | `python -m experiments.run_ridge` | Ridge leverage |
| Fig 11 | `python -m experiments.run_coverage` | Marginal coverage (1000 reps) |
| Fig 12 | `python -m experiments.run_nonlinear` | Non-linear predictors |

### Data sources

- **Diabetes**: `sklearn.datasets.load_diabetes()` (built-in)
- **CPU Activity**: OpenML dataset 562
- **Superconductor**: OpenML dataset 44065

All datasets are downloaded automatically on first run.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
lwcp/
    __init__.py          # Public API exports
    conformal.py         # LWCP class (fit / predict)
    leverage.py          # Leverage score computation (exact / approximate SVD)
    weights.py           # Weight functions and automatic selection
    diagnostics.py       # Weight alignment diagnostic
    _utils.py            # Conformal quantile helper

experiments/
    run_all.py           # Master script (runs all experiments)
    run_*.py             # Individual experiment scripts
    dgps.py              # Data generating processes
    baselines.py         # Baseline methods (Vanilla CP, CQR, Studentized, LWCP+)
    plotting.py          # Publication-quality matplotlib style
    metrics.py           # MSCE and WSC evaluation metrics
    results/             # Saved JSON results
    figures/             # Generated PDF and PNG figures

tests/
    test_conformal.py    # LWCP class tests
    test_leverage.py     # Leverage computation tests
    test_weights.py      # Weight function tests
    test_integration.py  # End-to-end integration tests
```

## Citation

```bibtex
@inproceedings{fadnavis2026lwcp,
  author    = {Fadnavis, Shreyas},
  title     = {Leverage-Weighted Conformal Prediction},
  booktitle = {Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026}
}
```

## License

MIT
