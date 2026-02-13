"""Conditional coverage metrics for conformal prediction evaluation.

MSCE (Mean Squared Calibration Error) and WSC (Worst-Slab Coverage)
measure conditional coverage quality across leverage bins.
"""

import numpy as np
from numpy.typing import NDArray


def compute_msce(
    h_values: NDArray,
    covered: NDArray,
    alpha: float = 0.1,
    n_bins: int = 10,
) -> float:
    """Mean Squared Calibration Error across leverage bins.

    MSCE = (1/K) sum_k (coverage_k - (1 - alpha))^2

    Parameters
    ----------
    h_values : ndarray
        Leverage scores for test points.
    covered : ndarray
        Boolean array indicating coverage for each test point.
    alpha : float
        Nominal miscoverage level.
    n_bins : int
        Number of leverage bins (quantile-based).

    Returns
    -------
    float
        MSCE value. Lower is better; 0 means perfect calibration.
    """
    h_values = np.asarray(h_values, dtype=np.float64)
    covered = np.asarray(covered, dtype=np.float64)
    target = 1.0 - alpha

    bin_edges = np.percentile(h_values, np.linspace(0, 100, n_bins + 1))
    msce = 0.0
    n_valid = 0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (h_values >= bin_edges[i]) & (h_values <= bin_edges[i + 1])
        else:
            mask = (h_values >= bin_edges[i]) & (h_values < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_cov = np.mean(covered[mask])
            msce += (bin_cov - target) ** 2
            n_valid += 1

    return msce / max(n_valid, 1)


def compute_wsc(
    h_values: NDArray,
    covered: NDArray,
    n_bins: int = 10,
) -> float:
    """Worst-Slab Coverage: minimum coverage across leverage bins.

    WSC = min_k coverage_k

    Parameters
    ----------
    h_values : ndarray
        Leverage scores for test points.
    covered : ndarray
        Boolean array indicating coverage for each test point.
    n_bins : int
        Number of leverage bins (quantile-based).

    Returns
    -------
    float
        Worst-slab coverage. Higher is better; ideal is 1 - alpha.
    """
    h_values = np.asarray(h_values, dtype=np.float64)
    covered = np.asarray(covered, dtype=np.float64)

    bin_edges = np.percentile(h_values, np.linspace(0, 100, n_bins + 1))
    min_cov = 1.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (h_values >= bin_edges[i]) & (h_values <= bin_edges[i + 1])
        else:
            mask = (h_values >= bin_edges[i]) & (h_values < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_cov = np.mean(covered[mask])
            min_cov = min(min_cov, bin_cov)

    return min_cov
