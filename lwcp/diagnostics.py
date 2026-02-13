"""Diagnostics for LWCP weight alignment and leverage analysis."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class WeightAlignmentResult(NamedTuple):
    """Result of weight alignment diagnostic."""
    slope: float
    p_value: float
    recommendation: str  # "use LWCP", "use vanilla", or "inconclusive"
    eta_hat: float  # leverage variation coefficient


def diagnose_weight_alignment(
    calibration_scores: NDArray[np.floating],
    leverage_scores: NDArray[np.floating],
    significance: float = 0.05,
) -> WeightAlignmentResult:
    """Test whether LWCP weight function is well-aligned with heteroscedasticity.

    Regresses |score - median(score)| on leverage h. A significantly positive
    slope indicates that weighted scores still have increasing dispersion with
    leverage, meaning the weight is anti-aligned (LWCP may hurt).

    Parameters
    ----------
    calibration_scores : ndarray of shape (n_cal,)
        Weighted calibration scores: |r_i| * w(h_i).
    leverage_scores : ndarray of shape (n_cal,)
        Leverage scores h_i for calibration points.
    significance : float, default=0.05
        Significance level for the slope test.

    Returns
    -------
    result : WeightAlignmentResult
        slope: OLS regression coefficient of |score - median| on h.
        p_value: two-sided p-value for slope != 0.
        recommendation: "use LWCP", "use vanilla", or "inconclusive".
        eta_hat: std(h) / mean(h), the leverage variation coefficient.
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    h = np.asarray(leverage_scores, dtype=np.float64)
    n = len(scores)

    # Leverage diagnostic
    eta_hat = float(np.std(h) / np.mean(h)) if np.mean(h) > 0 else 0.0

    # Dispersion vs leverage: regress |score - median| on h
    median_score = np.median(scores)
    dispersion = np.abs(scores - median_score)

    # OLS: dispersion = a + b * h
    h_centered = h - np.mean(h)
    slope = float(np.sum(h_centered * dispersion) / (np.sum(h_centered**2) + 1e-15))
    residuals = dispersion - (np.mean(dispersion) + slope * h_centered)
    se = float(np.sqrt(np.sum(residuals**2) / ((n - 2) * np.sum(h_centered**2) + 1e-15)))

    from scipy import stats
    t_stat = slope / (se + 1e-15)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))

    # Recommendation
    if p_value < significance and slope > 0:
        recommendation = "use vanilla"
    elif eta_hat < 0.3:
        recommendation = "inconclusive"
    else:
        recommendation = "use LWCP"

    return WeightAlignmentResult(
        slope=slope,
        p_value=p_value,
        recommendation=recommendation,
        eta_hat=eta_hat,
    )
