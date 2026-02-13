"""Internal utilities for LWCP."""

import math

import numpy as np
from numpy.typing import NDArray


def conformal_quantile(scores: NDArray[np.floating], alpha: float, n_cal: int) -> float:
    """Compute the conformal quantile.

    Returns the ceil((1-alpha)(n_cal+1))-th smallest score.

    Parameters
    ----------
    scores : ndarray of shape (n_cal,)
        Nonconformity scores from the calibration set.
    alpha : float
        Significance level in (0, 1).
    n_cal : int
        Number of calibration points.

    Returns
    -------
    q : float
        The conformal quantile threshold.
    """
    sorted_scores = np.sort(scores)
    rank = math.ceil((1 - alpha) * (n_cal + 1)) - 1  # 0-indexed
    rank = min(rank, n_cal - 1)  # clamp to valid index
    return float(sorted_scores[rank])
