"""Leverage-based weighting functions for LWCP nonconformity scores."""

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class WeightFunction(Protocol):
    """Protocol for leverage-based weighting functions.

    A weight function maps leverage scores h to positive weights w(h).
    Any callable with signature (NDArray) -> NDArray satisfies this.
    """

    def __call__(self, h: NDArray[np.floating]) -> NDArray[np.floating]: ...


class ConstantWeight:
    """w(h) = 1. Recovers vanilla split conformal prediction."""

    def __call__(self, h: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.ones_like(h, dtype=np.float64)

    def __repr__(self) -> str:
        return "ConstantWeight()"


class InverseRootLeverageWeight:
    """w(h) = (1 + h)^{-1/2}.

    Matches the prediction variance structure of OLS with homoscedastic
    errors, where Var(Y_new - Yhat_new) = sigma^2 (1 + h_new).
    """

    def __call__(self, h: NDArray[np.floating]) -> NDArray[np.floating]:
        return (1.0 + np.asarray(h, dtype=np.float64)) ** (-0.5)

    def __repr__(self) -> str:
        return "InverseRootLeverageWeight()"


class PowerLawWeight:
    """w(h) = (h + epsilon)^{-gamma}.

    Parameters
    ----------
    gamma : float
        Exponent. Must be positive. Larger gamma = more aggressive adaptation.
    epsilon : float
        Small constant added to h to avoid division by zero.
    """

    def __init__(self, gamma: float, epsilon: float = 1e-10):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma
        self.epsilon = epsilon

    def __call__(self, h: NDArray[np.floating]) -> NDArray[np.floating]:
        return (np.asarray(h, dtype=np.float64) + self.epsilon) ** (-self.gamma)

    def __repr__(self) -> str:
        return f"PowerLawWeight(gamma={self.gamma})"


class WeightSelector:
    """Data-driven weight selection via 3-way split validation.

    Selects from a discrete candidate set by minimizing conditional
    coverage gap on a held-out validation portion. Uses a 3-way split
    (train / validation / calibration) so that the selected weight is
    determined before the final calibration quantile, preserving
    marginal coverage.

    Parameters
    ----------
    candidates : list of weight functions, optional
        Candidate weights to evaluate. Defaults to:
        [ConstantWeight, InverseRootLeverageWeight,
         PowerLawWeight(0.3), PowerLawWeight(0.7)].
    val_fraction : float, default=0.3
        Fraction of calibration set to hold out for validation.
    n_bins : int, default=5
        Number of leverage bins for conditional coverage evaluation.
    random_state : int or None, default=None
        Random seed for validation split.
    """

    def __init__(
        self,
        candidates: Optional[List] = None,
        val_fraction: float = 0.3,
        n_bins: int = 5,
        random_state: Optional[int] = None,
    ):
        self.candidates = candidates or [
            ConstantWeight(),
            InverseRootLeverageWeight(),
            PowerLawWeight(gamma=0.3),
            PowerLawWeight(gamma=0.7),
        ]
        self.val_fraction = val_fraction
        self.n_bins = n_bins
        self.random_state = random_state
        self.selected_weight_ = None
        self.selection_scores_ = None

    def select(
        self,
        residuals: NDArray[np.floating],
        leverage_scores: NDArray[np.floating],
        alpha: float = 0.1,
    ) -> "WeightFunction":
        """Select best weight function from candidates.

        Parameters
        ----------
        residuals : ndarray of shape (n_cal,)
            Absolute residuals |y_i - f_hat(x_i)| on calibration set.
        leverage_scores : ndarray of shape (n_cal,)
            Leverage scores h_i for calibration points.
        alpha : float
            Significance level.

        Returns
        -------
        best_weight : WeightFunction
            The selected weight function.
        """
        from ._utils import conformal_quantile

        residuals = np.asarray(residuals, dtype=np.float64)
        h = np.asarray(leverage_scores, dtype=np.float64)
        n = len(residuals)

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = max(int(self.val_fraction * n), 2 * self.n_bins)
        val_idx = perm[:n_val]
        cal_idx = perm[n_val:]

        best_gap = np.inf
        best_weight = self.candidates[1]  # default: InverseRootLeverageWeight
        scores_dict = {}

        for w in self.candidates:
            # Compute scores on calibration portion
            cal_scores = residuals[cal_idx] * w(h[cal_idx])
            q_hat = conformal_quantile(cal_scores, alpha, len(cal_idx))

            # Evaluate on validation portion
            val_residuals = residuals[val_idx]
            val_h = h[val_idx]
            val_widths = q_hat / w(val_h)
            val_covered = val_residuals <= val_widths

            # Conditional coverage gap: bottom vs top quintile by leverage
            h_lo = np.percentile(val_h, 20)
            h_hi = np.percentile(val_h, 80)
            mask_lo = val_h <= h_lo
            mask_hi = val_h >= h_hi

            if mask_lo.sum() > 0 and mask_hi.sum() > 0:
                gap = abs(float(np.mean(val_covered[mask_lo]) - np.mean(val_covered[mask_hi])))
            else:
                gap = np.inf

            scores_dict[repr(w)] = gap

            if gap < best_gap:
                best_gap = gap
                best_weight = w

        self.selected_weight_ = best_weight
        self.selection_scores_ = scores_dict
        return best_weight
