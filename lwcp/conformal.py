"""Leverage-Weighted Conformal Prediction."""

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from ._utils import conformal_quantile
from .leverage import LeverageComputer
from .weights import ConstantWeight, InverseRootLeverageWeight, WeightFunction, WeightSelector


class LWCP(BaseEstimator):
    """Leverage-Weighted Conformal Prediction.

    Constructs prediction intervals whose width adapts to the statistical
    leverage of the test point. High-leverage points (far from training
    centroid) get wider intervals; low-leverage points get narrower intervals.

    Parameters
    ----------
    predictor : sklearn estimator
        Any estimator with fit(X, y) and predict(X). Will be cloned.
    weight_fn : WeightFunction, callable, or "auto", default=None
        Leverage-based weighting function w(h). If None, uses
        InverseRootLeverageWeight: w(h) = (1+h)^{-1/2}. If "auto",
        selects from a candidate set via 3-way split validation.
    alpha : float, default=0.1
        Significance level. Intervals target 1-alpha coverage.
    ridge : float, default=0.0
        Ridge regularization for leverage score computation.
    leverage_method : str, default="exact"
        "exact" or "approximate" (randomized SVD).
    n_components : int or None, default=None
        Components for randomized SVD.
    random_state : int or None, default=None
        Random seed for data splitting and randomized SVD.
    calibration_size : float or int, default=0.5
        Fraction (if float in (0,1)) or count (if int) of calibration samples.

    Attributes
    ----------
    predictor_ : fitted estimator
    leverage_computer_ : LeverageComputer
    q_hat_ : float
        Conformal quantile from calibration scores.
    calibration_scores_ : ndarray of shape (n_cal,)
    calibration_leverages_ : ndarray of shape (n_cal,)
    """

    def __init__(
        self,
        predictor,
        weight_fn: Optional[WeightFunction] = None,
        alpha: float = 0.1,
        ridge: float = 0.0,
        leverage_method: str = "exact",
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
        calibration_size: Union[float, int] = 0.5,
    ):
        self.predictor = predictor
        self.weight_fn = weight_fn
        self.alpha = alpha
        self.ridge = ridge
        self.leverage_method = leverage_method
        self.n_components = n_components
        self.random_state = random_state
        self.calibration_size = calibration_size

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "LWCP":
        """Split data, train predictor, compute calibration scores.

        Parameters
        ----------
        X : ndarray of shape (n, p)
        y : ndarray of shape (n,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if isinstance(self.calibration_size, float):
            cal_frac = self.calibration_size
        else:
            cal_frac = self.calibration_size / X.shape[0]

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=cal_frac, random_state=self.random_state
        )
        return self.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)

    def fit_with_precomputed_split(
        self,
        X_train: NDArray[np.floating],
        y_train: NDArray[np.floating],
        X_cal: NDArray[np.floating],
        y_cal: NDArray[np.floating],
    ) -> "LWCP":
        """Fit with a user-provided train/calibration split.

        Parameters
        ----------
        X_train, y_train : training data
        X_cal, y_cal : calibration data

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)

        # Train predictor
        self.predictor_ = clone(self.predictor)
        self.predictor_.fit(X_train, y_train)

        # Compute leverage
        self.leverage_computer_ = LeverageComputer(
            ridge=self.ridge,
            method=self.leverage_method,
            n_components=self.n_components,
            random_state=self.random_state,
        ).fit(X_train)

        # Calibration leverages and residuals
        self.calibration_leverages_ = self.leverage_computer_.leverage_scores(X_cal)
        y_pred_cal = self.predictor_.predict(X_cal)
        residuals = np.abs(y_cal - y_pred_cal)

        # Resolve weight function
        if self.weight_fn == "auto":
            selector = WeightSelector(random_state=self.random_state)
            w = selector.select(residuals, self.calibration_leverages_, self.alpha)
            self.weight_selector_ = selector
        elif self.weight_fn is not None:
            w = self.weight_fn
        else:
            w = InverseRootLeverageWeight()

        # Calibration scores
        weights = w(self.calibration_leverages_)
        self.calibration_scores_ = residuals * weights

        # Conformal quantile
        n_cal = X_cal.shape[0]
        self.q_hat_ = conformal_quantile(self.calibration_scores_, self.alpha, n_cal)
        self._weight_fn_resolved = w

        return self

    def predict(
        self, X: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Compute prediction intervals for new points.

        Parameters
        ----------
        X : ndarray of shape (m, p)

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Point predictions.
        lower : ndarray of shape (m,)
            Lower bounds of prediction intervals.
        upper : ndarray of shape (m,)
            Upper bounds of prediction intervals.
        """
        X = np.asarray(X, dtype=np.float64)
        w = self._weight_fn_resolved

        y_pred = self.predictor_.predict(X)
        h_new = self.leverage_computer_.leverage_scores(X)
        w_new = w(h_new)

        half_width = self.q_hat_ / w_new
        lower = y_pred - half_width
        upper = y_pred + half_width

        return y_pred, lower, upper
