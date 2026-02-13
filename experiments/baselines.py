"""Baseline conformal prediction methods for comparison.

All methods share the interface:
    fit(X_train, y_train, X_cal, y_cal, alpha)
    predict(X_test) -> (y_pred, lower, upper)
"""

import time

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from lwcp import LWCP, ConstantWeight, InverseRootLeverageWeight
from lwcp._utils import conformal_quantile


class VanillaCP:
    """Standard split conformal prediction (constant-width intervals)."""

    def __init__(self, predictor=None, alpha: float = 0.1):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.name = "Vanilla CP"

    def fit(self, X_train, y_train, X_cal, y_cal):
        self._model = LWCP(
            predictor=self.predictor,
            weight_fn=ConstantWeight(),
            alpha=self.alpha,
        )
        self._model.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)
        return self

    def predict(self, X_test):
        return self._model.predict(X_test)


class LWCPMethod:
    """LWCP with InverseRootLeverageWeight."""

    def __init__(self, predictor=None, alpha: float = 0.1, ridge: float = 0.0):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.ridge = ridge
        self.name = "LWCP"

    def fit(self, X_train, y_train, X_cal, y_cal):
        self._model = LWCP(
            predictor=self.predictor,
            weight_fn=InverseRootLeverageWeight(),
            alpha=self.alpha,
            ridge=self.ridge,
        )
        self._model.fit_with_precomputed_split(X_train, y_train, X_cal, y_cal)
        return self

    def predict(self, X_test):
        return self._model.predict(X_test)


class CQR:
    """Conformalized Quantile Regression using quantile forests.

    Romano et al. (2019). Intervals: [q_lo(x) - qhat, q_hi(x) + qhat]
    where qhat calibrates s_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)).
    """

    def __init__(self, predictor=None, alpha: float = 0.1, random_state: int = 42):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.random_state = random_state
        self.name = "CQR"

    def fit(self, X_train, y_train, X_cal, y_cal):
        from quantile_forest import RandomForestQuantileRegressor

        self._point_model = clone(self.predictor)
        self._point_model.fit(X_train, y_train)

        # Fit quantile forest for conditional quantiles
        self._qf = RandomForestQuantileRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._qf.fit(X_train, y_train)

        # Predict quantiles on calibration set (quantile_forest expects [0, 1])
        quantiles = [self.alpha / 2, 1 - self.alpha / 2]
        cal_quantiles = self._qf.predict(X_cal, quantiles=quantiles)
        q_lo_cal = cal_quantiles[:, 0]
        q_hi_cal = cal_quantiles[:, 1]

        # Conformity scores: max(q_lo - y, y - q_hi)
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
        n_cal = len(y_cal)
        self._q_hat = conformal_quantile(scores, self.alpha, n_cal)

        return self

    def predict(self, X_test):
        y_pred = self._point_model.predict(X_test)
        quantiles = [self.alpha / 2, 1 - self.alpha / 2]
        test_quantiles = self._qf.predict(X_test, quantiles=quantiles)
        q_lo = test_quantiles[:, 0]
        q_hi = test_quantiles[:, 1]
        lower = q_lo - self._q_hat
        upper = q_hi + self._q_hat
        return y_pred, lower, upper


class CQR_GBR:
    """CQR using GradientBoostingRegressor with quantile loss.

    A stronger CQR baseline that doesn't rely on quantile forests.
    Uses separate GBR models for lower and upper quantiles.
    """

    def __init__(self, predictor=None, alpha: float = 0.1, random_state: int = 42):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.random_state = random_state
        self.name = "CQR-GBR"

    def fit(self, X_train, y_train, X_cal, y_cal):
        from sklearn.ensemble import GradientBoostingRegressor

        self._point_model = clone(self.predictor)
        self._point_model.fit(X_train, y_train)

        # Fit separate GBR models for lower and upper quantiles
        self._qr_lo = GradientBoostingRegressor(
            loss="quantile", alpha=self.alpha / 2,
            n_estimators=100, max_depth=3,
            random_state=self.random_state,
        )
        self._qr_hi = GradientBoostingRegressor(
            loss="quantile", alpha=1 - self.alpha / 2,
            n_estimators=100, max_depth=3,
            random_state=self.random_state,
        )
        self._qr_lo.fit(X_train, y_train)
        self._qr_hi.fit(X_train, y_train)

        # Conformity scores: max(q_lo - y, y - q_hi)
        q_lo_cal = self._qr_lo.predict(X_cal)
        q_hi_cal = self._qr_hi.predict(X_cal)
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

        n_cal = len(y_cal)
        self._q_hat = conformal_quantile(scores, self.alpha, n_cal)

        return self

    def predict(self, X_test):
        y_pred = self._point_model.predict(X_test)
        q_lo = self._qr_lo.predict(X_test)
        q_hi = self._qr_hi.predict(X_test)
        lower = q_lo - self._q_hat
        upper = q_hi + self._q_hat
        return y_pred, lower, upper


class StudentizedCP:
    """Studentized (normalized) residual conformal prediction.

    Fits auxiliary RandomForest to predict |residuals|, then uses
    scores s_i = |y_i - f(x_i)| / sigma_hat(x_i).
    """

    def __init__(self, predictor=None, alpha: float = 0.1, random_state: int = 42):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.random_state = random_state
        self.name = "Studentized CP"

    def fit(self, X_train, y_train, X_cal, y_cal):
        # Fit point predictor
        self._point_model = clone(self.predictor)
        self._point_model.fit(X_train, y_train)

        # Fit auxiliary model for residual magnitude on training data
        train_residuals = np.abs(y_train - self._point_model.predict(X_train))
        self._sigma_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._sigma_model.fit(X_train, train_residuals)

        # Calibration scores: |residual| / sigma_hat
        cal_residuals = np.abs(y_cal - self._point_model.predict(X_cal))
        sigma_hat_cal = np.maximum(self._sigma_model.predict(X_cal), 1e-8)
        scores = cal_residuals / sigma_hat_cal

        n_cal = len(y_cal)
        self._q_hat = conformal_quantile(scores, self.alpha, n_cal)

        return self

    def predict(self, X_test):
        y_pred = self._point_model.predict(X_test)
        sigma_hat = np.maximum(self._sigma_model.predict(X_test), 1e-8)
        half_width = self._q_hat * sigma_hat
        lower = y_pred - half_width
        upper = y_pred + half_width
        return y_pred, lower, upper


class LWCPPlus:
    """LWCP+ = Leverage-Studentized Conformal Prediction.

    Combines leverage geometry (√(1+h) correction) with lightweight residual
    scale estimation (10-tree RF). Scores: |r| / (σ̂(x) · √(1+h(x))).

    Key insight: training residuals have Var ∝ (1-h), but prediction errors
    have Var ∝ (1+h). The leverage correction bridges this mismatch that
    Studentized CP alone cannot fix.
    """

    def __init__(self, predictor=None, alpha: float = 0.1,
                 n_trees: int = 10, random_state: int = 42):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.n_trees = n_trees
        self.random_state = random_state
        self.name = "LWCP+"

    def fit(self, X_train, y_train, X_cal, y_cal):
        from lwcp.leverage import LeverageComputer

        # Fit point predictor
        self._point_model = clone(self.predictor)
        self._point_model.fit(X_train, y_train)

        # Compute leverage scores
        self._lev = LeverageComputer().fit(X_train)

        # Train lightweight σ̂ on training |residuals|
        train_residuals = np.abs(y_train - self._point_model.predict(X_train))
        self._sigma_model = RandomForestRegressor(
            n_estimators=self.n_trees,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._sigma_model.fit(X_train, train_residuals)

        # Calibration scores: |r| / (σ̂(x) · √(1+h))
        cal_residuals = np.abs(y_cal - self._point_model.predict(X_cal))
        h_cal = self._lev.leverage_scores(X_cal)
        sigma_hat_cal = np.maximum(self._sigma_model.predict(X_cal), 1e-8)
        scores = cal_residuals / (sigma_hat_cal * np.sqrt(1.0 + h_cal))

        n_cal = len(y_cal)
        self._q_hat = conformal_quantile(scores, self.alpha, n_cal)

        return self

    def predict(self, X_test):
        y_pred = self._point_model.predict(X_test)
        h_test = self._lev.leverage_scores(X_test)
        sigma_hat = np.maximum(self._sigma_model.predict(X_test), 1e-8)
        half_width = self._q_hat * sigma_hat * np.sqrt(1.0 + h_test)
        lower = y_pred - half_width
        upper = y_pred + half_width
        return y_pred, lower, upper


class LWCPMethod_NonLinear:
    """LWCP with a non-linear predictor and optional feature-space leverage.

    Parameters
    ----------
    predictor : sklearn estimator
        Any regressor with fit/predict (e.g., RandomForestRegressor, MLPRegressor).
    feature_extractor : callable or None
        Maps X -> Φ(X) for feature-space leverage. If None, uses raw-X leverage.
    alpha : float
        Significance level.
    ridge : float
        Ridge regularization for leverage computation.
    """

    def __init__(self, predictor, feature_extractor=None,
                 alpha: float = 0.1, ridge: float = 0.0):
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.ridge = ridge
        self.name = "LWCP (non-linear)"

    def fit(self, X_train, y_train, X_cal, y_cal):
        from lwcp.leverage import FeatureSpaceLeverageComputer, LeverageComputer

        self._point_model = clone(self.predictor)
        self._point_model.fit(X_train, y_train)

        if self.feature_extractor is not None:
            self._lev = FeatureSpaceLeverageComputer(
                feature_extractor=self.feature_extractor,
                ridge=self.ridge,
            ).fit(X_train)
        else:
            self._lev = LeverageComputer(ridge=self.ridge).fit(X_train)

        self._w = InverseRootLeverageWeight()

        # Calibration scores
        cal_residuals = np.abs(y_cal - self._point_model.predict(X_cal))
        h_cal = self._lev.leverage_scores(X_cal)
        weights = self._w(h_cal)
        scores = cal_residuals * weights

        n_cal = len(y_cal)
        self._q_hat = conformal_quantile(scores, self.alpha, n_cal)
        return self

    def predict(self, X_test):
        y_pred = self._point_model.predict(X_test)
        h_test = self._lev.leverage_scores(X_test)
        w_test = self._w(h_test)
        half_width = self._q_hat / w_test
        lower = y_pred - half_width
        upper = y_pred + half_width
        return y_pred, lower, upper


class LocalizedCP:
    """Localized Conformal Prediction (Guan, 2023).

    Uses a Gaussian kernel to localize conformal quantiles around each
    test point, producing adaptive intervals that respond to local structure.

    Parameters
    ----------
    predictor : sklearn estimator
        Any regressor with fit/predict.
    alpha : float
        Significance level.
    bandwidth : float or None
        Gaussian kernel bandwidth. If None, uses median heuristic.
    """

    def __init__(self, predictor=None, alpha: float = 0.1, bandwidth=None):
        self.predictor = predictor if predictor is not None else LinearRegression()
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.name = "Localized CP"

    def fit(self, X_train, y_train, X_cal, y_cal):
        self._point_model = clone(self.predictor)
        self._point_model.fit(X_train, y_train)

        # Calibration residuals
        cal_preds = self._point_model.predict(X_cal)
        self._cal_scores = np.abs(y_cal - cal_preds)
        self._X_cal = np.asarray(X_cal, dtype=np.float64)

        # Bandwidth via median heuristic if not provided
        if self.bandwidth is None:
            from scipy.spatial.distance import pdist
            dists = pdist(self._X_cal, metric='euclidean')
            self._bw = np.median(dists) if len(dists) > 0 else 1.0
        else:
            self._bw = self.bandwidth

        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        y_pred = self._point_model.predict(X_test)
        n_test = X_test.shape[0]
        n_cal = self._X_cal.shape[0]

        lower = np.empty(n_test)
        upper = np.empty(n_test)

        for i in range(n_test):
            # Gaussian kernel weights
            diffs = self._X_cal - X_test[i]
            sq_dists = np.sum(diffs**2, axis=1)
            kernel_weights = np.exp(-sq_dists / (2.0 * self._bw**2))

            # Weighted conformal quantile
            total_weight = np.sum(kernel_weights)
            if total_weight < 1e-15:
                # Fallback: uniform weights
                q = np.quantile(self._cal_scores, min(1.0, (1 - self.alpha) * (n_cal + 1) / n_cal))
            else:
                # Sort scores and compute weighted quantile
                sorted_idx = np.argsort(self._cal_scores)
                sorted_scores = self._cal_scores[sorted_idx]
                sorted_weights = kernel_weights[sorted_idx]
                sorted_weights = sorted_weights / (np.sum(sorted_weights) + 1e-15)

                cum_weights = np.cumsum(sorted_weights)
                target = 1.0 - self.alpha
                idx = np.searchsorted(cum_weights, target)
                idx = min(idx, n_cal - 1)
                q = sorted_scores[idx]

            lower[i] = y_pred[i] - q
            upper[i] = y_pred[i] + q

        return y_pred, lower, upper


def run_method_timed(method, X_train, y_train, X_cal, y_cal, X_test):
    """Run a method and return (y_pred, lower, upper, fit_time, predict_time)."""
    t0 = time.perf_counter()
    method.fit(X_train, y_train, X_cal, y_cal)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred, lower, upper = method.predict(X_test)
    predict_time = time.perf_counter() - t0

    return y_pred, lower, upper, fit_time, predict_time
