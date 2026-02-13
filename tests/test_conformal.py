"""Tests for the LWCP conformal prediction class."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from lwcp import LWCP, ConstantWeight, InverseRootLeverageWeight


def _make_linear_data(n, p, sigma=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + sigma * rng.standard_normal(n)
    return X, y, beta


class TestLWCPBasic:
    def test_fit_predict_shapes(self):
        X, y, _ = _make_linear_data(200, 5)
        model = LWCP(predictor=LinearRegression(), alpha=0.1, random_state=42)
        model.fit(X[:150], y[:150])
        y_pred, lower, upper = model.predict(X[150:])
        assert y_pred.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)

    def test_lower_less_than_upper(self):
        X, y, _ = _make_linear_data(200, 5)
        model = LWCP(predictor=LinearRegression(), alpha=0.1, random_state=42)
        model.fit(X[:150], y[:150])
        _, lower, upper = model.predict(X[150:])
        assert np.all(lower < upper)

    def test_prediction_in_interval(self):
        """Point prediction should be inside the interval."""
        X, y, _ = _make_linear_data(200, 5)
        model = LWCP(predictor=LinearRegression(), alpha=0.1, random_state=42)
        model.fit(X[:150], y[:150])
        y_pred, lower, upper = model.predict(X[150:])
        assert np.all(y_pred >= lower)
        assert np.all(y_pred <= upper)


class TestLWCPCoverage:
    def test_marginal_coverage(self):
        """Empirical coverage should be >= 1-alpha on average."""
        rng = np.random.default_rng(0)
        n, p = 1000, 5
        alpha = 0.1
        coverages = []
        for seed in range(50):
            X, y, _ = _make_linear_data(n, p, sigma=0.5, seed=seed)
            model = LWCP(
                predictor=LinearRegression(),
                alpha=alpha,
                random_state=seed,
                calibration_size=0.5,
            )
            model.fit(X[:800], y[:800])
            _, lower, upper = model.predict(X[800:])
            cov = np.mean((y[800:] >= lower) & (y[800:] <= upper))
            coverages.append(cov)
        mean_cov = np.mean(coverages)
        # Should be close to 0.9, allow some slack
        assert mean_cov >= 0.85, f"Mean coverage {mean_cov:.3f} too low"

    def test_coverage_with_constant_weight(self):
        """Vanilla CP (constant weight) should also have valid coverage."""
        coverages = []
        for seed in range(50):
            X, y, _ = _make_linear_data(1000, 5, sigma=0.5, seed=seed)
            model = LWCP(
                predictor=LinearRegression(),
                weight_fn=ConstantWeight(),
                alpha=0.1,
                random_state=seed,
            )
            model.fit(X[:800], y[:800])
            _, lower, upper = model.predict(X[800:])
            cov = np.mean((y[800:] >= lower) & (y[800:] <= upper))
            coverages.append(cov)
        assert np.mean(coverages) >= 0.85


class TestLWCPAdaptivity:
    def test_constant_weight_gives_constant_width(self):
        """With ConstantWeight, all intervals have the same width."""
        X, y, _ = _make_linear_data(500, 5)
        model = LWCP(
            predictor=LinearRegression(),
            weight_fn=ConstantWeight(),
            alpha=0.1,
            random_state=42,
        )
        model.fit(X[:400], y[:400])
        _, lower, upper = model.predict(X[400:])
        widths = upper - lower
        np.testing.assert_allclose(widths, widths[0], atol=1e-12)

    def test_lwcp_width_varies_with_leverage(self):
        """LWCP intervals should have varying width."""
        X, y, _ = _make_linear_data(500, 5)
        model = LWCP(
            predictor=LinearRegression(),
            weight_fn=InverseRootLeverageWeight(),
            alpha=0.1,
            random_state=42,
        )
        model.fit(X[:400], y[:400])
        _, lower, upper = model.predict(X[400:])
        widths = upper - lower
        # Widths should not all be the same
        assert np.std(widths) > 1e-10

    def test_high_leverage_gets_wider_interval(self):
        """Points with higher leverage should get wider intervals."""
        rng = np.random.default_rng(42)
        n_train, p = 200, 3
        X_train = rng.standard_normal((n_train, p))
        beta = np.ones(p)
        y_train = X_train @ beta + 0.1 * rng.standard_normal(n_train)

        # Low-leverage test point (near centroid)
        x_low = np.array([[0.0, 0.0, 0.0]])
        # High-leverage test point (far from centroid)
        x_high = np.array([[10.0, 10.0, 10.0]])

        model = LWCP(
            predictor=LinearRegression(),
            weight_fn=InverseRootLeverageWeight(),
            alpha=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        _, low_lo, low_hi = model.predict(x_low)
        _, high_lo, high_hi = model.predict(x_high)

        width_low = (low_hi - low_lo)[0]
        width_high = (high_hi - high_lo)[0]
        assert width_high > width_low


class TestLWCPPrecomputedSplit:
    def test_precomputed_split(self):
        X, y, _ = _make_linear_data(300, 5)
        model = LWCP(predictor=LinearRegression(), alpha=0.1)
        model.fit_with_precomputed_split(
            X[:200], y[:200], X[200:250], y[200:250]
        )
        y_pred, lower, upper = model.predict(X[250:])
        assert y_pred.shape == (50,)
        assert np.all(lower < upper)


class TestLWCPWithRidge:
    def test_ridge_predictor_with_ridge_leverage(self):
        X, y, _ = _make_linear_data(300, 5)
        model = LWCP(
            predictor=Ridge(alpha=1.0),
            alpha=0.1,
            ridge=1.0,
            random_state=42,
        )
        model.fit(X[:250], y[:250])
        y_pred, lower, upper = model.predict(X[250:])
        assert y_pred.shape == (50,)
        assert np.all(lower < upper)
