"""Tests for leverage score computation."""

import numpy as np
import pytest

from lwcp.leverage import LeverageComputer


class TestLeverageComputerExact:
    def test_trace_equals_p(self):
        """sum(h_i) = p for training data."""
        rng = np.random.default_rng(42)
        n, p = 100, 5
        X = rng.standard_normal((n, p))
        lc = LeverageComputer().fit(X)
        h = lc.leverage_scores(X)
        np.testing.assert_allclose(h.sum(), p, atol=1e-10)

    def test_bounded_0_1(self):
        """Training leverage scores are in [0, 1]."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        lc = LeverageComputer().fit(X)
        h = lc.leverage_scores(X)
        assert np.all(h >= -1e-12)
        assert np.all(h <= 1.0 + 1e-12)

    def test_identity_design(self):
        """When X = I, all training leverage scores are 1 (since p = n)."""
        X = np.eye(5)
        lc = LeverageComputer().fit(X)
        h = lc.leverage_scores(X)
        np.testing.assert_allclose(h, np.ones(5), atol=1e-12)

    def test_single_feature(self):
        """With p=1, h_i = x_i^2 / sum(x_j^2)."""
        X = np.array([[1.0], [2.0], [3.0]])
        lc = LeverageComputer().fit(X)
        h = lc.leverage_scores(X)
        sum_sq = 1 + 4 + 9
        expected = np.array([1, 4, 9]) / sum_sq
        np.testing.assert_allclose(h, expected, atol=1e-12)

    def test_test_point_leverage(self):
        """Leverage of out-of-sample points can exceed 1."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 3))
        lc = LeverageComputer().fit(X_train)
        # A point far from the centroid
        x_extreme = np.array([[100.0, 100.0, 100.0]])
        h = lc.leverage_scores(x_extreme)
        assert h[0] > 1.0

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((50, 4))
        X_test = rng.standard_normal((20, 4))
        lc = LeverageComputer().fit(X_train)
        h = lc.leverage_scores(X_test)
        assert h.shape == (20,)

    def test_not_fitted_raises(self):
        lc = LeverageComputer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            lc.leverage_scores(np.ones((5, 3)))


class TestRidgeLeverage:
    def test_ridge_reduces_leverage(self):
        """Ridge regularization should reduce leverage scores."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        h_exact = LeverageComputer(ridge=0.0).fit(X).leverage_scores(X)
        h_ridge = LeverageComputer(ridge=1.0).fit(X).leverage_scores(X)
        assert np.all(h_ridge <= h_exact + 1e-12)

    def test_ridge_trace(self):
        """sum(h_i^lambda) = d_lambda = tr(X(X^TX + lambda I)^{-1}X^T)."""
        rng = np.random.default_rng(42)
        n, p = 100, 5
        X = rng.standard_normal((n, p))
        lam = 2.0
        lc = LeverageComputer(ridge=lam).fit(X)
        h = lc.leverage_scores(X)
        # Expected: tr(X @ (X^T X + lambda I)^{-1} @ X^T)
        XtX = X.T @ X
        gram_inv = np.linalg.inv(XtX + lam * np.eye(p))
        expected_trace = np.trace(X @ gram_inv @ X.T)
        np.testing.assert_allclose(h.sum(), expected_trace, atol=1e-10)

    def test_large_ridge_shrinks_to_zero(self):
        """With very large lambda, leverages approach 0."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        h = LeverageComputer(ridge=1e10).fit(X).leverage_scores(X)
        np.testing.assert_allclose(h, 0.0, atol=1e-5)


class TestApproximateLeverage:
    def test_approximate_matches_exact(self):
        """Randomized SVD with full rank should closely match exact."""
        rng = np.random.default_rng(42)
        n, p = 200, 10
        X = rng.standard_normal((n, p))
        h_exact = LeverageComputer(ridge=0.0).fit(X).leverage_scores(X)
        h_approx = LeverageComputer(
            method="approximate", n_components=p, random_state=42
        ).fit(X).leverage_scores(X)
        np.testing.assert_allclose(h_approx, h_exact, atol=1e-4)

    def test_approximate_trace(self):
        """Approximate leverage scores should approximately sum to p."""
        rng = np.random.default_rng(42)
        n, p = 200, 10
        X = rng.standard_normal((n, p))
        h = LeverageComputer(
            method="approximate", n_components=p, random_state=42
        ).fit(X).leverage_scores(X)
        np.testing.assert_allclose(h.sum(), p, atol=0.5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            LeverageComputer(method="invalid").fit(np.ones((5, 2)))
