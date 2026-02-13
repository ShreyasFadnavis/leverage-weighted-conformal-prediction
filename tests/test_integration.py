"""Integration tests for the full LWCP pipeline."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from lwcp import LWCP, ConstantWeight, InverseRootLeverageWeight, PowerLawWeight


class TestFullPipeline:
    def test_ridge_approximate_pipeline(self):
        """Full pipeline: Ridge predictor + ridge leverage + approximate SVD."""
        rng = np.random.default_rng(42)
        n, p = 500, 10
        X = rng.standard_normal((n, p))
        beta = rng.standard_normal(p)
        y = X @ beta + 0.5 * rng.standard_normal(n)

        model = LWCP(
            predictor=Ridge(alpha=1.0),
            weight_fn=InverseRootLeverageWeight(),
            ridge=1.0,
            leverage_method="approximate",
            n_components=10,
            alpha=0.05,
            random_state=0,
        )
        model.fit(X[:400], y[:400])
        y_pred, lower, upper = model.predict(X[400:])

        assert y_pred.shape == (100,)
        assert np.all(lower < upper)
        coverage = np.mean((y[400:] >= lower) & (y[400:] <= upper))
        assert coverage >= 0.80  # Generous bound for small test set

    def test_custom_lambda_weight(self):
        """A user-provided lambda should work as weight_fn."""
        rng = np.random.default_rng(42)
        n, p = 300, 5
        X = rng.standard_normal((n, p))
        y = X @ np.ones(p) + 0.1 * rng.standard_normal(n)

        model = LWCP(
            predictor=LinearRegression(),
            weight_fn=lambda h: np.exp(-h),
            alpha=0.1,
            random_state=42,
        )
        model.fit(X[:250], y[:250])
        y_pred, lower, upper = model.predict(X[250:])
        assert y_pred.shape == (50,)
        assert np.all(lower < upper)

    def test_power_law_weight(self):
        """PowerLawWeight should work in the full pipeline."""
        rng = np.random.default_rng(42)
        n, p = 300, 5
        X = rng.standard_normal((n, p))
        y = X @ np.ones(p) + 0.1 * rng.standard_normal(n)

        model = LWCP(
            predictor=LinearRegression(),
            weight_fn=PowerLawWeight(gamma=0.5),
            alpha=0.1,
            random_state=42,
        )
        model.fit(X[:250], y[:250])
        y_pred, lower, upper = model.predict(X[250:])
        assert np.all(lower < upper)

    def test_all_weight_functions_produce_valid_intervals(self):
        """Every built-in weight function should produce valid intervals."""
        rng = np.random.default_rng(42)
        n, p = 300, 5
        X = rng.standard_normal((n, p))
        y = X @ np.ones(p) + 0.1 * rng.standard_normal(n)

        weight_fns = [
            ConstantWeight(),
            InverseRootLeverageWeight(),
            PowerLawWeight(gamma=0.5),
            PowerLawWeight(gamma=1.0),
        ]

        for w in weight_fns:
            model = LWCP(
                predictor=LinearRegression(),
                weight_fn=w,
                alpha=0.1,
                random_state=42,
            )
            model.fit(X[:250], y[:250])
            y_pred, lower, upper = model.predict(X[250:])
            assert np.all(lower < upper), f"Failed for {w!r}"

    def test_different_alpha_levels(self):
        """Wider intervals for smaller alpha (higher confidence)."""
        rng = np.random.default_rng(42)
        n, p = 500, 5
        X = rng.standard_normal((n, p))
        y = X @ np.ones(p) + 0.5 * rng.standard_normal(n)

        widths = {}
        for alpha in [0.3, 0.1, 0.01]:
            model = LWCP(
                predictor=LinearRegression(),
                alpha=alpha,
                random_state=42,
            )
            model.fit(X[:400], y[:400])
            _, lower, upper = model.predict(X[400:])
            widths[alpha] = np.mean(upper - lower)

        # Smaller alpha → wider intervals
        assert widths[0.01] > widths[0.1] > widths[0.3]
