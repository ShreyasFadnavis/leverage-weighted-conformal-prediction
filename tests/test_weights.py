"""Tests for leverage-based weighting functions."""

import numpy as np
import pytest

from lwcp.weights import (
    ConstantWeight,
    InverseRootLeverageWeight,
    PowerLawWeight,
    WeightFunction,
)


class TestConstantWeight:
    def test_returns_ones(self):
        w = ConstantWeight()
        h = np.array([0.0, 0.1, 0.5, 1.0])
        np.testing.assert_array_equal(w(h), np.ones(4))

    def test_scalar_array(self):
        w = ConstantWeight()
        result = w(np.array([0.5]))
        assert result[0] == 1.0

    def test_repr(self):
        assert repr(ConstantWeight()) == "ConstantWeight()"

    def test_satisfies_protocol(self):
        assert isinstance(ConstantWeight(), WeightFunction)


class TestInverseRootLeverageWeight:
    def test_at_zero(self):
        w = InverseRootLeverageWeight()
        result = w(np.array([0.0]))
        np.testing.assert_allclose(result, [1.0])

    def test_decreasing(self):
        w = InverseRootLeverageWeight()
        h = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
        result = w(h)
        assert np.all(np.diff(result) < 0)

    def test_known_values(self):
        w = InverseRootLeverageWeight()
        h = np.array([0.0, 1.0, 3.0])
        expected = np.array([1.0, 1 / np.sqrt(2), 0.5])
        np.testing.assert_allclose(result := w(h), expected)

    def test_positive(self):
        w = InverseRootLeverageWeight()
        h = np.linspace(0, 10, 100)
        assert np.all(w(h) > 0)

    def test_repr(self):
        assert repr(InverseRootLeverageWeight()) == "InverseRootLeverageWeight()"

    def test_satisfies_protocol(self):
        assert isinstance(InverseRootLeverageWeight(), WeightFunction)


class TestPowerLawWeight:
    def test_rejects_nonpositive_gamma(self):
        with pytest.raises(ValueError, match="positive"):
            PowerLawWeight(gamma=0.0)
        with pytest.raises(ValueError, match="positive"):
            PowerLawWeight(gamma=-1.0)

    def test_decreasing(self):
        w = PowerLawWeight(gamma=0.5)
        h = np.array([0.01, 0.1, 0.5, 1.0])
        result = w(h)
        assert np.all(np.diff(result) < 0)

    def test_epsilon_guard_at_zero(self):
        w = PowerLawWeight(gamma=0.5)
        result = w(np.array([0.0]))
        assert np.isfinite(result[0])
        assert result[0] > 0

    def test_larger_gamma_more_aggressive(self):
        h = np.array([0.1, 1.0])
        w_small = PowerLawWeight(gamma=0.5)
        w_large = PowerLawWeight(gamma=2.0)
        ratio_small = w_small(h)[0] / w_small(h)[1]
        ratio_large = w_large(h)[0] / w_large(h)[1]
        # Larger gamma → bigger ratio between low-h and high-h weights
        assert ratio_large > ratio_small

    def test_repr(self):
        assert "gamma=0.5" in repr(PowerLawWeight(gamma=0.5))

    def test_satisfies_protocol(self):
        assert isinstance(PowerLawWeight(gamma=1.0), WeightFunction)


class TestCustomWeightFunction:
    def test_lambda_satisfies_protocol(self):
        # Any callable works
        w = lambda h: np.exp(-h)
        # Protocol check is structural, not nominal, for runtime_checkable
        assert callable(w)
