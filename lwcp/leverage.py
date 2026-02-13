"""Leverage score computation via SVD."""

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


class FeatureSpaceLeverageComputer:
    """Computes leverage scores in a learned feature space.

    Given a feature extractor φ: X -> Φ(X), computes leverage as
    h(x) = φ(x)^T (Φ^T Φ + λI)^{-1} φ(x) where Φ is the feature matrix
    of the training data. This enables LWCP with non-linear predictors
    (e.g., neural networks via last-layer features, kernel methods).

    Parameters
    ----------
    feature_extractor : callable
        Maps ndarray (n, p) -> ndarray (n, d) where d is the feature dimension.
    ridge : float, default=0.0
        Ridge regularization for the feature-space Gram matrix.
    """

    def __init__(
        self,
        feature_extractor: Callable[[NDArray], NDArray],
        ridge: float = 0.0,
    ):
        self.feature_extractor = feature_extractor
        self.ridge = ridge
        self._gram_inv: Optional[NDArray] = None

    def fit(self, X_train: NDArray[np.floating]) -> "FeatureSpaceLeverageComputer":
        """Compute (Φ^T Φ + λI)^{-1} from training features.

        Parameters
        ----------
        X_train : ndarray of shape (n_train, p)

        Returns
        -------
        self
        """
        Phi = np.asarray(self.feature_extractor(X_train), dtype=np.float64)
        _, s, Vt = linalg.svd(Phi, full_matrices=False)
        d = s**2 + self.ridge
        inv_d = np.where(d > 1e-15, 1.0 / d, 0.0)
        self._gram_inv = (Vt.T * inv_d) @ Vt
        return self

    def leverage_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute feature-space leverage scores.

        h(x) = φ(x)^T (Φ_train^T Φ_train + λI)^{-1} φ(x)

        Parameters
        ----------
        X : ndarray of shape (m, p)

        Returns
        -------
        h : ndarray of shape (m,)
        """
        if self._gram_inv is None:
            raise RuntimeError("FeatureSpaceLeverageComputer not fitted. Call fit() first.")
        Phi = np.asarray(self.feature_extractor(X), dtype=np.float64)
        PhiG = Phi @ self._gram_inv
        h = np.sum(PhiG * Phi, axis=1)
        return h


def mlp_feature_extractor(mlp_model):
    """Create a feature extractor from a fitted sklearn MLPRegressor.

    Extracts the penultimate (last hidden) layer activations.

    Parameters
    ----------
    mlp_model : sklearn.neural_network.MLPRegressor
        A fitted MLP model.

    Returns
    -------
    extractor : callable
        Maps X (n, p) -> Φ(X) (n, d) where d is the last hidden layer size.
    """
    def _extract(X):
        X = np.asarray(X, dtype=np.float64)
        activations = [X]
        for i, (W, b) in enumerate(zip(mlp_model.coefs_, mlp_model.intercepts_)):
            z = activations[-1] @ W + b
            if i < len(mlp_model.coefs_) - 1:
                # Hidden layer: apply activation
                z = np.maximum(z, 0)  # relu (default for MLPRegressor)
            activations.append(z)
        # Return penultimate layer (last hidden layer activations)
        return activations[-2]
    return _extract


class LeverageComputer:
    """Computes leverage scores from a design matrix.

    Supports exact computation via SVD and approximate computation
    via randomized SVD. Optionally uses ridge regularization.

    Parameters
    ----------
    ridge : float, default=0.0
        Ridge regularization parameter lambda.
        When 0, computes standard leverage: h_i = x_i^T (X^T X)^{-1} x_i.
        When > 0, computes ridge leverage: h_i = x_i^T (X^T X + lambda I)^{-1} x_i.
    method : str, default="exact"
        "exact" uses full SVD, "approximate" uses randomized SVD.
    n_components : int or None, default=None
        Number of components for randomized SVD. Required when
        method="approximate". Defaults to min(n, p) if None.
    random_state : int or None, default=None
        Random seed for reproducibility of randomized SVD.
    """

    def __init__(
        self,
        ridge: float = 0.0,
        method: str = "exact",
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.ridge = ridge
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self._gram_inv: Optional[NDArray] = None

    def fit(self, X_train: NDArray[np.floating]) -> "LeverageComputer":
        """Compute and store (X_train^T X_train + lambda I)^{-1}.

        Parameters
        ----------
        X_train : ndarray of shape (n_train, p)

        Returns
        -------
        self
        """
        X = np.asarray(X_train, dtype=np.float64)
        n, p = X.shape

        if self.method == "exact":
            # X = U @ diag(s) @ Vt
            # X^T X = V @ diag(s^2) @ V^T
            # (X^T X + lambda I)^{-1} = V @ diag(1/(s^2 + lambda)) @ V^T
            _, s, Vt = linalg.svd(X, full_matrices=False)
            d = s**2 + self.ridge
            # Pseudoinverse: zero singular values → 0 in inverse (not inf)
            inv_d = np.where(d > 1e-15, 1.0 / d, 0.0)
            self._gram_inv = (Vt.T * inv_d) @ Vt

        elif self.method == "approximate":
            from sklearn.utils.extmath import randomized_svd

            k = self.n_components if self.n_components is not None else min(n, p)
            _, s, Vt = randomized_svd(X, n_components=k, random_state=self.random_state)
            d = s**2 + self.ridge
            inv_d = np.where(d > 1e-15, 1.0 / d, 0.0)
            self._gram_inv = (Vt.T * inv_d) @ Vt

        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'exact' or 'approximate'.")

        return self

    def leverage_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute leverage scores for rows of X.

        h_i = x_i^T (X_train^T X_train + lambda I)^{-1} x_i

        Parameters
        ----------
        X : ndarray of shape (m, p)

        Returns
        -------
        h : ndarray of shape (m,)
        """
        if self._gram_inv is None:
            raise RuntimeError("LeverageComputer has not been fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        # h_i = sum_j (X @ gram_inv)_{ij} * X_{ij}  — avoids forming m x m matrix
        XG = X @ self._gram_inv
        h = np.sum(XG * X, axis=1)
        return h
