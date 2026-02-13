"""Data-Generating Processes for LWCP experiments.

Each DGP returns (X_train, y_train, X_cal, y_cal, X_test, y_test, metadata).
"""

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def _compute_leverage(X_ref: NDArray, X_query: NDArray) -> NDArray:
    """Compute leverage scores of X_query w.r.t. X_ref design matrix."""
    _, s, Vt = linalg.svd(X_ref, full_matrices=False)
    inv_d = np.where(s > 1e-15, 1.0 / s**2, 0.0)
    gram_inv = (Vt.T * inv_d) @ Vt
    XG = X_query @ gram_inv
    return np.sum(XG * X_query, axis=1)


def _make_covariance(p: int) -> NDArray:
    """Decaying eigenvalue covariance: Sigma_jj = 1/j."""
    return np.diag(1.0 / np.arange(1, p + 1))


def dgp_textbook(
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    p: int = 30,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """DGP 1: Gaussian X, leverage-dependent heteroscedasticity.

    Y = X^T beta + eps, eps ~ N(0, sigma^2 (1+h(X)))
    where g(h) = 1 + h matches the OLS prediction variance structure.
    With p/n_train = 0.1, estimation error is small relative to
    noise heteroscedasticity, putting us in the asymptotic regime
    where InverseRootLeverageWeight is near-optimal.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test
    Sigma = _make_covariance(p)
    L = np.linalg.cholesky(Sigma)
    X_all = rng.standard_normal((n_total, p)) @ L.T
    beta = rng.standard_normal(p)

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    # Compute leverage w.r.t. training design for all points
    h_train = _compute_leverage(X_train, X_train)
    h_cal = _compute_leverage(X_train, X_cal)
    h_test = _compute_leverage(X_train, X_test)

    # g(h) = 1+h: matches InverseRootLeverageWeight w(h) = (1+h)^{-1/2}
    g = lambda h: 1.0 + h

    y_train = X_train @ beta + sigma * np.sqrt(g(h_train)) * rng.standard_normal(n_train)
    y_cal = X_cal @ beta + sigma * np.sqrt(g(h_cal)) * rng.standard_normal(n_cal)
    y_test = X_test @ beta + sigma * np.sqrt(g(h_test)) * rng.standard_normal(n_test)

    metadata = {
        "dgp_name": "Textbook (g(h)=1+h)",
        "g_func": g,
        "beta": beta,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


def dgp_heavy_tailed(
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    p: int = 30,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """DGP 2: Heavy-tailed errors (t_3), leverage-dependent heteroscedasticity.

    Y = X^T beta + eps, eps ~ t_3 * sqrt(1+h(X)).
    """
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test
    Sigma = _make_covariance(p)
    L = np.linalg.cholesky(Sigma)
    X_all = rng.standard_normal((n_total, p)) @ L.T
    beta = rng.standard_normal(p)

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    h_train = _compute_leverage(X_train, X_train)
    h_cal = _compute_leverage(X_train, X_cal)
    h_test = _compute_leverage(X_train, X_test)

    g = lambda h: 1.0 + h
    df = 3
    # t_3 has variance df/(df-2) = 3; normalize so marginal variance of base noise is 1
    t_scale = np.sqrt((df - 2) / df)

    y_train = X_train @ beta + sigma * np.sqrt(g(h_train)) * rng.standard_t(df, n_train) * t_scale
    y_cal = X_cal @ beta + sigma * np.sqrt(g(h_cal)) * rng.standard_t(df, n_cal) * t_scale
    y_test = X_test @ beta + sigma * np.sqrt(g(h_test)) * rng.standard_t(df, n_test) * t_scale

    metadata = {
        "dgp_name": "Heavy-tailed (t_3, g(h)=1+h)",
        "g_func": g,
        "beta": beta,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


def dgp_polynomial(
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    degree: int = 8,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """DGP 3: Polynomial regression. X in R, polynomial features up to degree.

    Leverage is naturally high at boundaries of X range. With degree=8
    and n_train=100, high-degree polynomial features create wide leverage spread.
    g(h) = 1 + h matches InverseRootLeverageWeight.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test

    # Raw X ~ Uniform(-1, 1)
    x_raw = rng.uniform(-1, 1, n_total)

    # Polynomial features: [x, x^2, ..., x^degree]
    X_all = np.column_stack([x_raw**k for k in range(1, degree + 1)])
    p = degree

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    h_train = _compute_leverage(X_train, X_train)
    h_cal = _compute_leverage(X_train, X_cal)
    h_test = _compute_leverage(X_train, X_test)

    g = lambda h: 1.0 + h

    # True function: sum of polynomial basis
    beta = rng.standard_normal(p)
    y_train = X_train @ beta + sigma * np.sqrt(g(h_train)) * rng.standard_normal(n_train)
    y_cal = X_cal @ beta + sigma * np.sqrt(g(h_cal)) * rng.standard_normal(n_cal)
    y_test = X_test @ beta + sigma * np.sqrt(g(h_test)) * rng.standard_normal(n_test)

    metadata = {
        "dgp_name": f"Polynomial (deg={degree}, g(h)=1+h)",
        "g_func": g,
        "beta": beta,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
        "x_raw_test": x_raw[n_train + n_cal :],
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


def dgp_homoscedastic(
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    p: int = 30,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """DGP 4: Null case — homoscedastic errors, g(h) = 1.

    LWCP should perform roughly the same as vanilla CP.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test
    Sigma = _make_covariance(p)
    L = np.linalg.cholesky(Sigma)
    X_all = rng.standard_normal((n_total, p)) @ L.T
    beta = rng.standard_normal(p)

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    h_test = _compute_leverage(X_train, X_test)
    h_cal = _compute_leverage(X_train, X_cal)

    y_train = X_train @ beta + sigma * rng.standard_normal(n_train)
    y_cal = X_cal @ beta + sigma * rng.standard_normal(n_cal)
    y_test = X_test @ beta + sigma * rng.standard_normal(n_test)

    metadata = {
        "dgp_name": "Homoscedastic (g(h)=1)",
        "g_func": lambda h: np.ones_like(h),
        "beta": beta,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


def dgp_adversarial(
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    p: int = 30,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """DGP 5: Adversarial — heteroscedasticity depends on ||X||, not leverage.

    Var(Y|X) = sigma^2 (1 + ||X||^2 / p). This is NOT well-captured by leverage
    when the covariance has decaying eigenvalues.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test
    Sigma = _make_covariance(p)
    L = np.linalg.cholesky(Sigma)
    X_all = rng.standard_normal((n_total, p)) @ L.T
    beta = rng.standard_normal(p)

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    h_test = _compute_leverage(X_train, X_test)
    h_cal = _compute_leverage(X_train, X_cal)

    # Heteroscedasticity based on norm, not leverage
    norm_sq_train = np.sum(X_train**2, axis=1) / p
    norm_sq_cal = np.sum(X_cal**2, axis=1) / p
    norm_sq_test = np.sum(X_test**2, axis=1) / p

    g_norm = lambda ns: 1.0 + ns

    y_train = X_train @ beta + sigma * np.sqrt(g_norm(norm_sq_train)) * rng.standard_normal(n_train)
    y_cal = X_cal @ beta + sigma * np.sqrt(g_norm(norm_sq_cal)) * rng.standard_normal(n_cal)
    y_test = X_test @ beta + sigma * np.sqrt(g_norm(norm_sq_test)) * rng.standard_normal(n_test)

    metadata = {
        "dgp_name": "Adversarial (Var ~ ||X||^2, not h)",
        "g_func": g_norm,
        "beta": beta,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


def dgp_custom_g(
    g_func,
    g_name: str = "custom",
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    p: int = 30,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """Generic DGP with user-specified g(h) for the heteroscedasticity sweep."""
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test
    Sigma = _make_covariance(p)
    L = np.linalg.cholesky(Sigma)
    X_all = rng.standard_normal((n_total, p)) @ L.T
    beta = rng.standard_normal(p)

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    h_train = _compute_leverage(X_train, X_train)
    h_cal = _compute_leverage(X_train, X_cal)
    h_test = _compute_leverage(X_train, X_test)

    y_train = X_train @ beta + sigma * np.sqrt(g_func(h_train)) * rng.standard_normal(n_train)
    y_cal = X_cal @ beta + sigma * np.sqrt(g_func(h_cal)) * rng.standard_normal(n_cal)
    y_test = X_test @ beta + sigma * np.sqrt(g_func(h_test)) * rng.standard_normal(n_test)

    metadata = {
        "dgp_name": f"Custom ({g_name})",
        "g_func": g_func,
        "beta": beta,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


def dgp_nonlinear(
    n_train: int = 300,
    n_cal: int = 500,
    n_test: int = 500,
    p: int = 30,
    sigma: float = 1.0,
    random_state: int = 42,
):
    """DGP 6: Non-linear response with leverage-dependent heteroscedasticity.

    Y = sum_j sin(X_j) + eps, eps ~ N(0, sigma^2 (1+h(X))).
    The true function is non-linear, so OLS is misspecified but
    non-linear predictors (RF, MLP) can capture the signal.
    Leverage still captures the geometric structure of X.
    """
    rng = np.random.default_rng(random_state)
    n_total = n_train + n_cal + n_test
    Sigma = _make_covariance(p)
    L = np.linalg.cholesky(Sigma)
    X_all = rng.standard_normal((n_total, p)) @ L.T

    X_train = X_all[:n_train]
    X_cal = X_all[n_train : n_train + n_cal]
    X_test = X_all[n_train + n_cal :]

    h_train = _compute_leverage(X_train, X_train)
    h_cal = _compute_leverage(X_train, X_cal)
    h_test = _compute_leverage(X_train, X_test)

    g = lambda h: 1.0 + h

    # Non-linear true function: sum of sines
    f = lambda X: np.sum(np.sin(X), axis=1)

    y_train = f(X_train) + sigma * np.sqrt(g(h_train)) * rng.standard_normal(n_train)
    y_cal = f(X_cal) + sigma * np.sqrt(g(h_cal)) * rng.standard_normal(n_cal)
    y_test = f(X_test) + sigma * np.sqrt(g(h_test)) * rng.standard_normal(n_test)

    metadata = {
        "dgp_name": "Non-linear (f=Σsin, g(h)=1+h)",
        "g_func": g,
        "sigma": sigma,
        "h_test": h_test,
        "h_cal": h_cal,
    }
    return X_train, y_train, X_cal, y_cal, X_test, y_test, metadata


# Registry for convenience
ALL_DGPS = {
    "textbook": dgp_textbook,
    "heavy_tailed": dgp_heavy_tailed,
    "polynomial": dgp_polynomial,
    "homoscedastic": dgp_homoscedastic,
    "adversarial": dgp_adversarial,
    "nonlinear": dgp_nonlinear,
}
