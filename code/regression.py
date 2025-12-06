"""Reference implementations for Part 2 – non-linear regression."""

from __future__ import annotations

import numpy as np

from linear_regression import LinearRegression
from polynomial_transformer import PolynomialTransformer


def _ensure_column(vector) -> np.ndarray:
    """
    Convert a 1-D array-like input into a single-column matrix.

    Args:
        vector (array-like): Input values.

    Returns:
        np.ndarray: Shape (n_samples, 1).

    Raises:
        ValueError: If the input cannot be coerced into a column vector.
    """
    arr = np.asarray(vector, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    raise ValueError("Input must be 1D or 2D")
    # raise NotImplementedError("Implement _ensure_column.")


def _stack_features(*features) -> np.ndarray:
    """
    Stack multiple feature vectors into a single 2-D array.

    Args:
        *features (array-like): Vectors of equal length.

    Returns:
        np.ndarray: Matrix whose columns correspond to the input vectors.

    Raises:
        ValueError: If feature vectors have different lengths.
    """
    columns = [_ensure_column(f) for f in features]
    if columns:
        n = columns[0].shape[0]
        for col in columns[1:]:
            if col.shape[0] != n:
                raise ValueError("Feature vectors have different lengths.")
    return np.hstack(columns)
    # raise NotImplementedError("Implement _stack_features.")


def polynomial_features(x, degree: int) -> np.ndarray:
    """
    Create polynomial design matrix for a single predictor.

    Args:
        x (array-like): Predictor values.
        degree (int): Maximum polynomial degree (>= 0).

    Returns:
        np.ndarray: Polynomial feature matrix including bias column.
    """
    x_col = _ensure_column(x)
    transformer = PolynomialTransformer(degree=degree, include_bias=True)
    return transformer.fit_transform(x_col)
    # raise NotImplementedError("Implement polynomial_features.")


def fit_polynomial_regression(
    x,
    y,
    degree: int = 2,
    learning_rate: float = 0.01,
    epochs: int = 2000,
) -> np.ndarray:
    """
    Fit polynomial regression coefficients via closed-form least squares.

    Args:
        x (array-like): Predictor values.
        y (array-like): Target values.
        degree (int): Polynomial degree.
        learning_rate (float): Ignored; kept for parity with student API.
        epochs (int): Ignored; kept for parity with student API.

    Returns:
        np.ndarray: Learned weights (including bias).
    """
    X_poly = polynomial_features(x, degree)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)
    return model.coef_
    # raise NotImplementedError("Implement fit_polynomial_regression.")


def predict_polynomial(
    x,
    weights,
) -> np.ndarray:
    """
    Evaluate a polynomial model at the provided inputs.

    Args:
        x (array-like): Predictor values.
        weights (array-like): Weight vector including bias.

    Returns:
        np.ndarray: Predicted responses.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float).ravel()
    if len(weights) == 0:
        return np.zeros_like(x_arr)
    degree = len(weights) - 1
    X_poly = polynomial_features(x, degree)
    return (X_poly @ weights).ravel()
    # raise NotImplementedError("Implement predict_polynomial.")


def fit_surface_regression(
    x1,
    x2,
    y,
    learning_rate: float = 0.01,
    epochs: int = 2500,
) -> np.ndarray:
    """
    Fit a quadratic surface regression with two predictors.

    Args:
        x1 (array-like): First predictor.
        x2 (array-like): Second predictor.
        y (array-like): Target values.
        learning_rate (float): Ignored; API compatibility only.
        epochs (int): Ignored; API compatibility only.

    Returns:
        np.ndarray: Learned weight vector.
    """
    X = _stack_features(x1, x2)
    polynomial_transformer = PolynomialTransformer(degree=2, include_bias=True)
    X_design = polynomial_transformer.fit_transform(X)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_design, y)
    return model.coef_
    # raise NotImplementedError("Implement fit_surface_regression.")


def predict_surface(
    x1,
    x2,
    weights,
) -> np.ndarray:
    """
    Predict outputs from a quadratic surface regression model.

    Args:
        x1 (array-like): First predictor.
        x2 (array-like): Second predictor.
        weights (array-like): Weight vector learned by fit_surface_regression.

    Returns:
        np.ndarray: Predicted responses.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    X = _stack_features(x1, x2)
    if len(weights) == 0:
        return np.zeros(X.shape[0])
    transformer = PolynomialTransformer(degree=2, include_bias=True)
    X_design = transformer.fit_transform(X)
    return (X_design @ weights).ravel()
    # raise NotImplementedError("Implement predict_surface.")

