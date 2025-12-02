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
    arr = np.asarray(vector)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    else:
        raise ValueError("Input cannot be coerced into a column vector.")
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
    # Check all have the same length
    if len(columns) > 0:
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
    # 1. Use _ensure_column to create a column vector for x
    x_col = _ensure_column(x)
    # 2. Construct a PolynomialTransformer with the desired degree and include_bias=True
    transformer = PolynomialTransformer(degree=degree, include_bias=True)
    # 3. Fit the transformer on x and return transform(x)
    transformer.fit(x_col)
    return transformer.transform(x_col)
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
    # 1. Build features with polynomial_features
    X_poly = polynomial_features(x, degree)
    # 2. Instantiate a LinearRegression with fit_intercept=False (bias already in feature matrix)
    model = LinearRegression(fit_intercept=False)
    # 3. Fit the model to (X_poly, y) and return the learned weight vector
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
    weights = np.asarray(weights)
    degree = len(weights) - 1
    X_poly = polynomial_features(x, degree)
    return X_poly @ weights
    raise NotImplementedError("Implement predict_polynomial.")


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
    x1 = _ensure_column(x1)
    x2 = _ensure_column(x2)
    X1_squared = x1 ** 2
    X2_squared = x2 ** 2
    X1_X2 = x1 * x2
    X_design = _stack_features(np.ones(x1.shape[0]), x1, x2, X1_squared, X1_X2, X2_squared)
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
    weights = np.asarray(weights)
    x1 = _ensure_column(x1)
    x2 = _ensure_column(x2)
    X1_squared = x1 ** 2
    X2_squared = x2 ** 2
    X1_X2 = x1 * x2
    X_design = _stack_features(np.ones(x1.shape[0]), x1, x2, X1_squared, X1_X2, X2_squared)
    return X_design @ weights
    # raise NotImplementedError("Implement predict_surface.")

