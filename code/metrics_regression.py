"""Reference implementation for regression metrics (Part 1B)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _prepare_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert regression targets/predictions into aligned 1-D float arrays.

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pair of flattened float arrays.

    Raises:
        ValueError: If the arrays have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return np.asarray(y_true).flatten().astype(float), np.asarray(y_pred).flatten().astype(float)
    # raise NotImplementedError("Implement _prepare_inputs.")


def mean_absolute_error(y_true, y_pred) -> float:
    """
    Mean absolute error (MAE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Average absolute deviation between prediction and truth.
    """
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true_arr - y_pred_arr))
    # raise NotImplementedError("Implement mean_absolute_error.")


def mean_squared_error(y_true, y_pred) -> float:
    """
    Mean squared error (MSE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Average squared deviation between prediction and truth.
    """
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    return np.mean((y_true_arr - y_pred_arr) ** 2)
    # raise NotImplementedError("Implement mean_squared_error.")


def root_mean_squared_error(y_true, y_pred) -> float:
    """
    Root mean squared error (RMSE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Square root of the mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
    # raise NotImplementedError("Implement root_mean_squared_error.")


def r2_score(y_true, y_pred) -> float:
    """
    Coefficient of determination (R²).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: R² score, 1.0 for perfect predictions.
    """
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    ss_total = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    ss_residual = np.sum((y_true_arr - y_pred_arr) ** 2)
    if ss_total == 0:
        return 0.0
    return 1 - (ss_residual / ss_total)
    # raise NotImplementedError("Implement r2_score.")


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """
    Aggregate common regression metrics into a dictionary.

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        Dict[str, float]: Keys "mae", "mse", "rmse", and "r2".
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    # raise NotImplementedError("Implement regression_report.")

