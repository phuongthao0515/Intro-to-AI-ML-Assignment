"""Closed-form linear regression with optional intercept and L2 regularisation."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept: bool = True, reg_strength: float = 0.0) -> None:
        """
        Closed-form linear regression solver with optional L2 regularization.

        Args:
            fit_intercept (bool): Whether to augment X with a bias column.
            reg_strength (float): Ridge penalty applied to coefficients
                (intercept excluded).
        """
        if reg_strength < 0:
            raise ValueError("reg_strength cannot be negative.")
        self.fit_intercept = fit_intercept
        self.reg_strength = reg_strength
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Solve the normal equations and store weights/intercept.

        Args:
            X (array-like): Training design matrix (n_samples, n_features).
            y (array-like): Target vector (n_samples,) or (n_samples, 1).

        Returns:
            LinearRegression: Fitted estimator (self).

        Raises:
            ValueError: If X and y have different numbers of samples.
        """
        X_array = self._ensure_2d(X)
        y_array = np.asarray(y, dtype=float).reshape(-1,1)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if self.fit_intercept:
            X_array = self._augment_features(X_array)
        XTX = X_array.T @ X_array
        XTy = X_array.T @ y_array
        n_columns = XTX.shape[0]
        reg_matrix = self.reg_strength * np.eye(n_columns)
        if self.fit_intercept:
            reg_matrix[0, 0] = 0.0
        try:
            weights = np.linalg.solve(XTX + reg_matrix, XTy)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(XTX + reg_matrix) @ XTy
        if self.fit_intercept:
            self.intercept_ = weights[0, 0]
            self.coef_ = weights[1:, 0]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights[:, 0]
        return self
        # raise NotImplementedError("Implement LinearRegression.fit.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for new samples.

        Args:
            X (array-like): Input matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: 1-D array of predictions.

        Raises:
            RuntimeError: If called before fit.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_array = self._ensure_2d(X)
        return X_array @ self.coef_ + self.intercept_
        # raise NotImplementedError("Implement LinearRegression.predict.")

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """
        Optionally prepend a bias column to X.
        """
        X = self._ensure_2d(X)
        if not self.fit_intercept:
            return X
        return np.hstack((np.ones((X.shape[0], 1), dtype=float), X))
        # raise NotImplementedError("Implement LinearRegression._augment_features.")

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        """
        Coerce the input into a 2-D NumPy array of floats.
        """
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim != 2:
            raise ValueError("Input array must be 1D or 2D.")
        return X_array
        raise NotImplementedError("Implement LinearRegression._ensure_2d.")

