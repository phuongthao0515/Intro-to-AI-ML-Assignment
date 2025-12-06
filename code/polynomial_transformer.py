"""Reusable polynomial feature transformer supporting n-dimensional inputs."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class PolynomialTransformer:
    """
    Generate polynomial feature expansions similar to sklearn's PolynomialFeatures.

    Attributes
    ----------
    degree : int
        Maximum total degree of the polynomial terms.
    include_bias : bool
        Whether to prepend a column of ones.
    n_features_in_ : int | None
        Number of input features seen during fit.
    combinations_ : list[tuple[int, ...]]
        Cached index combinations for generating monomials (excluding the bias term).
    """

    def __init__(self, degree: int = 2, include_bias: bool = True) -> None:
        """
        Create a transformer that expands inputs into polynomial feature space.

        Args:
            degree (int): Maximum total polynomial degree (>= 0).
            include_bias (bool): When True, prepend a constant column of ones.
        """
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        self.degree = degree
        self.include_bias = include_bias
        self.n_features_in_: int | None = None
        self.combinations_: List[Tuple[int, ...]] | None = None

    def fit(self, X: Sequence[Sequence[float]]) -> "PolynomialTransformer":
        """
        Learn the dimensionality of the input and cache index combinations.

        Args:
            X (array-like): Training data with shape (n_samples, n_features).

        Returns:
            PolynomialTransformer: The fitted transformer (self).
        """
        X_array = self._validate_input(X)
        self.n_features_in_ = X_array.shape[1]
        self.combinations_ = self._generate_combinations(self.n_features_in_)
        return self
        # raise NotImplementedError("Implement PolynomialTransformer.fit.")

    def transform(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Apply the learned polynomial expansion to new data.

        Args:
            X (array-like): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Dense design matrix containing all polynomial terms.

        Raises:
            ValueError: If transform is called with a different feature count
                than was seen during fit.
        """
        X_array = self._validate_input(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit.")
        n_samples = X_array.shape[0]
        columns: List[np.ndarray] = []
        if self.include_bias:
            columns.append(np.ones((n_samples, 1), dtype=float))
        for combination in self.combinations_:
            col = np.ones(n_samples)
            for index in combination:
                col *= X_array[:, index]
            columns.append(col.reshape(-1, 1))
        if not columns:
            return np.empty((n_samples, 0), dtype=float)
        return np.hstack(columns)
        # raise NotImplementedError("Implement PolynomialTransformer.transform.")

    def fit_transform(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Fit the transformer on X and immediately return the transformed matrix.
        """
        return self.fit(X).transform(X)
        # raise NotImplementedError("Implement PolynomialTransformer.fit_transform.")

    def _generate_combinations(self, n_features: int) -> List[Tuple[int, ...]]:
        """
        Enumerate all index tuples representing monomials up to self.degree.
        """
        combinations: List[Tuple[int, ...]] = []
        for degree in range(1, self.degree + 1):
            combination = combinations_with_replacement(range(n_features), degree)
            combinations.extend(combination)
        return combinations
        # raise NotImplementedError("Implement PolynomialTransformer._generate_combinations.")

    @staticmethod
    def _validate_input(X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Ensure X can be interpreted as a 2-D NumPy array of floats.

        Returns:
            np.ndarray: Copy/view of X with shape (n_samples, n_features).

        Raises:
            ValueError: If X cannot be reshaped into 2 dimensions.
        """
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim != 2:
            raise ValueError("Input X must be a 2-D array.")
        return X_array
        # raise NotImplementedError("Implement PolynomialTransformer._validate_input.")

