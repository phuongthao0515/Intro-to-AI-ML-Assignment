"""Reference implementations for Part 3 – logistic and softmax regression."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply the logistic sigmoid element-wise.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid outputs.
    """
    return 1 / (1 + np.exp(-z))
    # raise NotImplementedError("Implement _sigmoid.")


def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply a numerically stable softmax across rows.

    Args:
        z (np.ndarray): Logit matrix of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Probabilities for each class per sample.
    """
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / sum_exp_z
    # raise NotImplementedError("Implement _softmax.")


class LogisticRegression:
    """Binary logistic regression trained via batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1500,
        reg_strength: float = 0.0,
        random_state: int | None = 0,
    ) -> None:
        """
        Args:
            learning_rate (float): Step size for gradient updates (> 0).
            epochs (int): Number of passes over the training data (> 0).
            reg_strength (float): L2 regularisation strength (>= 0).
            random_state (int | None): Seed passed to NumPy default RNG.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if reg_strength < 0:
            raise ValueError("reg_strength cannot be negative.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y) -> None:
        """
        Train the classifier using batch gradient descent.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Binary labels of shape (n_samples,).
        """
        # 1. Convert X and y to arrays
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        n_samples, n_features = X.shape

        # 2. Initialize parameters
        self._initialize_parameters(n_features)

        # 3. Training loop
        for _ in range(self.epochs):
            # (a) Forward pass
            _, probs = self._forward(X)
            # (b) Backward pass
            grad_w, grad_b = self._backward(X, y, probs)
            # (c) Update parameters
            self._update(grad_w, grad_b)
        # raise NotImplementedError("Implement LogisticRegression.fit.")

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for each sample.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Probabilities for the positive class.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        # Convert to 2D numpy array
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Compute logits z = X @ w + b
        # Apply sigmoid to get P(y=1|x)
        _, probs = self._forward(X)
        # Return array of shape (n_samples,) with P(y=1)
        return probs.flatten()
        # raise NotImplementedError("Implement LogisticRegression.predict_proba.")

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels (0 or 1) using a 0.5 threshold.
        """
        proba = self.predict_proba(X)
        # proba is P(y=1), shape (n_samples,)
        return (proba >= 0.5).astype(int)
        # raise NotImplementedError("Implement LogisticRegression.predict.")

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and probabilities for the current parameters.
        """
        z = X @ self.weights + self.bias
        p = _sigmoid(z)
        return z, p
        # raise NotImplementedError("Implement LogisticRegression._forward.")

    def _backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients of the loss with respect to weights and bias.
        """
        n = X.shape[0]
        e = probs - y_true  # error: p - y
        grad_w = (1 / n) * (X.T @ e) + self.reg_strength * self.weights
        grad_b = (1 / n) * np.sum(e)
        return grad_w, grad_b
        # raise NotImplementedError("Implement LogisticRegression._backward.")

    def _update(self, grad_w: np.ndarray, grad_b: float) -> None:
        """
        Apply one gradient descent step.
        """
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b
        # raise NotImplementedError("Implement LogisticRegression._update.")

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialise weights from a small Gaussian and zero bias.
        """
        self.weights = self._rng.normal(0, 0.01, size=(n_features, 1))
        self.bias = 0.0
        # raise NotImplementedError("Implement LogisticRegression._initialize_parameters.")


class SoftmaxRegression:
    """Multiclass generalisation of logistic regression with softmax output."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 2000,
        reg_strength: float = 0.0,
        random_state: int | None = 0,
    ) -> None:
        """
        Args:
            learning_rate (float): Step size for gradient updates (> 0).
            epochs (int): Number of iterations (> 0).
            reg_strength (float): L2 penalty applied to weights (>= 0).
            random_state (int | None): Seed for reproducible initialisation.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if reg_strength < 0:
            raise ValueError("reg_strength cannot be negative.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y) -> None:
        """
        Train the model using gradient descent on the cross-entropy loss.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Class labels (hashable) of shape (n_samples,).
        """
        # Convert X and y to arrays
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Compute unique classes and store in self.classes_
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Create one-hot encoding Y_onehot
        # Map each label to its index in classes_
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[label] for label in y])
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_indices] = 1

        # Initialize parameters
        self._initialize_parameters(n_features, n_classes)

        # Training loop
        for _ in range(self.epochs):
            # Forward pass
            _, probs = self._forward(X)
            # Backward pass
            grad_w, grad_b = self._backward(X, Y_onehot, probs)
            # Update parameters
            self._update(grad_w, grad_b)
        # raise NotImplementedError("Implement SoftmaxRegression.fit.")

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for each sample.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Shape (n_samples, n_classes) with row sums equal to 1.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        # Convert to 2D numpy array
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Compute logits Z = X @ W + b
        # Apply softmax to get probabilities
        _, probs = self._forward(X)
        # Return array of shape (n_samples, n_classes)
        return probs
        # raise NotImplementedError("Implement SoftmaxRegression.predict_proba.")

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels via argmax over predicted probabilities.
        """
        proba = self.predict_proba(X)
        # Get index of max probability for each sample
        indices = np.argmax(proba, axis=1)
        # Map indices back to original class labels
        return self.classes_[indices]
        # raise NotImplementedError("Implement SoftmaxRegression.predict.")

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and softmax probabilities for the current parameters.
        """
        Z = X @ self.weights + self.bias
        P = _softmax(Z)
        return Z, P
        # raise NotImplementedError("Implement SoftmaxRegression._forward.")

    def _backward(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and bias given softmax probabilities.
        """
        n = X.shape[0]
        E = probs - y_onehot  
        grad_w = (1 / n) * (X.T @ E) + self.reg_strength * self.weights
        grad_b = (1 / n) * np.sum(E, axis=0)
        return grad_w, grad_b
        # raise NotImplementedError("Implement SoftmaxRegression._backward.")

    def _update(self, grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        """
        Apply one gradient descent update.
        """
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b
        # raise NotImplementedError("Implement SoftmaxRegression._update.")

    def _initialize_parameters(self, n_features: int, n_classes: int) -> None:
        """
        Initialise weights and biases for a given feature/class configuration.
        """
        self.weights = self._rng.normal(0, 0.01, size=(n_features, n_classes))
        self.bias = np.zeros(n_classes)
        # raise NotImplementedError("Implement SoftmaxRegression._initialize_parameters.")

