"""Reference implementation for classification metrics (Part 1A)."""

from __future__ import annotations

import numpy as np


def _safe_divide(num: float, denom: float) -> float:
    """
    Divide two floats safely, returning 0.0 when the denominator is zero.

    Args:
        num (float): Numerator.
        denom (float): Denominator.

    Returns:
        float: num / denom when denom != 0, otherwise 0.0.
    """
    return 0.0 if denom == 0 else float(num / denom)
    # raise NotImplementedError("Implement _safe_divide.")


def _prepare_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classification targets/predictions into 1-D NumPy arrays.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pair of flattened arrays (y_true, y_pred).

    Raises:
        ValueError: If the arrays do not share the same length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    # raise NotImplementedError("Implement _prepare_inputs.")


def _resolve_labels(labels, y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> list:
    """
    Determine the ordered label set used to build the confusion matrix.

    Args:
        labels (array-like | None): Explicit label ordering or None to infer.
        y_true_arr (np.ndarray): Flattened true labels.
        y_pred_arr (np.ndarray): Flattened predicted labels.

    Returns:
        list: Ordered list of unique labels.

    Raises:
        ValueError: If the final label list is empty.
    """
    if labels is not None:
        result = list(labels)
        if not result:
            raise ValueError("No labels found.")
        return result
    combined = np.concatenate([y_true_arr, y_pred_arr])
    unique_labels = []
    for label in combined:
        if label not in unique_labels:
            unique_labels.append(label)
    if not unique_labels:
        raise ValueError("No labels found.")
    return unique_labels


def confusion_matrix(
    y_true,
    y_pred,
    labels=None,
) -> np.ndarray:
    """
    Build the confusion matrix for multi-class classification.

    Args:
        y_true (array-like): True labels, convertible to a 1-D NumPy array.
        y_pred (array-like): Predicted labels, same length as `y_true`.
        labels (array-like | None): Optional ordered list of label values. When
            None, the union of labels from y_true and y_pred (in encounter order)
            is used. Every element must be hashable.

    Returns:
        np.ndarray: Square matrix of shape (n_classes, n_classes) containing
            integer counts. Rows correspond to true labels and columns to
            predicted labels.

    Example:
        >>> confusion_matrix(["cat", "dog"], ["cat", "cat"], labels=["cat", "dog"])
        array([[1, 0],
               [1, 0]])
    """
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    label_list = _resolve_labels(labels, y_true_arr, y_pred_arr)
    n_labels = len(label_list)
    confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(label_list)}
    for true, pred in zip(y_true_arr, y_pred_arr):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        confusion_matrix[true_index, pred_index] += 1
    return confusion_matrix
    # raise NotImplementedError("Implement confusion_matrix.")


def accuracy_score(y_true, y_pred) -> float:
    """
    Compute overall accuracy from the confusion matrix.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Ratio of correctly predicted samples over total samples.

    Example:
        >>> accuracy_score(["cat", "dog"], ["cat", "cat"])
        0.5
    """
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    if n == 0:
        return 0.0
    correct = np.trace(cm)
    return _safe_divide(correct, n)


def precision_score(y_true, y_pred, positive_label) -> float:
    """
    Precision for a single positive class, computed from the confusion matrix.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        positive_label: Label treated as the positive class.

    Returns:
        float: True positives divided by predicted positives.

    Example:
        >>> precision_score(["cat", "dog"], ["cat", "cat"], positive_label="cat")
        0.5
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    labels = _resolve_labels(None, np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())
    pos_idx = labels.index(positive_label)
    TP = conf_matrix[pos_idx, pos_idx]
    FP = conf_matrix[:, pos_idx].sum() - TP
    return _safe_divide(TP, TP + FP)


def recall_score(y_true, y_pred, positive_label) -> float:
    """
    Recall for a single positive class, computed from the confusion matrix.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        positive_label: Label treated as the positive class.

    Returns:
        float: True positives divided by actual positives (support).

    Example:
        >>> recall_score(["cat", "cat"], ["cat", "dog"], positive_label="cat")
        0.5
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    labels = _resolve_labels(None, np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())
    pos_idx = labels.index(positive_label)
    TP = conf_matrix[pos_idx, pos_idx]
    FN = conf_matrix[pos_idx, :].sum() - TP
    return _safe_divide(TP, TP + FN)
    # raise NotImplementedError("Implement recall_score.")


def f1_score(y_true, y_pred, positive_label) -> float:
    """
    Harmonic mean of precision and recall for a single positive class.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        positive_label: Label treated as the positive class.

    Returns:
        float: 2 * precision * recall / (precision + recall).

    Example:
        >>> f1_score(["cat", "dog"], ["cat", "cat"], positive_label="cat")
        0.6666666666666666
    """
    prec = precision_score(y_true, y_pred, positive_label)
    rec = recall_score(y_true, y_pred, positive_label)
    return _safe_divide(2 * prec * rec, prec + rec)
    # raise NotImplementedError("Implement f1_score.")


def macro_f1_score(y_true, y_pred, labels) -> float:
    """
    Average F1 score across all specified classes (unweighted macro average).

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        labels (array-like): Ordered list of labels to include in the macro
            computation.

    Returns:
        float: Mean of per-class F1 scores.

    Example:
        >>> macro_f1_score(["cat", "dog"], ["cat", "cat"], labels=["cat", "dog"])
        0.5
    """
    sum_f1 = 0.0
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    for label in labels:
        f1 = f1_score(y_true_arr, y_pred_arr, label)
        sum_f1 += f1
    return _safe_divide(sum_f1, len(labels))
    # raise NotImplementedError("Implement macro_f1_score.")


def micro_f1_score(y_true, y_pred, labels) -> float:
    """
    Micro-averaged F1 score aggregated over all specified classes.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        labels (array-like): Label set to include when computing micro F1.

    Returns:
        float: F1 score derived from global TP/FP/FN sums.

    Example:
        >>> micro_f1_score(["cat", "dog"], ["cat", "cat"], labels=["cat", "dog"])
        0.5
    """
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true_arr, y_pred_arr, labels)
    total_TP = 0
    total_FP = 0
    total_FN = 0
    for i in range(len(labels)):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        total_TP += TP
        total_FP += FP
        total_FN += FN
    precision = _safe_divide(total_TP, total_TP + total_FP)
    recall = _safe_divide(total_TP, total_TP + total_FN)
    return _safe_divide(2 * precision * recall, precision + recall)
    # raise NotImplementedError("Implement micro_f1_score.")

