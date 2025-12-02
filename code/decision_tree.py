"""Reference utilities for decision tree metrics and best-split selection."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import math
import pandas as pd


def entropy(labels: Sequence[str]) -> float:
    """
    Compute Shannon entropy (base 2) from a multiset of labels.

    Args:
        labels (Sequence[str]): Iterable of categorical labels.

    Returns:
        float: Entropy value in bits. Returns 0.0 for empty input.
    """
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent
    # raise NotImplementedError("Implement entropy.")


def gini(labels: Sequence[str]) -> float:
    """
    Compute the Gini impurity for the provided labels.

    Args:
        labels (Sequence[str]): Iterable of categorical labels.

    Returns:
        float: Gini impurity (0.0 indicates pure set).
    """
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0.0
    sum_p_squared = 0.0
    for count in counts.values():
        p = count / total
        sum_p_squared += p ** 2
    return 1 - sum_p_squared
    # raise NotImplementedError("Implement gini.")


def partition_dataset(df: pd.DataFrame, feature: str) -> Dict[str, pd.DataFrame]:
    """
    Group rows of a dataframe by a categorical feature.

    Args:
        df (pd.DataFrame): Input dataset.
        feature (str): Column name to partition on.

    Returns:
        Dict[str, pd.DataFrame]: Mapping from feature value to subset dataframe
        (reindexed from 0).
    """
    result = {}
    for value, group in df.groupby(feature):
        result[value] = group.reset_index(drop=True)
    return result
    # raise NotImplementedError("Implement partition_dataset.")


def information_gain(
    df: pd.DataFrame,
    feature: str,
    target: str = "play",
) -> float:
    """
    Compute information gain of splitting on a categorical feature.

    Args:
        df (pd.DataFrame): Dataset containing feature and target columns.
        feature (str): Feature to evaluate.
        target (str): Target column name (default "play").

    Returns:
        float: Information gain in bits.
    """
    # 1. Compute base entropy H(S)
    base_entropy = entropy(df[target])

    # 2. Partition the dataset
    parts = partition_dataset(df, feature)
    total = len(df)

    # 3. Compute weighted sum of child entropies
    weighted_entropy = 0.0
    for subset in parts.values():
        weight = len(subset) / total
        weighted_entropy += weight * entropy(subset[target])

    # 4. Return information gain
    return base_entropy - weighted_entropy
    # raise NotImplementedError("Implement information_gain.")


def _split_info(partitions: Dict[str, pd.DataFrame], total_rows: int) -> float:
    """
    Compute the split information term used in gain ratio.

    Args:
        partitions (Dict[str, pd.DataFrame]): Subsets after splitting.
        total_rows (int): Total number of rows in the original dataset.

    Returns:
        float: Split information (entropy of partition proportions).
    """
    if total_rows == 0:
        return 0.0
    split_info = 0.0
    for subset in partitions.values():
        p = len(subset) / total_rows
        if p > 0:
            split_info -= p * math.log2(p)
    return split_info
    # raise NotImplementedError("Implement _split_info.")


def gain_ratio(
    df: pd.DataFrame,
    feature: str,
    target: str = "play",
) -> float:
    """
    Compute the gain ratio of splitting on `feature`.

    Args:
        df (pd.DataFrame): Dataset containing feature and target columns.
        feature (str): Feature to evaluate.
        target (str): Target column name.

    Returns:
        float: Gain ratio (0 when split information is zero).
    """
    # 1. Compute information gain
    ig = information_gain(df, feature, target)

    # 2. Compute split info
    parts = partition_dataset(df, feature)
    si = _split_info(parts, len(df))

    # 3. If split info is 0, return 0
    if si == 0:
        return 0.0

    # 4. Otherwise return IG / SI
    return ig / si
    # raise NotImplementedError("Implement gain_ratio.")


def best_split(
    df: pd.DataFrame,
    candidate_features: Iterable[str],
    target: str = "play",
    criterion: str = "gain_ratio",
) -> str:
    """
    Select the best feature to split on using the specified criterion.

    Args:
        df (pd.DataFrame): Dataset including candidate feature columns.
        candidate_features (Iterable[str]): Feature names to evaluate.
        target (str): Target column name.
        criterion (str): One of {"gain_ratio", "information_gain", "gini"}.

    Returns:
        str: Feature name with highest score.

    Raises:
        ValueError: If an invalid criterion is provided or no candidates exist.
    """
    # Validate criterion
    valid_criteria = {"gain_ratio", "information_gain", "gini"}
    if criterion not in valid_criteria:
        raise ValueError(f"Invalid criterion: {criterion}. Must be one of {valid_criteria}.")

    # Convert to list to check if empty
    features = list(candidate_features)
    if len(features) == 0:
        raise ValueError("No candidate features provided.")

    best_feature = None
    best_score = -float('inf')

    for feature in features:
        if criterion == "gain_ratio":
            score = gain_ratio(df, feature, target)
        elif criterion == "information_gain":
            score = information_gain(df, feature, target)
        elif criterion == "gini":
            # Compute Gini reduction: G(S) - sum(|Sv|/|S| * G(Sv))
            base_gini = gini(df[target])
            parts = partition_dataset(df, feature)
            total = len(df)
            weighted_gini = 0.0
            for subset in parts.values():
                weight = len(subset) / total
                weighted_gini += weight * gini(subset[target])
            score = base_gini - weighted_gini

        if score > best_score:
            best_score = score
            best_feature = feature

    return best_feature
    # raise NotImplementedError("Implement best_split.")

