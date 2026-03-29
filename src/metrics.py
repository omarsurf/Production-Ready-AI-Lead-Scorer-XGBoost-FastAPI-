"""
Business metrics for lead scoring evaluation and hyperparameter tuning.

This module centralizes ranking-based metrics like Precision@K that align
with business objectives (contact top X% of leads).
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import make_scorer


def precision_at_k_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_fraction: float = 0.1,
) -> float:
    """
    Compute precision in the top k% of predictions.

    This metric directly measures: "If we contact the top k% of leads
    ranked by predicted score, what fraction will convert?"

    Args:
        y_true: Binary ground truth labels (0/1).
        y_score: Predicted probabilities or scores (higher = more likely positive).
        k_fraction: Fraction of samples to consider (e.g., 0.1 = top 10%).

    Returns:
        Precision (conversion rate) in the top k% of predictions.

    Examples:
        >>> y_true = np.array([1, 1, 0, 0, 0])
        >>> y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        >>> precision_at_k_score(y_true, y_score, k_fraction=0.4)
        1.0  # Top 40% = 2 samples, both are positive

    Note:
        When k_fraction results in fewer than 1 sample, k is set to 1.
        This ensures the metric is always computable on small datasets.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0:
        return 0.0

    n_samples = len(y_true)
    k = max(1, int(n_samples * k_fraction))

    # Sort by score descending, take top k
    top_k_indices = np.argsort(y_score)[::-1][:k]

    # Handle edge case: all same class in top k
    top_k_labels = y_true[top_k_indices]
    return float(top_k_labels.mean())


def precision_at_k_proba(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k_fraction: float = 0.1,
) -> float:
    """
    Compute precision@k using probability predictions.

    This is a wrapper for use with sklearn scorers that receive
    predict_proba output (2D array with shape [n_samples, n_classes]).

    Args:
        y_true: Binary ground truth labels (0/1).
        y_proba: Probability predictions from predict_proba.
                 Can be 1D (positive class proba) or 2D (all classes).
        k_fraction: Fraction of samples to consider.

    Returns:
        Precision in the top k% of predictions.
    """
    y_proba = np.asarray(y_proba)

    # Handle 2D output from predict_proba (take positive class column)
    if y_proba.ndim == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba

    return precision_at_k_score(y_true, y_score, k_fraction)


def make_precision_at_k_scorer(k_fraction: float = 0.1):
    """
    Create a sklearn-compatible scorer for Precision@K.

    This scorer can be used directly in RandomizedSearchCV, GridSearchCV,
    or cross_val_score.

    Args:
        k_fraction: Fraction of samples to consider (e.g., 0.1 = top 10%).

    Returns:
        A sklearn scorer object that uses predict_proba.

    Example:
        >>> from sklearn.model_selection import RandomizedSearchCV
        >>> scorer = make_precision_at_k_scorer(0.1)
        >>> search = RandomizedSearchCV(
        ...     estimator=model,
        ...     param_distributions=params,
        ...     scoring={"roc_auc": "roc_auc", "precision_at_10": scorer},
        ...     refit="precision_at_10",
        ... )
    """

    # Create a partial function with fixed k_fraction
    # Use **kwargs to absorb any extra sklearn-internal arguments
    def _precision_at_k(y_true, y_proba, **kwargs):
        return precision_at_k_proba(y_true, y_proba, k_fraction=k_fraction)

    return make_scorer(
        _precision_at_k,
        needs_proba=True,
        greater_is_better=True,
    )


def get_tuning_scorers(k_fraction: float = 0.1) -> Tuple[dict, str]:
    """
    Get the multi-metric scoring dict and refit metric for tuning.

    Returns both ROC-AUC (statistical guard) and Precision@K (business objective).

    Args:
        k_fraction: Fraction for Precision@K scorer.

    Returns:
        Tuple of (scoring_dict, refit_metric_name)

    Example:
        >>> scoring, refit = get_tuning_scorers(0.1)
        >>> search = RandomizedSearchCV(..., scoring=scoring, refit=refit)
    """
    k_percent = int(k_fraction * 100)
    precision_key = f"precision_at_{k_percent}"

    scoring = {
        "roc_auc": "roc_auc",
        precision_key: make_precision_at_k_scorer(k_fraction),
    }

    return scoring, precision_key


# Pre-built scorers for common K values
precision_at_10_scorer = make_precision_at_k_scorer(0.10)
precision_at_20_scorer = make_precision_at_k_scorer(0.20)
precision_at_30_scorer = make_precision_at_k_scorer(0.30)
