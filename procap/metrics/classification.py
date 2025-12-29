"""
Classification metrics for ProCap benchmark.

Supports multi-label and binary classification metrics.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)


def f1_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute maximum F1 score over threshold sweep.

    For multi-label classification, finds the threshold that maximizes
    the micro-averaged F1 score.

    Args:
        y_true: Binary ground truth labels [n_samples, n_labels]
        y_score: Prediction scores [n_samples, n_labels]

    Returns:
        Maximum F1 score
    """
    best_f1 = 0.0
    thresholds = np.arange(0.05, 1.0, 0.05)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        try:
            f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
            best_f1 = max(best_f1, f1)
        except Exception:
            pass

    return float(best_f1)


def f1_micro(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute micro-averaged F1 score.

    Args:
        y_true: Binary ground truth labels
        y_score: Prediction scores
        threshold: Classification threshold

    Returns:
        Micro-averaged F1 score
    """
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def f1_macro(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        y_true: Binary ground truth labels
        y_score: Prediction scores
        threshold: Classification threshold

    Returns:
        Macro-averaged F1 score
    """
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def auprc_micro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute micro-averaged Area Under Precision-Recall Curve.

    Args:
        y_true: Binary ground truth labels [n_samples, n_labels]
        y_score: Prediction scores [n_samples, n_labels]

    Returns:
        Micro-averaged AUPRC
    """
    # Flatten for micro-averaging
    y_true_flat = y_true.ravel()
    y_score_flat = y_score.ravel()

    # Filter out samples where true label is neither 0 nor 1
    valid_mask = np.isin(y_true_flat, [0, 1])
    if not valid_mask.any():
        return 0.0

    try:
        return float(average_precision_score(y_true_flat[valid_mask], y_score_flat[valid_mask]))
    except Exception:
        return 0.0


def auprc_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute macro-averaged Area Under Precision-Recall Curve.

    Args:
        y_true: Binary ground truth labels [n_samples, n_labels]
        y_score: Prediction scores [n_samples, n_labels]

    Returns:
        Macro-averaged AUPRC
    """
    n_labels = y_true.shape[1]
    auprcs = []

    for i in range(n_labels):
        y_true_i = y_true[:, i]
        y_score_i = y_score[:, i]

        # Skip labels with no positive samples
        if y_true_i.sum() == 0:
            continue

        try:
            auprc = average_precision_score(y_true_i, y_score_i)
            auprcs.append(auprc)
        except Exception:
            pass

    return float(np.mean(auprcs)) if auprcs else 0.0


def auroc_micro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute micro-averaged Area Under ROC Curve.

    Args:
        y_true: Binary ground truth labels [n_samples, n_labels]
        y_score: Prediction scores [n_samples, n_labels]

    Returns:
        Micro-averaged AUROC
    """
    y_true_flat = y_true.ravel()
    y_score_flat = y_score.ravel()

    # Need both positive and negative samples
    if len(np.unique(y_true_flat)) < 2:
        return 0.0

    try:
        return float(roc_auc_score(y_true_flat, y_score_flat))
    except Exception:
        return 0.0


def auroc_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute macro-averaged Area Under ROC Curve.

    Args:
        y_true: Binary ground truth labels [n_samples, n_labels]
        y_score: Prediction scores [n_samples, n_labels]

    Returns:
        Macro-averaged AUROC
    """
    n_labels = y_true.shape[1]
    aurocs = []

    for i in range(n_labels):
        y_true_i = y_true[:, i]
        y_score_i = y_score[:, i]

        # Need both positive and negative samples
        if len(np.unique(y_true_i)) < 2:
            continue

        try:
            auroc = roc_auc_score(y_true_i, y_score_i)
            aurocs.append(auroc)
        except Exception:
            pass

    return float(np.mean(aurocs)) if aurocs else 0.0


def precision_micro(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute micro-averaged precision.

    Args:
        y_true: Binary ground truth labels
        y_score: Prediction scores
        threshold: Classification threshold

    Returns:
        Micro-averaged precision
    """
    y_pred = (y_score >= threshold).astype(int)
    return float(precision_score(y_true, y_pred, average="micro", zero_division=0))


def recall_micro(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute micro-averaged recall.

    Args:
        y_true: Binary ground truth labels
        y_score: Prediction scores
        threshold: Classification threshold

    Returns:
        Micro-averaged recall
    """
    y_pred = (y_score >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, average="micro", zero_division=0))
