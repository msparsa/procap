"""
Regression metrics for ProCap benchmark.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as sklearn_r2


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman's rank correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2:
        return 0.0

    try:
        corr, _ = stats.spearmanr(y_true, y_pred)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Pearson correlation coefficient
    """
    if len(y_true) < 2:
        return 0.0

    try:
        corr, _ = stats.pearsonr(y_true, y_pred)
        return float(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    try:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float("inf")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    try:
        return float(mean_absolute_error(y_true, y_pred))
    except Exception:
        return float("inf")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        R² value
    """
    try:
        r2 = sklearn_r2(y_true, y_pred)
        return float(r2) if not np.isnan(r2) else 0.0
    except Exception:
        return 0.0


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Kendall's tau correlation coefficient.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Kendall's tau coefficient
    """
    if len(y_true) < 2:
        return 0.0

    try:
        tau, _ = stats.kendalltau(y_true, y_pred)
        return float(tau) if not np.isnan(tau) else 0.0
    except Exception:
        return 0.0
