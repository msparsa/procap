"""Evaluation metrics for ProCap benchmark."""

from procap.metrics.classification import (
    f1_max,
    f1_micro,
    f1_macro,
    auprc_micro,
    auprc_macro,
    auroc_micro,
    auroc_macro,
    precision_micro,
    recall_micro,
)
from procap.metrics.regression import (
    spearman_correlation,
    pearson_correlation,
    rmse,
    mae,
    r2_score,
)
from procap.metrics.generation import (
    rouge_l,
    bert_score,
    novelty_50,
)

__all__ = [
    # Classification
    "f1_max",
    "f1_micro",
    "f1_macro",
    "auprc_micro",
    "auprc_macro",
    "auroc_micro",
    "auroc_macro",
    "precision_micro",
    "recall_micro",
    # Regression
    "spearman_correlation",
    "pearson_correlation",
    "rmse",
    "mae",
    "r2_score",
    # Generation
    "rouge_l",
    "bert_score",
    "novelty_50",
]
