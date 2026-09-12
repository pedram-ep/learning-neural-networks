"""Evaluation metrics for both heads."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics_a(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Head A (3-class).
    y_true: (N,) int class indices
    y_pred: (N,) int predicted class indices
    """
    return {
        "acc_a": accuracy_score(y_true, y_pred),
        "f1_macro_a": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_metrics_b(y_true: np.ndarray, y_prob: np.ndarray,
                      threshold: float = 0.5) -> dict:
    """
    Head B (binary).
    y_true:    (N,) int binary labels (0/1)
    y_prob:    (N,) float predicted probabilities
    threshold: decision threshold (default 0.5)
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "acc_b": accuracy_score(y_true, y_pred),
        "f1_b": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc_b"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_b"] = float("nan")
    return metrics