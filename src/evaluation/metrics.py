"""
Evaluation metrics: accuracy, precision, recall, specificity, F1, AUC-ROC, MCC.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_all_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Compute all 7 evaluation metrics.

    Args:
        labels: Ground-truth labels (0 or 1).
        predictions: Predicted labels (0 or 1).
        probabilities: Predicted probability of positive class (class 1).

    Returns:
        Dict with test_accuracy, test_precision, test_recall,
        test_specificity, test_f1, test_auc_roc, test_mcc.
    """
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    mcc = matthews_corrcoef(labels, predictions)

    # AUC-ROC
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.0

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_specificity": round(specificity, 4),
        "test_f1": round(f1, 4),
        "test_auc_roc": round(auc, 4),
        "test_mcc": round(mcc, 4),
    }
