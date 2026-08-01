"""
Evaluation metrics for binary classification.

Returns a comprehensive metric dict suitable for paper-grade reporting:
  - point classification metrics (accuracy, precision, recall, specificity, F1, MCC)
  - threshold-free ranking metrics (AUC-ROC, AUC-PR)
  - calibration metrics (NLL, Brier, ECE)
  - optimal threshold (Youden's J and a fixed-sensitivity operating point)
  - confusion-matrix elements at the default 0.5 threshold and at the
    optimal threshold
  - raw arrays for ROC and PR curves (callers save these for figures)
  - bootstrap 95% CI for AUC-ROC

The ``compute_all_metrics`` function takes ``labels`` and ``probs``
(probability of class 1) as primary inputs and uses an internal
threshold to derive predictions. Pass ``threshold`` to override.
"""

from typing import Dict, Optional, Tuple
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# NumPy compatibility: np.trapz was deprecated in NumPy 2.0 in favour of
# np.trapezoid. Use whichever is available.
# ---------------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
if _trapz is None:
    raise ImportError("NumPy ≥ 1.20 is required (need np.trapz or np.trapezoid)")


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def youdens_j(thresholds: np.ndarray, sensitivities: np.ndarray,
              specificities: np.ndarray) -> Tuple[float, int]:
    """Find the threshold that maximises Youden's J = sensitivity + specificity - 1.

    Returns ``(best_threshold, best_index)``.
    """
    j = sensitivities + specificities - 1.0
    j = np.nan_to_num(j, nan=-1.0)
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), best_idx


def find_threshold_for_sensitivity(
    thresholds: np.ndarray,
    sensitivities: np.ndarray,
    target_sensitivity: float,
) -> Tuple[float, int]:
    """Find the highest threshold that still achieves ``target_sensitivity``.

    Useful for clinical operating points (e.g. 95% sensitivity for
    screening, even at the cost of specificity).
    """
    mask = sensitivities >= target_sensitivity
    if not mask.any():
        return float(thresholds[0]), 0
    valid_thresholds = thresholds[mask]
    valid_indices = np.where(mask)[0]
    # Pick the highest threshold = most specific at >= target sensitivity
    best_idx = int(valid_indices[np.argmax(valid_thresholds)])
    return float(thresholds[best_idx]), best_idx


# ---------------------------------------------------------------------------
# Bootstrap CI for AUC
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap (percentile) 95% CI for AUC-ROC.

    Args:
        labels: Shape (N,).
        probs: Shape (N,). Probability of class 1.
        n_boot: Number of bootstrap resamples.
        alpha: 1 - CI level (default 0.05 → 95% CI).
        seed: Seed for reproducibility.

    Returns:
        (auc_point, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = roc_auc_score(labels[idx], probs[idx])
            if np.isfinite(a):
                aucs.append(a)
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    aucs = np.array(aucs)
    lo = float(np.percentile(aucs, 100 * alpha / 2.0))
    hi = float(np.percentile(aucs, 100 * (1.0 - alpha / 2.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            auc_point = float(roc_auc_score(labels, probs))
        except ValueError:
            auc_point = float("nan")
    if not np.isfinite(auc_point):
        return float("nan"), float("nan"), float("nan")
    return auc_point, lo, hi


# ---------------------------------------------------------------------------
# DeLong test for AUC comparison
# ---------------------------------------------------------------------------

def _placement_values(labels: np.ndarray, probs: np.ndarray):
    """DeLong placement values V10 (for positives) and V01 (for negatives).

    Returns:
        (auc, V10, V01) where V10/V01 are length-n1 / length-n0 arrays of
        per-example structural components.
    """
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return 0.5, np.array([]), np.array([])
    sorted_probs = np.sort(probs)
    # V10_i = fraction of negatives with score < pos_i
    V10 = np.searchsorted(sorted_probs, pos, side="right") / n
    # V01_j = fraction of positives with score < neg_j (i.e. 1 - rank/m)
    V01 = 1.0 - np.searchsorted(sorted_probs, neg, side="right") / m
    auc = float(V10.mean())
    return auc, V10, V01


def delong_auc_variance(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute DeLong's structural components for AUC variance.

    Returns:
        (auc, placement_values) where placement_values is a length-2
        numpy array ``[V10, V01]`` of per-example placement values.
    """
    auc, V10, V01 = _placement_values(labels, probs)
    return float(auc), np.array([V10, V01], dtype=object)


def delong_auc_test(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sided DeLong test for difference between two AUCs on the same labels.

    Implementation follows DeLong, DeLong, Clarke-Pearson (1988). For each
    classifier we compute the per-example placement values V10, V01, then
    the variance of the AUC is
        S = S10/m + S01/n,
    where S10, S01 are the variances of V10 / V01 across examples.

    Returns:
        Dict with keys: auc_a, auc_b, auc_diff, se_diff, z, p_value.
    """
    auc_a, V_a = delong_auc_variance(labels, probs_a)
    auc_b, V_b = delong_auc_variance(labels, probs_b)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0 or len(V_a[0]) == 0 or len(V_b[0]) == 0:
        return {"auc_a": auc_a, "auc_b": auc_b,
                "auc_diff": float("nan"), "se_diff": float("nan"),
                "z": float("nan"), "p_value": float("nan")}
    # Sun & Xu (2014) pooled-variance formulation of DeLong
    s10 = np.var(V_a[0], ddof=1) / n1 + np.var(V_a[1], ddof=1) / n0
    s20 = np.var(V_b[0], ddof=1) / n1 + np.var(V_b[1], ddof=1) / n0
    var_diff = s10 + s20
    if var_diff <= 0 or not np.isfinite(var_diff):
        return {"auc_a": auc_a, "auc_b": auc_b,
                "auc_diff": auc_a - auc_b, "se_diff": 0.0,
                "z": float("nan"), "p_value": float("nan")}
    se_diff = float(np.sqrt(var_diff))
    diff = auc_a - auc_b
    z = diff / se_diff
    from scipy.stats import norm
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(diff),
        "se_diff": se_diff,
        "z": float(z),
        "p_value": float(p),
    }


# ---------------------------------------------------------------------------
# Curve arrays
# ---------------------------------------------------------------------------

def compute_curves(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute ROC and PR curves as raw arrays.

    Returns:
        Dict with:
          - fpr, tpr, roc_thresholds (from sklearn.roc_curve)
          - precision, recall, pr_thresholds (from sklearn.precision_recall_curve)
          - auprc (area under PR curve, computed via trapezoidal rule)
    """
    fpr, tpr, roc_th = roc_curve(labels, probs)
    precision, recall, pr_th = precision_recall_curve(labels, probs)
    # Trapezoidal approximation of AUC-PR
    sorted_idx = np.argsort(recall)
    auprc = float(_trapz(precision[sorted_idx], recall[sorted_idx]))
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_th,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_th,
        "auprc": auprc,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    compute_bootstrap_ci: bool = True,
    n_bootstrap: int = 2000,
    target_sensitivity: float = 0.95,
) -> Dict[str, float]:
    """Compute the full metric stack for binary classification.

    Args:
        labels: Ground-truth labels (0 or 1), shape (N,).
        predictions: Predicted labels (0 or 1), shape (N,). Used as-is for
            point metrics; the optimal threshold is computed separately.
        probabilities: Predicted probability of class 1, shape (N,).
        threshold: Threshold used to derive ``predictions`` (default 0.5).
        compute_bootstrap_ci: Whether to compute bootstrap CI for AUC.
        n_bootstrap: Number of bootstrap resamples.
        target_sensitivity: For the clinical operating point.

    Returns:
        Dict of metric_name → value. All values rounded to 4 decimals.
    """
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    # ---- Point metrics at the supplied threshold ----
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    mcc = matthews_corrcoef(labels, predictions) if len(np.unique(labels)) > 1 else 0.0

    # ---- Confusion matrix ----
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback: only one class present — fill missing counts with 0
        tn = fp = fn = tp = 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ---- Threshold-free ranking metrics ----
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.5
    # Newer sklearn warns (UndefinedMetricWarning) and returns NaN rather
    # than raising when only one class is present. Normalise that to 0.5.
    if not np.isfinite(auc):
        auc = 0.5
    try:
        pr_auc = float(_trapz(
            *reversed(_precision_recall_area(probabilities, labels))
        ))
    except Exception:
        pr_auc = 0.0

    # ---- Calibration / proper scoring ----
    eps = 1e-15
    p_clipped = np.clip(probabilities, eps, 1 - eps)
    try:
        nll = float(-np.mean(
            labels * np.log(p_clipped) + (1 - labels) * np.log(1 - p_clipped)
        ))
    except Exception:
        nll = float("nan")
    try:
        brier = float(brier_score_loss(labels, probabilities))
    except Exception:
        brier = float("nan")

    # ---- Curves and optimal threshold ----
    curves = compute_curves(labels, probabilities)
    auprc = curves["auprc"]
    sens = curves["tpr"]
    specs = 1.0 - curves["fpr"]
    youden_th, youden_idx = youdens_j(
        curves["roc_thresholds"], sens, specs,
    )
    # At Youden's J
    youden_preds = (probabilities >= youden_th).astype(int)
    youden_acc = accuracy_score(labels, youden_preds)
    youden_sens = recall_score(labels, youden_preds, zero_division=0)
    youden_spec = (
        float(((youden_preds == 0) & (labels == 0)).sum() /
              max(((labels == 0)).sum(), 1))
    )
    youden_f1 = f1_score(labels, youden_preds, zero_division=0)

    # Fixed-sensitivity operating point
    op_th, op_idx = find_threshold_for_sensitivity(
        curves["roc_thresholds"], sens, target_sensitivity,
    )
    op_preds = (probabilities >= op_th).astype(int)
    op_spec = (
        float(((op_preds == 0) & (labels == 0)).sum() /
              max(((labels == 0)).sum(), 1))
    )
    op_f1 = f1_score(labels, op_preds, zero_division=0)

    # ---- Bootstrap CI for AUC ----
    if compute_bootstrap_ci:
        auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(
            labels, probabilities, n_boot=n_bootstrap,
        )
        # ``bootstrap_auc_ci`` returns NaN when the test set is
        # degenerate (only one class present). Fall back to the inline
        # AUC (already guarded against single-class via the 0.5 fallback
        # above) so downstream consumers never see a NaN headline metric.
        if not np.isfinite(auc_pt):
            auc_pt = float(auc)
            auc_lo = float("nan")
            auc_hi = float("nan")
    else:
        auc_pt, auc_lo, auc_hi = auc, float("nan"), float("nan")

    return {
        # Sample size
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "threshold_used": float(threshold),
        # Point metrics at default threshold
        "test_accuracy": round(float(acc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_specificity": round(float(specificity), 4),
        "test_npv": round(float(npv), 4),
        "test_f1": round(float(f1), 4),
        "test_mcc": round(float(mcc), 4),
        "test_fpr": round(float(fpr), 4),
        "test_fnr": round(float(fnr), 4),
        # Threshold-free
        "test_auc_roc": round(float(auc_pt), 4),
        "test_auc_roc_ci_low": round(float(auc_lo), 4),
        "test_auc_roc_ci_high": round(float(auc_hi), 4),
        "test_auc_pr": round(float(auprc), 4),
        # Calibration / proper scoring
        "test_nll": round(float(nll), 4),
        "test_brier": round(float(brier), 4),
        # Confusion matrix at default threshold
        "test_tp": int(tp), "test_fp": int(fp),
        "test_tn": int(tn), "test_fn": int(fn),
        # Optimal threshold (Youden's J)
        "youden_threshold": round(float(youden_th), 4),
        "youden_accuracy": round(float(youden_acc), 4),
        "youden_sensitivity": round(float(youden_sens), 4),
        "youden_specificity": round(float(youden_spec), 4),
        "youden_f1": round(float(youden_f1), 4),
        # Fixed-sensitivity operating point
        "op_threshold_at_sens": round(float(op_th), 4),
        "op_target_sensitivity": float(target_sensitivity),
        "op_specificity_at_target": round(float(op_spec), 4),
        "op_f1_at_target": round(float(op_f1), 4),
    }


def _precision_recall_area(probs: np.ndarray, labels: np.ndarray):
    """Return (recall, precision) sorted for AUPRC trapezoidal integration."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    return recall, precision


# ---------------------------------------------------------------------------
# Cheap per-epoch metrics (no bootstrap, no curve arrays — for log speed)
# ---------------------------------------------------------------------------

def compute_epoch_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Light per-epoch metric dict. No bootstrap, no curve arrays.

    Used by the trainer's per-epoch log to keep the CSV lean.
    """
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    mcc = matthews_corrcoef(labels, predictions) if len(np.unique(labels)) > 1 else 0.0

    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.5
    if not np.isfinite(auc):
        auc = 0.5

    eps = 1e-15
    p_clipped = np.clip(probabilities, eps, 1 - eps)
    try:
        nll = float(-np.mean(
            labels * np.log(p_clipped) + (1 - labels) * np.log(1 - p_clipped)
        ))
    except Exception:
        nll = float("nan")

    return {
        "acc": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "mcc": round(float(mcc), 4),
        "auc": round(float(auc), 4),
        "nll": round(float(nll), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
