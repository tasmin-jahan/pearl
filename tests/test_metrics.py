"""
Pytest suite — paper-grade evaluation metrics.

Covers:
  - compute_all_metrics shape + key presence
  - compute_epoch_metrics lightweight variant
  - bootstrap_auc_ci returns sane CI bounds
  - youdens_j finds a reasonable optimal threshold
  - find_threshold_for_sensitivity honours target
  - delong_auc_test can distinguish two AUCs on the same labels
  - compute_curves returns ROC/PR raw arrays
"""

import numpy as np

from src.evaluation.metrics import (
    bootstrap_auc_ci,
    compute_all_metrics,
    compute_curves,
    compute_epoch_metrics,
    delong_auc_test,
    find_threshold_for_sensitivity,
    youdens_j,
)


def _toy_data(n: int = 200, seed: int = 0):
    """Generate a separable binary-classification dataset.

    Class 1 probs are drawn from N(0.7, 0.1); class 0 from N(0.3, 0.1).
    This gives AUC well above 0.5 so all assertions have headroom.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    probs = np.where(
        labels == 1,
        rng.normal(0.7, 0.1, size=n).clip(0.01, 0.99),
        rng.normal(0.3, 0.1, size=n).clip(0.01, 0.99),
    )
    preds = (probs >= 0.5).astype(int)
    return labels, preds, probs


def test_compute_all_metrics_keys_and_ranges():
    labels, preds, probs = _toy_data()
    m = compute_all_metrics(
        labels, preds, probs,
        compute_bootstrap_ci=False,  # keep test fast
    )
    # ---- Sample-size keys ----
    assert m["n_samples"] == len(labels)
    assert m["n_positive"] == int(labels.sum())
    assert m["n_negative"] == int((1 - labels).sum())
    # ---- Point metrics in [0, 1] ----
    for k in (
        "test_accuracy", "test_precision", "test_recall",
        "test_specificity", "test_f1", "test_mcc",
        "test_auc_roc", "test_auc_pr",
    ):
        assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"
    # ---- Separable data should give AUC near 1.0 ----
    assert m["test_auc_roc"] > 0.85
    # ---- Confusion matrix is non-negative integers ----
    for k in ("test_tp", "test_fp", "test_tn", "test_fn"):
        assert isinstance(m[k], int)
        assert m[k] >= 0
    # ---- Youden threshold + operating point ----
    assert 0.0 <= m["youden_threshold"] <= 1.0
    assert 0.0 <= m["youden_sensitivity"] <= 1.0
    assert 0.0 <= m["youden_specificity"] <= 1.0
    assert 0.0 <= m["op_threshold_at_sens"] <= 1.0


def test_compute_epoch_metrics_keys():
    labels, preds, probs = _toy_data(n=80)
    m = compute_epoch_metrics(labels, preds, probs)
    for k in ("acc", "precision", "recall", "specificity",
              "f1", "mcc", "auc", "nll", "tp", "fp", "tn", "fn"):
        assert k in m


def test_bootstrap_auc_ci_bounds():
    labels, preds, probs = _toy_data(n=300)
    auc_pt, lo, hi = bootstrap_auc_ci(
        labels, probs, n_boot=200, seed=0,
    )
    # CI should bracket the point estimate and live in [0, 1]
    assert 0.0 <= lo <= auc_pt + 1e-6
    assert auc_pt - 1e-6 <= hi <= 1.0
    assert 0.0 <= auc_pt <= 1.0
    # The CI width should be positive for non-degenerate data
    assert hi - lo > 0.0


def test_youdens_j_returns_valid_threshold():
    labels, preds, probs = _toy_data()
    curves = compute_curves(labels, probs)
    th, idx = youdens_j(
        curves["roc_thresholds"], curves["tpr"], 1 - curves["fpr"],
    )
    assert 0.0 <= th <= 1.0
    assert 0 <= idx < len(curves["roc_thresholds"])


def test_find_threshold_for_sensitivity():
    labels, preds, probs = _toy_data()
    curves = compute_curves(labels, probs)
    th, idx = find_threshold_for_sensitivity(
        curves["roc_thresholds"], curves["tpr"], target_sensitivity=0.95,
    )
    assert 0.0 <= th <= 1.0
    # The chosen operating point should reach the target sensitivity
    assert curves["tpr"][idx] >= 0.95 - 1e-6


def test_delong_auc_test_distinguishes_models():
    """DeLong should detect a significant difference between a strong
    and a noisy classifier on the same labels."""
    labels, _, probs_strong = _toy_data(n=400, seed=1)
    rng = np.random.default_rng(2)
    probs_noisy = rng.uniform(0.2, 0.8, size=len(labels))
    res = delong_auc_test(labels, probs_strong, probs_noisy)
    assert res["auc_a"] > res["auc_b"]
    assert res["z"] > 0
    # Strong model should clearly beat noise
    assert res["p_value"] < 0.05


def test_compute_curves_returns_raw_arrays():
    labels, _, probs = _toy_data()
    c = compute_curves(labels, probs)
    assert "fpr" in c and "tpr" in c and "roc_thresholds" in c
    assert "precision" in c and "recall" in c and "pr_thresholds" in c
    assert "auprc" in c
    # FPR / TPR monotonically non-decreasing
    assert np.all(np.diff(c["fpr"]) >= -1e-12)
    assert np.all(np.diff(c["tpr"]) >= -1e-12)
    # AUPRC in [0, 1]
    assert 0.0 <= c["auprc"] <= 1.0


def test_compute_all_metrics_handles_single_class():
    """Edge case: only one class present in labels."""
    labels = np.zeros(50, dtype=int)
    preds = np.zeros(50, dtype=int)
    probs = np.full(50, 0.3)
    m = compute_all_metrics(
        labels, preds, probs, compute_bootstrap_ci=False,
    )
    # AUC is undefined → should fall back to 0.5
    assert m["test_auc_roc"] == 0.5
    # Specificity well-defined (all TNs)
    assert m["test_specificity"] == 1.0


def test_compute_all_metrics_single_class_with_bootstrap():
    """Regression: bootstrap_auc_ci returns NaN on single-class data;
    compute_all_metrics must NOT propagate that NaN to test_auc_roc."""
    labels = np.zeros(50, dtype=int)
    preds = np.zeros(50, dtype=int)
    probs = np.full(50, 0.3)
    m = compute_all_metrics(
        labels, preds, probs, compute_bootstrap_ci=True, n_bootstrap=50,
    )
    assert m["test_auc_roc"] == 0.5, (
        f"expected 0.5 fallback, got {m['test_auc_roc']!r}"
    )
    # CI is also degenerate → NaN is acceptable here, but the headline
    # point estimate must be the inline fallback.
    assert np.isnan(m["test_auc_roc_ci_low"])
    assert np.isnan(m["test_auc_roc_ci_high"])