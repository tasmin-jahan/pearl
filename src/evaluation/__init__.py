"""
Evaluation modules.
"""
from src.evaluation.evaluator import evaluate_model, collect_logits_and_labels
from src.evaluation.metrics import compute_all_metrics, compute_epoch_metrics
from src.ensemble._07_evaluate_ensemble import (
    main as run_evaluate_ensemble_main,
)
from src.ensemble import _07_evaluate_ensemble as run_evaluate_ensemble

__all__ = [
    "evaluate_model",
    "collect_logits_and_labels",
    "compute_all_metrics",
    "compute_epoch_metrics",
    "run_evaluate_ensemble",
    "run_evaluate_ensemble_main",
]
