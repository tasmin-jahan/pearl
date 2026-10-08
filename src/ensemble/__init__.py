"""
src.ensemble — Full ensemble subsystem.

Files are ordered by execution:
    _00_load.py                   — multi-checkpoint loader
    _01_predict.py                — probability-averaging inference
    _02_per_model_calibration.py  — Pass 1 of two-pass calibration
    _03_ensemble_calibration.py   — Pass 2 of two-pass calibration
    _04_uncertainty.py            — MC-Dropout over the ensemble
    _05_cli_calibration.py        — orchestrator: ensemble calibration
    _06_cli_uncertainty.py        — orchestrator: ensemble uncertainty
    _07_evaluate_ensemble.py      — probability-averaged ensemble evaluation CLI
"""
from src.ensemble._00_load import load_ensemble
from src.ensemble._01_predict import predict_ensemble, evaluate_ensemble
from src.ensemble._02_per_model_calibration import (
    apply_per_model_temperature,
    collect_logits,
    fit_per_model_temperature,
)
from src.ensemble._03_ensemble_calibration import (
    apply_ensemble_temperature,
    fit_ensemble_temperature,
)
from src.ensemble._04_uncertainty import mc_dropout_ensemble
from src.ensemble._05_cli_calibration import run_calibration_ensemble
from src.ensemble._06_cli_uncertainty import run_uncertainty_ensemble
from src.ensemble import _07_evaluate_ensemble as run_evaluate_ensemble

__all__ = [
    "load_ensemble",
    "predict_ensemble",
    "evaluate_ensemble",
    "collect_logits",
    "fit_per_model_temperature",
    "apply_per_model_temperature",
    "fit_ensemble_temperature",
    "apply_ensemble_temperature",
    "mc_dropout_ensemble",
    "run_calibration_ensemble",
    "run_uncertainty_ensemble",
    "run_evaluate_ensemble",
]