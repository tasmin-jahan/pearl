"""
src.ensemble — Full ensemble subsystem.

Files are ordered by execution so that ``import`` statements trace the
ensemble pipeline::

    _00_load.py                       — multi-checkpoint loader
    _01_predict.py                    — probability-averaging inference
    _02_per_model_calibration.py      — Pass 1 of two-pass calibration
    _03_ensemble_calibration.py       — Pass 2 of two-pass calibration
    _04_uncertainty.py                — MC-Dropout over the ensemble
    _05_cli_calibration.py            — orchestrator: ensemble calibration
    _06_cli_uncertainty.py            — orchestrator: ensemble uncertainty

The leading underscore is required because Python module names cannot
start with a digit. The numeric prefix is preserved for execution-order
readability.

This package is the single place where ensemble-related code lives.
Single-model code stays in ``src.calibration``, ``src.uncertainty``,
``src.xai``, ``src.evaluation`` etc.

The CLIs in ``src/calibration/run_calibration.py`` and
``src/uncertainty/run_uncertainty.py`` delegate to ``_05`` and ``_06``
when their ``--ensemble`` flag (or multi-``--model``) is set.
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
]
