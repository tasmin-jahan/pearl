#!/usr/bin/env python3
"""
CLI entrypoint for Optuna hyperparameter tuning.

Usage:
    python scripts/tune.py \
        --experiment configs/experiment/tune_best.yaml \
        --study_name swin_tiny_srad \
        --storage sqlite:///results/tuning/optuna.db
"""

import argparse
import os
import sys

import optuna
import pandas as pd
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.training.tuner import make_objective
from src.model.builder import build_model
from src.training.checkpoint import save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--study_name", type=str, default="pcos_tuning")
    parser.add_argument("--storage", type=str, default=None)
    args = parser.parse_args()

    exp_config = load_config(args.experiment)
    seed = exp_config.get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and preprocessing configs
    model_name = exp_config["model"]
    preproc_name = exp_config["preprocessing"]
    model_config = load_config(f"configs/model/{model_name}.yaml")
    preproc_config = load_config(f"configs/preprocessing/{preproc_name}.yaml")

    n_trials = exp_config.get("n_trials", 50)
    results_dir = exp_config.get("results_dir", "results/tuning/")
    os.makedirs(results_dir, exist_ok=True)

    # Storage
    storage = args.storage
    if storage and storage.startswith("sqlite:///"):
        db_dir = os.path.dirname(storage.replace("sqlite:///", ""))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    # Create study
    pruner_name = exp_config.get("pruner", "median")
    warmup = exp_config.get("pruner_warmup_steps", 5)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    # Objective
    objective = make_objective(model_config, preproc_config, exp_config, device)

    print(f"[Tune] Starting {n_trials} trials for {model_name} + {preproc_name}")
    study.optimize(objective, n_trials=n_trials)

    # Results
    print(f"\n[Tune] Best trial:")
    print(f"  Value (val_auc): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Save best params
    best_params_path = os.path.join(results_dir, "best_params.yaml")
    with open(best_params_path, "w") as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)
    print(f"[Tune] Best params saved to {best_params_path}")

    # Save all trials CSV
    trials_data = []
    for trial in study.trials:
        row = {"trial_id": trial.number, "val_auc": trial.value,
               "pruned": trial.state == optuna.trial.TrialState.PRUNED,
               "duration_sec": (trial.datetime_complete - trial.datetime_start).total_seconds()
               if trial.datetime_complete else None}
        row.update(trial.params)
        trials_data.append(row)

    df = pd.DataFrame(trials_data)
    trials_path = os.path.join(results_dir, "all_trials.csv")
    df.to_csv(trials_path, index=False)
    print(f"[Tune] All trials saved to {trials_path}")

    # Retrain best model and save checkpoint
    print(f"\n[Tune] Retraining best model with optimal hyperparameters...")
    import copy
    best_cfg = copy.deepcopy(model_config)
    best_cfg["freeze_fraction"] = study.best_trial.params["freeze_fraction"]
    best_cfg["head"]["dropout"] = study.best_trial.params["dropout"]

    best_model = build_model(best_cfg)
    ckpt_dir = os.path.join(
        os.path.dirname(results_dir.rstrip("/")),
        "checkpoints",
        preproc_name,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}__tuned.pt")
    save_checkpoint(best_model, torch.optim.AdamW(best_model.parameters()),
                    0, study.best_trial.value, ckpt_path)

    print(f"[Tune] Done!")


if __name__ == "__main__":
    main()
