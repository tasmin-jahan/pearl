#!/usr/bin/env python3
"""
v3 Phase 6.1 — full Optuna sweep across all 9 architectures.

Iterates over each architecture in the experiment config, runs an
Optuna study for that architecture with the requested trial budget,
saves per-architecture best params + trial history.

Defaults to using the preprocessing config marked as winner in the
Phase 0.4 ablation (set the experiment config's ``preprocessing``
field to that name). Only one preprocessing config is swept — the
ablation already decided between SRAD and Gaussian.

Usage:
    python scripts/sweep_hpo.py \
        --experiment configs/experiment/tune_per_arch.yaml \
        --out_dir results/sweep_hpo/
"""

import argparse
import json
import os
import sys

import optuna
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, apply_overrides
from src.utils.seed import set_seed
from src.training.tuner import make_objective, _NullLogger


def main():
    parser = argparse.ArgumentParser(description="Per-architecture Optuna HPO sweep")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/sweep_hpo/")
    args = parser.parse_args()

    exp_config = load_config(args.experiment)
    os.makedirs(args.out_dir, exist_ok=True)
    seed = exp_config.get("seed", 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = exp_config.get("models", [])
    n_trials = exp_config.get("n_trials_per_arch", 20)
    pruner_name = exp_config.get("pruner", "median")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=exp_config.get("pruner_warmup_steps", 5),
    )

    preproc_name = exp_config.get("preprocessing", "srad")
    preproc_config = load_config(f"configs/preprocessing/{preproc_name}.yaml")

    summary_rows = []

    for model_name in models:
        print(f"\n{'='*60}\n[HPO-Sweep] Architecture: {model_name}\n{'='*60}")
        model_config = load_config(f"configs/model/{model_name}.yaml")

        # Each model gets a fresh Optuna study (separate seed for HPO diversity)
        set_seed(seed)
        study = optuna.create_study(
            study_name=model_name,
            direction="maximize",
            pruner=pruner,
            load_if_exists=True,
        )

        objective = make_objective(
            model_config=model_config,
            preproc_config=preproc_config,
            experiment_config=exp_config,
            device=device,
            logger_factory=_NullLogger,
        )

        try:
            study.optimize(objective, n_trials=n_trials)
        except Exception as e:
            print(f"[HPO-Sweep] {model_name} aborted: {e}")
            continue

        # ---- Per-architecture outputs ----
        arch_dir = os.path.join(args.out_dir, model_name)
        os.makedirs(arch_dir, exist_ok=True)

        # Best params
        with open(os.path.join(arch_dir, "best_params.yaml"), "w") as f:
            yaml.dump(study.best_trial.params, f, default_flow_style=False)

        # Trial history
        rows = []
        for t in study.trials:
            rows.append({
                "trial_id": t.number,
                "val_auc": t.value,
                "pruned": t.state == optuna.trial.TrialState.PRUNED,
                **t.params,
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(arch_dir, "all_trials.csv"), index=False,
        )

        # Summary row
        summary_rows.append({
            "arch": model_name,
            "best_val_auc": study.best_trial.value,
            "best_params": json.dumps(study.best_trial.params),
            "n_trials": n_trials,
            "n_pruned": sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ),
        })

    # ---- Final summary ----
    summary_path = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(summary_rows).sort_values(
        "best_val_auc", ascending=False,
    ).to_csv(summary_path, index=False)
    print(f"\n[HPO-Sweep] Summary saved to {summary_path}")

    # ---- Top-k finalists ----
    top_k = exp_config.get("top_k_finalists", 3)
    print(f"\n[HPO-Sweep] Top-{top_k} finalists:")
    finalists = (
        pd.DataFrame(summary_rows)
        .sort_values("best_val_auc", ascending=False)
        .head(top_k)
    )
    finalists.to_csv(os.path.join(args.out_dir, "finalists.csv"), index=False)
    print(finalists[["arch", "best_val_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
