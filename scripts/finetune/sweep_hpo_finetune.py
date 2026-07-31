#!/usr/bin/env python3
"""Optuna HPO sweep for a single (preprocessing, arch) fine-tune pair.

This is a custom variant of scripts/sweep_hpo.py that:
  - Loads the Zenodo splits (not preprocessed .npy files).
  - Uses the in-memory ZenodoDataset (no preprocessing cache).
  - Resumes each trial from the fine-tuned checkpoint (via
    the experiment config's `fine_tune_resume_from` field).
  - Outputs a per-trial log + best params YAML.

Usage:
    python scripts/sweep_hpo_finetune.py \
        --experiment configs/experiment/finetune_hpo_srad_nopad_densenet121.yaml
"""
import argparse
import json
import os
import sys
from typing import Dict

import optuna
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer
from src.training.tuner import OptunaCallback
from src.data.zenodo_dataset import build_zenodo_loader


def _load_split(path: str):
    with open(path) as f:
        d = json.load(f)
    return [(item["path"], int(item["label"])) for item in d["items"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out_dir", default="results/finetune_hpo/")
    args = parser.parse_args()

    exp_config = load_config(args.experiment)
    seed = exp_config.get("seed", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = exp_config.get("models", [])
    if not models:
        raise ValueError("experiment config must have 'models' list")
    if len(models) > 1:
        print(f"[HPO] WARN: multi-model HPO not supported; using first: {models[0]}")
    model_name = models[0]
    preproc_name = exp_config["preprocessing"]

    model_config = load_config(f"configs/model/{model_name}.yaml")
    preproc_config = load_config(f"configs/preprocessing/{preproc_name}.yaml")
    search_space = exp_config.get("search_space", {})
    training_defaults = exp_config.get("training", {})

    n_trials = exp_config.get("n_trials_per_arch", 25)
    fine_tune_ckpt = exp_config.get("data", {}).get("fine_tune_resume_from")
    if not fine_tune_ckpt or not os.path.isfile(fine_tune_ckpt):
        raise FileNotFoundError(
            f"fine_tune_resume_from not set or missing: {fine_tune_ckpt}"
        )

    # Splits
    train_pairs = _load_split(exp_config["data"]["train_split"])
    val_pairs = _load_split(exp_config["data"]["val_split"])
    print(f"[HPO] {model_name} + {preproc_name}")
    print(f"[HPO] Resume from: {fine_tune_ckpt}")
    print(f"[HPO] Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Output dirs
    out_dir = os.path.join(args.out_dir, f"{preproc_name}_{model_name}")
    arch_dir = os.path.join(out_dir, model_name)
    os.makedirs(arch_dir, exist_ok=True)

    # Save the resolved experiment config for reproducibility
    with open(os.path.join(arch_dir, "experiment_config.yaml"), "w") as f:
        yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)

    # Pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=exp_config.get("pruner_warmup_steps", 3),
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=f"{preproc_name}_{model_name}",
    )

    def objective(trial: optuna.trial.Trial) -> float:
        # Sample hyperparameters
        params = {
            "lr": trial.suggest_float(
                "lr", search_space["lr"]["low"], search_space["lr"]["high"], log=True,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", search_space["weight_decay"]["low"],
                search_space["weight_decay"]["high"], log=True,
            ),
            "dropout": trial.suggest_float(
                "dropout", search_space["dropout"]["low"], search_space["dropout"]["high"],
            ),
            "rotation": trial.suggest_int(
                "rotation", int(search_space["rotation"]["low"]),
                int(search_space["rotation"]["high"]),
            ),
            "label_smoothing": trial.suggest_float(
                "label_smoothing", search_space["label_smoothing"]["low"],
                search_space["label_smoothing"]["high"],
            ),
            "freeze_fraction": trial.suggest_categorical(
                "freeze_fraction", search_space["freeze_fraction"]["choices"],
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", search_space["batch_size"]["choices"],
            ),
        }

        # Build a per-trial cfg
        cfg = dict(training_defaults)
        cfg.update({
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
            "batch_size": int(params["batch_size"]),
            "monitor": "val_auc",
            "bf16": True,
            "channels_last": True,
            "grad_clip_norm": 1.0,
            "ema": True,
            "sampler": "weighted",
            "keep_last_n": 1,
            "augmentation": {
                "rotation": params["rotation"],
                "horizontal_flip": True,
                "scale": 0.1,
            },
        })

        # Build model — apply per-trial dropout
        cfg_model = dict(model_config)
        cfg_model["freeze_fraction"] = params["freeze_fraction"]
        head = dict(cfg_model.get("head", {}))
        head["dropout"] = params["dropout"]
        cfg_model["head"] = head

        model = build_model(cfg_model)

        # Resume from fine-tuned checkpoint
        from src.training.checkpoint import load_checkpoint
        load_checkpoint(
            model, fine_tune_ckpt,
            optimizer=None, scheduler=None, ema=None, device=device,
        )

        # Build data
        train_loader, val_loader = build_zenodo_loader(
            train_pairs=train_pairs, val_pairs=val_pairs,
            preproc_config=preproc_config,
            batch_size=int(params["batch_size"]),
            input_size=cfg_model.get("input_size", 224),
            sampler="weighted",
        )

        # Loss
        class_weights = train_loader.dataset.get_class_weights()
        criterion = build_weighted_loss(class_weights, device=device)

        # Trainer
        callback = OptunaCallback(trial)
        checkpoint_path = os.path.join(arch_dir, f"trial_{trial.number}.pt")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            criterion=criterion,
            config=cfg,
            logger=_NullLogger(),
            device=device,
            checkpoint_path=checkpoint_path,
            on_epoch_end=callback,
            silent=True,
        )

        try:
            metrics = trainer.train()
            val_auc = metrics.get("val_auc_best", 0.0)
            return float(val_auc)
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"[HPO] Trial {trial.number} failed: {e}")
            return 0.0
        finally:
            # Free disk: delete trial checkpoint + rolling files.
            for p in (
                os.path.join(arch_dir, f"trial_{trial.number}.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e1.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e2.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e3.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e4.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e5.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e6.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e7.pt"),
                os.path.join(arch_dir, f"trial_{trial.number}_e8.pt"),
            ):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            torch.cuda.empty_cache()

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Save best params
    if study.best_trial is None:
        raise RuntimeError("No successful trials — HPO failed.")
    best_params = study.best_trial.params
    with open(os.path.join(arch_dir, "best_params.yaml"), "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)
    print(f"[HPO] Best val_auc: {study.best_trial.value:.4f}")
    print(f"[HPO] Best params: {best_params}")

    # Save all trials
    rows = []
    for t in study.trials:
        rows.append({
            "trial_id": t.number,
            "val_auc": t.value,
            "pruned": t.state == optuna.trial.TrialState.PRUNED,
            "complete": t.state == optuna.trial.TrialState.COMPLETE,
            **t.params,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(arch_dir, "all_trials.csv"), index=False,
    )

    # Summary
    summary = {
        "arch": model_name,
        "preprocessing": preproc_name,
        "best_val_auc": study.best_trial.value,
        "best_params": best_params,
        "n_trials": n_trials,
        "n_complete": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ),
        "n_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
    }
    with open(os.path.join(arch_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[HPO] Summary: {os.path.join(arch_dir, 'summary.json')}")


class _NullLogger:
    """No-op logger to suppress per-trial disk spam."""

    def log_epoch(self, *args, **kwargs): pass
    def log_final_metrics(self, *args, **kwargs): pass
    def plot_training_curves(self, *args, **kwargs): pass
    def close(self, *args, **kwargs): pass


if __name__ == "__main__":
    main()