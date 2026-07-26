#!/usr/bin/env python3
"""
CLI entrypoint for matrix sweeps (e.g. the v3 Phase 0 ablation:
9 models × 2 denoising configs = 18 runs, default hyperparameters).

Usage:
    python scripts/sweep.py --experiment configs/experiment/ablation_18.yaml
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger, make_run_dir
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Run full model × preprocessing sweep")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config YAML")
    args = parser.parse_args()

    experiment_config = load_config(args.experiment)
    training_config = experiment_config.get("training", {})
    seed = experiment_config.get("seed", 42)
    results_dir = experiment_config.get("results_dir", "results/")

    models = experiment_config.get("models", [])
    preprocessings = experiment_config.get("preprocessing", [])

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_runs = len(models) * len(preprocessings)
    print(f"[Sweep] {len(models)} models × {len(preprocessings)} preprocessing = {total_runs} runs")
    print(f"[Sweep] Device: {device}\n")

    all_results = []
    run_idx = 0

    for model_name in models:
        model_config_path = os.path.join("configs", "model", f"{model_name}.yaml")
        model_config = load_config(model_config_path)

        for preproc_name in preprocessings:
            run_idx += 1
            preproc_config_path = os.path.join("configs", "preprocessing", f"{preproc_name}.yaml")
            preproc_config = load_config(preproc_config_path)

            print(f"\n{'='*60}")
            print(f"[Sweep] Run {run_idx}/{total_runs}: {model_name} + {preproc_name}")
            print(f"{'='*60}")

            set_seed(seed)

            # Run directory
            run_dir = make_run_dir(results_dir, model_name, preproc_name)
            full_config = {
                "model": model_config,
                "preprocessing": preproc_config,
                "experiment": experiment_config,
                "seed": seed,
            }
            logger = ExperimentLogger(run_dir, full_config)

            # Data
            input_size = model_config.get("input_size", 224)
            batch_size = training_config.get("batch_size", 32)
            sampler = training_config.get("sampler", "shuffle")
            train_loader, val_loader, test_loader = build_dataloaders(
                preproc_config, batch_size=batch_size, input_size=input_size,
                sampler=sampler,
            )

            # Model
            model = build_model(model_config)

            # Loss
            class_weights = train_loader.dataset.get_class_weights()
            criterion = build_weighted_loss(class_weights, device=device)

            # Checkpoint
            checkpoint_path = os.path.join(
                results_dir, "checkpoints", f"{model_name}__{preproc_name}.pt"
            )

            # Train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                criterion=criterion,
                config=training_config,
                logger=logger,
                device=device,
                checkpoint_path=checkpoint_path,
            )

            metrics = trainer.train()
            metrics["arch"] = model_name
            metrics["preprocessing"] = preproc_name
            all_results.append(metrics)

            # Free GPU memory
            del model, trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build sweep matrix CSV
    df = pd.DataFrame(all_results)
    matrix_path = os.path.join(results_dir, "sweep_matrix.csv")
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    df.to_csv(matrix_path, index=False)
    print(f"\n[Sweep] Results matrix saved to {matrix_path}")
    print(f"[Sweep] All {total_runs} runs complete!")


if __name__ == "__main__":
    main()
