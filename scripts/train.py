#!/usr/bin/env python3
"""
CLI entrypoint for a single training run.

Usage:
    python scripts/train.py \
        --model configs/model/efficientnet_b4.yaml \
        --preprocessing configs/preprocessing/full_ad.yaml \
        --experiment configs/experiment/sweep_all_54.yaml \
        --run_dir results/runs/efficientnet_b4__full_ad__20260518  # optional
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, load_experiment_config
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger, make_run_dir
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a single PCOS model")
    parser.add_argument("--model", type=str, required=True, help="Model config YAML")
    parser.add_argument("--preprocessing", type=str, required=True, help="Preprocessing config YAML")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config YAML")
    parser.add_argument("--run_dir", type=str, default=None, help="Override run directory")
    args = parser.parse_args()

    # Load configs
    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    experiment_config = load_config(args.experiment)
    training_config = experiment_config.get("training", {})

    # Seed
    seed = experiment_config.get("seed", 42)
    set_seed(seed)

    # Device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # Run directory
    arch = model_config["name"]
    preproc_name = preproc_config["name"]
    results_dir = experiment_config.get("results_dir", "results/")
    run_dir = args.run_dir or make_run_dir(results_dir, arch, preproc_name)

    # Merged config for logging
    full_config = {
        "model": model_config,
        "preprocessing": preproc_config,
        "experiment": experiment_config,
        "seed": seed,
    }

    # Logger
    logger = ExperimentLogger(run_dir, full_config)

    # Data
    input_size = model_config.get("input_size", 224)
    batch_size = training_config.get("batch_size", 32)
    train_loader, val_loader, test_loader = build_dataloaders(
        preproc_config, batch_size=batch_size, input_size=input_size,
    )

    # Model
    model = build_model(model_config)

    # Loss
    class_weights = train_loader.dataset.get_class_weights()
    criterion = build_weighted_loss(class_weights, device=device)

    # Checkpoint path
    checkpoint_path = os.path.join(
        results_dir, "checkpoints", f"{arch}__{preproc_name}.pt"
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

    final_metrics = trainer.train()

    # Add metadata
    final_metrics["arch"] = arch
    final_metrics["preprocessing"] = preproc_name
    final_metrics["seed"] = seed

    # Re-save with metadata
    logger.log_final_metrics(final_metrics)

    print(f"\n[Train] Done. Results in {run_dir}")
    return final_metrics


if __name__ == "__main__":
    main()
