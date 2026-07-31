#!/usr/bin/env python3
"""
CLI entrypoint for a single training run.

Usage:
    python scripts/train.py \
        --model configs/model/swin_tiny.yaml \
        --preprocessing configs/preprocessing/srad.yaml \
        --experiment configs/experiment/best_model_xai.yaml \
        --set training.lr=5e-4 training.batch_size=16 \
        --run_dir results/ablation/checkpoints/srad/swin_tiny  # optional

Dynamic overrides: ``--set <dotted.key.path>=<value>`` (repeatable).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import (
    load_config, load_experiment_config,
    parse_overrides, apply_overrides,
)
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger, make_arch_dir, make_run_dir
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
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Config override, e.g. --set training.lr=5e-4 (repeatable)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configs
    full = load_experiment_config(args.model, args.preprocessing, args.experiment)
    experiment_config = full["experiment"]
    model_config = full["model"]
    preproc_config = full["preprocessing"]
    training_config = full.get("training", experiment_config.get("training", {}))

    # Dynamic overrides
    if args.overrides:
        overrides = parse_overrides(args.overrides)
        training_config = apply_overrides(training_config, overrides)
        experiment_config = apply_overrides(experiment_config, overrides)
        model_config = apply_overrides(model_config, overrides)

    # Seed
    seed = experiment_config.get("seed", 42)
    set_seed(seed)

    # Device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # Per-(prep, arch) directory — holds all artifacts for this run.
    arch = model_config["name"]
    preproc_name = preproc_config["name"]
    results_dir = experiment_config.get("results_dir", "results/")
    arch_dir = make_arch_dir(results_dir, preproc_name, arch)

    # --run_dir override (for resume): can point at either the per-arch
    # directory or a legacy timestamped runs/ directory.
    run_dir = args.run_dir or arch_dir

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

    # Checkpoint path lives inside the per-arch directory.
    checkpoint_path = os.path.join(arch_dir, "best.pt")

    # Train (with optional resume)
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

    final_metrics = trainer.train(resume_from=args.resume)

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
