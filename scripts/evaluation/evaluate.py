#!/usr/bin/env python3
"""
CLI entrypoint for standalone evaluation of a trained model.

Usage:
    python scripts/evaluate.py \
        --model configs/model/swin_tiny.yaml \
        --preprocessing configs/preprocessing.yaml \
        --checkpoint results/ablation/checkpoints/srad/swin_tiny/best.pt

Preprocessing is now a single unified config (configs/preprocessing.yaml).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.evaluation.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PCOS model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    input_size = model_config.get("input_size", 224)
    _, _, test_loader = build_dataloaders(preproc_config, batch_size=32, input_size=input_size)

    # Model
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device=device)
    metrics["arch"] = model_config["name"]
    metrics["preprocessing"] = preproc_config["name"]

    # Print
    print("\n[Evaluate] Test Set Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Evaluate] Saved to {args.output}")


if __name__ == "__main__":
    main()
