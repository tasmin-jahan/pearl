#!/usr/bin/env python3
"""Stage 06: Create & Evaluate Ensemble Model.

Aggregates individual models into a calibrated, probability-averaged ensemble.
Supports auto-discovery of all models from a model directory or explicit
member specification.

Usage::

    # Auto-discover all trained models from results/figshare:
    python scripts/06_create_ensemble.py \
        --model-dir results/figshare \
        --test-dataset-dir data/preprocessed/figshare/test \
        --output results/ensemble/figshare_ensemble_metrics.json

    # Evaluate ensemble on external PCOSgen test set:
    python scripts/06_create_ensemble.py \
        --model-dir results/figshare \
        --test-dataset-dir data/preprocessed/pcosgen/test \
        --output results/ensemble/pcosgen_zero_shot_ensemble.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PEARL ensemble.")
    parser.add_argument(
        "--test-dataset-dir",
        default="data/preprocessed/figshare/test",
        help="Path to preprocessed test dataset split.",
    )
    parser.add_argument(
        "--model-dir",
        default="results/figshare",
        help="Root directory containing <arch>/best.pt models.",
    )
    parser.add_argument(
        "--output",
        default="results/ensemble/ensemble_metrics.json",
        help="Path to output ensemble metrics JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.ensemble._07_evaluate_ensemble",
        "--test-dataset-dir", args.test_dataset_dir,
        "--model-dir", args.model_dir,
        "--output", args.output,
        "--batch-size", str(args.batch_size),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("========================================")
    print("        PEARL Pipeline - Ensemble       ")
    print("========================================")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if res.returncode == 0:
        print("\n========================================")
        print("  STATUS: Ensemble creation & evaluation complete!")
        print("  Next Step: python scripts/07_uncertainty_analysis.py")
        print("========================================")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
