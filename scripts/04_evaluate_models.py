#!/usr/bin/env python3
"""Stage 04: Evaluate Trained Models on Test Set.

Runs zero-shot or in-distribution evaluation across one or all trained models.
Computes:
  - Test Accuracy, AUC-ROC, Sensitivity, Specificity
  - F1-Score, Matthews Correlation Coefficient (MCC), Brier Score
  - Bootstrapped 95% Confidence Intervals

Usage::

    # Multi-model evaluation across all checkpoints:
    python scripts/04_evaluate_models.py --model-dir results/figshare \
        --test-dataset-dir data/preprocessed/figshare/test

    # Cross-dataset (zero-shot) evaluation on PCOSgen:
    python scripts/04_evaluate_models.py --model-dir results/figshare \
        --test-dataset-dir data/preprocessed/pcosgen/test \
        --output-dir results/evaluation/zero_shot_pcosgen

    # Single-model evaluation:
    python scripts/04_evaluate_models.py --checkpoint-dir results/figshare/convnext_tiny \
        --model-config configs/model/convnext_tiny.yaml \
        --test-dataset-dir data/preprocessed/figshare/test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PEARL models on test data.")
    parser.add_argument(
        "--test-dataset-dir",
        default="data/preprocessed/figshare/test",
        help="Path to preprocessed test dataset split.",
    )
    parser.add_argument(
        "--model-dir",
        default="results/figshare",
        help="Directory containing <arch>/best.pt subdirectories.",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing best.pt for single-model mode.")
    parser.add_argument("--model-config", default=None, help="Model YAML config for single-model mode.")
    parser.add_argument("--output-dir", default=None, help="Output directory for evaluation metrics.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    test_path = Path(args.test_dataset_dir)
    if not test_path.exists():
        print(f"[ERROR] Test dataset not found: {args.test_dataset_dir}")
        return 1

    cmd = [
        sys.executable, "-m", "src.evaluation.evaluate",
        "--test-dataset-dir", args.test_dataset_dir,
        "--batch-size", str(args.batch_size),
    ]

    if args.checkpoint_dir:
        cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
        if args.model_config:
            cmd.extend(["--model-config", args.model_config])
    else:
        cmd.extend(["--model-dir", args.model_dir])
        out_dir = args.output_dir or f"results/evaluation/{Path(args.model_dir).name}"
        cmd.extend(["--output-dir", out_dir])

    if args.device:
        cmd.extend(["--device", args.device])

    print("========================================")
    print("      PEARL Pipeline - Evaluation       ")
    print("========================================")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if res.returncode == 0:
        print("\n========================================")
        print("  STATUS: Evaluation completed!")
        print("  Next Step: python scripts/05_calibration_analysis.py")
        print("========================================")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
