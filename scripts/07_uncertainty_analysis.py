#!/usr/bin/env python3
"""Stage 07: MC-Dropout Uncertainty Quantification & Risk-Coverage Analysis.

Executes Monte Carlo Dropout (default: 50 stochastic passes) to compute:
  - Predictive entropy (in nats)
  - Risk-coverage trade-off curves
  - Selective classification performance

Usage::

    # Run uncertainty analysis on a single model:
    python scripts/07_uncertainty_analysis.py \
        --model convnext_tiny \
        --checkpoint results/figshare/convnext_tiny/best.pt \
        --dataset-dir data/preprocessed/figshare \
        --out-dir results/figshare/convnext_tiny/uncertainty

    # Run uncertainty analysis across all discovered models in a folder:
    python scripts/07_uncertainty_analysis.py \
        --model-dir results/figshare \
        --dataset-dir data/preprocessed/figshare
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MC-Dropout uncertainty analysis.")
    parser.add_argument("--dataset-dir", default="data/preprocessed/figshare", help="Preprocessed dataset root.")
    parser.add_argument("--model-dir", default="results/figshare", help="Root directory with trained models.")
    parser.add_argument("--model", default=None, help="Specific model name.")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint path.")
    parser.add_argument("--out-dir", default=None, help="Output directory for uncertainty results.")
    parser.add_argument("--mc-passes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    print("========================================")
    print("      PEARL Pipeline - Uncertainty      ")
    print("========================================")

    if args.model and args.checkpoint:
        out_dir = args.out_dir or f"{Path(args.checkpoint).parent}/uncertainty"
        cmd = [
            sys.executable, "-m", "src.uncertainty.run_uncertainty",
            "--dataset-dir", args.dataset_dir,
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--out-dir", out_dir,
            "--mc-passes", str(args.mc_passes),
            "--batch-size", str(args.batch_size),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        res = subprocess.run(cmd, cwd=str(REPO_ROOT))
        return res.returncode

    model_dir = Path(args.model_dir)
    discovered = [p for p in model_dir.iterdir() if p.is_dir() and (p / "best.pt").is_file()]
    if not discovered:
        print(f"[ERROR] No model checkpoints found in {args.model_dir}")
        return 1

    for p in discovered:
        arch = p.name
        ckpt = str(p / "best.pt")
        out_dir = str(p / "uncertainty")
        print(f"\nEvaluating uncertainty for {arch} ({args.mc_passes} MC passes)...")
        cmd = [
            sys.executable, "-m", "src.uncertainty.run_uncertainty",
            "--dataset-dir", args.dataset_dir,
            "--model", arch,
            "--checkpoint", ckpt,
            "--out-dir", out_dir,
            "--mc-passes", str(args.mc_passes),
            "--batch-size", str(args.batch_size),
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    print("\n========================================")
    print("  STATUS: Uncertainty analysis complete!")
    print("  Next Step: python scripts/08_xai_analysis.py")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
