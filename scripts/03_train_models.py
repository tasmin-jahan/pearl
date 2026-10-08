#!/usr/bin/env python3
"""Stage 03: Train Individual or All PEARL Models.

Supports training a single model individually or all five architectures
sequentially:
  - convnext_tiny
  - densenet169
  - efficientnet_b0
  - swin_tiny
  - vit_base

Optimized for Nvidia T4 GPU (16GB VRAM) and cross-platform execution.

Usage::

    # Train a single model individually:
    python scripts/03_train_models.py --model convnext_tiny
    python scripts/03_train_models.py --model densenet169
    python scripts/03_train_models.py --model vit_base --batch-size 16

    # Train all 5 architectures sequentially:
    python scripts/03_train_models.py --model all

    # Fine-tune models on target dataset:
    python scripts/03_train_models.py --dataset-dir data/preprocessed/pcosgen \
        --model-dir results/figshare --output-dir results/finetune_pcosgen
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_MODELS = [
    "convnext_tiny",
    "densenet169",
    "efficientnet_b0",
    "swin_tiny",
    "vit_base",
]


def train_single_model(
    model_name: str,
    dataset_dir: str,
    output_dir: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    fp16: bool,
    checkpoint: str | None,
    model_dir: str | None,
    seed: int,
) -> int:
    print(f"\n========================================")
    print(f"  Training Model: {model_name}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"========================================")

    cmd = [
        sys.executable, "-m", "src.train",
        "--dataset-dir", dataset_dir,
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
    ]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if model_dir:
        cmd.extend(["--model-dir", model_dir])
    if fp16:
        cmd.append("--fp16")

    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode == 0:
        print(f"  [OK] Successfully completed training for {model_name}")
    else:
        print(f"  [ERROR] Training failed for {model_name} (code: {res.returncode})")
    return res.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PEARL Transfer Learning Models.")
    parser.add_argument(
        "--model",
        default="all",
        help="Model to train: convnext_tiny, densenet169, efficientnet_b0, swin_tiny, vit_base, or all.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/preprocessed/figshare",
        help="Path to preprocessed dataset split directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory for checkpoints (default: results/<dataset_name>).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 mixed precision (recommended for T4).")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint to resume / fine-tune from.")
    parser.add_argument("--model-dir", default=None, help="Directory containing <model>/best.pt for multi-model fine-tuning.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        print("        Run python scripts/02_prepare_data.py first.")
        return 1

    targets = ALL_MODELS if args.model.lower() == "all" else [args.model]
    failures = []

    for name in targets:
        code = train_single_model(
            model_name=name,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            fp16=args.fp16,
            checkpoint=args.checkpoint,
            model_dir=args.model_dir,
            seed=args.seed,
        )
        if code != 0:
            failures.append(name)

    print("\n========================================")
    if failures:
        print(f"  Training finished with failures: {failures}")
        return 1
    else:
        print("  STATUS: All models trained successfully!")
        print("  Next Step: python scripts/04_evaluate_models.py")
        print("========================================")
        return 0


if __name__ == "__main__":
    sys.exit(main())
