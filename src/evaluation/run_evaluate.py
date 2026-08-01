"""
Single-model evaluator CLI.

Loads a checkpoint, runs inference on a preprocessed PNG test set,
and writes one JSON of per-model metrics. No aggregation across
checkpoints — call this once per model you want to score.

Usage::

    python -m src.evaluation.run_evaluate \
        --checkpoint-dir results/figshare/swin_tiny \
        --test-dataset-dir data/preprocessed/figshare \
        --model-config configs/model/swin_tiny.yaml \
        --preprocessing-config configs/preprocessing.yaml \
        --output results/eval/figshare_swin_tiny.json

The checkpoint directory is expected to contain ``best.pt``. The test
dataset directory is the preprocessed PNG tree produced by
``src.preprocessing.run_preprocessing``. Both arguments are required so
this CLI never reads raw data or hard-codes a default split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from src.data.dataloader import build_dataloaders
from src.evaluation.evaluator import evaluate_model
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import set_seed


def _resolve_checkpoint(checkpoint_dir: str) -> str:
    """Return the ``best.pt`` path inside ``checkpoint_dir`` or fail."""
    ckpt = Path(checkpoint_dir) / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Expected {ckpt} but it does not exist. "
            f"--checkpoint-dir must contain best.pt."
        )
    return str(ckpt)


def _load_model(model_config_path: str, checkpoint: str, device: str) -> torch.nn.Module:
    cfg = load_config(model_config_path)
    model = build_model(cfg)
    load_checkpoint(model, checkpoint, device=device)
    model.to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a single PCOS checkpoint on a preprocessed test set.",
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Directory containing best.pt (e.g. results/figshare/swin_tiny).",
    )
    parser.add_argument(
        "--test-dataset-dir", required=True,
        help="Preprocessed PNG dataset root with train/val/test (e.g. data/preprocessed/figshare).",
    )
    parser.add_argument(
        "--model-config", required=True,
        help="Path to the model YAML (e.g. configs/model/swin_tiny.yaml).",
    )
    parser.add_argument(
        "--preprocessing-config", default="configs/preprocessing.yaml",
        help="Preprocessing YAML (default: configs/preprocessing.yaml).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=None,
        help="Where to write the metrics JSON. Default: <checkpoint-dir>/metrics.json",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve paths
    checkpoint = _resolve_checkpoint(args.checkpoint_dir)
    preproc_config = load_config(args.preprocessing_config)
    output = args.output or os.path.join(args.checkpoint_dir, "metrics.json")

    # Build test loader from the preprocessed dataset
    _, _, test_loader = build_dataloaders(
        args.test_dataset_dir,
        preproc_config,
        batch_size=args.batch_size,
        num_workers=2,
        sampler="none",
        drop_last=False,
    )

    # Load model + run inference
    model = _load_model(args.model_config, checkpoint, device)
    metrics = evaluate_model(model, test_loader, device=device)

    # Tag with provenance
    metrics["checkpoint_dir"] = args.checkpoint_dir
    metrics["test_dataset_dir"] = args.test_dataset_dir
    metrics["model_config"] = args.model_config
    metrics["preprocessing_config"] = args.preprocessing_config

    # Write per-model metrics JSON
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Evaluate] Saved {output}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
