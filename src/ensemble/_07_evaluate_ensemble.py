"""
Ensemble evaluator CLI: probability-averages N models on a single
preprocessed test set and writes one metrics JSON.

Probability-averaging means: each member produces a softmax probability
vector for every test image; we average those vectors across members
and take the argmax. This is the headline ensemble used in
``docs/fydp-thesis.md`` (pre-HPO ensemble: external AUC 0.9487).

Unlike :mod:`src.evaluation.run_evaluate` (which scores one model at
a time), this CLI always aggregates all passed checkpoints. Pass 2
or 50; the average is over all of them with equal weight.

Usage::

    python -m src.evaluation.run_evaluate_ensemble \
        --test-dataset-dir data/preprocessed/pcosgen \
        --preprocessing-config configs/preprocessing.yaml \
        --output results/ensemble_pcosgen.json \
        --checkpoint results/stage2/swin_tiny/best.pt \
        --model swin_tiny \
        --checkpoint results/stage2/vit_base/best.pt \
        --model vit_base \
        ...

Each ``--checkpoint`` is paired 1:1 with the preceding ``--model``.
At least two pairs are required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

from src.data.dataloader import build_test_loader
from src.ensemble import evaluate_ensemble, load_ensemble
from src.utils.config import load_config
from src.utils.seed import set_seed


def _resolve_model_config_path(model_configs_dir: str, arch: str) -> str:
    """Map an arch short-name to its YAML under configs/model/."""
    path = Path(model_configs_dir) / f"{arch}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find model config for {arch!r}: expected {path}."
        )
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score a probability-averaged ensemble on a preprocessed test "
            "set. Always aggregates every --model/--checkpoint pair you "
            "pass; no member selection."
        ),
    )
    parser.add_argument(
        "--test-dataset-dir", required=True,
        help=(
            "Preprocessed PNG test set. Accepts either the dataset root "
            "(e.g. data/preprocessed/pcosgen) or the test split directory "
            "(e.g. data/preprocessed/pcosgen/test)."
        ),
    )
    parser.add_argument(
        "--preprocessing-config", default="configs/preprocessing.yaml",
    )
    parser.add_argument(
        "--model-configs-dir", default="configs/model",
        help="Directory holding <arch>.yaml files.",
    )
    parser.add_argument(
        "--model", action="append", default=[],
        help="Model short name. Repeat for each ensemble member. "
             "Optional when --model-dir is given (auto-discovers).",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="Checkpoint path; pair 1:1 with each --model.",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Root containing <arch>/best.pt. Auto-discovers and "
             "fine-tunes every model; equivalent to listing each "
             "--model/--checkpoint pair manually.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", required=True,
        help="Where to write the ensemble metrics JSON.",
    )
    args = parser.parse_args(argv)

    # Auto-discover from --model-dir when --model/--checkpoint were omitted.
    if not args.model and args.model_dir:
        if not os.path.isdir(args.model_dir):
            parser.error(f"--model-dir {args.model_dir} is not a directory")
        discovered = sorted(
            p.name for p in Path(args.model_dir).iterdir()
            if p.is_dir() and (p / "best.pt").is_file()
        )
        if not discovered:
            parser.error(
                f"--model-dir {args.model_dir} has no <arch>/best.pt "
                "subdirectories."
            )
        args.model = discovered
        args.checkpoint = [str(Path(args.model_dir) / a / "best.pt")
                           for a in discovered]
        print(f"[Auto-discover] Using {len(args.model)} ensemble members: "
              f"{args.model}")

    if len(args.model) != len(args.checkpoint):
        parser.error(
            f"--model ({len(args.model)}) and --checkpoint "
            f"({len(args.checkpoint)}) must appear the same number of times."
        )
    if len(args.model) < 2:
        parser.error(
            "Probability-averaging needs at least two members; "
            "use src.evaluation.run_evaluate for a single model."
        )

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    preproc_config = load_config(args.preprocessing_config)

    # Build a single test loader — same images, same normalization.
    test_loader = build_test_loader(
        args.test_dataset_dir, preproc_config,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Resolve and load each member. We do this once up front so the
    # ensemble pass is a single forward sweep.
    model_configs = [
        load_config(_resolve_model_config_path(args.model_configs_dir, m))
        for m in args.model
    ]
    print(f"[Ensemble] Loading {len(args.model)} members on {device}…")
    models = load_ensemble(model_configs, args.checkpoint, device=device)

    # Probability-averaged inference + full metric stack.
    metrics = evaluate_ensemble(models, test_loader, device=device)

    # Tag with provenance for downstream comparison.
    metrics["ensemble_members"] = [
        {"arch": arch, "checkpoint": ckpt}
        for arch, ckpt in zip(args.model, args.checkpoint)
    ]
    metrics["test_dataset_dir"] = args.test_dataset_dir
    metrics["preprocessing_config"] = args.preprocessing_config
    metrics["aggregation"] = "probability_mean"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Ensemble] Saved {args.output}")
    interesting = ["test_auc_roc", "test_accuracy", "test_f1",
                   "test_precision", "test_recall", "test_specificity",
                   "youden_accuracy", "youden_threshold"]
    for k in interesting:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())