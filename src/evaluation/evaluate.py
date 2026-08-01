"""
Single- and multi-model evaluator CLI.

Single model::

    python -m src.evaluation.run_evaluate \
        --checkpoint-dir results/figshare/swin_tiny \
        --test-dataset-dir data/preprocessed/figshare \
        --model-config configs/model/swin_tiny.yaml

Multi-model (zero-shot or any cross-checkpoint eval)::

    python -m src.evaluation.run_evaluate \
        --model-dir results/stage1 \
        --test-dataset-dir data/preprocessed/pcosgen \
        --model-configs-dir configs/model \
        --output-dir results/zero_shot_pcosgen

For multi-model, every ``<arch>/best.pt`` under ``--model-dir`` is
evaluated; each architecture's YAML is resolved by name from
``--model-configs-dir``. Results are written as
``<output-dir>/<arch>/metrics.json`` and one aggregated
``<output-dir>/all_metrics.json``.

The CLI is dataset-agnostic: it never inspects which dataset the
checkpoint was trained on. The same command works for in-distribution
eval, zero-shot transfer, and re-scoring after fine-tuning.
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


def _discover_architectures(model_dir: str) -> List[str]:
    """Find every ``<arch>`` directory containing ``best.pt``."""
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--model-dir does not exist: {model_dir}")
    archs = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "best.pt").is_file():
            archs.append(child.name)
    if not archs:
        raise FileNotFoundError(
            f"No '<arch>/best.pt' directories found under {model_dir}."
        )
    return archs


def _resolve_model_config(model_configs_dir: str, arch: str) -> str:
    """Find configs/model/<arch>.yaml or fail with a clear error."""
    path = Path(model_configs_dir) / f"{arch}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find model config for arch {arch!r}: expected {path}. "
            "Pass --model-configs-dir pointing at the directory containing "
            "<arch>.yaml files."
        )
    return str(path)


def _load_model(model_config_path: str, checkpoint: str, device: str) -> torch.nn.Module:
    cfg = load_config(model_config_path)
    model = build_model(cfg)
    load_checkpoint(model, checkpoint, device=device)
    model.to(device)
    model.eval()
    return model


def _evaluate_one(
    *,
    arch: str,
    checkpoint: str,
    model_config: str,
    test_loader,
    device: str,
    test_dataset_dir: str,
    preproc_config_path: str,
) -> Dict:
    model = _load_model(model_config, checkpoint, device)
    metrics = evaluate_model(model, test_loader, device=device)
    metrics["arch"] = arch
    metrics["checkpoint"] = checkpoint
    metrics["model_config"] = model_config
    metrics["test_dataset_dir"] = test_dataset_dir
    metrics["preprocessing_config"] = preproc_config_path
    return metrics


def _print_metrics(label: str, metrics: Dict) -> None:
    print(f"[Evaluate] {label}")
    interesting = [
        "n_samples", "test_auc_roc", "test_accuracy", "test_f1",
        "test_precision", "test_recall", "test_specificity",
    ]
    for k in interesting:
        if k in metrics:
            v = metrics[k]
            print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more PCOS checkpoints on a preprocessed test "
            "set. The CLI is dataset-agnostic: pass any preprocessed "
            "dataset dir to --test-dataset-dir."
        ),
    )

    # ---- Single-model arguments ----
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory containing best.pt (single-model mode).",
    )
    parser.add_argument(
        "--model-config", default=None,
        help="Path to the model YAML for the single checkpoint.",
    )

    # ---- Multi-model arguments ----
    parser.add_argument(
        "--model-dir", default=None,
        help="Root containing <arch>/best.pt (multi-model mode).",
    )
    parser.add_argument(
        "--model-configs-dir", default="configs/model",
        help="Directory holding <arch>.yaml files (multi-model mode).",
    )

    # ---- Shared arguments ----
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
        help="Preprocessing YAML (default: configs/preprocessing.yaml).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=None,
        help=(
            "Single-model mode: where to write metrics JSON "
            "(default: <checkpoint-dir>/metrics.json). "
            "Multi-model mode: this is ignored; use --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Multi-model mode: writes <output-dir>/<arch>/metrics.json "
            "and an aggregated <output-dir>/all_metrics.json. "
            "Single-model mode: ignored."
        ),
    )

    args = parser.parse_args(argv)

    # ---- Mode validation ----
    has_single = args.checkpoint_dir is not None
    has_multi = args.model_dir is not None
    if has_single == has_multi:
        parser.error(
            "Pass exactly one of --checkpoint-dir (single-model) or "
            "--model-dir (multi-model)."
        )
    if has_single and args.model_config is None:
        parser.error("--model-config is required with --checkpoint-dir")

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    preproc_config = load_config(args.preprocessing_config)

    # Build the test loader once — reused across all architectures in
    # multi-model mode.
    test_loader = build_test_loader(
        args.test_dataset_dir,
        preproc_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if has_single:
        checkpoint = _resolve_checkpoint(args.checkpoint_dir)
        metrics = _evaluate_one(
            arch=Path(args.checkpoint_dir).name,
            checkpoint=checkpoint,
            model_config=args.model_config,
            test_loader=test_loader,
            device=device,
            test_dataset_dir=args.test_dataset_dir,
            preproc_config_path=args.preprocessing_config,
        )
        output = args.output or os.path.join(args.checkpoint_dir, "metrics.json")
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2)
        _print_metrics(f"Wrote {output}", metrics)
        return 0

    # ---- Multi-model mode ----
    archs = _discover_architectures(args.model_dir)
    output_root = args.output_dir or os.path.join(args.model_dir, "_eval")
    all_metrics: Dict[str, Dict] = {}
    for arch in archs:
        ckpt_dir = Path(args.model_dir) / arch
        checkpoint = _resolve_checkpoint(str(ckpt_dir))
        model_config = _resolve_model_config(args.model_configs_dir, arch)
        print(f"\n[Evaluate] >>> {arch}")
        try:
            metrics = _evaluate_one(
                arch=arch,
                checkpoint=checkpoint,
                model_config=model_config,
                test_loader=test_loader,
                device=device,
                test_dataset_dir=args.test_dataset_dir,
                preproc_config_path=args.preprocessing_config,
            )
        except Exception as e:
            print(f"[Evaluate] {arch} FAILED: {e}")
            all_metrics[arch] = {"error": repr(e)}
            continue

        per_arch_dir = os.path.join(output_root, arch)
        os.makedirs(per_arch_dir, exist_ok=True)
        per_arch_path = os.path.join(per_arch_dir, "metrics.json")
        with open(per_arch_path, "w") as f:
            json.dump(metrics, f, indent=2)
        _print_metrics(f"{arch} -> {per_arch_path}", metrics)
        all_metrics[arch] = metrics

    summary_path = os.path.join(output_root, "all_metrics.json")
    os.makedirs(output_root, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Evaluate] Wrote aggregate summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())