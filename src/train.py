"""
Unified PEARL training entry point.

This is the only command needed for both foundation training and
checkpoint-initialized training. It expects a fully preprocessed dataset::

    data/preprocessed/<dataset>/
        train/{images/*.png,label.csv}
        val/{images/*.png,label.csv}
        test/{images/*.png,label.csv}

Examples::

    # Train one model from ImageNet initialization (e.g. Figshare)
    python -m src.train \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny \
        --output-dir results/figshare

    # Train several models at once — saved under the names you passed
    python -m src.train \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny vit_base convnext_tiny densenet169 efficientnet_b0 \
        --output-dir results/figshare

    # Initialize one model from a checkpoint (fine-tune)
    python -m src.train \
        --dataset-dir data/preprocessed/pcosgen \
        --model swin_tiny \
        --checkpoint results/figshare/swin_tiny/best.pt \
        --output-dir results/pcosgen

    # Fine-tune all models in one invocation. Each model loads its own
    # matching best.pt from --model-dir.
    python -m src.train \
        --dataset-dir data/preprocessed/pcosgen \
        --model swin_tiny vit_base convnext_tiny densenet169 efficientnet_b0 \
        --model-dir results/figshare \
        --output-dir results/pcosgen

    # Dataset-specific Optuna study (single model only)
    python -m src.train \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny \
        --hpo --search-space configs/search_space/figshare.yaml \
        --n-trials 30 \
        --output-dir results/hpo/figshare/swin_tiny

The presence of ``--checkpoint`` does not activate a separate fine-tuning
pipeline; it simply changes model initialization. All runs use the same
DataLoader, Trainer, validation, early-stopping, and checkpoint code.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import yaml

from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer, TrialDivergedError
from src.utils.config import DEFAULT_PREPROCESSING_CONFIG, load_config
from src.utils.logging import ExperimentLogger
from src.utils.seed import set_seed


SUPPORTED_MODELS = (
    "swin_tiny",
    "vit_base",
    "convnext_tiny",
    "densenet169",
    "efficientnet_b0",
)
DEFAULT_TRAINING_CONFIG: Dict[str, object] = {
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 32,
    "sampler": "weighted",
    "max_epochs": 100,
    "early_stopping_patience": 20,
    "warmup_epochs": 2,
    "freeze_epochs": 0,
    "ema": True,
    "ema_decay": 0.999,
    "bf16": True,
    "channels_last": True,
    "compile": False,
    "grad_clip_norm": 1.0,
    "save_rng_state": True,
}


class _NullLogger:
    """No-op logger used inside Optuna trials."""

    def log_epoch(self, *args, **kwargs):
        pass

    def log_final_metrics(self, *args, **kwargs):
        pass

    def plot_training_curves(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class OptunaCallback:
    """Report Trainer validation AUC to Optuna after every epoch."""

    def __init__(self, trial):
        self.trial = trial

    def __call__(self, epoch: int, val_auc: float) -> None:
        import optuna

        self.trial.report(val_auc, epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def _resolve_model_path(model: str) -> str:
    """Accept a short model name or an explicit YAML path."""
    if model.endswith((".yaml", ".yml")):
        path = model
    else:
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model {model!r}. Expected one of: "
                f"{', '.join(SUPPORTED_MODELS)}"
            )
        path = os.path.join("configs", "model", f"{model}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model config not found: {path}")
    return path


def _load_model_config(model: str) -> dict:
    return load_config(_resolve_model_path(model))


def _load_initial_weights(model: torch.nn.Module, checkpoint: str, device: str) -> None:
    """Load model weights before Trainer constructs its optimizer."""
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    load_checkpoint(
        model,
        checkpoint,
        optimizer=None,
        scheduler=None,
        ema=None,
        device=device,
    )


def run_training(
    *,
    dataset_dir: str,
    model_name: str,
    preproc_config: dict,
    training_config: dict,
    output_dir: str,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    on_epoch_end: Optional[Callable[[int, float], None]] = None,
    silent: bool = False,
    logger=None,
) -> dict:
    """Run one training job through the canonical PEARL pipeline.

    The dataset must already contain train/val/test PNG splits. No split
    logic exists here or in the dataloader.

    ``model_name`` is the model identifier (or YAML path) used to
    resolve the architecture config; the output directory is decided
    by the caller, so this function does not derive it from
    ``model_name``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = _load_model_config(model_name)
    model_cfg = copy.deepcopy(model_cfg)

    # Model settings (dropout, freeze fraction) may be supplied by HPO or
    # CLI. Training-only keys are kept in training_config.
    if "dropout" in training_config:
        model_cfg.setdefault("head", {})["dropout"] = float(
            training_config["dropout"]
        )
    if "freeze_fraction" in training_config:
        model_cfg["freeze_fraction"] = float(training_config["freeze_fraction"])

    batch_size = int(training_config.get("batch_size", 32))
    sampler = str(training_config.get("sampler", "weighted"))
    num_workers = int(training_config.get("num_workers", 4))
    pin_memory = bool(training_config.get("pin_memory", device.startswith("cuda")))

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_dir,
        preproc_config,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )
    model = build_model(model_cfg)
    if checkpoint:
        _load_initial_weights(model, checkpoint, device)

    class_weights = train_loader.dataset.get_class_weights()
    criterion = build_weighted_loss(class_weights, device=device)

    os.makedirs(output_dir, exist_ok=True)
    run_config = {
        "dataset_dir": dataset_dir,
        "model": model_cfg,
        "preprocessing": preproc_config,
        "training": training_config,
        "checkpoint": checkpoint,
    }
    if logger is None:
        logger = ExperimentLogger(output_dir, run_config, resume=False)

    checkpoint_path = os.path.join(output_dir, "best.pt")
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
        on_epoch_end=on_epoch_end,
        silent=silent,
    )
    metrics = trainer.train()
    try:
        logger.log_final_metrics(metrics)
        logger.plot_training_curves()
        logger.close()
    except Exception:
        pass
    return metrics


def _suggest(trial, key: str, spec: dict):
    """Sample one value from a compact search-space schema."""
    kind = spec.get("type", "float")
    if kind == "categorical" or "choices" in spec:
        return trial.suggest_categorical(key, spec["choices"])
    if kind == "int":
        return trial.suggest_int(
            key,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    return trial.suggest_float(
        key,
        float(spec["low"]),
        float(spec["high"]),
        step=spec.get("step"),
        log=bool(spec.get("log", False)),
    )


def run_hpo(
    *,
    dataset_dir: str,
    model_name: str,
    preproc_config: dict,
    base_training_config: dict,
    search_space_path: str,
    output_dir: str,
    checkpoint: Optional[str] = None,
    n_trials: int = 30,
    study_name: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Run a dataset-specific Optuna study around ``run_training``."""
    import optuna

    search_cfg = load_config(search_space_path)
    search_space = search_cfg.get("search_space", search_cfg)
    if not search_space:
        raise ValueError(f"Empty search space: {search_space_path}")

    os.makedirs(output_dir, exist_ok=True)
    study_name = study_name or f"{Path(dataset_dir).name}_{model_name}"
    storage = f"sqlite:///{os.path.abspath(os.path.join(output_dir, 'study.db'))}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )

    def objective(trial):
        trial_cfg = copy.deepcopy(base_training_config)
        for key, spec in search_space.items():
            trial_cfg[key] = _suggest(trial, key, spec)

        trial_dir = os.path.join(output_dir, f"trial_{trial.number:04d}")
        callback = OptunaCallback(trial)
        try:
            metrics = run_training(
                dataset_dir=dataset_dir,
                model_name=model_name,
                preproc_config=preproc_config,
                training_config=trial_cfg,
                output_dir=trial_dir,
                checkpoint=checkpoint,
                device=device,
                on_epoch_end=callback,
                silent=True,
                logger=_NullLogger(),
            )
            return float(metrics["val_auc_best"])
        except optuna.exceptions.TrialPruned:
            raise
        except TrialDivergedError as exc:
            print(f"[HPO] Trial {trial.number} pruned: {exc}")
            raise optuna.exceptions.TrialPruned() from exc

    study.optimize(objective, n_trials=n_trials)
    result = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    with open(os.path.join(output_dir, "best_params.yaml"), "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(f"[HPO] Best val AUC: {study.best_value:.6f}")
    print(f"[HPO] Best params: {study.best_params}")
    return result


def _build_training_config(args) -> dict:
    cfg = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    cfg.update(
        {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "sampler": args.sampler,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "num_workers": args.num_workers,
            "dropout": args.dropout,
            "freeze_fraction": args.freeze_fraction,
            "warmup_epochs": args.warmup_epochs,
            "freeze_epochs": args.freeze_epochs,
            "ema": args.ema,
            "ema_decay": args.ema_decay,
            "bf16": args.bf16,
            "channels_last": args.channels_last,
            "compile": args.compile,
        }
    )
    return cfg


def _resolve_model_dir_checkpoint(model_dir: str, model_name: str) -> Optional[str]:
    """Find ``<model_dir>/<model_name>/best.pt`` if it exists.

    Used for fine-tune-all mode: each model loads its own checkpoint
    from a shared root.
    """
    candidate = os.path.join(model_dir, model_name, "best.pt")
    return candidate if os.path.isfile(candidate) else None


def _output_name(model_name: str) -> str:
    """Resolve the on-disk directory name for a model spec.

    Bare names like ``swin_tiny`` are used verbatim. YAML paths
    collapse to the file stem so the output tree stays clean.
    """
    if model_name.endswith((".yaml", ".yml")):
        return Path(model_name).stem
    return model_name


def run_multi(
    *,
    dataset_dir: str,
    model_names: List[str],
    preproc_config: dict,
    training_config: dict,
    output_dir: str,
    checkpoint: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, dict]:
    """Train or fine-tune multiple models sequentially.

    Each model is saved under ``<output_dir>/<model_name>/``. The
    ``model_name`` is used verbatim — that's also the directory the
    caller passes to ``--model-dir`` later for fine-tune-all mode.

    Args:
        dataset_dir: Preprocessed PNG dataset root.
        model_names: List of model short names (or YAML paths).
        preproc_config: Loaded preprocessing config.
        training_config: Built training config dict.
        output_dir: Root under which each model gets its own folder.
        checkpoint: Single checkpoint applied to every model.
        model_dir: Root searched per-model for a ``best.pt``
            (``<model_dir>/<model_name>/best.pt``). Ignored if
            ``checkpoint`` is also given.
        device: Device string.
        seed: Seed (reset per model for determinism).

    Returns:
        Mapping ``model_name → metrics dict``.
    """
    results: Dict[str, dict] = {}
    for name in model_names:
        ckpt = checkpoint
        out_name = _output_name(name)
        if ckpt is None and model_dir is not None:
            ckpt = _resolve_model_dir_checkpoint(model_dir, out_name)
            if ckpt is not None:
                print(f"[Multi] {out_name}: initializing from {ckpt}")
            else:
                print(f"[Multi] {out_name}: no checkpoint under "
                      f"{model_dir!r}, training from scratch")
        set_seed(seed)
        run_dir = os.path.join(output_dir, out_name)
        try:
            metrics = run_training(
                dataset_dir=dataset_dir,
                model_name=name,
                preproc_config=preproc_config,
                training_config=training_config,
                output_dir=run_dir,
                checkpoint=ckpt,
                device=device,
            )
            results[out_name] = metrics
        except Exception as exc:
            print(f"[Multi] {out_name} FAILED: {exc}")
            results[out_name] = {"error": str(exc)}
    print(f"\n[Multi] Done. {len(results)} models run. "
          f"Output root: {output_dir}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PEARL on preprocessed PNG data.")
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Preprocessed dataset root with train/val/test (e.g. data/preprocessed/figshare).",
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help=(
            "One or more model names or YAML paths (space-separated). "
            "Each model is saved under <output-dir>/<name>/. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}."
        ),
    )
    parser.add_argument(
        "--preprocessing", default=DEFAULT_PREPROCESSING_CONFIG,
        help=f"Preprocessing config (default: {DEFAULT_PREPROCESSING_CONFIG}).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional checkpoint used to initialize model weights "
             "(applied to every model when listing several).",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help=(
            "Root directory searched per-model for a best.pt to "
            "initialize from (e.g. results/figshare). With several "
            "--model names this fine-tunes every model from its own "
            "matching checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output root. Each model gets <output-dir>/<name>/. "
             "Default: results/<dataset_name>.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--freeze-fraction", type=float, default=0.60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--sampler", choices=("weighted", "shuffle", "none"), default="weighted"
    )

    # ---- Training-loop behavior flags (Phase 3 throughput opts) ----
    parser.add_argument(
        "--ema", dest="ema", action="store_true", default=True,
        help="Maintain EMA shadow weights and evaluate test set with them (default: True).",
    )
    parser.add_argument(
        "--no-ema", dest="ema", action="store_false",
        help="Disable EMA shadow weights.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--bf16", dest="bf16", action="store_true", default=True,
        help="Use bfloat16 autocast on CUDA (default: True).",
    )
    parser.add_argument(
        "--fp16", dest="bf16", action="store_false",
        help="Use float16 autocast instead of bfloat16.",
    )
    parser.add_argument(
        "--channels-last", dest="channels_last", action="store_true", default=True,
        help="Use channels_last memory format on CUDA (default: True).",
    )
    parser.add_argument(
        "--no-channels-last", dest="channels_last", action="store_false",
        help="Disable channels_last memory format.",
    )
    parser.add_argument(
        "--compile", action="store_true", default=False,
        help="Wrap the model in torch.compile (max-autotune).",
    )
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--freeze-epochs", type=int, default=0)

    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO.")
    parser.add_argument(
        "--search-space",
        help="Dataset-specific search-space YAML (required with --hpo).",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    if not os.path.isdir(args.dataset_dir):
        print(f"ERROR: --dataset-dir {args.dataset_dir} is not a directory")
        return 1
    if args.hpo and not args.search_space:
        print("ERROR: --search-space is required with --hpo")
        return 1
    if args.hpo and len(args.model) > 1:
        print("ERROR: --hpo supports a single --model.")
        return 1

    dataset_name = Path(args.dataset_dir).name
    output_dir = args.output_dir or os.path.join("results", dataset_name)
    preproc_config = load_config(args.preprocessing)
    training_config = _build_training_config(args)

    if args.hpo:
        run_hpo(
            dataset_dir=args.dataset_dir,
            model_name=args.model[0],
            preproc_config=preproc_config,
            base_training_config=training_config,
            search_space_path=args.search_space,
            output_dir=os.path.join(output_dir, args.model[0]),
            checkpoint=args.checkpoint,
            n_trials=args.n_trials,
            study_name=args.study_name,
            device=args.device,
            seed=args.seed,
        )
    elif len(args.model) == 1:
        # Single-model fast path.
        out_name = _output_name(args.model[0])
        run_dir = os.path.join(output_dir, out_name)
        metrics = run_training(
            dataset_dir=args.dataset_dir,
            model_name=args.model[0],
            preproc_config=preproc_config,
            training_config=training_config,
            output_dir=run_dir,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        print(json.dumps(metrics, indent=2, default=str))
    else:
        # Multi-model: train or fine-tune each model sequentially.
        run_multi(
            dataset_dir=args.dataset_dir,
            model_names=args.model,
            preproc_config=preproc_config,
            training_config=training_config,
            output_dir=output_dir,
            checkpoint=args.checkpoint,
            model_dir=args.model_dir,
            device=args.device,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())