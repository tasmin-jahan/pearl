#!/usr/bin/env python3
"""Fine-tune Figshare-trained no-pad checkpoints on Zenodo PCOSgen.

For each (preprocessing, architecture) pair, this script:
  1. Loads the matching Figshare checkpoint from results/ablation_nopad/.
  2. Builds in-memory train/val loaders from
     data_external/zenodo_splits/{train,val}.json using the same
     preprocessing as the Figshare checkpoint (srad_nopad or gauss_nopad).
  3. Initializes a Trainer with the new LR / batch size / epochs.
  4. Loads ONLY the Figshare model weights (not optimizer state) and
     starts training from epoch 1 with the new optimizer.
  5. Skips runs whose <run_dir>/best.pt already exists.
  6. Aggregates results into results/finetune_zenodo/sweep_matrix.csv.

Usage:
    python scripts/sweep_finetune.py --experiment configs/experiment/finetune_zenodo.yaml
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger, make_arch_dir
from src.data.zenodo_dataset import build_zenodo_loader
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer
from src.training.checkpoint import load_checkpoint

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.table import Table as RichTable
    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False


def _load_split(path: str) -> List[Tuple[str, int]]:
    with open(path) as f:
        d = json.load(f)
    return [(item["path"], int(item["label"])) for item in d["items"]]


def _rich_supported() -> bool:
    if not _RICH_AVAILABLE:
        return False
    if not sys.stdout.isatty():
        return False
    return True


def _make_summary_table(results: list) -> "RichTable":
    tbl = RichTable(
        title="[bold cyan]Fine-tune Sweep — Zenodo Test[/]",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True,
    )
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("arch", style="bold")
    tbl.add_column("prep")
    tbl.add_column("best_epoch", justify="right")
    tbl.add_column("epochs", justify="right")
    tbl.add_column("zenodo_val_auc", justify="right")
    tbl.add_column("zenodo_test_auc", justify="right")
    tbl.add_column("zenodo_test_acc", justify="right")
    tbl.add_column("zenodo_test_f1", justify="right")
    tbl.add_column("zenodo_test_mcc", justify="right")
    tbl.add_column("time(s)", justify="right", style="dim")
    tbl.add_column("ckpt", justify="left", style="dim cyan")

    for i, m in enumerate(results, 1):
        auc = m.get("external_test_auc", m.get("test_auc_roc", float("nan")))
        val_auc = m.get("val_auc_best", float("nan"))
        acc = m.get("test_accuracy", float("nan"))
        f1 = m.get("test_f1", float("nan"))
        mcc = m.get("test_mcc", float("nan"))

        if auc >= 0.95:
            auc_str = f"[bold green]{auc:.4f}[/]"
        elif auc >= 0.85:
            auc_str = f"[green]{auc:.4f}[/]"
        elif auc >= 0.70:
            auc_str = f"[yellow]{auc:.4f}[/]"
        else:
            auc_str = f"[red]{auc:.4f}[/]"

        tbl.add_row(
            str(i),
            m.get("arch", "?"),
            m.get("preprocessing", "?"),
            str(m.get("best_epoch", "?")),
            str(m.get("epochs_trained", "?")),
            f"{val_auc:.4f}" if val_auc == val_auc else "—",
            auc_str,
            f"{acc:.4f}" if acc == acc else "—",
            f"{f1:.4f}" if f1 == f1 else "—",
            f"{mcc:.4f}" if mcc == mcc else "—",
            f"{m.get('total_train_time_sec', 0):.0f}",
            f"{m.get('preprocessing', '?')}/{m.get('arch', '?')}/best.pt",
        )

    return tbl


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sweep on Zenodo PCOSgen")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would run without launching training.")
    args = parser.parse_args()

    exp_config = load_config(args.experiment)
    training_config = exp_config.get("training", {})
    data_config = exp_config.get("data", {})
    seed = exp_config.get("seed", 42)
    results_dir = exp_config.get("results_dir", "results/finetune_zenodo/")

    models = exp_config.get("models", [])
    preprocessings = exp_config.get("preprocessing", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SweepFinetune] {len(models)} models × {len(preprocessings)} preprocessing = "
          f"{len(models) * len(preprocessings)} runs")
    print(f"[SweepFinetune] Device: {device}, seed: {seed}")
    print(f"[SweepFinetune] Resume root: {data_config.get('resume_root')}")
    print(f"[SweepFinetune] Train split: {data_config.get('train_split')}")
    print(f"[SweepFinetune] Val split:   {data_config.get('val_split')}")
    print(f"[SweepFinetune] External dir:{data_config.get('external_dir')} "
          f"(layout={data_config.get('external_layout')})")

    # Load splits ONCE (shared across all runs)
    train_pairs = _load_split(data_config["train_split"])
    val_pairs = _load_split(data_config["val_split"])
    print(f"[SweepFinetune] Loaded {len(train_pairs)} train + {len(val_pairs)} val pairs")
    print(f"  train class dist: {dict(Counter(l for _, l in train_pairs))}")
    print(f"  val   class dist: {dict(Counter(l for _, l in val_pairs))}")

    if args.dry_run:
        return

    use_rich = _rich_supported()
    console = Console() if use_rich else None
    started_at = time.time()
    all_results = []

    if use_rich:
        console.print(Panel(
            f"[bold]{len(models)}[/] models × [bold]{len(preprocessings)}[/] preprocessing "
            f"= [bold cyan]{len(models) * len(preprocessings)}[/] runs\n"
            f"device: [yellow]{device}[/]    seed: [yellow]{seed}[/]\n"
            f"results: [dim]{results_dir}[/]",
            title="[bold magenta]◆ PEARL Fine-tune Sweep[/]",
            border_style="magenta",
        ))

    total_runs = len(models) * len(preprocessings)
    run_idx = 0

    for model_name in models:
        model_config_path = os.path.join("configs", "model", f"{model_name}.yaml")
        model_config = load_config(model_config_path)

        for preproc_name in preprocessings:
            run_idx += 1
            preproc_config_path = os.path.join(
                "configs", "preprocessing", f"{preproc_name}.yaml",
            )
            preproc_config = load_config(preproc_config_path)

            arch_dir = make_arch_dir(results_dir, preproc_name, model_name)
            ckpt_path = os.path.join(arch_dir, "best.pt")

            # Skip if already done.
            if os.path.isfile(ckpt_path):
                print(f"[SweepFinetune] Run {run_idx}/{total_runs}: "
                      f"{model_name} + {preproc_name} — skipping (best.pt exists)")
                # Try to read the existing final_metrics.json so we can include
                # the run in the sweep_matrix without re-training.
                existing = os.path.join(arch_dir, "final_metrics.json")
                if os.path.isfile(existing):
                    with open(existing) as f:
                        m = json.load(f)
                    m["arch"] = model_name
                    m["preprocessing"] = preproc_name
                    m["skipped"] = True
                    all_results.append(m)
                continue

            print(f"\n{'='*60}")
            print(f"[SweepFinetune] Run {run_idx}/{total_runs}: "
                  f"{model_name} + {preproc_name}")
            print(f"{'='*60}")

            set_seed(seed)

            # Build data
            input_size = model_config.get("input_size", 224)
            batch_size = training_config.get("batch_size", 32)
            sampler = training_config.get("sampler", "weighted")
            train_loader, val_loader = build_zenodo_loader(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                preproc_config=preproc_config,
                batch_size=batch_size,
                input_size=input_size,
                sampler=sampler,
            )
            # We pass `val_loader` as `test_loader` for the trainer. Internal
            # "test" metrics here mean Zenodo val (640). The Zenodo test
            # (1468) is evaluated separately by evaluate_external.py after
            # the fine-tune.

            full_config = {
                "model": model_config,
                "preprocessing": preproc_config,
                "experiment": exp_config,
                "seed": seed,
            }
            logger = ExperimentLogger(arch_dir, full_config)

            # Model — override freeze_fraction: fine-tuning wants the entire
            # backbone trainable. The default 0.60 (set in per-arch model
            # configs) is right for from-scratch ImageNet warmup but wrong
            # here: we want to ADAPT the no-pad features to the new domain.
            model_config = dict(model_config)
            model_config["freeze_fraction"] = 0.0
            model = build_model(model_config)

            # Load ONLY Figshare weights — we want a fresh optimizer with the
            # new LR (3e-5), not the Figshare optimizer state.
            resume_path = os.path.join(
                data_config["resume_root"], preproc_name, model_name, "best.pt",
            )
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_path}. "
                    "Run the no-pad ablation first."
                )
            print(f"[SweepFinetune] Loading Figshare weights from {resume_path}")
            load_checkpoint(model, resume_path, optimizer=None, scheduler=None,
                            ema=None, device=device)

            # Loss
            class_weights = train_loader.dataset.get_class_weights()
            criterion = build_weighted_loss(class_weights, device=device)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=val_loader,  # internal "test" = Zenodo val
                criterion=criterion,
                config=training_config,
                logger=logger,
                device=device,
                checkpoint_path=ckpt_path,
            )
            metrics = trainer.train()  # no resume_from: weights already loaded
            metrics["arch"] = model_name
            metrics["preprocessing"] = preproc_name
            metrics["skipped"] = False
            metrics["resume_from"] = resume_path
            all_results.append(metrics)

            # Free GPU memory
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

    total_time = time.time() - started_at

    # Sweep matrix CSV
    matrix_path = os.path.join(results_dir, "sweep_matrix.csv")
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(matrix_path, index=False)
    print(f"\n[SweepFinetune] Sweep matrix written to {matrix_path}")

    if use_rich and console is not None:
        console.print(_make_summary_table(all_results))
        console.print(
            f"\n[bold green]✓ Fine-tune sweep complete[/] "
            f"[dim]({total_runs} runs in {total_time/60:.1f}m "
            f"avg {total_time/max(total_runs,1):.0f}s/run)[/]"
        )
    else:
        print(f"\n[SweepFinetune] Done. {total_runs} runs in {total_time/60:.1f}m")


if __name__ == "__main__":
    main()