#!/usr/bin/env python3
"""
CLI entrypoint for matrix sweeps (e.g. the v3 Phase 0 ablation:
9 models × 2 denoising configs = 18 runs, default hyperparameters).

Usage:
    python scripts/sweep.py --experiment configs/experiment/ablation_18.yaml
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger, make_run_dir
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer

# Rich UI for the sweep outer loop (ETA dashboard + summary table).
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.table import Table as RichTable
    from rich.text import Text as RichText
    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    _RICH_AVAILABLE = False


def _rich_supported() -> bool:
    if not _RICH_AVAILABLE:
        return False
    if "pytest" in sys.modules:
        return False
    if not sys.stdout.isatty():
        return False
    return True


def _make_summary_table(results: list) -> "RichTable":
    """Build a richly-colored summary table from collected run metrics."""
    tbl = RichTable(
        title="[bold cyan]Sweep Results — Test Set[/]",
        title_justify="left",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        expand=True,
    )
    tbl.add_column("#", justify="right", style="dim")
    tbl.add_column("arch", style="bold")
    tbl.add_column("preproc")
    tbl.add_column("best_epoch", justify="right")
    tbl.add_column("epochs", justify="right")
    tbl.add_column("test_auc", justify="right")
    tbl.add_column("test_acc", justify="right")
    tbl.add_column("test_f1", justify="right")
    tbl.add_column("test_mcc", justify="right")
    tbl.add_column("sens", justify="right")
    tbl.add_column("spec", justify="right")
    tbl.add_column("time(s)", justify="right", style="dim")
    tbl.add_column("ckpt", justify="left", style="dim cyan")

    for i, m in enumerate(results, 1):
        auc = m.get("test_auc_roc", float("nan"))
        f1 = m.get("test_f1", float("nan"))
        mcc = m.get("test_mcc", float("nan"))
        acc = m.get("test_accuracy", float("nan"))
        sens = m.get("test_recall", float("nan"))
        spec = m.get("test_specificity", float("nan"))

        # Color the AUC value: green/yellow/red thresholds.
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
            auc_str,
            f"{acc:.4f}" if not _isnan(acc) else "—",
            f"{f1:.4f}" if not _isnan(f1) else "—",
            f"{mcc:.4f}" if not _isnan(mcc) else "—",
            f"{sens:.4f}" if not _isnan(sens) else "—",
            f"{spec:.4f}" if not _isnan(spec) else "—",
            f"{m.get('total_train_time_sec', 0):.0f}",
            f"{m.get('preprocessing', '?')}/{m.get('arch', '?')}.pt",
        )

    return tbl


def _isnan(x):
    try:
        import math
        return math.isnan(x)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full model × preprocessing sweep")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config YAML")
    args = parser.parse_args()

    experiment_config = load_config(args.experiment)
    training_config = experiment_config.get("training", {})
    seed = experiment_config.get("seed", 42)
    results_dir = experiment_config.get("results_dir", "results/")

    models = experiment_config.get("models", [])
    preprocessings = experiment_config.get("preprocessing", [])

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_runs = len(models) * len(preprocessings)
    use_rich = _rich_supported()
    console = Console() if use_rich else None

    if use_rich:
        console.print(
            Panel(
                f"[bold]{len(models)}[/] models × [bold]{len(preprocessings)}[/] preprocessing "
                f"= [bold cyan]{total_runs}[/] runs\n"
                f"device: [yellow]{device}[/]    seed: [yellow]{seed}[/]\n"
                f"results: [dim]{results_dir}/runs/[/]",
                title="[bold magenta]◆ PEARL Sweep[/]",
                border_style="magenta",
            )
        )
    else:
        print(f"[Sweep] {len(models)} models × {len(preprocessings)} preprocessing = {total_runs} runs")
        print(f"[Sweep] Device: {device}\n")

    all_results = []
    run_idx = 0

    # Outer-loop progress bar — tracks completion across the whole sweep.
    outer_progress = None
    outer_task = None
    outer_live = None
    started_at = time.time()
    run_started_at = None

    if use_rich:
        outer_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Sweep[/]"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )
        outer_task = outer_progress.add_task("starting…", total=total_runs)
        outer_live = Live(
            outer_progress, console=console,
            refresh_per_second=4, transient=False,
        )
        outer_live.__enter__()

    for model_name in models:
        model_config_path = os.path.join("configs", "model", f"{model_name}.yaml")
        model_config = load_config(model_config_path)

        for preproc_name in preprocessings:
            run_idx += 1
            preproc_config_path = os.path.join("configs", "preprocessing", f"{preproc_name}.yaml")
            preproc_config = load_config(preproc_config_path)

            if outer_progress is not None and outer_task is not None:
                # Update ETA estimate: average across runs seen so far.
                if run_idx > 1 and run_started_at is not None:
                    avg = (time.time() - started_at) / (run_idx - 1)
                    eta_total = avg * (total_runs - run_idx + 1)
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_total))
                else:
                    eta_str = "—"
                outer_progress.update(
                    outer_task, advance=1,
                    description=f"[{model_name} + {preproc_name}] | ETA {eta_str}",
                )

            if not use_rich:
                print(f"\n{'='*60}")
                print(f"[Sweep] Run {run_idx}/{total_runs}: {model_name} + {preproc_name}")
                print(f"{'='*60}")

            run_started_at = time.time()
            set_seed(seed)

            # Run directory
            run_dir = make_run_dir(results_dir, model_name, preproc_name)
            full_config = {
                "model": model_config,
                "preprocessing": preproc_config,
                "experiment": experiment_config,
                "seed": seed,
            }
            logger = ExperimentLogger(run_dir, full_config)

            # Data
            input_size = model_config.get("input_size", 224)
            batch_size = training_config.get("batch_size", 32)
            sampler = training_config.get("sampler", "shuffle")
            train_loader, val_loader, test_loader = build_dataloaders(
                preproc_config, batch_size=batch_size, input_size=input_size,
                sampler=sampler,
            )

            # Model
            model = build_model(model_config)

            # Loss
            class_weights = train_loader.dataset.get_class_weights()
            criterion = build_weighted_loss(class_weights, device=device)

            # Checkpoint path: <results>/checkpoints/<prep>/<model>.pt
            checkpoint_dir = os.path.join(results_dir, "checkpoints", preproc_name)
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pt")

            # Train
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
            )

            metrics = trainer.train()
            metrics["arch"] = model_name
            metrics["preprocessing"] = preproc_name
            metrics["preprocessing_full"] = preproc_name  # for table
            all_results.append(metrics)

            # Free GPU memory
            del model, trainer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Close the outer Live context.
    if outer_live is not None:
        outer_live.__exit__(None, None, None)

    # Build sweep matrix CSV
    df = pd.DataFrame(all_results)
    matrix_path = os.path.join(results_dir, "sweep_matrix.csv")
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    df.to_csv(matrix_path, index=False)

    # Render the final summary table.
    total_time = time.time() - started_at
    if use_rich and console is not None:
        console.print(_make_summary_table(all_results))
        console.print(
            f"\n[bold green]✓ Sweep complete[/] "
            f"[dim]({total_runs} runs in {total_time/60:.1f}m "
            f"avg {total_time/max(total_runs,1):.0f}s/run)[/]"
        )
        console.print(f"[dim]Results matrix saved to {matrix_path}[/]")
    else:
        print(f"\n[Sweep] Results matrix saved to {matrix_path}")
        print(f"[Sweep] All {total_runs} runs complete!")


if __name__ == "__main__":
    main()
