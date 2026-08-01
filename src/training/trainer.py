"""
Training loop with AdamW + cosine annealing + early stopping, v3 features:
  - LR warmup + two-phase fine-tuning (freeze backbone initially, ramp LR)
  - Cosine annealing warm restarts + gradient clipping
  - No-decay parameter groups (BN/bias excluded from weight_decay)
  - NaN/Inf divergence guard (aborts the whole trial)
  - Throughput optimizations: torch.compile, bf16 autocast, TF32 matmul,
    channels_last, pin_memory + non_blocking transfers
  - EMA shadow weights (Phase 3.1)
  - RNG + dataloader state resumption (Phase 3.3)
  - Self-contained checkpoint with arch + class_names (Phase 3.2)
  - Rich UI: per-epoch progress bar + colored epoch summary panel

The trainer writes only ``best.pt``; no rolling-window checkpoints are
retained. Each per-epoch checkpoint can easily weigh hundreds of MB, so
keeping every recent epoch fills disk for no paper-grade benefit.
"""

import copy
import json
import os
import time
import warnings
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.metrics import compute_epoch_metrics
from src.training.checkpoint import save_checkpoint
from src.utils.logging import ExperimentLogger

# Rich UI — graceful fallback if the user doesn't have rich installed.
try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - rich is optional
    _RICH_AVAILABLE = False


def _rich_supported() -> bool:
    """Return True if rich output is usable in the current environment.

    The trainer is also invoked under pytest collection (where stdout is
    captured) and inside Optuna trials (where a live display is unwanted).
    Both callers can pass ``silent=True`` to disable this layer.
    """
    if not _RICH_AVAILABLE:
        return False
    # Disable rich when running under pytest (no TTY, no point in live UI).
    import sys as _sys
    if "pytest" in _sys.modules:
        return False
    # Disable when stdout is not a TTY (e.g. redirected to a file).
    if not _sys.stdout.isatty():
        return False
    return True


# Module-level handshake for nested Live contexts. The sweep.py outer
# loop sets ``_OUTER_LIVE_ACTIVE = True`` before entering its Live
# context and resets it on exit. The trainer checks this flag at
# __init__ time to avoid opening an inner Live that would jitter
# against the outer one. Callers wanting to opt back in (rare) can
# pass ``config={"use_inner_live": True}``.
_OUTER_LIVE_ACTIVE = False


def set_outer_live_active(active: bool) -> None:
    """Toggle the module-level outer-Live-active flag.

    Sweep scripts call ``set_outer_live_active(True)`` before entering
    their Rich ``Live`` context manager, and ``set_outer_live_active(False)``
    on exit. Trainer instances created while this flag is True skip
    their own per-epoch Live and fall back to plain console.print().
    """
    global _OUTER_LIVE_ACTIVE
    _OUTER_LIVE_ACTIVE = active


def _outer_live_active() -> bool:
    return _OUTER_LIVE_ACTIVE


class TrialDivergedError(RuntimeError):
    """Raised when loss/grad-norm goes non-finite mid-training.

    Callers (Optuna sweep, smoke tests) should prune the trial rather
    than try to recover: a NaN mid-run almost always indicates a
    diverging configuration, not a one-off bad batch.
    """


class EMA:
    """Exponential moving average of model parameters.

    Holds a slow-moving shadow copy of active weights. The shadow is
    updated each step; the swap-by-context-manager interface lets you
    evaluate using EMA weights without permanently modifying the model.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module):
        """Snapshot current weights so we can restore them after evaluation."""
        self._backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters() if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if hasattr(self, "_backup") and name in self._backup:
                p.data.copy_(self._backup[name])


class Trainer:
    """Full training loop manager.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        criterion: Loss function.
        config: Training config dict.
        logger: ExperimentLogger instance.
        device: Device string ('cuda' or 'cpu').
        checkpoint_path: Where to save the best checkpoint.
        on_epoch_end: Optional callback(epoch, val_auc) called after validation.
            Used by Optuna for trial.report/should_prune.
        silent: If True, suppress per-epoch console printing (useful inside sweeps).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        criterion: nn.Module,
        config: dict,
        logger: ExperimentLogger,
        device: str = "cuda",
        checkpoint_path: str = "results/checkpoints/best.pt",
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
        silent: bool = False,
    ):
        self.config = config
        self.silent = silent
        self.on_epoch_end = on_epoch_end

        # ---- Throughput optimizations (Phase 2.5) ----
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.model = model.to(device)
        if config.get("channels_last", True) and device.startswith("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)

        # Optional torch.compile — may fail on some environments; warn, don't error.
        if config.get("compile", False):
            try:
                self.model = torch.compile(
                    self.model, mode=config.get("compile_mode", "max-autotune"),
                )
            except Exception as e:
                warnings.warn(f"torch.compile failed ({e}); continuing without compile.")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.logger = logger
        self.device = device
        self.checkpoint_path = checkpoint_path

        # ---- Hyperparameters ----
        self.lr = config.get("lr", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-2)
        self.beta1 = config.get("beta1", 0.9)
        self.beta2 = config.get("beta2", 0.999)
        self.eps = config.get("eps", 1e-8)

        self.warmup_epochs = config.get("warmup_epochs", 2)
        self.freeze_epochs = config.get("freeze_epochs", 0)
        self.max_epochs = config.get("max_epochs", 100)
        self.patience = config.get("early_stopping_patience", 20)
        self.ema_decay = config.get("ema_decay", 0.999)
        self.use_ema = config.get("ema", True)
        self.amp_dtype = (
            torch.bfloat16 if config.get("bf16", True) else torch.float16
        )

        # ---- Optimizer (no-decay groups — Phase 2.3) ----
        self.optimizer = self._build_optimizer()

        # ---- Scheduler (CosineAnnealingWarmRestarts — Phase 2.2) ----
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=max(self.max_epochs // 4, 1), T_mult=2,
        )

        # ---- Grad clipping (Phase 2.2) ----
        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)

        # ---- State ----
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.start_epoch = 1
        self.global_step = 0

        # ---- EMA ----
        self.ema: Optional[EMA] = None
        if self.use_ema and device.startswith("cuda"):
            try:
                self.ema = EMA(self.model, decay=self.ema_decay)
            except Exception:
                self.ema = None

        # ---- RNG state (Phase 3.3) — restored on resume ----
        self._save_rng = config.get("save_rng_state", True)

        # Optional initial backbone freeze
        if self.freeze_epochs > 0:
            self._freeze_backbone(True)

        # ---- Rich console (created lazily; reused across epochs) ----
        self._console = Console() if _rich_supported() else None

        # ---- Inner Live control ----
        # When running inside an outer Rich Live context (e.g. the
        # sweep.py outer loop), nested Live contexts cause terminal
        # jitter because both fight for the cursor. Callers can either:
        #   (a) pass ``silent=True`` (no per-epoch printing at all), or
        #   (b) pass ``use_inner_live=False`` to fall back to plain
        #       ``console.print()`` of the epoch panel — which renders
        #       cleanly even when an outer Live is wrapping the sweep.
        self._use_inner_live = not (
            self.silent or _outer_live_active() or not config.get("use_inner_live", True)
        )

    # ------------------------------------------------------------------
    # Optimizer with no-decay param groups
    # ------------------------------------------------------------------
    def _build_optimizer(self):
        """AdamW with two parameter groups: decay vs no-decay.

        No-decay group includes biases, BatchNorm scales/shifts, and
        LayerNorm scales/shifts. Decaying these hurts calibration.
        """
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                p.ndim <= 1  # biases and 1D parameters
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "bn" in name.lower()
            ):
                no_decay.append(p)
            else:
                decay.append(p)

        param_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            param_groups, lr=self.lr,
            betas=(self.beta1, self.beta2), eps=self.eps,
        )

    def _freeze_backbone(self, freeze: bool):
        """Freeze/unfreeze the backbone (everything except the head)."""
        for name, p in self.model.named_parameters():
            if "head" not in name and not name.startswith("head."):
                p.requires_grad = not freeze

    # ------------------------------------------------------------------
    # Rich UI rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _color_auc(auc: float) -> str:
        """Map AUC to a color: green for high, yellow for mid, red for low."""
        if auc >= 0.95:
            return "bold green"
        if auc >= 0.85:
            return "green"
        if auc >= 0.70:
            return "yellow"
        return "red"

    @staticmethod
    def _color_loss(loss: float) -> str:
        if loss < 0.2:
            return "bold green"
        if loss < 0.5:
            return "green"
        if loss < 1.0:
            return "yellow"
        return "red"

    def _render_epoch_panel(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_auc: float,
        val_f1: float,
        val_mcc: float,
        lr: float,
        epoch_time: float,
        epochs_without_improvement: int = 0,
    ):
        """Build a richly-colored epoch summary panel."""
        # Compute delta from best so the user can see improvement at a glance.
        star = "  "
        if val_auc > self.best_val_auc:
            star = "★ "
        if val_auc >= self.best_val_auc and self.best_val_auc > 0.0:
            star = "★ "  # already best

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="dim")
        grid.add_column(justify="left")

        grid.add_row("train_loss", f"[{self._color_loss(train_loss)}]{train_loss:.4f}[/]")
        grid.add_row("val_loss",   f"[{self._color_loss(val_loss)}]{val_loss:.4f}[/]")
        grid.add_row("train_acc",   f"{train_acc:.4f}")
        grid.add_row("val_acc",     f"{val_acc:.4f}")
        grid.add_row("val_auc",     f"[{self._color_auc(val_auc)}]{val_auc:.4f}[/]")
        grid.add_row("val_f1",      f"[{self._color_auc(val_f1)}]{val_f1:.4f}[/]")
        grid.add_row("val_mcc",     f"[{self._color_auc(val_mcc)}]{val_mcc:.4f}[/]")
        grid.add_row("lr",          f"{lr:.2e}")
        grid.add_row("epoch_time",  f"{epoch_time:.1f}s")
        if epochs_without_improvement > 0:
            grid.add_row(
                "patience",
                f"[yellow]{epochs_without_improvement}/{self.patience}[/]",
            )

        best_str = f"best={self.best_val_auc:.4f}@e{self.best_epoch}"
        title = f"[bold cyan]{star}Epoch {epoch:03d}[/]  [dim]{best_str}[/]"

        return Panel(
            grid,
            title=title,
            border_style="cyan",
            padding=(0, 1),
            expand=False,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, resume_from: Optional[str] = None) -> dict:
        """Run the full training loop.

        Args:
            resume_from: Optional path to a checkpoint to resume from.
                Restores model/optimizer/scheduler/RNG/dataloader state.

        Returns:
            Dict of final test-set metrics.
        """
        if resume_from is not None:
            self._resume_state(resume_from)

        use_rich = (not self.silent) and _rich_supported()
        if not self.silent:
            print(f"\n{'='*60}")
            print(f"Training for up to {self.max_epochs} epochs "
                  f"(patience={self.patience}, warmup={self.warmup_epochs})")
            print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            epoch_start = time.time()

            # ---- Unfreeze backbone after freeze_epochs ----
            if self.freeze_epochs > 0 and epoch == self.freeze_epochs + 1:
                if not self.silent:
                    print(f"[Trainer] Unfreezing backbone at epoch {epoch}")
                self._freeze_backbone(False)
                # Re-build optimizer with newly trainable params (lower LR)
                self.lr *= 0.1
                self.optimizer = self._build_optimizer()
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=max((self.max_epochs - epoch) // 4, 1),
                    T_mult=2,
                )

            # ---- LR warmup ----
            if epoch <= self.warmup_epochs:
                warmup_lr = self.lr * (epoch / max(1, self.warmup_epochs))
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Rich Live progress for the whole epoch (train + validate).
            # Skip the inner Live if either the caller is silent OR an
            # outer Rich Live is already running (e.g. sweep.py) — nested
            # Lives cause terminal jitter.
            train_progress = None
            live_ctx = None
            epoch_table = None
            if use_rich and self._use_inner_live:
                train_progress = Progress(
                    SpinnerColumn(),
                    TextColumn(f"[bold cyan]E{epoch:03d}"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    console=self._console,
                    transient=True,
                )
                epoch_table = Table.grid(padding=(0, 1))
                epoch_table.add_row(train_progress)
                live_ctx = Live(
                    epoch_table, console=self._console,
                    refresh_per_second=8, transient=False,
                )
                live_ctx.__enter__()
                train_task = train_progress.add_task(
                    "training", total=len(self.train_loader),
                )
                self._live_ctx = live_ctx
                self._train_progress = train_progress
                self._train_task = train_task
            else:
                self._live_ctx = None
                self._train_progress = None
                self._train_task = None
                # If rich is available but inner Live is disabled, log
                # a one-line per-epoch progress hint so the user still
                # sees something moving in the terminal.
                if use_rich and not self.silent and self._console is not None:
                    self._console.print(
                        f"[bold cyan]E{epoch:03d}[/] training {len(self.train_loader)} batches…"
                    )

            try:
                train_loss, train_acc = self._train_one_epoch(epoch)
                val_loss, val_acc, val_auc, val_f1, val_mcc, val_metrics = self._validate()
            except TrialDivergedError as e:
                if live_ctx is not None:
                    live_ctx.__exit__(None, None, None)
                    self._live_ctx = None
                if not self.silent:
                    print(f"\n[Trainer] TrialDiverged at epoch {epoch}: {e}")
                raise
            except Exception:
                if live_ctx is not None:
                    live_ctx.__exit__(None, None, None)
                    self._live_ctx = None
                raise

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Render the rich epoch summary panel before exiting Live context
            if use_rich and live_ctx is not None:
                panel = self._render_epoch_panel(
                    epoch, train_loss, train_acc,
                    val_loss, val_acc, val_auc, val_f1, val_mcc,
                    current_lr, epoch_time,
                    epochs_without_improvement=self.epochs_without_improvement,
                )
                # Stop the progress bar so the panel sits at the bottom.
                if train_progress is not None:
                    train_progress.update(self._train_task, completed=len(self.train_loader))
                # Add panel as the second row of the table.
                epoch_table.add_row(panel)
                live_ctx.__exit__(None, None, None)
                self._live_ctx = None
            elif use_rich and live_ctx is None and not self.silent and self._console is not None:
                # Inner Live is disabled (e.g. running under an outer Live
                # from sweep.py). Print the same panel as a static block
                # — no cursor-racing, no jitter.
                panel = self._render_epoch_panel(
                    epoch, train_loss, train_acc,
                    val_loss, val_acc, val_auc, val_f1, val_mcc,
                    current_lr, epoch_time,
                    epochs_without_improvement=self.epochs_without_improvement,
                )
                self._console.print(panel)

            self.logger.log_epoch(
                epoch, train_loss, val_loss,
                train_acc, val_acc, val_auc, val_f1, val_mcc,
                current_lr, epoch_time,
                val_precision=val_metrics["precision"],
                val_recall=val_metrics["recall"],
                val_specificity=val_metrics["specificity"],
                val_nll=val_metrics["nll"],
                val_tp=val_metrics["tp"],
                val_fp=val_metrics["fp"],
                val_tn=val_metrics["tn"],
                val_fn=val_metrics["fn"],
            )

            # ---- EMA update ----
            if self.ema is not None:
                self.ema.update(self.model)

            # ---- Improvement check + checkpoint ----
            improved = val_auc > self.best_val_auc
            if improved:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_best(epoch, val_auc)
            else:
                self.epochs_without_improvement += 1

            # ---- Scheduler step (only after warmup) ----
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            # ---- Per-epoch callback (Optuna pruning) ----
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, val_auc)

            # ---- Early stopping ----
            if self.epochs_without_improvement >= self.patience:
                if use_rich:
                    self._console.print(
                        f"[bold yellow]⏹  Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)[/]"
                    )
                elif not self.silent:
                    print(f"\n[Trainer] Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        stopped_early = self.epochs_without_improvement >= self.patience
        epochs_trained = epoch

        if not self.silent:
            if use_rich and self._console is not None:
                self._console.print(
                    f"\n[bold green]✓ Training complete[/] "
                    f"[dim]({total_time:.0f}s, {epochs_trained} epochs)[/]\n"
                    f"  best val AUC = [bold cyan]{self.best_val_auc:.4f}[/] "
                    f"@ epoch [bold]{self.best_epoch}[/]"
                    + ("  [yellow](early-stopped)[/]" if stopped_early else "")
                )
            else:
                print(f"\n[Trainer] Training complete in {total_time:.0f}s")
                print(f"[Trainer] Best val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

        # ---- Load best (or EMA-weighted) checkpoint for test eval ----
        eval_metrics, (labels_np, probs_np, preds_np) = self._evaluate_test_with_ema()

        # ---- Save ROC / PR curves + confusion matrices to disk ----
        try:
            self._save_curve_artifacts(labels_np, probs_np, preds_np)
        except Exception as e:
            if not self.silent:
                print(f"[Trainer] curve-artifact save skipped: {e}")

        # ---- Build final metrics dict ----
        final_metrics = {
            "best_epoch": self.best_epoch,
            "epochs_trained": epochs_trained,
            "stopped_early": stopped_early,
            "val_auc_best": self.best_val_auc,
            "total_train_time_sec": round(total_time, 1),
            "params_total": sum(p.numel() for p in self.model.parameters()),
            "params_trainable": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "n_train": len(self.train_loader.dataset),
            "n_val": len(self.val_loader.dataset),
            "n_test": len(self.test_loader.dataset),
        }
        final_metrics.update(eval_metrics)

        self.logger.log_final_metrics(final_metrics)
        try:
            self.logger.plot_training_curves()
        except Exception as e:
            if not self.silent:
                print(f"[Trainer] plot_training_curves skipped: {e}")
        self.logger.close()

        # ---- Rich final summary panel ----
        if use_rich and self._console is not None:
            self._render_final_panel(final_metrics, total_time)

        return final_metrics

    def _render_final_panel(self, final_metrics: dict, total_time: float):
        """Render a colored test-set summary panel at the end of training."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="right", style="dim")
        grid.add_column(justify="left")

        key_metrics = [
            ("test_auc_roc",  "AUC-ROC"),
            ("test_accuracy", "Accuracy"),
            ("test_f1",       "F1"),
            ("test_mcc",      "MCC"),
            ("test_precision", "Precision"),
            ("test_recall",   "Sensitivity"),
            ("test_specificity", "Specificity"),
            ("test_nll",      "NLL"),
            ("test_brier",    "Brier"),
        ]
        for k, label in key_metrics:
            v = final_metrics.get(k, float("nan"))
            if isinstance(v, float):
                if "auc" in k or "accuracy" in k or "f1" in k or "mcc" in k \
                        or "precision" in k or "recall" in k or "specificity" in k:
                    grid.add_row(label, f"[{self._color_auc(v)}]{v:.4f}[/]")
                else:
                    grid.add_row(label, f"{v:.4f}")

        grid.add_row("---", "---")
        grid.add_row("epochs", f"{final_metrics['epochs_trained']}")
        grid.add_row("best_epoch", f"{final_metrics['best_epoch']}")
        grid.add_row("early_stopped",
                     "[yellow]yes[/]" if final_metrics["stopped_early"] else "[green]no[/]")
        grid.add_row("wall_time", f"{total_time:.0f}s ({total_time/60:.1f}m)")

        self._console.print(
            Panel(
                grid,
                title="[bold green]✓ Test Results[/]",
                border_style="green",
                padding=(0, 1),
            )
        )

    def _save_curve_artifacts(self, labels, probs, preds):
        """Save ROC curve, PR curve, and confusion matrix PNGs.

        Produces only PNGs. Raw arrays are not persisted on disk: the
        downstream CLIs (calibration, uncertainty, XAI) re-run inference
        from ``best.pt`` rather than relying on cached predictions.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix as sk_cm

        from src.evaluation.metrics import compute_curves

        curves = compute_curves(labels, probs)
        run_dir = self.logger.run_dir
        os.makedirs(run_dir, exist_ok=True)

        # ---- ROC curve ----
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(curves["fpr"], curves["tpr"],
                color="tab:blue", linewidth=2,
                label=f"AUC = {curves.get('auprc', 0):.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Test ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- PR curve ----
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(curves["recall"], curves["precision"],
                color="tab:orange", linewidth=2,
                label=f"AUC-PR = {curves['auprc']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Test Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Confusion matrix ----
        cm = sk_cm(labels, preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Test Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(
            os.path.join(run_dir, "confusion_matrix.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

        if not self.silent:
            print(f"[Trainer] Saved curve artifacts to {run_dir}")

    # ------------------------------------------------------------------
    # Per-epoch train + validate
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int):
        """Train for one epoch. Returns (avg_loss, accuracy).

        Implements:
          - bf16 autocast (Phase 2.5)
          - grad clipping + NaN guard (Phase 2.2 + 2.4)
          - channels_last + non_blocking transfers
          - EMA param update
          - Rich progress bar advance (when rich UI is active)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        use_amp = self.device.startswith("cuda")

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, dtype=torch.long)
            labels = labels.to(self.device, non_blocking=True)

            if use_amp and hasattr(self.model, "to"):
                # channels_last is a property of the underlying storage; safe
                # to set per-batch for the input.
                images = images.contiguous(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if not torch.isfinite(loss):
                raise TrialDivergedError(
                    f"Non-finite loss at epoch {epoch}, step {self.global_step}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_norm,
            )
            if not torch.isfinite(grad_norm):
                raise TrialDivergedError(
                    f"Non-finite grad norm at epoch {epoch}, step {self.global_step}"
                )

            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
            self.global_step += 1

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            # ---- Rich progress bar advance ----
            if self._train_progress is not None and self._train_task is not None:
                # Update description with running loss for live feedback.
                running_avg_loss = running_loss / max(total, 1)
                running_acc = correct / max(total, 1)
                self._train_progress.update(
                    self._train_task, advance=1,
                    description=(
                        f"loss={running_avg_loss:.4f} acc={running_acc:.4f}"
                    ),
                )

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self):
        """Validate and return (loss, acc, auc, f1, mcc, metrics_dict).

        The metrics_dict carries the full per-epoch stack so the trainer
        can log precision/recall/specificity/NLL/confusion-matrix counts
        alongside the headline AUC/F1.
        """
        self.model.eval()
        running_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []
        use_amp = self.device.startswith("cuda")

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, dtype=torch.long)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = F.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        total = len(all_labels)
        avg_loss = running_loss / max(total, 1)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        all_preds_np = np.array(all_preds)

        m = compute_epoch_metrics(
            all_labels_np, all_preds_np, all_probs_np,
        )
        # Headline scalars (kept for early-stopping convenience)
        return (
            avg_loss,
            m["acc"],
            m["auc"],
            m["f1"],
            m["mcc"],
            m,  # full metrics dict for richer logging
        )

    @torch.no_grad()
    def _evaluate_test(self):
        """Evaluate on test set and return (metrics_dict, raw_arrays).

        The raw arrays (labels, probs, preds) are needed downstream for
        saving ROC / PR curves and confusion matrices to disk. Keeping
        them in the return value avoids a second pass over the loader.
        """
        self.model.eval()
        all_labels, all_probs, all_preds = [], [], []
        use_amp = self.device.startswith("cuda")

        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, dtype=torch.long)
            labels = labels.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp):
                logits = self.model(images)
            probs = F.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        labels_np = np.array(all_labels)
        probs_np = np.array(all_probs)
        preds_np = np.array(all_preds)

        from src.evaluation.metrics import compute_all_metrics
        metrics = compute_all_metrics(labels_np, preds_np, probs_np)
        return metrics, (labels_np, probs_np, preds_np)

    def _evaluate_test_with_ema(self):
        """Evaluate on test set with EMA weights if available.

        Loads best checkpoint, swaps in EMA weights, runs test eval,
        then restores the original (non-EMA) weights for the
        best-checkpoint state.

        Returns:
            (metrics_dict, raw_arrays) — see :meth:`_evaluate_test`.
        """
        # Load best model state
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False,
        )
        model_inner = self._unwrap_compiled()
        model_inner.load_state_dict(ckpt["model_state_dict"])

        # Snapshot current weights and swap in EMA (if available)
        if self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
            result = self._evaluate_test()
            self.ema.restore(self.model)
            return result
        return self._evaluate_test()

    def _unwrap_compiled(self):
        """Get the inner nn.Module when self.model is torch.compile-wrapped."""
        model = self.model
        if hasattr(model, "_orig_mod"):
            return model._orig_mod
        return model

    # ------------------------------------------------------------------
    # Checkpoint helpers (Phase 3.2 + 3.4)
    # ------------------------------------------------------------------
    def _save_best(self, epoch: int, val_auc: float):
        save_checkpoint(
            self._unwrap_compiled(), self.optimizer, epoch, val_auc,
            self.checkpoint_path, ema=self.ema,
        )

    # ------------------------------------------------------------------
    # Resume support (Phase 3.3)
    # ------------------------------------------------------------------
    def _resume_state(self, ckpt_path: str):
        """Restore model/optim/sched/RNG/dataloader state from a checkpoint.

        Without RNG/dataloader state, a resumed run won't reproduce the
        same augmentation sequence, breaking the resume-correctness test.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_inner = self._unwrap_compiled()
        model_inner.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                warnings.warn(f"Could not restore optimizer state: {e}")
        if "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        if "ema_state_dict" in ckpt and self.ema is not None:
            try:
                self.ema.shadow = ckpt["ema_state_dict"]
            except Exception:
                pass

        # RNG state (Phase 3.3)
        if self._save_rng and "rng_state" in ckpt:
            try:
                torch.set_rng_state(ckpt["rng_state"]["torch"])
                if ckpt["rng_state"].get("cuda") and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
                if "numpy" in ckpt["rng_state"]:
                    np.random.set_state(ckpt["rng_state"]["numpy"])
                import random as _random
                if "python" in ckpt["rng_state"]:
                    _random.setstate(ckpt["rng_state"]["python"])
            except Exception as e:
                warnings.warn(f"Could not restore RNG state: {e}")

        # Dataloader state: skipper-friendly approach — re-emit to caller
        # by setting self.start_epoch. Sampling-completeness of the
        # loader is approximated (full dataloader RNG tracking is harder).
        self.start_epoch = ckpt.get("epoch", 1) + 1
        self.best_val_auc = ckpt.get("val_auc", 0.0)
        self.best_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        if not self.silent:
            print(f"[Trainer] Resumed from {ckpt_path} at epoch {self.start_epoch} "
                  f"(best_val_auc={self.best_val_auc:.4f})")
