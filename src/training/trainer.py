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
  - Rolling-window checkpoint retention (Phase 3.4)
  - Self-contained checkpoint with arch + class_names (Phase 3.2)
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
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

from src.training.checkpoint import save_checkpoint
from src.utils.logging import ExperimentLogger


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

        # ---- Rolling-window checkpoints (Phase 3.4) ----
        self.keep_last_n = config.get("keep_last_n", 3)

        # ---- RNG state (Phase 3.3) — restored on resume ----
        self._save_rng = config.get("save_rng_state", True)

        # Optional initial backbone freeze
        if self.freeze_epochs > 0:
            self._freeze_backbone(True)

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

            try:
                train_loss, train_acc = self._train_one_epoch(epoch)
                val_loss, val_acc, val_auc, val_f1, val_mcc = self._validate()
            except TrialDivergedError as e:
                if not self.silent:
                    print(f"\n[Trainer] TrialDiverged at epoch {epoch}: {e}")
                raise

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.log_epoch(
                epoch, train_loss, val_loss,
                train_acc, val_acc, val_auc, val_f1, val_mcc,
                current_lr, epoch_time,
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

            # Always retain the last N checkpoints (rolling window)
            self._save_recent(epoch, val_auc)

            # ---- Scheduler step (only after warmup) ----
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            # ---- Per-epoch callback (Optuna pruning) ----
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, val_auc)

            # ---- Early stopping ----
            if self.epochs_without_improvement >= self.patience:
                if not self.silent:
                    print(f"\n[Trainer] Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        stopped_early = self.epochs_without_improvement >= self.patience
        epochs_trained = epoch

        if not self.silent:
            print(f"\n[Trainer] Training complete in {total_time:.0f}s")
            print(f"[Trainer] Best val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")

        # ---- Load best (or EMA-weighted) checkpoint for test eval ----
        eval_metrics = self._evaluate_test_with_ema()

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

        return final_metrics

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

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self):
        """Validate and return (loss, acc, auc, f1, mcc)."""
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

        accuracy = (all_preds_np == all_labels_np).mean()
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except Exception:
            auc = 0.5
        f1 = f1_score(all_labels_np, all_preds_np, zero_division=0)
        mcc = matthews_corrcoef(all_labels_np, all_preds_np) if total > 1 else 0.0

        return avg_loss, accuracy, auc, f1, mcc

    @torch.no_grad()
    def _evaluate_test(self):
        """Evaluate on test set and return metrics dict."""
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

        from src.evaluation.metrics import compute_all_metrics
        return compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs),
        )

    def _evaluate_test_with_ema(self):
        """Evaluate on test set with EMA weights if available.

        Loads best checkpoint, swaps in EMA weights, runs test eval,
        then restores the original (non-EMA) weights for the
        best-checkpoint state.
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
            metrics = self._evaluate_test()
            self.ema.restore(self.model)
            return metrics
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

    def _save_recent(self, epoch: int, val_auc: float):
        """Save a numbered checkpoint in the same dir; prune to keep_last_n."""
        base_dir = os.path.dirname(self.checkpoint_path)
        recent_path = os.path.join(
            base_dir, f"{os.path.splitext(os.path.basename(self.checkpoint_path))[0]}__epoch{epoch}.pt",
        )
        save_checkpoint(
            self._unwrap_compiled(), self.optimizer, epoch, val_auc,
            recent_path, ema=self.ema,
        )
        self._prune_old_checkpoints(base_dir)

    def _prune_old_checkpoints(self, base_dir: str):
        """Keep only the keep_last_n most recent __epoch*.pt files."""
        prefix = os.path.splitext(os.path.basename(self.checkpoint_path))[0]
        files = []
        for f in os.listdir(base_dir):
            if f.startswith(prefix + "__epoch") and f.endswith(".pt"):
                files.append(os.path.join(base_dir, f))
        files.sort(key=lambda p: os.path.getmtime(p))
        for old in files[:-self.keep_last_n]:
            try:
                os.remove(old)
            except OSError:
                pass

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
