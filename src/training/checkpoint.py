"""
Checkpoint utilities: save and load model checkpoints by validation AUC.

v3 features:
  - Self-contained: stores architecture info, class names, RNG state,
    EMA shadow, and enriched metadata so a checkpoint alone can
    reconstruct everything needed for evaluation/resume.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    path: str,
    ema: Optional[object] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    global_step: int = 0,
    arch_name: str = "",
    class_names: Optional[list] = None,
    save_rng: bool = True,
    extra: Optional[dict] = None,
) -> None:
    """Save a self-contained checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch (1-indexed).
        val_auc: Validation AUC.
        path: File path.
        ema: Optional EMA instance.
        scheduler: Optional LR scheduler.
        global_step: Optimizer step count.
        arch_name: Short architecture name (e.g. "efficientnet_b4").
        class_names: Class index → name mapping.
        save_rng: Whether to include RNG state for resume reproducibility.
        extra: Any additional metadata to embed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "val_auc": val_auc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "arch_name": arch_name,
        "class_names": class_names or ["noninfected", "infected"],
    }

    if scheduler is not None:
        try:
            state["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            pass

    if ema is not None:
        try:
            state["ema_state_dict"] = {
                k: v.cpu() for k, v in ema.shadow.items()
            }
        except Exception:
            pass

    if save_rng:
        state["rng_state"] = _capture_rng_state()

    if extra:
        state["extra"] = extra

    torch.save(state, path)
    print(f"[Checkpoint] Saved (epoch={epoch}, val_auc={val_auc:.4f}) → {path}")


def _capture_rng_state() -> dict:
    """Capture Python, NumPy, and PyTorch RNG state for resume reproducibility."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema: Optional[object] = None,
    device: str = "cuda",
) -> dict:
    """Load a model checkpoint.

    Args:
        model: Model to load weights into.
        path: Path to checkpoint file.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        ema: Optional EMA instance to restore shadow.
        device: Device to map tensors to.

    Returns:
        Checkpoint metadata dict.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            pass
    if ema is not None and "ema_state_dict" in checkpoint:
        try:
            ema.shadow = {
                k: v.to(device) for k, v in checkpoint["ema_state_dict"].items()
            }
        except Exception:
            pass

    print(
        f"[Checkpoint] Loaded from {path} "
        f"(epoch={checkpoint.get('epoch', '?')}, val_auc={checkpoint.get('val_auc', '?')})"
    )
    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "val_auc": checkpoint.get("val_auc", 0.0),
        "arch_name": checkpoint.get("arch_name", ""),
        "class_names": checkpoint.get("class_names", []),
        "extra": checkpoint.get("extra", {}),
    }
