"""
Stochastic Weight Averaging (SWA) helper.

SWA averages model weights across multiple checkpoints to find flatter
minima, often yielding a 0.5–1% AUC improvement at no extra training
cost. Implemented as a post-training step that loads each checkpoint
and averages its parameters.

This is complementary to EMA (which the trainer maintains in-shadow
during training) — both are cheap, and they operate on different
statistics. EMA is on by default and is what the trainer uses at test
time; SWA is opt-in and lives here for callers that want to average
multiple independent training runs.

Note: the trainer no longer keeps rolling-window per-epoch checkpoints,
so ``run_swa_on_recent`` cannot discover them automatically. Callers
should pass explicit checkpoint paths via :func:`average_checkpoints`
instead.
"""

import glob
import os
from typing import List, Optional

import torch
import torch.nn as nn


def average_checkpoints(
    model: nn.Module,
    checkpoint_paths: List[str],
    device: str = "cpu",
) -> nn.Module:
    """Load and average parameters of multiple checkpoints into model.

    Args:
        model: Target model. Its parameters will be overwritten in-place
            with the averaged state.
        checkpoint_paths: List of .pt file paths to average.
        device: Device to map tensors to during loading.

    Returns:
        The same model instance, with averaged parameters.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must be non-empty")
    state_dicts = []
    for p in checkpoint_paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        state_dicts.append(ckpt["model_state_dict"])
    avg_state = _average_state_dicts(state_dicts)
    model.load_state_dict(avg_state)
    return model


def _average_state_dicts(state_dicts: List[dict]) -> dict:
    """Element-wise average of state dicts."""
    avg = {}
    for key in state_dicts[0].keys():
        if state_dicts[0][key].dtype.is_floating_point:
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            avg[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
        else:
            # Integer buffers (e.g. num_batches_tracked): take the last
            # to keep them consistent (they're metadata, not stats).
            avg[key] = state_dicts[-1][key]
    return avg


def run_swa_on_recent(
    model: nn.Module,
    checkpoint_path: str,
    keep_last_n: int = 5,
    device: str = "cpu",
) -> nn.Module:
    """Average the most recent ``keep_last_n`` numbered checkpoints.

    Looks for files matching ``{checkpoint_path_stem}_e*.pt`` in
    the same directory as ``checkpoint_path``. With the current trainer
    this glob is usually empty (the trainer only writes ``best.pt``);
    prefer :func:`average_checkpoints` with explicit paths for new code.

    Args:
        model: Model to update with averaged weights.
        checkpoint_path: Path to the best.pt (used to find the prefix).
        keep_last_n: How many recent checkpoints to average.
        device: Device to map tensors to.

    Returns:
        Model with SWA-averaged weights.
    """
    base_dir = os.path.dirname(checkpoint_path)
    prefix = os.path.splitext(os.path.basename(checkpoint_path))[0]
    pattern = os.path.join(base_dir, f"{prefix}_e*.pt")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    paths = paths[-keep_last_n:]
    if not paths:
        print(f"[SWA] No matching checkpoints for {pattern}, skipping.")
        return model
    print(f"[SWA] Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p}")
    return average_checkpoints(model, paths, device=device)
