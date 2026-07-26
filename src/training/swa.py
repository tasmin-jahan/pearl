"""
Stochastic Weight Averaging (SWA) helper for Phase 6.4.

SWA averages model weights over a window of training epochs to find
flatter minima, often yielding a 0.5–1% AUC improvement at no extra
training cost. Implemented as a post-training step that loads each
fold's recent checkpoints and averages their parameters.

This is complementary to EMA (which the trainer maintains in-shadow
during training) — both are cheap, and they operate on different
statistics.
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

    Looks for files matching ``{checkpoint_path_stem}__epoch*.pt`` in
    the same directory as ``checkpoint_path``.

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
    pattern = os.path.join(base_dir, f"{prefix}__epoch*.pt")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    paths = paths[-keep_last_n:]
    if not paths:
        print(f"[SWA] No matching checkpoints for {pattern}, skipping.")
        return model
    print(f"[SWA] Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p}")
    return average_checkpoints(model, paths, device=device)
