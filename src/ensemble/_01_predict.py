"""
01_predict — Step 1 of the ensemble pipeline.

Probability-averaging inference across N models. Each model's softmax
output is averaged; this preserves each member's softmax range and is
the canonical PEARL ensembling convention.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics


@torch.no_grad()
def predict_ensemble(
    models: Sequence[torch.nn.Module],
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple:
    """Run probability-averaging inference across an ensemble.

    Returns:
        (mean_probs, all_labels) — both numpy arrays of shape
        ``(N_samples, C)`` and ``(N_samples,)``.
    """
    all_labels = []
    sum_probs = []
    n_models = len(models)
    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long)
        batch_probs = []
        for m in models:
            logits = m(images)
            batch_probs.append(F.softmax(logits.float(), dim=1))
        avg = torch.stack(batch_probs, dim=0).mean(dim=0)
        sum_probs.append(avg.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    probs = np.concatenate(sum_probs, axis=0)
    labels = np.array(all_labels)
    return probs, labels


def evaluate_ensemble(
    models: Sequence[torch.nn.Module],
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict:
    """Compute all metrics on the ensemble's averaged probabilities."""
    probs, labels = predict_ensemble(models, dataloader, device=device)
    preds = probs.argmax(axis=1)
    return compute_all_metrics(labels, preds, probs[:, 1])
