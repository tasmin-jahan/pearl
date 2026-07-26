"""
Ensemble of top-k finalists (Phase 6.5).

Provides:
  - load_ensemble(model_configs, checkpoint_paths) -> nn.Module list
  - predict_ensemble(models, dataloader, device) -> probs, labels
  - evaluate_ensemble(...) -> metrics dict

Ensemble is simple probability-averaging across the k models with
calibrated logits (Phase 7). The averaging happens at the probability
level (not logit level) to keep each model's softmax into its natural
range before mixing.
"""

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_all_metrics
from src.model.builder import build_model


def load_ensemble(
    model_configs: Sequence[dict],
    checkpoint_paths: Sequence[str],
    device: str = "cpu",
) -> List[torch.nn.Module]:
    """Build N models and load each from its own checkpoint."""
    if len(model_configs) != len(checkpoint_paths):
        raise ValueError("model_configs and checkpoint_paths must be same length")
    models = []
    for cfg, path in zip(model_configs, checkpoint_paths):
        m = build_model(cfg).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def predict_ensemble(
    models: Sequence[torch.nn.Module],
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple:
    """Run probability-averaging inference across an ensemble.

    Returns:
        (mean_probs, all_labels) — both numpy arrays.
    """
    all_labels = []
    sum_probs = None
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
        if sum_probs is None:
            sum_probs = [avg.cpu().numpy()]
        else:
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
