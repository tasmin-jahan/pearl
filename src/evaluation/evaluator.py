"""
Test set evaluator: loads a checkpoint, runs inference, and returns metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import compute_all_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str = "cuda",
) -> dict:
    """Run evaluation on a test set and return all metrics.

    Args:
        model: Trained model (already loaded with best weights).
        test_loader: Test DataLoader.
        device: Device string.

    Returns:
        Dict of metric_name → value.
    """
    model.eval()
    model.to(device)

    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in test_loader:
        images = images.to(device)
        labels_t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels_t.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    return metrics


def collect_logits_and_labels(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
):
    """Collect raw logits and labels from a dataloader.

    Useful for calibration and uncertainty analysis.

    Args:
        model: Trained model.
        dataloader: DataLoader to iterate.
        device: Device string.

    Returns:
        Tuple of (logits_tensor, labels_tensor).
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            labels_t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
            all_labels.append(labels_t.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)
