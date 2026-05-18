"""
Class-weighted CrossEntropyLoss for imbalanced PCOS dataset.

Weight formula: weight_c = n_total / (2 * n_c)
"""

import torch
import torch.nn as nn


def build_weighted_loss(class_weights: torch.Tensor, device: str = "cuda") -> nn.Module:
    """Build a class-weighted CrossEntropyLoss.

    Args:
        class_weights: Tensor of shape (num_classes,) with per-class weights.
        device: Device to place weights on.

    Returns:
        nn.CrossEntropyLoss with class weights.
    """
    weights = class_weights.to(device)
    print(f"[Loss] Class weights: {weights.tolist()}")
    return nn.CrossEntropyLoss(weight=weights)
