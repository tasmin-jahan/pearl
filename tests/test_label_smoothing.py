"""
Tests for label-smoothing support in build_weighted_loss.

The smoothing epsilon is the only knob that affects whether the model
falls into the "always predict positive" shortcut on imbalanced
datasets. Pin the behaviour so a future refactor can't silently drop it.
"""

import torch
import torch.nn as nn

from src.training.losses import build_weighted_loss


def test_default_label_smoothing_is_0_05():
    weights = torch.tensor([1.0, 1.0])
    loss = build_weighted_loss(weights, device="cpu")
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.label_smoothing == 0.05


def test_label_smoothing_zero_disables_smoothing():
    weights = torch.tensor([1.0, 1.0])
    loss = build_weighted_loss(weights, device="cpu", label_smoothing=0.0)
    assert loss.label_smoothing == 0.0


def test_smoothing_changes_loss_value():
    weights = torch.tensor([1.0, 1.0])
    loss_smooth = build_weighted_loss(weights, device="cpu", label_smoothing=0.05)
    loss_plain = build_weighted_loss(weights, device="cpu", label_smoothing=0.0)

    logits = torch.tensor([[2.0, 1.0], [0.5, 0.2]])
    labels = torch.tensor([0, 1])
    # Smoothing raises loss on confident-but-not-overwhelming logits
    # because the target distribution is mixed with uniform.
    assert loss_smooth(logits, labels) > loss_plain(logits, labels)


def test_class_weights_still_applied_with_smoothing():
    weights = torch.tensor([2.0, 1.0])
    loss = build_weighted_loss(weights, device="cpu", label_smoothing=0.1)
    assert torch.allclose(loss.weight, torch.tensor([2.0, 1.0]))
    assert loss.label_smoothing == 0.1
