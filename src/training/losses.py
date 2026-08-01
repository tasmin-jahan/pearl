"""
Class-weighted CrossEntropyLoss for imbalanced PCOS dataset, with
optional label smoothing to discourage the "always predict positive"
shortcut that transfer-learning models tend to fall into when the
source-domain class balance is heavily skewed (Figshare is 78% positive;
PCOSGen is 37% positive).

Label smoothing mixes the one-hot target with a uniform distribution::

    loss = (1 - ε) * CE(one_hot(y), p) + ε * CE(uniform, p)

This forces the model to assign non-trivial mass to the wrong class even
on its most confident predictions, which:

  - Keeps logits less peaky (smoother probability surface).
  - Helps cross-dataset transfer because a smaller per-sample logit range
    is less sensitive to distribution shift.
  - Costs ~zero accuracy in the well-calibrated regime; helps a lot in
    the collapsed regime.

The default ε = 0.05 follows the original Szegedy et al. (2016) recipe
and is essentially free for clean datasets. Set ``label_smoothing=0.0``
to recover the original behaviour.
"""

import torch
import torch.nn as nn


def build_weighted_loss(
    class_weights: torch.Tensor,
    device: str = "cuda",
    label_smoothing: float = 0.05,
) -> nn.Module:
    """Build a class-weighted CrossEntropyLoss with optional label smoothing.

    Args:
        class_weights: Tensor of shape (num_classes,) with per-class weights.
        device: Device to place weights on.
        label_smoothing: ε in [0, 1). 0.0 disables smoothing. 0.05 is the
            standard default; 0.1 is more aggressive.

    Returns:
        nn.CrossEntropyLoss with class weights and label smoothing applied.
    """
    weights = class_weights.to(device)
    print(
        f"[Loss] Class weights: {weights.tolist()}  "
        f"label_smoothing={label_smoothing}"
    )
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
    )
