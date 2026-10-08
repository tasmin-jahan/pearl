"""
MC Dropout inference for uncertainty quantification.

Runs 50 stochastic forward passes with dropout active to estimate
predictive uncertainty via entropy.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def mc_dropout_inference(
    model: torch.nn.Module,
    dataloader,
    n_passes: int = 50,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MC Dropout inference.

    Keeps model in train mode (dropout active) but disables gradient
    computation for efficiency.

    Args:
        model: Trained model with dropout layers.
        dataloader: Test DataLoader.
        n_passes: Number of stochastic forward passes.
        device: Device string.

    Returns:
        Tuple of:
          - mean_probs: (N, 2) mean predicted probabilities
          - entropy: (N,) predictive entropy in nats
    """
    model.to(device)
    model.eval()
    # Activate only dropout layers so that normalization layers (BatchNorm, LayerNorm)
    # keep using their frozen running statistics instead of corrupting them
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()

    all_probs = []  # will be (n_passes, N, 2)

    for t in range(n_passes):
        pass_probs = []
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            pass_probs.append(probs.cpu())
        all_probs.append(torch.cat(pass_probs, dim=0))

        if (t + 1) % 10 == 0:
            print(f"  MC pass {t + 1}/{n_passes}")

    # Stack: (n_passes, N, 2) → transpose to (N, n_passes, 2)
    all_probs = torch.stack(all_probs, dim=1)  # (N, n_passes, 2)

    # Mean across passes
    mean_probs = all_probs.mean(dim=1)  # (N, 2)

    # Predictive entropy: H = -sum(p * log(p))
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)  # (N,)

    model.eval()
    return mean_probs, entropy
