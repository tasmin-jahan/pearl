"""
Layer-wise Relevance Propagation (LRP) using the zennit library.

Computes pixel-level relevance scores showing which input regions
contribute most to the model's prediction.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from zennit.composites import EpsilonPlusFlat
    from zennit.attribution import Gradient
    HAS_ZENNIT = True
except ImportError:
    HAS_ZENNIT = False


def generate_lrp(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None,
    device: str = "cuda",
) -> np.ndarray:
    """Generate LRP relevance map for a single image.

    Args:
        model: Trained model.
        image_tensor: Single image tensor (1, C, H, W).
        target_class: Target class. If None, uses predicted class.
        device: Device string.

    Returns:
        Relevance map as HxW float32 array.
    """
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device).requires_grad_(True)

    if HAS_ZENNIT:
        return _lrp_zennit(model, image_tensor, target_class, device)
    else:
        return _lrp_gradient_fallback(model, image_tensor, target_class, device)


def _lrp_zennit(model, image_tensor, target_class, device):
    """LRP using zennit library."""
    composite = EpsilonPlusFlat()

    with Gradient(model=model, composite=composite) as attributor:
        logits = model(image_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Create one-hot target
        target = torch.zeros_like(logits)
        target[0, target_class] = 1.0

        logits.backward(gradient=target)
        relevance = image_tensor.grad.detach().cpu().numpy()[0]

    # Sum over channels
    if relevance.ndim == 3:
        relevance = relevance.sum(axis=0)

    return relevance


def _lrp_gradient_fallback(model, image_tensor, target_class, device):
    """Fallback: gradient × input (when zennit is unavailable)."""
    logits = model(image_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    target = torch.zeros_like(logits)
    target[0, target_class] = 1.0

    logits.backward(gradient=target)
    grad = image_tensor.grad.detach().cpu().numpy()[0]
    inp = image_tensor.detach().cpu().numpy()[0]

    relevance = (grad * inp).sum(axis=0)
    return relevance


def save_lrp_visualization(
    relevance: np.ndarray,
    save_dir: str,
    sample_id: int,
) -> None:
    """Save LRP relevance heatmap.

    Args:
        relevance: HxW relevance map.
        save_dir: Output directory.
        sample_id: Sample identifier.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(abs(relevance.min()), abs(relevance.max()))
    ax.imshow(relevance, cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.axis("off")
    ax.set_title("LRP Relevance")
    fig.savefig(os.path.join(save_dir, f"sample_{sample_id:04d}_relevance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
