"""
SHAP-based explanations using DeepExplainer.

Uses a background set of 50-100 training images to compute
SHAP values for test samples.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def generate_shap_values(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    background_tensor: torch.Tensor,
    device: str = "cuda",
) -> np.ndarray:
    """Generate SHAP values for a single image.

    Args:
        model: Trained model.
        image_tensor: Single image (1, C, H, W).
        background_tensor: Background images (B, C, H, W), typically 50-100.
        device: Device string.

    Returns:
        SHAP values as (C, H, W) array.
    """
    if not HAS_SHAP:
        print("[SHAP] shap not installed, returning gradient-based fallback")
        return _gradient_fallback(model, image_tensor, device)

    model.eval()
    model.to(device)
    background_tensor = background_tensor.to(device)
    image_tensor = image_tensor.to(device)

    explainer = shap.DeepExplainer(model, background_tensor)
    shap_values = explainer.shap_values(image_tensor)

    # shap_values is a list of arrays (one per class)
    # Use the predicted class
    with torch.no_grad():
        pred_class = model(image_tensor).argmax(dim=1).item()

    sv = shap_values[pred_class][0]  # (C, H, W)
    return sv


def _gradient_fallback(model, image_tensor, device):
    """Gradient-based fallback when SHAP is unavailable."""
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device).requires_grad_(True)

    logits = model(image_tensor)
    pred_class = logits.argmax(dim=1).item()
    logits[0, pred_class].backward()

    grad = image_tensor.grad.detach().cpu().numpy()[0]
    return grad


def get_background_samples(train_loader, n_samples: int = 100) -> torch.Tensor:
    """Collect background samples from training DataLoader.

    Args:
        train_loader: Training DataLoader.
        n_samples: Number of background images to collect.

    Returns:
        Tensor of shape (n_samples, C, H, W).
    """
    samples = []
    for images, _ in train_loader:
        samples.append(images)
        if sum(s.shape[0] for s in samples) >= n_samples:
            break
    all_samples = torch.cat(samples, dim=0)[:n_samples]
    return all_samples


def save_shap_visualization(
    shap_values: np.ndarray,
    original_image: np.ndarray,
    save_dir: str,
    sample_id: int,
) -> None:
    """Save SHAP value visualization.

    Args:
        shap_values: SHAP values (C, H, W).
        original_image: Original image (H, W, C) or (C, H, W).
        save_dir: Output directory.
        sample_id: Sample identifier.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Sum over channels for visualization
    if shap_values.ndim == 3:
        sv_2d = np.abs(shap_values).sum(axis=0)
    else:
        sv_2d = np.abs(shap_values)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sv_2d, cmap="hot")
    ax.axis("off")
    ax.set_title("SHAP Values (|importance|)")
    fig.savefig(os.path.join(save_dir, f"sample_{sample_id:04d}_shap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mean_shap_summary(
    all_shap_values: list,
    save_path: str,
) -> None:
    """Save population-level mean SHAP summary.

    Args:
        all_shap_values: List of (C, H, W) SHAP arrays.
        save_path: Output path for the summary plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mean_importance = np.mean([np.abs(sv).sum(axis=0) for sv in all_shap_values], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_importance, cmap="hot")
    ax.axis("off")
    ax.set_title("Mean SHAP Importance (across samples)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
