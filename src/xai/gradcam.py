"""
Grad-CAM visualization using torchcam.

Generates heatmaps from the final convolutional block of the model.
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchcam.methods import GradCAM


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int = None,
    device: str = "cuda",
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single image.

    Args:
        model: Trained model.
        image_tensor: Single image tensor of shape (1, C, H, W).
        target_class: Target class index. If None, uses predicted class.
        device: Device string.

    Returns:
        Heatmap as HxW float32 array in [0, 1].
    """
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Find the last conv layer in the backbone
    target_layer = _find_last_conv_layer(model)

    cam_extractor = GradCAM(model, target_layer=[target_layer])

    # Forward pass
    logits = model(image_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Generate CAM
    activation_map = cam_extractor(target_class, logits)

    # Get the heatmap
    heatmap = activation_map[0].squeeze().cpu().numpy()

    # Resize to input size
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    heatmap = cv2.resize(heatmap, (w, h))

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    cam_extractor.remove_hooks()

    return heatmap


def _find_last_conv_layer(model) -> str:
    """Find the name of the last Conv2d layer in the backbone."""
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
    if last_conv_name is None:
        raise ValueError("No Conv2d layer found in model")
    return last_conv_name


def save_gradcam_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    save_dir: str,
    sample_id: int,
) -> None:
    """Save original, heatmap, and overlay images.

    Args:
        original_image: Original image as HxWxC uint8 or float.
        heatmap: Grad-CAM heatmap as HxW float in [0,1].
        save_dir: Directory to save images.
        sample_id: Sample identifier for filenames.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Normalize original to [0, 255] uint8
    if original_image.dtype != np.uint8:
        img = original_image - original_image.min()
        img = (img / (img.max() + 1e-8) * 255).astype(np.uint8)
    else:
        img = original_image.copy()

    # Save original
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Original")
    fig.savefig(os.path.join(save_dir, f"sample_{sample_id:04d}_orig.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap, cmap="jet")
    ax.axis("off")
    ax.set_title("Grad-CAM Heatmap")
    fig.savefig(os.path.join(save_dir, f"sample_{sample_id:04d}_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save overlay
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title("Grad-CAM Overlay")
    fig.savefig(os.path.join(save_dir, f"sample_{sample_id:04d}_overlay.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
