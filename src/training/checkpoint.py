"""
Checkpoint utilities: save and load model checkpoints by validation AUC.
"""

import os

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    path: str,
) -> None:
    """Save model checkpoint.

    Args:
        model: The model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        val_auc: Validation AUC at this checkpoint.
        path: File path for the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_auc": val_auc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"[Checkpoint] Saved (epoch={epoch}, val_auc={val_auc:.4f}) → {path}")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cuda",
) -> dict:
    """Load a model checkpoint.

    Args:
        model: Model to load weights into.
        path: Path to checkpoint file.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Checkpoint metadata dict with 'epoch' and 'val_auc'.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"[Checkpoint] Loaded from {path} "
        f"(epoch={checkpoint.get('epoch', '?')}, val_auc={checkpoint.get('val_auc', '?')})"
    )
    return {
        "epoch": checkpoint.get("epoch", 0),
        "val_auc": checkpoint.get("val_auc", 0.0),
    }
