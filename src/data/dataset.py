"""
PCOSDataset — PyTorch Dataset class for preprocessed PCOS images.

Loads preprocessed numpy arrays from disk and applies optional
augmentation at training time via the Preprocessor.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PCOSDataset(Dataset):
    """PyTorch Dataset for preprocessed PCOS ultrasound images.

    Expects directory structure:
        root_dir/
            {split}/
                infected/       ← .npy files (label=1)
                notinfected/    ← .npy files (label=0)

    Args:
        root_dir: Path to the preprocessed data root.
        split: One of 'train', 'val', 'test'.
        preprocessor: Optional Preprocessor instance for augmentation.
        augment: Whether to apply augmentation (True only for training).
    """

    CLASS_MAP = {"notinfected": 0, "infected": 1}

    def __init__(
        self,
        root_dir: str,
        split: str,
        preprocessor=None,
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.preprocessor = preprocessor
        self.augment = augment

        self.samples = []  # list of (path, label)
        split_dir = os.path.join(root_dir, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name, label in self.CLASS_MAP.items():
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

        self.labels = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load a preprocessed image and return as tensor with label.

        Returns:
            Tuple of (image_tensor [C, H, W], label_int).
        """
        path, label = self.samples[idx]
        image = np.load(path)  # HxWxC float32

        # Apply augmentation if requested (training only)
        if self.augment and self.preprocessor is not None:
            image = self.preprocessor._apply_augmentation(image)

        # Convert HWC → CHW and to torch tensor
        if len(image.shape) == 2:
            # Grayscale: add channel dim
            image = image[np.newaxis, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        image_tensor = torch.from_numpy(image.copy()).float()
        return image_tensor, label

    def get_class_counts(self):
        """Return dict of class_name → count."""
        counts = {}
        for class_name, label in self.CLASS_MAP.items():
            counts[class_name] = sum(1 for _, l in self.samples if l == label)
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced loss.

        weight_c = n_total / (2 * n_c)

        Returns:
            Tensor of shape (num_classes,) with weights.
        """
        n_total = len(self.samples)
        counts = self.get_class_counts()
        n_class_0 = counts.get("notinfected", 1)
        n_class_1 = counts.get("infected", 1)

        w0 = n_total / (2.0 * n_class_0)
        w1 = n_total / (2.0 * n_class_1)
        return torch.tensor([w0, w1], dtype=torch.float32)
