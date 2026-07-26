"""
DataLoader builder for PCOS dataset.

Constructs train/val/test DataLoaders from a preprocessing config,
with augmentation enabled only for training.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import PCOSDataset
from src.preprocessing.preprocess import Preprocessor


def build_dataloaders(
    preprocessing_config: dict,
    batch_size: int = 32,
    input_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from preprocessed files on disk.

    Args:
        preprocessing_config: Preprocessing config dict.
        batch_size: Batch size for all loaders.
        input_size: Input image size (for augmentation transforms).
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory for GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    output_dir = preprocessing_config.get("output_dir", "results/preprocessed/default")

    preprocessor = Preprocessor(preprocessing_config, input_size=input_size)

    train_dataset = PCOSDataset(
        root_dir=output_dir, split="train",
        preprocessor=preprocessor, augment=True,
    )
    val_dataset = PCOSDataset(
        root_dir=output_dir, split="val",
        preprocessor=None, augment=False,
    )
    test_dataset = PCOSDataset(
        root_dir=output_dir, split="test",
        preprocessor=None, augment=False,
    )

    print(f"[DataLoader] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


class _InMemoryDataset(Dataset):
    """Dataset that loads+preprocesses images on-the-fly from raw paths.

    Used by k-fold CV (Phase 6.3) so we don't need to write each fold's
    preprocessed data to disk.
    """

    def __init__(self, paths, labels, preprocessor, augment):
        self.paths = list(paths)
        self.labels = list(labels)
        self.preprocessor = preprocessor
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            # Return a black image as a fail-safe; surfacing in logs.
            img = np.zeros((self.preprocessor.input_size,
                            self.preprocessor.input_size, 3), dtype=np.uint8)
        x = self.preprocessor.apply(img, augment=self.augment)
        if x.ndim == 2:
            x = x[None, :, :]
        else:
            x = np.transpose(x, (2, 0, 1))
        import torch
        return torch.from_numpy(x.copy()).float(), self.labels[idx]

    def get_class_counts(self):
        import collections
        return dict(collections.Counter(self.labels))

    def get_class_weights(self):
        import torch
        n = len(self.paths)
        counts = self.get_class_counts()
        w0 = n / (2.0 * max(counts.get(0, 1), 1))
        w1 = n / (2.0 * max(counts.get(1, 1), 1))
        return torch.tensor([w0, w1], dtype=torch.float32)


def _make_dataset_and_loaders(
    train_paths: Sequence[str], train_labels: Sequence[int],
    val_paths: Sequence[str], val_labels: Sequence[int],
    preprocessing_config: dict,
    batch_size: int = 32,
    input_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val loaders that load+preprocess raw images in-memory.

    Used by Phase 6.3 (k-fold) so each fold can use its own train/val
    split without round-tripping through the disk preprocessor.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    preprocessor = Preprocessor(preprocessing_config, input_size=input_size)
    preprocessor_eval = Preprocessor(preprocessing_config, input_size=input_size)

    train_ds = _InMemoryDataset(train_paths, train_labels, preprocessor, augment=True)
    val_ds = _InMemoryDataset(val_paths, val_labels, preprocessor_eval, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader
