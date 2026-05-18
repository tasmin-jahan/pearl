"""
DataLoader builder for PCOS dataset.

Constructs train/val/test DataLoaders from a preprocessing config,
with augmentation enabled only for training.
"""

from typing import Tuple

from torch.utils.data import DataLoader

from src.data.dataset import PCOSDataset
from src.preprocessing.preprocess import Preprocessor


def build_dataloaders(
    preprocessing_config: dict,
    batch_size: int = 32,
    input_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders.

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

    # Preprocessor (used only for augmentation at training time)
    preprocessor = Preprocessor(preprocessing_config, input_size=input_size)

    # Datasets
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
