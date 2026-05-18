"""
Global random seed setter for reproducibility.

Sets seeds for Python, NumPy, PyTorch (CPU + CUDA), and enables
deterministic cuDNN behavior.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)
