"""DataLoader builder for batched transition training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from .dataset import KaggricultureDataset


def create_dataloader(
    cache_dir: Path | str,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """Create a high-performance PyTorch DataLoader serving transition batches.

    Args:
        cache_dir: Directory containing .npz cache files.
        batch_size: Number of transitions per batch.
        shuffle: Whether to shuffle transitions.
        num_workers: Number of background worker processes.
        pin_memory: Whether to pin memory for accelerated GPU transfers.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        PyTorch DataLoader instance.
    """
    dataset = KaggricultureDataset(cache_dir=cache_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )
