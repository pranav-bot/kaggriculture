"""Tier 2 Boundary and Corner Case Tests: R4 PyTorch Dataset & DataLoader.

Validates:
- Single-element dataset boundary (dataset len = 1 with batch_size = 64)
- Uneven chunk boundaries across sharded files
- DataLoader with num_workers = 0 (synchronous fallback)
- Out-of-bounds dataset indexing (IndexError)
- Empty cache directory handling
- CPU fallback and non-CUDA execution safety
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for R4 Dataset/DataLoader tests")

try:
    from kaggriculture.data.dataset import KaggricultureDataset
except ImportError:
    KaggricultureDataset = None

try:
    from kaggriculture.data.dataloader import create_dataloader
except ImportError:
    create_dataloader = None


def _populate_dummy_chunk(path: Path, n: int):
    """Helper to write dummy npz chunk."""
    np.savez(
        path,
        state_spatial=np.zeros((n, 23, 10, 10), dtype=np.float32),
        state_global=np.zeros((n, 30), dtype=np.float32),
        opponent_spatial=np.zeros((n, 20, 10, 10), dtype=np.float32),
        opponent_global=np.zeros((n, 12), dtype=np.float32),
        market=np.zeros((n, 35), dtype=np.float32),
        action=np.zeros((n,), dtype=np.int64),
        reward=np.zeros((n,), dtype=np.float32),
        return_to_go=np.zeros((n,), dtype=np.float32),
        strategy_cluster=np.zeros((n,), dtype=np.int64),
        counterfactual_value=np.zeros((n,), dtype=np.float32),
    )


def test_r4_single_element_dataset_boundary(temp_cache_dir: Path):
    """Test 1 (R4 Boundary): Single sample dataset with batch_size=64 yields single batch of size 1."""
    _populate_dummy_chunk(temp_cache_dir / "single.npz", n=1)

    if KaggricultureDataset is None or create_dataloader is None:
        pytest.skip("kaggriculture.data pending M4 implementation")

    ds = KaggricultureDataset(cache_dir=temp_cache_dir)
    assert len(ds) == 1

    loader = create_dataloader(cache_dir=temp_cache_dir, batch_size=64, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1
    assert batches[0][0].shape[0] == 1


def test_r4_uneven_chunk_sizes_boundary(temp_cache_dir: Path):
    """Test 2 (R4 Boundary): Multiple chunks with uneven sizes (50 and 33 samples) map seamlessly."""
    _populate_dummy_chunk(temp_cache_dir / "chunk_a.npz", n=50)
    _populate_dummy_chunk(temp_cache_dir / "chunk_b.npz", n=33)

    if KaggricultureDataset is None:
        pytest.skip("kaggriculture.data.dataset pending M4 implementation")

    ds = KaggricultureDataset(cache_dir=temp_cache_dir)
    assert len(ds) == 83

    # Sample right at chunk boundary
    item_49 = ds[49]
    item_50 = ds[50]
    assert item_49 is not None and item_50 is not None


def test_r4_dataloader_num_workers_zero_boundary(temp_cache_dir: Path):
    """Test 3 (R4 Boundary): num_workers=0 executes synchronously on main thread without crash."""
    _populate_dummy_chunk(temp_cache_dir / "chunk.npz", n=32)

    if create_dataloader is None:
        pytest.skip("kaggriculture.data.dataloader pending M4 implementation")

    loader = create_dataloader(cache_dir=temp_cache_dir, batch_size=16, num_workers=0)
    count = sum(b[0].shape[0] for b in loader)
    assert count == 32


def test_r4_dataset_index_out_of_bounds_boundary(temp_cache_dir: Path):
    """Test 4 (R4 Boundary): Accessing index >= len(dataset) raises IndexError."""
    _populate_dummy_chunk(temp_cache_dir / "chunk.npz", n=10)

    if KaggricultureDataset is None:
        pytest.skip("kaggriculture.data.dataset pending M4 implementation")

    ds = KaggricultureDataset(cache_dir=temp_cache_dir)
    with pytest.raises(IndexError):
        _ = ds[10]
    with pytest.raises(IndexError):
        _ = ds[999]


def test_r4_empty_cache_directory_boundary(tmp_path: Path):
    """Test 5 (R4 Boundary): Dataset raises clear error when cache directory has no chunk files."""
    empty_dir = tmp_path / "empty_cache"
    empty_dir.mkdir()

    if KaggricultureDataset is None:
        pytest.skip("kaggriculture.data.dataset pending M4 implementation")

    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        _ = KaggricultureDataset(cache_dir=empty_dir)


def test_r4_cpu_non_cuda_safety_boundary(temp_cache_dir: Path):
    """Test 6 (R4 Boundary): Running DataLoader without CUDA operates with zero device assertions."""
    _populate_dummy_chunk(temp_cache_dir / "chunk.npz", n=16)

    if create_dataloader is None:
        pytest.skip("kaggriculture.data.dataloader pending M4 implementation")

    loader = create_dataloader(cache_dir=temp_cache_dir, batch_size=8, pin_memory=False, num_workers=0)
    batch = next(iter(loader))
    state_tensor = batch[0]
    assert state_tensor.device.type == "cpu"
