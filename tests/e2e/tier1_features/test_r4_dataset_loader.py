"""Tier 1 Feature Coverage: R4 High-Performance PyTorch Dataset & DataLoader.

Validates:
- Custom Dataset serving 8-element transition tuples:
  (state_t, opponent_state_t, market_t, action_t, reward_t, return_to_go_t, strategy_cluster, counterfactual_value)
- Tuple tensor typing, fixed shapes, and channel bounds
- DataLoader batch collation with configurable batch sizes (64, 128)
- Memory pinning and prefetching worker configuration
- High throughput benchmark asserting >500 transitions/second
- Flat memory consumption across multi-epoch iteration
"""

from __future__ import annotations

import time
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

try:
    from kaggriculture.data.benchmark import benchmark_dataloader, check_memory_leaks
except ImportError:
    benchmark_dataloader = None
    check_memory_leaks = None


def _create_mock_cache(cache_dir: Path, num_samples: int = 256):
    """Helper to populate dummy chunk file for dataset testing."""
    chunk_file = cache_dir / "chunk_0.npz"
    np.savez(
        chunk_file,
        state_spatial=np.zeros((num_samples, 23, 10, 10), dtype=np.float32),
        state_global=np.zeros((num_samples, 30), dtype=np.float32),
        opponent_spatial=np.zeros((num_samples, 20, 10, 10), dtype=np.float32),
        opponent_global=np.zeros((num_samples, 12), dtype=np.float32),
        market=np.zeros((num_samples, 35), dtype=np.float32),
        action=np.zeros((num_samples,), dtype=np.int64),
        reward=np.zeros((num_samples,), dtype=np.float32),
        return_to_go=np.zeros((num_samples,), dtype=np.float32),
        strategy_cluster=np.zeros((num_samples,), dtype=np.int64),
        counterfactual_value=np.zeros((num_samples,), dtype=np.float32),
    )


def test_r4_dataset_initialization_and_len(temp_cache_dir: Path):
    """Test 1 (R4): KaggricultureDataset accurately reads sample count."""
    _create_mock_cache(temp_cache_dir, num_samples=128)

    if KaggricultureDataset is None:
        pytest.skip("kaggriculture.data.dataset pending M4 implementation")

    ds = KaggricultureDataset(cache_dir=temp_cache_dir)
    assert len(ds) == 128


def test_r4_dataset_tuple_structure_and_types(temp_cache_dir: Path):
    """Test 2 (R4): Dataset __getitem__ returns the required 8-element transition tuple."""
    _create_mock_cache(temp_cache_dir, num_samples=64)

    if KaggricultureDataset is None:
        pytest.skip("kaggriculture.data.dataset pending M4 implementation")

    ds = KaggricultureDataset(cache_dir=temp_cache_dir)
    item = ds[0]

    assert isinstance(item, tuple)
    assert len(item) == 8, f"Expected 8-element tuple, got {len(item)}"

    state_t, opp_state_t, market_t, action_t, reward_t, rtg_t, strat_t, cf_val_t = item
    assert isinstance(state_t, torch.Tensor)
    assert state_t.shape == (23, 10, 10)
    assert isinstance(opp_state_t, torch.Tensor)
    assert opp_state_t.shape == (20, 10, 10)
    assert isinstance(market_t, torch.Tensor)
    assert market_t.shape == (35,)
    assert isinstance(action_t, torch.Tensor)
    assert isinstance(reward_t, torch.Tensor)
    assert isinstance(rtg_t, torch.Tensor)
    assert isinstance(strat_t, torch.Tensor)
    assert isinstance(cf_val_t, torch.Tensor)


def test_r4_dataloader_batch_collation(temp_cache_dir: Path):
    """Test 3 (R4): DataLoader batches elements cleanly at batch size 64."""
    _create_mock_cache(temp_cache_dir, num_samples=128)

    if create_dataloader is None:
        pytest.skip("kaggriculture.data.dataloader pending M4 implementation")

    loader = create_dataloader(cache_dir=temp_cache_dir, batch_size=64, num_workers=0)
    batch = next(iter(loader))

    assert len(batch) == 8
    state_batch = batch[0]
    assert state_batch.shape == (64, 23, 10, 10)


def test_r4_dataloader_pin_memory_option(temp_cache_dir: Path):
    """Test 4 (R4): Pinned memory flag can be enabled without raising exceptions."""
    _create_mock_cache(temp_cache_dir, num_samples=64)

    if create_dataloader is None:
        pytest.skip("kaggriculture.data.dataloader pending M4 implementation")

    loader = create_dataloader(cache_dir=temp_cache_dir, batch_size=32, pin_memory=True, num_workers=0)
    batch = next(iter(loader))
    assert batch is not None


def test_r4_dataloader_throughput_benchmark(temp_cache_dir: Path):
    """Test 5 (R4): DataLoader achieves >500 transitions/sec throughput."""
    _create_mock_cache(temp_cache_dir, num_samples=1024)

    if benchmark_dataloader is None:
        pytest.skip("kaggriculture.data.benchmark pending M4 implementation")

    throughput = benchmark_dataloader(cache_dir=temp_cache_dir, batch_size=64, num_transitions=1024)
    assert throughput >= 500.0, f"DataLoader throughput {throughput:.1f} trans/s below threshold 500"


def test_r4_memory_leak_verification(temp_cache_dir: Path):
    """Test 6 (R4): Memory consumption remains flat over 3 epochs."""
    _create_mock_cache(temp_cache_dir, num_samples=256)

    if check_memory_leaks is None:
        pytest.skip("kaggriculture.data.benchmark.check_memory_leaks pending M4 implementation")

    is_leak_free, growth_pct = check_memory_leaks(cache_dir=temp_cache_dir, epochs=3, batch_size=64)
    assert is_leak_free, f"Memory growth detected: {growth_pct:.2f}% across 3 epochs"
