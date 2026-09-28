"""Throughput benchmarking and memory leak verification for PyTorch DataLoader."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
import psutil

from .dataloader import create_dataloader


def benchmark_dataloader(
    cache_dir: Path | str,
    batch_size: int = 64,
    num_transitions: int = 1024,
) -> float:
    """Benchmark transitions served per second.

    Args:
        cache_dir: Cache directory containing transitions.
        batch_size: DataLoader batch size.
        num_transitions: Target number of transitions to benchmark.

    Returns:
        Throughput in transitions per second.
    """
    loader = create_dataloader(cache_dir=cache_dir, batch_size=batch_size, num_workers=0)
    count = 0
    start = time.perf_counter()

    while count < num_transitions:
        for batch in loader:
            count += batch[0].shape[0]
            if count >= num_transitions:
                break

    elapsed = max(1e-6, time.perf_counter() - start)
    return float(count / elapsed)


def check_memory_leaks(
    cache_dir: Path | str,
    epochs: int = 3,
    batch_size: int = 64,
) -> tuple[bool, float]:
    """Verify that memory consumption remains stable over multiple epochs.

    Args:
        cache_dir: Cache directory containing transitions.
        epochs: Number of full passes over dataset.
        batch_size: Batch size.

    Returns:
        Tuple of (is_leak_free, growth_percentage).
    """
    process = psutil.Process(os.getpid())
    gc.collect()
    initial_rss = process.memory_info().rss

    loader = create_dataloader(cache_dir=cache_dir, batch_size=batch_size, num_workers=0)
    for _ in range(epochs):
        for batch in loader:
            _ = batch[0].shape

    gc.collect()
    final_rss = process.memory_info().rss

    growth_pct = max(0.0, ((final_rss - initial_rss) / max(1.0, initial_rss)) * 100.0)
    # Less than 20% growth across multiple epochs indicates no significant leak
    is_leak_free = growth_pct < 20.0

    return is_leak_free, float(growth_pct)
