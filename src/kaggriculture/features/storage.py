"""Chunked memory-mapped storage and reader for transition batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np


def save_transitions_chunk(chunk_path: Path | str, transitions: Mapping[str, np.ndarray]) -> None:
    """Save a dictionary of transition numpy arrays into a compressed .npz chunk file.

    Args:
        chunk_path: Destination file path (ending in .npz).
        transitions: Dictionary mapping field names to numpy arrays of equal first dimension.
    """
    path = Path(chunk_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **transitions)


class TransitionChunkReader:
    """Indexed reader over directory of chunked .npz transition files."""

    def __init__(self, chunk_dir: Path | str):
        self.chunk_dir = Path(chunk_dir)
        self.chunk_files = sorted(list(self.chunk_dir.glob("*.npz")))
        self._chunks: list[dict[str, np.ndarray]] = []
        self._offsets: list[int] = [0]
        self._total_samples = 0

        for cf in self.chunk_files:
            data = np.load(cf, mmap_mode="r")
            # Extract first array to check batch length
            keys = list(data.keys())
            if not keys:
                continue
            chunk_len = len(data[keys[0]])
            chunk_dict = {k: data[k] for k in keys}
            self._chunks.append(chunk_dict)
            self._total_samples += chunk_len
            self._offsets.append(self._total_samples)

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if idx < 0:
            idx = self._total_samples + idx
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_samples - 1}]")

        # Find which chunk contains this index
        for c_idx in range(len(self._chunks)):
            start = self._offsets[c_idx]
            end = self._offsets[c_idx + 1]
            if start <= idx < end:
                local_idx = idx - start
                chunk = self._chunks[c_idx]
                return {k: chunk[k][local_idx] for k in chunk}

        raise IndexError(f"Index {idx} not found in any chunk")
