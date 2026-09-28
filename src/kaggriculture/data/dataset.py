"""PyTorch Dataset implementation for offline transition chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset


class KaggricultureDataset(Dataset):
    """High-performance PyTorch Dataset serving preprocessed offline transitions."""

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.chunk_files = sorted(list(self.cache_dir.glob("*.npz")))
        if not self.chunk_files:
            raise FileNotFoundError(f"No .npz chunk files found in {self.cache_dir}")
        self._chunks: list[dict[str, np.ndarray]] = []
        self._offsets: list[int] = [0]
        self._total_samples = 0

        for cf in self.chunk_files:
            try:
                # Load arrays into memory for high-throughput collation
                data = np.load(cf)
                keys = list(data.keys())
                if not keys:
                    continue
                first_arr = data[keys[0]]
                chunk_len = len(first_arr)
                chunk_dict = {k: np.array(data[k]) for k in keys}
                self._chunks.append(chunk_dict)
                self._total_samples += chunk_len
                self._offsets.append(self._total_samples)
            except Exception:
                continue

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if idx < 0:
            idx = self._total_samples + idx
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self._total_samples}")

        # Locate chunk containing this index
        for c_idx in range(len(self._chunks)):
            start = self._offsets[c_idx]
            end = self._offsets[c_idx + 1]
            if start <= idx < end:
                local_idx = idx - start
                chunk = self._chunks[c_idx]

                state_t = torch.from_numpy(chunk["state_spatial"][local_idx]).float()
                opp_state_t = torch.from_numpy(chunk["opponent_spatial"][local_idx]).float()
                market_t = torch.from_numpy(chunk["market"][local_idx]).float()
                action_t = torch.tensor(chunk["action"][local_idx], dtype=torch.long)
                reward_t = torch.tensor(chunk["reward"][local_idx], dtype=torch.float32)
                rtg_t = torch.tensor(chunk["return_to_go"][local_idx], dtype=torch.float32)
                strat_t = torch.tensor(chunk["strategy_cluster"][local_idx], dtype=torch.long)
                cf_val_t = torch.tensor(chunk["counterfactual_value"][local_idx], dtype=torch.float32)

                return (
                    state_t,
                    opp_state_t,
                    market_t,
                    action_t,
                    reward_t,
                    rtg_t,
                    strat_t,
                    cf_val_t,
                )

        raise IndexError(f"Index {idx} could not be resolved across chunks")
