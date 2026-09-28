#!/usr/bin/env python3
"""Continually fine-tune the IQL value network from live replay buffers.

The input format is deliberately permissive: ``.pt``/``.pth`` files may
contain a bundle, a list of records, or a dict of column tensors; JSONL/JSON
and NPZ files are also accepted.  A record must contain ``spatial``, ``vec``
and a return field (``return``, ``return_to_go`` or ``rtg``).  ``action`` is
ingested and retained for schema compatibility, although the IQL value model
does not use actions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:  # Works both as ``python scripts/...`` and as ``import scripts...``.
        from .train_iql_value import IQLValueNet, VEC_DIM
except ImportError:
        from train_iql_value import IQLValueNet, VEC_DIM

LR = 1e-5
DEFAULT_BATCH_SIZE = 256
HIGH_LOSS_WEIGHT = 6.0
BANKRUPTCY_WEIGHT = 10.0


def _records_from_object(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        if "state_dict" in obj:
            raise ValueError("checkpoint supplied where replay data was expected")
        # Columnar buffers are common in torch/numpy data exports.
        return _columnar_records(obj) if any(k in obj for k in ("spatial", "state_spatial")) else [obj]
    if isinstance(obj, (list, tuple)):
        records = []
        for item in obj:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                records.append({"spatial": item[0], "vec": item[1], "return": item[2],
                                "action": item[3] if len(item) > 3 else 0})
        return records
    raise ValueError(f"unsupported replay payload: {type(obj).__name__}")


def _columnar_records(columns: dict[str, Any]) -> list[dict[str, Any]]:
    aliases = {"spatial": ("spatial", "state_spatial"), "vec": ("vec", "state_global"),
               "return": ("return", "return_to_go", "rtg"), "action": ("action",)}
    key = next((k for k in aliases["spatial"] if k in columns), None)
    if key is None:
        return []
    n = len(columns[key])
    out = []
    for i in range(n):
        rec: dict[str, Any] = {}
        for name, keys in aliases.items():
            for candidate in keys:
                if candidate in columns:
                    rec[name] = columns[candidate][i]
                    break
        out.append(rec)
    return out


def load_replay_records(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Read replay records while skipping malformed files/rows safely."""
    records: list[dict[str, Any]] = []
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.suffix.lower() in {".pt", ".pth", ".json", ".jsonl", ".npz"})
        elif path.exists():
            files.append(path)
    for path in files:
        try:
            suffix = path.suffix.lower()
            if suffix in {".pt", ".pth"}:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    payload = {k: data[k] for k in data.files}
            elif suffix == ".jsonl":
                payload = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            else:
                payload = json.loads(path.read_text())
            records.extend(_records_from_object(payload))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return records


def _value(record: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


class ReplayDataset(Dataset):
    """Validated state/action/return tuples suitable for a PyTorch DataLoader."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        rows = []
        for record in records:
            state = record.get("state") if isinstance(record.get("state"), dict) else record
            spatial = _value(state, ("spatial", "state_spatial"))
            vec = _value(state, ("vec", "state_global"))
            target = _value(record, ("return", "return_to_go", "rtg"))
            if spatial is None or vec is None or target is None:
                continue
            try:
                s = torch.as_tensor(spatial, dtype=torch.float32)
                v = torch.as_tensor(vec, dtype=torch.float32)
                if s.shape != (23, 10, 10) or v.numel() != VEC_DIM:
                    continue
                rows.append((s, v.reshape(VEC_DIM), torch.tensor(float(target)),
                             torch.tensor(int(_value(record, ("action",), 0))),
                             float(_value(record, ("prediction_loss", "loss", "high_prediction_loss"), 0.0) or 0.0),
                             torch.tensor(bool(_value(record, ("bankruptcy", "bankrupt"), False)))))
            except (TypeError, ValueError, RuntimeError):
                continue
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def importance_weights(
    prediction_loss: torch.Tensor,
    bankruptcy: torch.Tensor,
    *,
    prediction: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Emphasize hallucinated high-value predictions and bad outcomes.

    Replays may carry a cached prediction loss, but the current model's
    prediction error is preferred when available so weights track v1's actual
    overestimation bias during fine-tuning.
    """
    difficult = torch.nan_to_num(
        prediction_loss, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp_min(0).clamp_max(1).to(torch.float32)
    if prediction is not None and target is not None:
        difficult = torch.maximum(
            difficult,
            (prediction.detach() - target.detach()).abs().clamp_max(1.0),
        )
    weights = 1.0 + difficult * HIGH_LOSS_WEIGHT
    weights = weights + bankruptcy.to(torch.float32) * BANKRUPTCY_WEIGHT
    if target is not None:
        weights = weights + (target.detach() <= 0).to(torch.float32) * 2.0
    return weights


def weighted_expectile_loss(
    pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, tau: float = 0.7
) -> torch.Tensor:
    diff = target - pred
    expectile = torch.where(diff >= 0, torch.full_like(diff, tau), torch.full_like(diff, 1 - tau))
    return (weights * expectile * diff.square()).sum() / weights.sum().clamp_min(1e-8)


def load_model(path: str | os.PathLike[str]) -> tuple[IQLValueNet, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model = IQLValueNet(vec_dim=int(config.get("vec_dim", VEC_DIM)), hidden=int(config.get("hidden", 64)))
    model.load_state_dict(state, strict=True)
    return model, config


def evaluate_mse(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = count = 0
    with torch.no_grad():
        for spatial, vec, target, *_ in loader:
            total += float((model(spatial, vec) - target).square().sum())
            count += len(target)
    return total / count if count else float("inf")


def train_continuous(
    v1_path: str,
    v2_path: str,
    replay_paths: Iterable[str | os.PathLike[str]],
    *,
    epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = LR,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Fine-tune and atomically publish v2 only when holdout MSE improves."""
    if not Path(v1_path).is_file():
        raise FileNotFoundError(f"v1 checkpoint not found: {v1_path}")
    records = load_replay_records(replay_paths)
    dataset = ReplayDataset(records)
    if len(dataset) < 2:
        return {"status": "blocked", "reason": "no_valid_replay_samples", "samples": len(dataset), "saved": False}
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    split = min(max(1, int(len(indices) * (1 - holdout_fraction))), len(indices) - 1)
    train_set = torch.utils.data.Subset(dataset, indices[:split])
    holdout_set = torch.utils.data.Subset(dataset, indices[split:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    holdout_loader = DataLoader(holdout_set, batch_size=batch_size)
    model, config = load_model(v1_path)
    baseline = evaluate_mse(model, holdout_loader)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(max(1, epochs)):
        for spatial, vec, target, _action, prediction_loss, bankruptcy in train_loader:
            optimizer.zero_grad()
            pred = model(spatial, vec)
            loss = weighted_expectile_loss(
                pred,
                target,
                importance_weights(
                    prediction_loss,
                    bankruptcy,
                    prediction=pred,
                    target=target,
                ),
                                            tau=float(config.get("tau", 0.7)))
            loss.backward()
            optimizer.step()
    candidate = evaluate_mse(model, holdout_loader)
    result = {"status": "improved" if candidate < baseline else "unchanged", "baseline_mse": baseline,
              "candidate_mse": candidate, "samples": len(dataset), "saved": False}
    if candidate < baseline:
        destination = Path(v2_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".tmp")
        torch.save({"state_dict": model.state_dict(), "config": {**config, "learning_rate": lr,
                   "holdout_mse": candidate, "base_checkpoint": str(v1_path)}}, temp)
        os.replace(temp, destination)
        result["saved"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", default="experiments/iql_value/iql_weights_v1.pt")
    parser.add_argument("--v2", default="experiments/iql_value/iql_weights_v2.pt")
    parser.add_argument("--replay", nargs="+", default=["replays", "live_losses", "live_wins"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train_continuous(args.v1, args.v2, args.replay, epochs=args.epochs,
                                       batch_size=args.batch_size, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
