#!/usr/bin/env python3
"""Continually fine-tune the IQL value network from live Kaggle replay buffers.

Ingests newly parsed state-action-return tuples from `./replays/live_losses/`
and `./replays/live_wins/` (as well as pre-extracted .pt/.npz/.jsonl datasets).
Fine-tunes the base IQL value network (`iql_weights_v1.pt`) using a low learning
rate (1e-5) and an importance-weighted expectile regression loss:
- States from live losses where the network overestimated terminal cash or where
  the game resulted in low score/bankruptcy receive high importance weights.
- Only outputs `iql_weights_v2.pt` if the fine-tuned model achieves a lower
  Mean Squared Error on a holdout validation set compared to the v1 model.

Usage:
    .venv/bin/python scripts/continuous_train_iql.py
    .venv/bin/python scripts/continuous_train_iql.py --v1 experiments/iql_value/iql_weights_v1.pt \
                                                    --v2 experiments/iql_value/iql_weights_v2.pt \
                                                    --losses-dir replays/live_losses \
                                                    --wins-dir replays/live_wins \
                                                    --epochs 3 --lr 1e-5
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from train_iql_value import IQLValueNet, VEC_DIM, SPATIAL_CHANNELS, TARGET_SCALE, TAU
except ImportError:
    from scripts.train_iql_value import IQLValueNet, VEC_DIM, SPATIAL_CHANNELS, TARGET_SCALE, TAU

try:
    from kaggriculture.features.spatial import extract_spatial_tensor
    from kaggriculture.features.global_state import extract_global_vector
    from kaggriculture.features.market import extract_market_vector
except ImportError:
    extract_spatial_tensor = None  # type: ignore
    extract_global_vector = None  # type: ignore
    extract_market_vector = None  # type: ignore

DEFAULT_LR: float = 1e-5
DEFAULT_BATCH_SIZE: int = 128
HIGH_LOSS_WEIGHT: float = 6.0
BANKRUPTCY_WEIGHT: float = 10.0
DEFAULT_V1_PATH: str = "experiments/iql_value/iql_weights_v1.pt"
DEFAULT_V2_PATH: str = "experiments/iql_value/iql_weights_v2.pt"
DEFAULT_LOSSES_DIR: str = "replays/live_losses"
DEFAULT_WINS_DIR: str = "replays/live_wins"


# =============================================================================
# 1. Raw Replay Feature Extraction
# =============================================================================

def extract_tuples_from_kaggle_replay(
    replay_doc: Dict[str, Any],
    is_loss: bool = False,
    stride: int = 2,
    player_query: str = "Pranav",
) -> List[Dict[str, Any]]:
    """Extracts (spatial, vec, return, action, is_loss, bankruptcy) from a raw Kaggle replay JSON."""
    if extract_spatial_tensor is None or extract_global_vector is None or extract_market_vector is None:
        return []

    steps = replay_doc.get("steps") or []
    if not steps or len(steps) < 2:
        return []

    info = replay_doc.get("info", {})
    team_names = info.get("TeamNames", ["Player 0", "Player 1"])

    # Determine our seat
    my_seat = 0
    for idx, name in enumerate(team_names):
        if player_query.lower() in str(name).lower():
            my_seat = idx
            break
    opp_seat = 1 - my_seat

    rewards = replay_doc.get("rewards") or [0.0, 0.0]
    my_final_cash = float(rewards[my_seat] if my_seat < len(rewards) else 0.0)
    opp_final_cash = float(rewards[opp_seat] if opp_seat < len(rewards) else 0.0)

    # Replay outcome flags
    match_is_loss = is_loss or (my_final_cash < opp_final_cash)
    is_bankrupt = (my_final_cash <= 3000.0)

    records: List[Dict[str, Any]] = []
    n = len(steps)

    for t in range(0, n, max(1, stride)):
        step_frames = steps[t]
        if not isinstance(step_frames, list) or len(step_frames) <= my_seat:
            continue

        frame = step_frames[my_seat]
        obs = frame.get("observation")
        if not isinstance(obs, dict):
            continue

        farms = obs.get("farms") or []
        farm = farms[my_seat] if 0 <= my_seat < len(farms) else {}
        current_cash = float(farm.get("money", 0.0) or 0.0)

        try:
            spatial = extract_spatial_tensor(obs, my_seat)  # (23, 10, 10)
            gv = extract_global_vector(obs, my_seat)        # (30,)
            mv = extract_market_vector(obs.get("market") or {})  # (35,)
            vec = np.concatenate([gv, mv]).astype(np.float32)  # (65,)
        except Exception:
            continue

        # Target: scaled return-to-go terminal cash delta
        rtg = float((my_final_cash - current_cash) / TARGET_SCALE)

        records.append({
            "spatial": spatial,
            "vec": vec,
            "return": rtg,
            "action": 0,
            "is_loss": match_is_loss,
            "bankruptcy": is_bankrupt,
            "terminal_cash": my_final_cash,
            "prediction_loss": 1.0 if (match_is_loss and is_bankrupt) else (0.5 if match_is_loss else 0.0),
        })

    return records


# =============================================================================
# 2. Replay Loader & Dataset
# =============================================================================

def _records_from_object(obj: Any, is_loss_hint: bool = False, stride: int = 2) -> List[Dict[str, Any]]:
    """Normalizes raw dictionary, list, or Kaggle replay document into records."""
    if isinstance(obj, dict):
        if "state_dict" in obj:
            raise ValueError("Checkpoint supplied where replay data was expected")
        # Raw Kaggle replay document
        if "steps" in obj and isinstance(obj["steps"], list):
            return extract_tuples_from_kaggle_replay(obj, is_loss=is_loss_hint, stride=stride)
        # Columnar buffer
        if any(k in obj for k in ("spatial", "state_spatial")):
            return _columnar_records(obj)
        return [obj]

    if isinstance(obj, (list, tuple)):
        records: List[Dict[str, Any]] = []
        for item in obj:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                records.append({
                    "spatial": item[0],
                    "vec": item[1],
                    "return": item[2],
                    "action": item[3] if len(item) > 3 else 0,
                })
        return records

    raise ValueError(f"Unsupported replay payload: {type(obj).__name__}")


def _columnar_records(columns: Dict[str, Any]) -> List[Dict[str, Any]]:
    aliases = {
        "spatial": ("spatial", "state_spatial"),
        "vec": ("vec", "state_global"),
        "return": ("return", "return_to_go", "rtg"),
        "action": ("action",),
        "bankruptcy": ("bankruptcy", "bankrupt"),
        "prediction_loss": ("prediction_loss", "loss", "high_prediction_loss"),
        "is_loss": ("is_loss", "loss_match"),
    }
    key = next((k for k in aliases["spatial"] if k in columns), None)
    if key is None:
        return []
    n = len(columns[key])
    out: List[Dict[str, Any]] = []
    for i in range(n):
        rec: Dict[str, Any] = {}
        for name, keys in aliases.items():
            for candidate in keys:
                if candidate in columns:
                    rec[name] = columns[candidate][i]
                    break
        out.append(rec)
    return out


def load_replay_records(
    paths: Iterable[Union[str, os.PathLike[str]]],
    stride: int = 2,
) -> List[Dict[str, Any]]:
    """Loads state-action-return records across replay files and directories safely."""
    records: List[Dict[str, Any]] = []
    files: List[Tuple[Path, bool]] = []  # (path, is_loss_hint)

    for raw in paths:
        path = Path(raw)
        is_loss_path = "loss" in str(path).lower()
        if path.is_dir():
            for p in sorted(path.rglob("*")):
                if p.suffix.lower() in {".pt", ".pth", ".json", ".jsonl", ".npz", ".gz"}:
                    files.append((p, is_loss_path or ("loss" in str(p).lower())))
        elif path.exists():
            files.append((path, is_loss_path))

    for path, is_loss_hint in files:
        try:
            suffix = path.suffix.lower()
            if suffix == ".gz" or str(path).endswith(".json.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
            elif suffix in {".pt", ".pth"}:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    payload = {k: data[k] for k in data.files}
            elif suffix == ".jsonl":
                payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))

            records.extend(_records_from_object(payload, is_loss_hint=is_loss_hint, stride=stride))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    return records


def _value(record: Dict[str, Any], names: Tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


class ReplayDataset(Dataset):
    """PyTorch Dataset serving validated (spatial, vec, return, action, prediction_loss, bankruptcy, is_loss) tuples."""

    def __init__(self, records: Iterable[Dict[str, Any]]) -> None:
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
                if s.shape != (SPATIAL_CHANNELS, 10, 10) or v.numel() != VEC_DIM:
                    continue
                tgt = torch.tensor(float(target), dtype=torch.float32)
                act = torch.tensor(int(_value(record, ("action",), 0)), dtype=torch.long)
                loss_metric = float(_value(record, ("prediction_loss", "loss", "high_prediction_loss"), 0.0) or 0.0)
                bankrupt = torch.tensor(bool(_value(record, ("bankruptcy", "bankrupt"), False)), dtype=torch.bool)
                is_loss = torch.tensor(bool(_value(record, ("is_loss", "loss"), False)), dtype=torch.bool)

                # Row schema: (spatial, vec, target, action, prediction_loss, bankruptcy, is_loss)
                rows.append((s, v.reshape(VEC_DIM), tgt, act, loss_metric, bankrupt, is_loss))
            except (TypeError, ValueError, RuntimeError):
                continue
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        return self.rows[index]


def create_live_replay_dataloader(
    losses_dir: Union[str, Path] = DEFAULT_LOSSES_DIR,
    wins_dir: Union[str, Path] = DEFAULT_WINS_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stride: int = 2,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, ReplayDataset]:
    """Creates train and holdout validation DataLoaders ingesting from live losses and wins."""
    dirs = []
    if Path(losses_dir).exists():
        dirs.append(losses_dir)
    if Path(wins_dir).exists():
        dirs.append(wins_dir)

    records = load_replay_records(dirs, stride=stride)
    dataset = ReplayDataset(records)

    if len(dataset) < 2:
        empty_loader = DataLoader(dataset, batch_size=batch_size)
        return empty_loader, empty_loader, dataset

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    split = min(max(1, int(len(indices) * (1 - holdout_fraction))), len(indices) - 1)
    train_set = Subset(dataset, indices[:split])
    holdout_set = Subset(dataset, indices[split:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    holdout_loader = DataLoader(holdout_set, batch_size=batch_size, shuffle=False)

    return train_loader, holdout_loader, dataset


# =============================================================================
# 3. Critical Weighting & Importance Sampling
# =============================================================================

def importance_weights(
    prediction_loss: torch.Tensor,
    bankruptcy: torch.Tensor,
    *,
    is_loss: Optional[torch.Tensor] = None,
    prediction: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes importance sampling weights to correct overestimation hallucinations in loss matches.

    States from live losses where our network predicted high terminal cash (pred > target)
    are heavily weighted in the expectile regression loss function to aggressively correct bias.
    """
    difficult = torch.nan_to_num(
        prediction_loss, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp_min(0).clamp_max(1).to(torch.float32)

    # Compute overestimation error: ReLU(pred - target)
    if prediction is not None and target is not None:
        overestimation = torch.relu(prediction.detach() - target.detach())
        # If is_loss is provided, weight loss overestimation heavily
        if is_loss is not None:
            difficult = torch.maximum(difficult, (is_loss.to(torch.float32) * overestimation).clamp_max(2.0))
        else:
            difficult = torch.maximum(difficult, overestimation.clamp_max(2.0))

    weights = 1.0 + difficult * HIGH_LOSS_WEIGHT
    weights = weights + bankruptcy.to(torch.float32) * BANKRUPTCY_WEIGHT

    if is_loss is not None:
        weights = weights + is_loss.to(torch.float32) * 2.0

    if target is not None:
        weights = weights + (target.detach() <= 0).to(torch.float32) * 2.0

    return weights


def weighted_expectile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    tau: float = TAU,
) -> torch.Tensor:
    """Weighted asymmetric L2 expectile regression loss."""
    diff = target - pred
    expectile = torch.where(diff >= 0, torch.full_like(diff, tau), torch.full_like(diff, 1.0 - tau))
    weighted_diff = weights * expectile * diff.square()
    return weighted_diff.sum() / weights.sum().clamp_min(1e-8)


# =============================================================================
# 4. Model Loading & Evaluation
# =============================================================================

def load_model(path: Union[str, os.PathLike[str]]) -> Tuple[IQLValueNet, Dict[str, Any]]:
    """Loads existing IQL Value Network from checkpoint."""
    p = Path(path)
    if not p.is_file():
        # Fallback check for iql_value_net.pt if v1 was passed
        alt = p.parent / "iql_value_net.pt"
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"IQL value model checkpoint not found: {path}")

    payload = torch.load(p, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    config = payload.get("config", {}) if isinstance(payload, dict) else {}

    model = IQLValueNet(
        vec_dim=int(config.get("vec_dim", VEC_DIM)),
        hidden=int(config.get("hidden", 64)),
    )
    model.load_state_dict(state, strict=True)
    return model, config


def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    """Calculates Mean Squared Error on validation/holdout dataloader."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            spatial, vec, target = batch[0], batch[1], batch[2]
            pred = model(spatial, vec)
            total_loss += float((pred - target).square().sum())
            count += len(target)
    return total_loss / count if count > 0 else float("inf")


# =============================================================================
# 5. Continuous Training Pipeline
# =============================================================================

def train_continuous(
    v1_path: str = DEFAULT_V1_PATH,
    v2_path: str = DEFAULT_V2_PATH,
    replay_paths: Optional[Sequence[Union[str, os.PathLike[str]]]] = None,
    *,
    epochs: int = 3,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    holdout_fraction: float = 0.2,
    stride: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fine-tunes IQL Value Network and atomically emits v2 only upon validation MSE improvement."""
    v1_file = Path(v1_path)
    if not v1_file.is_file():
        alt = v1_file.parent / "iql_value_net.pt"
        if alt.is_file():
            v1_file = alt
        else:
            raise FileNotFoundError(f"v1 checkpoint not found: {v1_path}")

    target_paths = list(replay_paths) if replay_paths else [DEFAULT_LOSSES_DIR, DEFAULT_WINS_DIR]
    records = load_replay_records(target_paths, stride=stride)
    dataset = ReplayDataset(records)

    if len(dataset) < 2:
        return {
            "status": "blocked",
            "reason": "insufficient_valid_replay_samples",
            "samples": len(dataset),
            "saved": False,
        }

    # Deterministic train/holdout split
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    split = min(max(1, int(len(indices) * (1 - holdout_fraction))), len(indices) - 1)
    train_set = Subset(dataset, indices[:split])
    holdout_set = Subset(dataset, indices[split:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    holdout_loader = DataLoader(holdout_set, batch_size=batch_size, shuffle=False)

    # 1. Load base v1 model
    model, config = load_model(v1_file)
    baseline_mse = evaluate_mse(model, holdout_loader)

    print(f"📊 Baseline v1 Holdout MSE: {baseline_mse:.6f} across {len(holdout_set)} validation samples.")

    # 2. Fine-tuning loop with very low learning rate (1e-5)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tau = float(config.get("tau", TAU))

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            spatial, vec, target = batch[0], batch[1], batch[2]
            prediction_loss, bankruptcy = batch[4], batch[5]
            is_loss = batch[6] if len(batch) > 6 else None

            optimizer.zero_grad()
            pred = model(spatial, vec)

            weights = importance_weights(
                prediction_loss,
                bankruptcy,
                is_loss=is_loss,
                prediction=pred,
                target=target,
            )

            loss = weighted_expectile_loss(pred, target, weights, tau=tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        print(f"   Epoch {epoch}/{epochs} - Weighted Expectile Loss: {avg_loss:.6f}")

    # 3. Validation MSE Comparison Gate
    candidate_mse = evaluate_mse(model, holdout_loader)
    improvement = baseline_mse - candidate_mse
    pct_improvement = (improvement / baseline_mse * 100.0) if baseline_mse > 0 else 0.0

    print(f"📊 Candidate v2 Holdout MSE: {candidate_mse:.6f} (Delta: {improvement:+.6f}, {pct_improvement:+.2f}%)")

    result: Dict[str, Any] = {
        "status": "improved" if candidate_mse < baseline_mse else "unchanged",
        "baseline_mse": baseline_mse,
        "candidate_mse": candidate_mse,
        "pct_improvement": pct_improvement,
        "train_samples": len(train_set),
        "holdout_samples": len(holdout_set),
        "samples": len(dataset),
        "total_samples": len(dataset),
        "saved": False,
    }

    # 4. Atomic output of iql_weights_v2.pt ONLY if MSE strictly improved
    if candidate_mse < baseline_mse:
        destination = Path(v2_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = destination.with_name(destination.name + ".tmp")

        saved_payload = {
            "state_dict": model.state_dict(),
            "config": {
                **config,
                "learning_rate": lr,
                "epochs": epochs,
                "holdout_mse": candidate_mse,
                "holdout_mse_v1": baseline_mse,
                "holdout_mse_v2": candidate_mse,
                "pct_improvement": pct_improvement,
                "base_checkpoint": str(v1_file),
            },
        }
        torch.save(saved_payload, temp_dest)
        os.replace(temp_dest, destination)
        result["saved"] = True
        print(f"🎉 Validation Gate PASSED: Model improved! Emitted: {destination.resolve()}")
    else:
        print(f"🛑 Validation Gate BLOCKED: Candidate MSE did not improve over v1 baseline. Did not save {v2_path}.")

    return result



def main() -> None:
    parser = argparse.ArgumentParser(description="Continually fine-tune IQL value network on live replays.")
    parser.add_argument("--v1", default=DEFAULT_V1_PATH, help="Path to base v1 checkpoint")
    parser.add_argument("--v2", default=DEFAULT_V2_PATH, help="Output path for fine-tuned v2 checkpoint")
    parser.add_argument("--losses-dir", default=DEFAULT_LOSSES_DIR, help="Path to live losses replays")
    parser.add_argument("--wins-dir", default=DEFAULT_WINS_DIR, help="Path to live wins replays")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (default: 1e-5)")
    parser.add_argument("--stride", type=int, default=2, help="Observation subsampling stride")
    parser.add_argument("--holdout", type=float, default=0.2, help="Holdout validation fraction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    replay_targets = [args.losses_dir, args.wins_dir]
    report = train_continuous(
        v1_path=args.v1,
        v2_path=args.v2,
        replay_paths=replay_targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        holdout_fraction=args.holdout,
        stride=args.stride,
        seed=args.seed,
    )
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
