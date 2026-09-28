#!/usr/bin/env python3
"""Implicit Q-Learning (IQL) Value Network for kaggriculture.

Offline state-value function V_psi(s) trained with expectile regression
(tau=0.7) on undiscounted terminal-cash return-to-go targets.

IQL key property: the loss NEVER queries an explicit policy for unseen
actions. Training uses only (state, return_to_go) pairs from the empirical
il dataset distribution -- no actor, no max-Q over OOD actions.

State representation (multi-modal):
  - spatial:  (23, 10, 10) farm grid tensor  (features.spatial)
  - vec:      65-dim continuous vector = concat(
                  global_30 (cash, time, labor, quadrants, inventory),
                  market_35 (prices, inventory deviations, ratios))
Target:
  - return_to_go_t = (terminal_cash - cash_t) / TARGET_SCALE
    (undiscounted terminal cash delta, scaled for stable optimization)

Usage:
  .venv/bin/python scripts/train_iql_value.py \
      --num-episodes 150 --stride 10 --epochs 10 --out experiments/iql_value/iql_value_net.pt
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kaggriculture.features.global_state import extract_global_vector
from kaggriculture.features.market import extract_market_vector
from kaggriculture.features.spatial import extract_spatial_tensor

TAU = 0.7
TARGET_SCALE = 100_000.0  # cash units -> scaled regression target
GLOBAL_DIM = 30
MARKET_DIM = 35
VEC_DIM = GLOBAL_DIM + MARKET_DIM  # 65
SPATIAL_CHANNELS = 23


# ----------------------------------------------------------------------------
# IQL expectile loss
# ----------------------------------------------------------------------------
def expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """Asymmetric L2 (expectile regression) loss for the IQL state-value V_psi.

    L_tau = E[ |tau - 1(target - V < 0)| * (target - V)^2 ]

    With tau=0.7, positive residuals (target above current value estimate)
    get weight 0.7 and negative residuals weight 0.3, so V learns an upper
    expectile of the return distribution -- the value of the best action
    available *within* the data distribution, without ever querying a policy
    on out-of-distribution actions.
    """
    diff = target - pred
    weight = torch.where(diff >= 0.0, torch.full_like(diff, tau), torch.full_like(diff, 1.0 - tau))
    return (weight * diff.pow(2)).mean()


# ----------------------------------------------------------------------------
# Multi-modal value network
# ----------------------------------------------------------------------------
class IQLValueNet(nn.Module):
    """V_psi(s): spatial CNN branch + continuous-vector MLP branch, fused head.

    Args:
        spatial: (B, 23, 10, 10) farm grid.
        vec:     (B, 65) concat(global_30, market_35).
    Returns:
        (B,) scaled return-to-go prediction. Terminal cash estimate is
        cash_now + V(s) * TARGET_SCALE.
    NOTE: no action input, no policy query -- value depends on state only.
    """

    def __init__(self, vec_dim: int = VEC_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(SPATIAL_CHANNELS, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, spatial: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        s = self.spatial(spatial).flatten(1)  # (B, 64)
        g = self.vec_mlp(vec)  # (B, hidden)
        return self.head(torch.cat([s, g], dim=1)).squeeze(-1)  # (B,)


# ----------------------------------------------------------------------------
# Offline dataset construction (empirical distribution only)
# ----------------------------------------------------------------------------
def episode_samples(path: str, stride: int) -> tuple[list, list, list]:
    """Extract (spatial, vec, rtg) samples for both seats of one episode."""
    with gzip.open(path, "rt") as f:
        ep = json.load(f)
    steps = ep.get("steps") or []
    if not steps:
        return [], [], []
    n = len(steps)
    # Terminal liquid cash per seat defines the undiscounted return anchor.
    final_cash = []
    for seat in (0, 1):
        try:
            farms = steps[-1][seat]["observation"]["farms"]
            final_cash.append(float(farms[seat].get("money", 0.0) or 0.0))
        except (IndexError, KeyError, TypeError):
            final_cash.append(0.0)

    spat, vec, rtg = [], [], []
    for t in range(0, n, stride):
        for seat in (0, 1):
            try:
                obs = steps[t][seat]["observation"]
            except (IndexError, KeyError, TypeError):
                continue
            try:
                s = extract_spatial_tensor(obs, seat)
                gv = extract_global_vector(obs, seat)
                mv = extract_market_vector(obs.get("market") or {})
            except Exception:
                continue
            farms = obs.get("farms") or []
            farm = farms[seat] if 0 <= seat < len(farms) else {}
            cash = float(farm.get("money", 0.0) or 0.0)
            spat.append(s)
            vec.append(np.concatenate([gv, mv]).astype(np.float32))
            rtg.append(np.float32((final_cash[seat] - cash) / TARGET_SCALE))
    return spat, vec, rtg


def build_dataset(files: list[str], stride: int, label: str):
    spat_all, vec_all, rtg_all = [], [], []
    t0 = time.time()
    for i, path in enumerate(files):
        s, v, r = episode_samples(path, stride)
        spat_all.extend(s)
        vec_all.extend(v)
        rtg_all.extend(r)
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i + 1}/{len(files)} eps -> {len(rtg_all)} samples", flush=True)
    print(f"  [{label}] done: {len(rtg_all)} samples in {time.time() - t0:.1f}s", flush=True)
    S = np.stack(spat_all).astype(np.float32)
    V = np.stack(vec_all).astype(np.float32)
    R = np.asarray(rtg_all, dtype=np.float32)
    print(f"  [{label}] rtg(scaled): mean={R.mean():+.4f} std={R.std():.4f} "
          f"min={R.min():+.4f} max={R.max():+.4f} (= cash x{TARGET_SCALE:.0f})", flush=True)
    return S, V, R


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train IQL expectile value network")
    ap.add_argument("--episodes-dir", default=os.path.join(ROOT, "datasets", "il", "episodes"))
    ap.add_argument("--num-episodes", type=int, default=150)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "iql_value", "iql_value_net.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    files = sorted(glob.glob(os.path.join(args.episodes_dir, "*", "*.json.gz")))
    if not files:
        raise SystemExit(f"no episode files under {args.episodes_dir}")
    files = files[: args.num_episodes]
    print(f"Episodes: {len(files)} | stride={args.stride} | tau={args.tau} | epochs={args.epochs}")

    n_val = max(5, len(files) // 10)
    train_files, val_files = files[:-n_val], files[-n_val:]
    print(f"Split: {len(train_files)} train / {len(val_files)} val episodes (by episode)")

    S_tr, V_tr, R_tr = build_dataset(train_files, args.stride, "train")
    S_va, V_va, R_va = build_dataset(val_files, args.stride, "val")

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(S_tr), torch.from_numpy(V_tr), torch.from_numpy(R_tr)),
        batch_size=args.batch, shuffle=True, drop_last=False,
    )
    S_va_t = torch.from_numpy(S_va)
    V_va_t = torch.from_numpy(V_va)
    R_va_t = torch.from_numpy(R_va)

    model = IQLValueNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} (~{n_params * 4 / 1e6:.2f} MB fp32)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, cnt = 0.0, 0
        for sb, vb, rb in train_dl:
            opt.zero_grad()
            # IQL V-update: regression of V_psi(s) onto in-sample RTG only.
            # No policy network is constructed or queried anywhere.
            loss = expectile_loss(model(sb, vb), rb, tau=args.tau)
            loss.backward()
            opt.step()
            tot += loss.item() * len(sb)
            cnt += len(sb)
        model.eval()
        with torch.no_grad():
            val_loss = expectile_loss(model(S_va_t, V_va_t), R_va_t, tau=args.tau).item()
            mae_cash = (model(S_va_t, V_va_t) - R_va_t).abs().mean().item() * TARGET_SCALE
        print(f"epoch {epoch:02d}: train_expectile={tot / cnt:.6f} "
              f"val_expectile={val_loss:.6f} val_MAE={mae_cash:,.0f} cash", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "config": {
            "spatial_channels": SPATIAL_CHANNELS,
            "spatial_size": 10,
            "vec_dim": VEC_DIM,
            "hidden": 64,
            "tau": args.tau,
            "target": "return_to_go_terminal_cash_delta",
            "target_scale": TARGET_SCALE,
            "stride": args.stride,
            "val_expectile": best_val,
        },
    }
    torch.save(bundle, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out} ({size_mb:.2f} MB, limit 20 MB) | best val expectile={best_val:.6f}")
    assert size_mb < 20.0, "weight file exceeds 20MB Kaggle tarball budget"


if __name__ == "__main__":
    main()
