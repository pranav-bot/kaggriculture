"""Compact public-information opponent intent model.

The model deliberately consumes only the opponent's public farm, cash, and
market tape.  Torch is optional so the policy remains usable in the small
submission/runtime environments; when it is unavailable a deterministic
feature-based fallback is used.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from kaggriculture.features.spatial import extract_opponent_spatial

OPPONENT_INTENTS = (
    "PREPARING_EXPANSION",
    "HOARDING_FERTILIZER",
    "LIVESTOCK_RUSH",
    "CROP_ROTATION",
)
HORIZON_HOURS = 24
_MARKET_ITEMS = ("WHEAT", "CORN", "SOY", "MELON", "STRAWBERRY", "MILK", "WOOL")


def _observation(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping) and isinstance(item.get("observation"), Mapping):
        return item["observation"]
    return item if isinstance(item, Mapping) else {}


def _market_features(obs: Mapping[str, Any], seat: int) -> np.ndarray:
    farms = obs.get("farms") or []
    opp = farms[1 - seat] if 0 <= 1 - seat < len(farms) else {}
    market = obs.get("market") or {}
    prices = market.get("prices") or market.get("price") or {}
    inventory = market.get("inventory") or market.get("stocks") or {}
    features = [
        np.log1p(max(0.0, float(opp.get("money", 0.0) or 0.0))) / 15.0,
        min(1.0, len(opp.get("unlocked_quadrants") or []) / 4.0),
        float(obs.get("hour", 0) or 0) / 24.0,
        float(obs.get("day", 0) or 0) / 30.0,
    ]
    for item in _MARKET_ITEMS:
        features.append(np.log1p(max(0.0, float(prices.get(item, 0.0) or 0.0))) / 10.0)
    for item in _MARKET_ITEMS:
        features.append(np.log1p(max(0.0, float(inventory.get(item, 0.0) or 0.0))) / 10.0)
    return np.asarray(features, dtype=np.float32)


def encode_opponent_observation(
    obs: Mapping[str, Any], seat: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the public spatial grid and market/cash feature vector."""
    return extract_opponent_spatial(obs, seat), _market_features(obs, seat)


class OpponentIntentModel:
    """Small causal spatial-plus-tape classifier for a 24-hour intent horizon.

    The torch implementation is a two-layer causal Transformer with a
    128-dimensional representation.  No training/checkpoint is assumed here:
    deployments can load a state dict, while the safe fallback still provides
    calibrated, non-zero probabilities from observable signals.
    """

    def __init__(
        self,
        *,
        seat: int = 0,
        max_history: int = 24,
        hidden_size: int = 128,
        num_layers: int = 2,
        use_torch: bool = True,
    ) -> None:
        self.seat = int(seat)
        self.max_history = max(1, int(max_history))
        self.hidden_size = int(hidden_size)
        self.num_layers = max(2, min(4, int(num_layers)))
        self._torch = None
        self.network = None
        if use_torch:
            try:
                import torch
                import torch.nn as nn

                torch.set_num_threads(1)
                self._torch = torch

                class _Net(nn.Module):
                    def __init__(self, channels: int, market_dim: int, hidden: int, layers: int):
                        super().__init__()
                        self.spatial = nn.Sequential(
                            nn.Conv2d(channels, 32, 3, padding=1),
                            nn.GELU(),
                            nn.Conv2d(32, 48, 3, stride=2, padding=1),
                            nn.GELU(),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(48, hidden // 2),
                            nn.GELU(),
                        )
                        self.public = nn.Sequential(
                            nn.Linear(market_dim, hidden // 2), nn.GELU()
                        )
                        layer = nn.TransformerEncoderLayer(
                            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                            dropout=0.0, batch_first=True, norm_first=False,
                        )
                        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
                        self.head = nn.Linear(hidden, len(OPPONENT_INTENTS))

                    def forward(self, grids, public):
                        batch, length, channels, height, width = grids.shape
                        spatial = self.spatial(
                            grids.reshape(batch * length, channels, height, width)
                        ).reshape(batch, length, -1)
                        x = torch.cat((spatial, self.public(public)), dim=-1)
                        length = x.shape[1]
                        mask = torch.triu(
                            torch.ones(length, length, device=x.device, dtype=torch.bool),
                            diagonal=1,
                        )
                        return self.head(self.temporal(x, mask=mask)[:, -1])

                sample_grid, sample_public = encode_opponent_observation({}, self.seat)
                self.network = _Net(
                    sample_grid.shape[0], sample_public.shape[0],
                    self.hidden_size, self.num_layers,
                ).eval()
            except (ImportError, RuntimeError):
                self._torch = None
                self.network = None

    def _fallback(self, grids: np.ndarray, public: np.ndarray) -> np.ndarray:
        """Stable prior for untrained/lightweight runtimes."""
        latest = grids[-1]
        scores = np.zeros(4, dtype=np.float32)
        # Public opponent grid channels: empty, pasture, plant, fertilized,
        # animals (13:16), plus public cash/quadrants/market tape.
        empty, pasture, plant = latest[1].mean(), latest[3].mean(), latest[4].mean()
        fertilized = latest[12].mean()
        livestock = latest[13:16].mean()
        cash = public[-1, 0]
        unlocked = public[-1, 1]
        scores[0] = 1.5 * unlocked + 1.2 * empty + 0.3 * cash
        scores[1] = 2.0 * fertilized + 0.4 * cash
        scores[2] = 2.0 * livestock + 0.5 * pasture
        scores[3] = 1.6 * plant + 0.4 * (1.0 - fertilized)
        scores -= scores.max()
        p = np.exp(scores)
        return (p / p.sum()).astype(np.float32)

    def predict_proba(self, historical_obs_buffer: Sequence[Any]) -> dict[str, float]:
        """Predict the four intents from the most recent public observations."""
        observations = [_observation(x) for x in (historical_obs_buffer or [])]
        if not observations:
            return {name: 0.25 for name in OPPONENT_INTENTS}
        observations = observations[-self.max_history:]
        encoded = [encode_opponent_observation(x, self.seat) for x in observations]
        grids = np.stack([x[0] for x in encoded]).astype(np.float32)
        public = np.stack([x[1] for x in encoded]).astype(np.float32)
        if self.network is not None:
            with self._torch.inference_mode():
                logits = self.network(
                    self._torch.from_numpy(grids[None]),
                    self._torch.from_numpy(public[None]),
                )[0]
                probs = self._torch.softmax(logits, dim=-1).numpy()
        else:
            probs = self._fallback(grids, public)
        probs = np.nan_to_num(probs, nan=0.25, posinf=1.0, neginf=0.0)
        probs = probs / max(float(probs.sum()), 1e-8)
        return {name: float(probs[i]) for i, name in enumerate(OPPONENT_INTENTS)}

    __call__ = predict_proba


_DEFAULT_MODEL: OpponentIntentModel | None = None


def get_opponent_intent(
    historical_obs_buffer: Sequence[Any],
    *,
    model: OpponentIntentModel | None = None,
) -> dict[str, float]:
    """Return the next-24-hour opponent intent distribution.

    A model can be supplied by a controller or test; otherwise a warmed,
    process-local default keeps repeated per-turn queries inexpensive.
    """
    global _DEFAULT_MODEL
    if model is None:
        if _DEFAULT_MODEL is None:
            _DEFAULT_MODEL = OpponentIntentModel()
        model = _DEFAULT_MODEL
    return model.predict_proba(historical_obs_buffer)
