"""Causal probabilistic market forecasting utilities.

The forecaster deliberately stays small: a GRU only consumes the current and
past hourly observations, then predicts a diagonal Gaussian for six horizons.
This makes it useful for offline experiments without introducing an attention
mask or a large transformer dependency.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from kaggriculture.env.items import MARKET_PARAMS, PRICE_FLOOR, SHOPS, market_price

HISTORY_LENGTH = 48
FORECAST_HORIZON = 6
FEATURE_NAMES = (
    "price",
    "inventory",
    "volume",
    "demand",
    "opponent_cash_reserves",
    "opponent_visible_cows",
    "hour_sin",
    "hour_cos",
    "price_delta",
    "inventory_delta",
)
ACTIVE_SHOP_NAMES = tuple(sorted(SHOPS))
MARKET_FEATURE_NAMES = FEATURE_NAMES[:6] + ("price_delta", "inventory_delta") + tuple(
    f"active_{name.lower()}" for name in ACTIVE_SHOP_NAMES
)


def encode_market_state(
    state: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
    active_shops: Sequence[str] = (),
) -> np.ndarray:
    """Encode one hourly observation, including active-shop one-hot features."""
    market = state.get("market", state)
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    volumes = market.get("volume", market.get("volumes", {})) or {}
    demand = market.get("demand", {}) or {}
    item = str(state.get("item", market.get("item", "MELON"))).upper()
    def value(source: Any, key: str, fallback: Any = 0.0) -> float:
        if isinstance(source, Mapping):
            source = source.get(key, fallback)
        return float(source or 0.0)

    price = value(prices, item, state.get("price", 0.0))
    inv = value(inventory, item, state.get("inventory", 0.0))
    volume = value(volumes, item, state.get("volume", 0.0))
    demand_value = value(demand, item, state.get("demand", 0.0))
    farms = state.get("farms", []) or []
    player = int(state.get("player", 0) or 0)
    opponent = farms[1 - player] if isinstance(farms, Sequence) and len(farms) > 1 else {}
    opponent_cash = value(
        opponent,
        "money",
        state.get("opponent_cash_reserves", state.get("opponent_cash", 0.0)),
    )
    animals = opponent.get("animals", {}) if isinstance(opponent, Mapping) else {}
    opponent_cows = state.get("opponent_visible_cows", 0.0)
    if isinstance(animals, Mapping):
        opponent_cows = animals.get("COW", animals.get("cow", opponent_cows))
    elif isinstance(opponent, Mapping):
        opponent_cows = opponent.get("cows", opponent_cows)
    hour = float(state.get("hour", state.get("step", 0))) % 24.0
    prev_market = (previous or {}).get("market", previous or {})
    prev_price = value(prev_market.get("prices", {}) or {}, item, price)
    prev_inv = value(prev_market.get("inventory", {}) or {}, item, inv)
    values = [price, inv, volume, demand_value, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    values += [price - prev_price, inv - prev_inv]
    if not active_shops:
        town = state.get("town", {}) or {}
        active_shops = (
            town.get("active_shops", town.get("unlocked_shops", ()))
            if isinstance(town, Mapping)
            else ()
        )
    selected = {str(name).upper() for name in active_shops}
    values[4:4] = [float(opponent_cash), float(opponent_cows)]
    values += [float(name in selected) for name in ACTIVE_SHOP_NAMES]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class MarketWindowDataset(torch.utils.data.Dataset):
    """Minimal leak-free sliding-window dataset for encoded market sequences."""

    def __init__(self, features: Tensor | np.ndarray, targets: Tensor | np.ndarray, window: int = HISTORY_LENGTH):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        if self.features.ndim != 2 or self.targets.ndim != 2 or len(self.features) != len(self.targets):
            raise ValueError("features and targets must be [time, feature] and [time, horizon]")
        if window < 1 or len(self.features) <= window:
            raise ValueError("sequence must contain more rows than window")
        self.window = int(window)

    def __len__(self) -> int:
        return len(self.features) - self.window

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.features[index:index + self.window], self.targets[index + self.window]


class MarketForecaster(nn.Module):
    """A lightweight causal GRU forecaster for 48 hourly market states."""

    history_length = HISTORY_LENGTH
    forecast_horizon = FORECAST_HORIZON

    def __init__(
        self,
        input_size: int = len(MARKET_FEATURE_NAMES),
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = FORECAST_HORIZON,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1 or horizon < 1:
            raise ValueError("input_size, hidden_size, num_layers, and horizon must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.encoder = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mean_head = nn.Linear(hidden_size, horizon)
        self.log_variance_head = nn.Linear(hidden_size, horizon)

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mean, log_variance)`` with shape ``[batch, horizon]``.

        A single ``[48, features]`` state is accepted for convenient inference.
        The final GRU state is causal by construction: no future row is read.
        """
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.ndim != 3:
            raise ValueError("states must have shape [batch, 48, features] or [48, features]")
        if states.shape[1] != HISTORY_LENGTH:
            raise ValueError(f"expected {HISTORY_LENGTH} hourly states, got {states.shape[1]}")
        if states.shape[-1] != self.input_size:
            raise ValueError(f"expected {self.input_size} features, got {states.shape[-1]}")
        _, hidden = self.encoder(states)
        representation = hidden[-1]
        mean = self.mean_head(representation)
        # Bounding variance avoids numerical explosions during early offline training.
        log_variance = self.log_variance_head(representation).clamp(-12.0, 8.0)
        return mean, log_variance

    def predict(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Inference alias matching common project model conventions."""
        return self(states)

    def loss(self, states: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        """Compute Gaussian NLL directly from a state/target batch."""
        mean, log_variance = self(states)
        return gaussian_nll_loss(mean, log_variance, targets, reduction=reduction)

    def get_optimal_liquidation_volume(
        self,
        current_inventory: float,
        predicted_price_curve: Sequence[float] | Tensor | np.ndarray,
        *,
        item: str = "MELON",
        max_volume: int | None = None,
        params: Mapping[str, Mapping[str, object]] | None = None,
    ) -> int:
        """Return a conservative volume whose marginal price beats its hold value."""
        return get_optimal_liquidation_volume(
            current_inventory,
            predicted_price_curve,
            item=item,
            max_volume=max_volume,
            params=params,
        )


def gaussian_nll_loss(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Diagonal Gaussian negative log likelihood (constant omitted).

    ``log_variance`` is used directly rather than a variance output, which keeps
    the head unconstrained while the bounded log-variance remains numerically
    stable.
    """
    if mean.shape != log_variance.shape or mean.shape != target.shape:
        raise ValueError("mean, log_variance, and target must have identical shapes")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    nll = 0.5 * (log_variance + (target - mean).square() * torch.exp(-log_variance))
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    return nll


# Descriptive alias used by training scripts.
probabilistic_nll_loss = gaussian_nll_loss


def get_optimal_liquidation_volume(
    current_inventory: float,
    predicted_price_curve: Sequence[float] | Tensor | np.ndarray,
    *,
    item: str = "MELON",
    max_volume: int | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> int:
    """Choose a safe sell quantity under the game's nonlinear price curve.

    The forecast curve is treated as the reservation value of holding one unit.
    A unit is liquidated only while its *actual* marginal market price exceeds
    the best forecast price.  Prices are evaluated through the canonical
    ``market_price`` function, so square/hinge curves and the hard ``$1`` floor
    are respected.  The bounded loop also makes this safe for huge inventories.
    """
    inventory = max(0, int(np.floor(float(current_inventory))))
    if inventory == 0:
        return 0
    curve = torch.as_tensor(predicted_price_curve, dtype=torch.float32).detach().cpu().flatten()
    curve = curve[torch.isfinite(curve)]
    if curve.numel() == 0:
        return 0
    reservation = max(float(curve.max().item()), float(PRICE_FLOOR))
    limit = inventory if max_volume is None else min(inventory, max(0, int(max_volume)))
    if limit == 0:
        return 0

    resolved = params if params is not None else MARKET_PARAMS
    item = str(item).upper()
    if item not in resolved:
        raise KeyError(f"unknown market item: {item}")

    sold = 0
    # Selling raises market inventory by one except at the floor.  Stop as soon
    # as the marginal quote is no longer above the opportunity cost.
    for offset in range(limit):
        quote = float(market_price(item, inventory + offset, resolved))
        if quote <= reservation or quote <= PRICE_FLOOR:
            break
        sold += 1
    return sold
