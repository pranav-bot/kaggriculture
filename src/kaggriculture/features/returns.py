"""Net-worth delta reward and return-to-go calculations."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

BASE_COMMODITY_PRICES = {
    "WHEAT": 25.0,
    "CORN": 30.0,
    "SOY": 35.0,
    "MELON": 40.0,
    "STRAWBERRY": 200.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "EGG": 20.0,
}

SEED_REPLACEMENT_COST = {
    "WHEAT": 8.0,
    "CORN": 10.0,
    "SOY": 12.0,
    "MELON": 15.0,
    "STRAWBERRY": 50.0,
}

ANIMAL_VALUE = {
    "COW": 1000.0,
    "SHEEP": 600.0,
    "CHICKEN": 100.0,
}


def compute_net_worth(obs: Mapping[str, Any], seat: int = 0) -> float:
    """Compute total net worth = cash + market value of physical assets.

    Physical assets include:
    - Shed stored commodities
    - Carried commodities
    - Stored seeds (at replacement cost)
    - Active livestock in pastures
    """
    farms = obs.get("farms") or []
    farm = farms[seat] if 0 <= seat < len(farms) else {}

    # 1. Liquid cash
    cash = float(farm.get("money", 0.0) or 0.0)

    # 2. Market prices for inventory valuation
    market = obs.get("market") or {}
    prices = market.get("prices") or {}

    def get_price(item: str) -> float:
        return float(prices.get(item, BASE_COMMODITY_PRICES.get(item, 25.0)) or BASE_COMMODITY_PRICES.get(item, 25.0))

    # 3. Shed inventory
    private = obs.get("private") or {}
    shed = private.get("shed") or {}
    shed_val = 0.0
    if isinstance(shed, Mapping):
        for item, qty in shed.items():
            shed_val += float(qty or 0.0) * get_price(item)

    # 4. Carried inventory
    carried_val = 0.0
    carried = private.get("carried")
    if isinstance(carried, Mapping):
        for item, qty in carried.items():
            carried_val += float(qty or 0.0) * get_price(item)
    else:
        for inv in private.get("inventories") or []:
            if isinstance(inv, Mapping):
                for item, qty in inv.items():
                    carried_val += float(qty or 0.0) * get_price(item)

    # 5. Seed stock
    seeds = private.get("seeds") or {}
    seeds_val = 0.0
    if isinstance(seeds, Mapping):
        for item, qty in seeds.items():
            cost = SEED_REPLACEMENT_COST.get(item, 10.0)
            seeds_val += float(qty or 0.0) * cost

    # 6. Livestock valuation on farm tiles
    livestock_val = 0.0
    tiles = farm.get("tiles") or []
    for row in tiles:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, Mapping) and cell.get("kind") == "PASTURE":
                animal = cell.get("animal")
                if animal in ANIMAL_VALUE:
                    livestock_val += ANIMAL_VALUE[animal]

    return float(cash + shed_val + carried_val + seeds_val + livestock_val)


def compute_reward(obs_t: Mapping[str, Any], obs_t1: Mapping[str, Any], seat: int = 0) -> float:
    """Compute step reward as the change in total net worth from t to t+1."""
    nw_t = compute_net_worth(obs_t, seat)
    nw_t1 = compute_net_worth(obs_t1, seat)
    return float(nw_t1 - nw_t)


def compute_return_to_go(rewards: np.ndarray | list[float]) -> np.ndarray:
    """Compute undiscounted return-to-go for a sequence of step rewards.

    rtg[t] = sum_{k=t}^{T-1} rewards[k]
    Telescoping property: rtg[t] - rtg[t+1] == rewards[t]
    rtg[T-1] == rewards[T-1]
    """
    rew = np.asarray(rewards, dtype=np.float32)
    n = len(rew)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    rtg = np.zeros(n, dtype=np.float32)
    running_sum = 0.0
    for t in reversed(range(n)):
        running_sum += float(rew[t])
        rtg[t] = running_sum

    return rtg
