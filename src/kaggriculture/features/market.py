"""Continuous market vector feature extraction for Kaggriculture."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

MARKET_COMMODITIES = [
    "WHEAT",
    "CORN",
    "SOY",
    "MELON",
    "STRAWBERRY",
    "MILK",
    "WOOL",
    "EGG",
]

BASE_PRICES = {
    "WHEAT": 25.0,
    "CORN": 30.0,
    "SOY": 35.0,
    "MELON": 40.0,
    "STRAWBERRY": 200.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "EGG": 20.0,
}

BASELINE_INVENTORY = 10000.0


def extract_market_vector(market_dict: Mapping[str, Any]) -> np.ndarray:
    """Extract a 35-dimensional continuous market vector.

    Includes:
    - Wholesale prices per commodity (normalized)
    - Exact inventory deviations relative to 10,000 unit baseline: (I - 10,000) / 10,000
    - Log-transformed inventories
    - Price ratios relative to equilibrium base prices
    - Aggregate market deviations and volume ratios
    """
    vec = np.zeros(35, dtype=np.float32)
    prices = market_dict.get("prices") or {}
    inventory = market_dict.get("inventory") or {}

    inv_deviations = []
    price_ratios = []
    total_inv = 0.0

    for i, item in enumerate(MARKET_COMMODITIES):
        price = float(prices.get(item, BASE_PRICES.get(item, 25.0)) or BASE_PRICES.get(item, 25.0))
        inv = float(inventory.get(item, BASELINE_INVENTORY) or 0.0)
        safe_inv = max(0.0, inv)
        total_inv += safe_inv

        base_p = BASE_PRICES.get(item, 25.0)

        # 1. Price normalized (0..8)
        vec[i] = price / 200.0

        # 2. Inventory deviation relative to 10,000 unit baseline (8..16)
        inv_dev = (safe_inv - BASELINE_INVENTORY) / BASELINE_INVENTORY
        inv_dev_clipped = np.clip(inv_dev, -1.0, 10.0)
        vec[8 + i] = inv_dev_clipped
        inv_deviations.append(inv_dev_clipped)

        # 3. Log inventory (16..24)
        vec[16 + i] = np.log1p(safe_inv) / 12.0

        # 4. Price ratio to baseline (24..32)
        ratio = price / max(1.0, base_p)
        ratio_clipped = np.clip(ratio, 0.0, 5.0)
        vec[24 + i] = ratio_clipped
        price_ratios.append(ratio_clipped)

    # 5. Summary metrics (32..35)
    vec[32] = float(np.mean(inv_deviations)) if inv_deviations else 0.0
    vec[33] = float(np.mean(price_ratios)) if price_ratios else 1.0
    vec[34] = np.clip(total_inv / (BASELINE_INVENTORY * len(MARKET_COMMODITIES)), 0.0, 10.0)

    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
