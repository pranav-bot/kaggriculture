"""Continuous global feature vector extraction for Kaggriculture."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

COMMODITIES = ["WHEAT", "CORN", "SOY", "MELON", "STRAWBERRY", "MILK", "WOOL", "EGG"]
QUADRANTS = ["NW", "NE", "SW", "SE"]


def extract_global_vector(obs: Mapping[str, Any], seat: int = 0) -> np.ndarray:
    """Extract a 30-dimensional continuous global feature vector.

    Includes normalized representations of:
    - Current liquid cash and opponent cash (log-scaled, bankruptcy safe)
    - Game time indicators (day, hour, step, overage)
    - Farmhand capacity and daily hiring
    - Unlocked quadrant status (player and opponent)
    - Private inventory balances in shed and carried
    """
    vec = np.zeros(30, dtype=np.float32)

    farms = obs.get("farms") or []
    farm = farms[seat] if 0 <= seat < len(farms) else {}
    opp_farm = farms[1 - seat] if 0 <= (1 - seat) < len(farms) else {}

    # 1. Cash (bankruptcy-safe, log1p scaled)
    money = float(farm.get("money", 0.0) or 0.0)
    opp_money = float(opp_farm.get("money", 0.0) or 0.0)
    safe_money = max(0.0, money)
    safe_opp_money = max(0.0, opp_money)
    vec[0] = np.log1p(safe_money) / 15.0
    vec[1] = np.log1p(safe_opp_money) / 15.0

    # 2. Time metrics
    day = float(obs.get("day", 0) or 0)
    hour = float(obs.get("hour", 0) or 0)
    step = float(obs.get("step", 0) or 0)
    overage = float(obs.get("remainingOverageTime", 60.0) or 0.0)
    vec[2] = day / 30.0
    vec[3] = hour / 24.0
    vec[4] = step / 720.0
    vec[5] = max(0.0, min(1.0, overage / 60.0))

    # 3. Labor capacity
    hands = farm.get("hands") or []
    opp_hands = opp_farm.get("hands") or []
    hires = float(farm.get("hires_today", 0) or 0)
    vec[6] = min(1.0, len(hands) / 10.0)
    vec[7] = min(1.0, len(opp_hands) / 10.0)
    vec[8] = min(1.0, hires / 5.0)

    # 4. Quadrants (NW, NE, SW, SE)
    quads = set(farm.get("unlocked_quadrants") or [])
    opp_quads = set(opp_farm.get("unlocked_quadrants") or [])
    for i, q in enumerate(QUADRANTS):
        if q in quads:
            vec[9 + i] = 1.0
        if q in opp_quads:
            vec[13 + i] = 1.0

    # 5. Private inventory (shed goods)
    private = obs.get("private") or {}
    shed = private.get("shed") or {}
    for i, item in enumerate(COMMODITIES):
        qty = float(shed.get(item, 0.0) or 0.0)
        vec[17 + i] = min(2.0, qty / 50.0)

    # 6. Carried inventory & seed stocks
    carried_total = 0.0
    for inv in private.get("inventories") or []:
        if isinstance(inv, Mapping):
            carried_total += sum(float(v or 0.0) for v in inv.values())
    vec[25] = min(2.0, carried_total / 50.0)

    seeds = private.get("seeds") or {}
    seeds_total = sum(float(v or 0.0) for v in seeds.values()) if isinstance(seeds, Mapping) else 0.0
    vec[26] = min(2.0, seeds_total / 50.0)

    # 7. Relative cash delta
    vec[27] = np.clip((safe_money - safe_opp_money) / 50000.0, -1.0, 1.0)

    # 8. Unlocked shops count in town
    town = obs.get("town") or {}
    shops = town.get("unlocked_shops") or []
    vec[28] = min(1.0, len(shops) / 10.0)

    # 9. Day normalized cycle (sine wave for hour)
    vec[29] = np.sin(2.0 * np.pi * hour / 24.0)

    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
