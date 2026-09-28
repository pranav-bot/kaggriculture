"""Spatial grid tensor encoding for Kaggriculture 10x10 farm observations."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

CROP_TYPES = ["WHEAT", "CORN", "SOY", "MELON", "STRAWBERRY"]
ANIMAL_TYPES = ["COW", "SHEEP", "CHICKEN"]


def _extract_farm_grid(farm: Mapping[str, Any], is_opponent: bool = False) -> np.ndarray:
    """Extract spatial tensor for a farm.

    Channels for agent (23 channels):
    0: LOCKED
    1: EMPTY
    2: WEED
    3: PASTURE
    4: PLANT
    5-9: Crop species one-hot (WHEAT, CORN, SOY, MELON, STRAWBERRY)
    10: Crop maturity / lifespan ratio [0, 1]
    11: Crop watered_today [0, 1]
    12: Crop consecutive_unwatered [0, 1]
    13: Crop fertilized_until_day [0, 1]
    14-16: Animal species one-hot (COW, SHEEP, CHICKEN)
    17: Animal fed_today [0, 1]
    18: Animal cared_today [0, 1]
    19: Animal consecutive_unfed [0, 1]
    20: Animal fertilizer_available [0, 1]
    21: Farmer position [0, 1]
    22: Farmhands positions [0, 1]

    Channels for opponent (20 channels):
    0: LOCKED
    1: EMPTY
    2: WEED
    3: PASTURE
    4: PLANT
    5-9: Crop species one-hot (5 channels)
    10: Crop maturity [0, 1]
    11: Crop watered_today [0, 1]
    12: Crop fertilized [0, 1]
    13-15: Animal species one-hot (3 channels)
    16: Animal cared [0, 1]
    17: Animal fed [0, 1]
    18: Opponent farmer position [0, 1]
    19: Opponent hands positions [0, 1]
    """
    num_channels = 20 if is_opponent else 23
    tensor = np.zeros((num_channels, 10, 10), dtype=np.float32)

    tiles = farm.get("tiles") or []
    for r in range(min(10, len(tiles))):
        row = tiles[r]
        if not isinstance(row, list):
            continue
        for c in range(min(10, len(row))):
            cell = row[c]
            if cell is None:
                tensor[1, r, c] = 1.0
            elif cell == "LOCKED":
                tensor[0, r, c] = 1.0
            elif isinstance(cell, Mapping):
                kind = cell.get("kind")
                if kind == "WEED":
                    tensor[2, r, c] = 1.0
                elif kind == "PASTURE":
                    tensor[3, r, c] = 1.0
                    animal = cell.get("animal")
                    if animal in ANIMAL_TYPES:
                        a_idx = ANIMAL_TYPES.index(animal)
                        if is_opponent:
                            tensor[13 + a_idx, r, c] = 1.0
                            if cell.get("cared_today"):
                                tensor[16, r, c] = 1.0
                            if cell.get("fed_today"):
                                tensor[17, r, c] = 1.0
                        else:
                            tensor[14 + a_idx, r, c] = 1.0
                            if cell.get("fed_today"):
                                tensor[17, r, c] = 1.0
                            if cell.get("cared_today"):
                                tensor[18, r, c] = 1.0
                            unfed = float(cell.get("consecutive_unfed", 0) or 0)
                            tensor[19, r, c] = min(1.0, unfed / 3.0)
                            if cell.get("fertilizer_available"):
                                tensor[20, r, c] = 1.0
                elif kind == "PLANT":
                    tensor[4, r, c] = 1.0
                    crop = cell.get("crop")
                    if crop in CROP_TYPES:
                        c_idx = CROP_TYPES.index(crop)
                        tensor[5 + c_idx, r, c] = 1.0

                    # Maturity / yield
                    yield_u = float(cell.get("yield_units", 0) or 0)
                    tensor[10, r, c] = min(1.0, yield_u / 5.0)

                    if cell.get("watered_today"):
                        tensor[11, r, c] = 1.0

                    if is_opponent:
                        fert_day = cell.get("fertilized_until_day", -1)
                        if fert_day is not None and fert_day >= 0:
                            tensor[12, r, c] = 1.0
                    else:
                        unwatered = float(cell.get("consecutive_unwatered", 0) or 0)
                        tensor[12, r, c] = min(1.0, unwatered / 3.0)
                        fert_day = cell.get("fertilized_until_day", -1)
                        if fert_day is not None and fert_day >= 0:
                            tensor[13, r, c] = 1.0

    # Farmer position
    farmer_pos = farm.get("farmer")
    farmer_chan = 18 if is_opponent else 21
    if isinstance(farmer_pos, (list, tuple)) and len(farmer_pos) >= 2:
        fr, fc = int(farmer_pos[0]), int(farmer_pos[1])
        if 0 <= fr < 10 and 0 <= fc < 10:
            tensor[farmer_chan, fr, fc] = 1.0

    # Farmhands positions
    hands = farm.get("hands") or []
    hands_chan = 19 if is_opponent else 22
    for hand_pos in hands:
        if isinstance(hand_pos, (list, tuple)) and len(hand_pos) >= 2:
            hr, hc = int(hand_pos[0]), int(hand_pos[1])
            if 0 <= hr < 10 and 0 <= hc < 10:
                tensor[hands_chan, hr, hc] = 1.0

    return np.clip(tensor, 0.0, 1.0)


def extract_spatial_tensor(obs: Mapping[str, Any], seat: int = 0) -> np.ndarray:
    """Extract spatial tensor of shape (23, 10, 10) for the specified seat."""
    farms = obs.get("farms") or []
    farm = farms[seat] if 0 <= seat < len(farms) else {}
    return _extract_farm_grid(farm, is_opponent=False)


def extract_opponent_spatial(obs: Mapping[str, Any], seat: int = 0) -> np.ndarray:
    """Extract opponent spatial tensor of shape (20, 10, 10)."""
    opp_seat = 1 - seat
    farms = obs.get("farms") or []
    farm = farms[opp_seat] if 0 <= opp_seat < len(farms) else {}
    return _extract_farm_grid(farm, is_opponent=True)
