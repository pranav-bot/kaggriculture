"""Observation and tile schema validation for Kaggle Kaggriculture 1.32.7."""

from __future__ import annotations

from typing import Any, Mapping

REQUIRED_OBS_KEYS = {
    "day",
    "hour",
    "step",
    "player",
    "remainingOverageTime",
    "farms",
    "market",
    "town",
    "private",
}

REQUIRED_FARM_KEYS = {
    "farmer",
    "hands",
    "hires_today",
    "money",
    "tiles",
    "unlocked_quadrants",
}

VALID_TILE_KINDS = {"WEED", "PASTURE", "PLANT"}


def validate_tile(tile: Any) -> bool:
    """Validate a single tile on the 10x10 farm grid.

    A tile may be:
    - None: Unlocked empty land.
    - "LOCKED": Locked quadrant tile string.
    - dict: A structured tile object with kind in {"WEED", "PASTURE", "PLANT"}.
    """
    if tile is None:
        return True
    if isinstance(tile, str):
        return tile == "LOCKED"
    if isinstance(tile, dict):
        kind = tile.get("kind")
        if kind not in VALID_TILE_KINDS:
            return False
        if kind == "PLANT":
            return "crop" in tile or "yield_units" in tile
        elif kind == "PASTURE":
            # Pasture can be empty or have an animal
            return True
        elif kind == "WEED":
            return True
    return False


def validate_observation(obs: Any) -> bool:
    """Validate observation dict conforms to Kaggle environment 1.32.7 schema."""
    if not isinstance(obs, Mapping):
        return False

    missing = REQUIRED_OBS_KEYS - set(obs.keys())
    if missing:
        return False

    farms = obs.get("farms")
    if not isinstance(farms, list) or len(farms) != 2:
        return False

    for farm in farms:
        if not isinstance(farm, Mapping):
            return False
        if not REQUIRED_FARM_KEYS.issubset(farm.keys()):
            return False
        tiles = farm.get("tiles")
        if not isinstance(tiles, list) or len(tiles) != 10:
            return False
        for row in tiles:
            if not isinstance(row, list) or len(row) != 10:
                return False
            for cell in row:
                if not validate_tile(cell):
                    return False

    market = obs.get("market")
    if not isinstance(market, Mapping):
        return False

    town = obs.get("town")
    if not isinstance(town, Mapping):
        return False

    private = obs.get("private")
    if not isinstance(private, Mapping):
        return False

    return True
