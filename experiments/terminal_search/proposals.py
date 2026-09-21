"""Harvest → return → DROP route proposals for terminal week."""

from __future__ import annotations

from typing import Any

from .routes import return_to_shed_drop, walk as _walk
from .unit_step import harvestable_units


def _flatten_unit_chain(parts: list[Any]) -> list[Any]:
    flat: list[Any] = []
    for step in parts:
        if isinstance(step, list) and len(step) == 1 and isinstance(step[0], str):
            flat.append(step[0])
        elif isinstance(step, str):
            flat.append(step)
        elif isinstance(step, list):
            flat.extend(step)
    return flat


def _board_size(farm: dict[str, Any]) -> int:
    tiles = farm.get("tiles") or []
    return len(tiles) if tiles else 10


def harvest_targets(farm: dict[str, Any], day: int) -> list[tuple[int, int, str, int]]:
    targets: list[tuple[int, int, str, int]] = []
    tiles = farm.get("tiles") or []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row or []):
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            qty = harvestable_units(tile, day)
            if qty <= 0:
                continue
            targets.append((x, y, str(tile.get("crop", "WHEAT")), qty))
    targets.sort(key=lambda t: (-t[3], t[0], t[1]))
    return targets


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def propose_harvest_drop_routes(
    obs: dict[str, Any],
    *,
    proposals_per_actor: int = 4,
    max_unit_steps: int = 12,
) -> list[dict[str, Any]]:
    """
    Build candidate farmer action chains: walk → HARVEST → walk → DROP.
    Returns partial action dicts with ``farmer`` key set.
    """
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or [{}]
    farm = farms[player] if len(farms) > player else farms[0]
    day = int(obs.get("day", 27) or 27)
    board = _board_size(farm)
    farmer = tuple(farm.get("farmer", [0, 0]))
    targets_by_dist = sorted(
        harvest_targets(farm, day),
        key=lambda t: _manhattan(farmer, (t[0], t[1])),
    )
    proposals: list[dict[str, Any]] = []
    for x, y, _crop, _qty in targets_by_dist:
        route = _walk(farmer, (x, y)) + [["HARVEST"]] + return_to_shed_drop((x, y), board)
        chain = _flatten_unit_chain(route)[:max_unit_steps]
        if not chain:
            continue
        proposals.append({"farmer": chain, "hands": [], "market": []})
        if len(proposals) >= proposals_per_actor:
            break
    return proposals
