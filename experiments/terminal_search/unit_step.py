"""Minimal per-turn unit command application for terminal search."""

from __future__ import annotations

from typing import Any

from kaggriculture.helpers.yield_check import check_max_yield_met

from .state import TerminalState
from .routes import shed_access_tiles as _shed_access_tiles


def _tile(state: TerminalState, x: int, y: int) -> Any:
    tiles = state.farm.get("tiles") or []
    if y < 0 or y >= len(tiles):
        return None
    row = tiles[y]
    if x < 0 or x >= len(row):
        return None
    return row[x]


def _set_tile(state: TerminalState, x: int, y: int, value: Any) -> None:
    state.farm["tiles"][y][x] = value


def _move(pos: tuple[int, int], direction: str) -> tuple[int, int]:
    x, y = pos
    d = direction.upper()
    if d == "EAST":
        return (x + 1, y)
    if d == "WEST":
        return (x - 1, y)
    if d == "SOUTH":
        return (y + 1, y) if False else (x, y + 1)
    if d == "NORTH":
        return (x, y - 1)
    return pos


def _is_shed_adjacent(pos: tuple[int, int], board_size: int = 10) -> bool:
    return pos in _shed_access_tiles(board_size)


def _plant_yield_state(tile: dict[str, Any], day: int) -> dict[str, Any]:
    planted = int(tile.get("planted_day", 0) or 0)
    age = max(0, day - planted)
    return {
        "plant_type": str(tile.get("crop", "WHEAT")),
        "age_in_days": age,
        "watered_bonus_days_count": int(tile.get("watered_bonus_days", tile.get("watered_days", 0)) or 0),
        "fertilized_bonus_days_count": int(tile.get("fertilized_bonus_days", 0) or 0),
        "harvests_collected": int(tile.get("harvests_collected", 0) or 0),
    }


def harvestable_units(tile: dict[str, Any], day: int) -> int:
    units = int(tile.get("yield_units", 0) or 0)
    if units > 0:
        return units
    if tile.get("kind") != "PLANT":
        return 0
    try:
        if check_max_yield_met(_plant_yield_state(tile, day)):
            crop = str(tile.get("crop", "WHEAT"))
            return int(tile.get("yield_units", 0) or 1)
    except (ValueError, KeyError):
        return 0
    return 0


def apply_unit_command(
    state: TerminalState,
    actor_index: int,
    command: list[Any],
    *,
    board_size: int = 10,
) -> None:
    if not command:
        return
    positions = state.actor_positions()
    if actor_index >= len(positions):
        return
    pos = positions[actor_index]
    invs = state.inventories()
    inv = invs[actor_index] if actor_index < len(invs) else {}
    cmd = str(command[0]).upper()

    if cmd in ("EAST", "WEST", "NORTH", "SOUTH"):
        new_pos = _move(pos, cmd)
        if actor_index == 0:
            state.farm["farmer"] = list(new_pos)
        else:
            hands = list(state.farm.get("hands") or [])
            while len(hands) <= actor_index - 1:
                hands.append(list(pos))
            hands[actor_index - 1] = list(new_pos)
            state.farm["hands"] = hands
        return

    if cmd == "HARVEST":
        x, y = pos
        tile = _tile(state, x, y)
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            return
        crop = str(tile.get("crop", "WHEAT"))
        qty = harvestable_units(tile, state.day)
        if qty <= 0:
            return
        inv[crop] = int(inv.get(crop, 0) or 0) + qty
        tile = dict(tile)
        tile["yield_units"] = max(0, int(tile.get("yield_units", 0) or 0) - qty)
        _set_tile(state, x, y, tile)
        state.private["inventories"] = invs
        return

    if cmd == "DROP" and _is_shed_adjacent(pos, board_size):
        shed = dict(state.private.get("shed") or {})
        for item, qty in list(inv.items()):
            if qty <= 0:
                continue
            shed[item] = int(shed.get(item, 0) or 0) + int(qty)
            inv[item] = 0
        state.private["shed"] = shed
        state.private["inventories"] = invs
        state.record_overflow()
        return

    if cmd == "PICKUP" and _is_shed_adjacent(pos, board_size):
        shed = dict(state.private.get("shed") or {})
        for item, qty in list(shed.items()):
            if qty <= 0:
                continue
            inv[item] = int(inv.get(item, 0) or 0) + int(qty)
            shed[item] = 0
        state.private["shed"] = shed
        state.private["inventories"] = invs


def _normalize_commands(chain: list[Any]) -> list[list[Any]]:
    if not chain:
        return []
    if isinstance(chain[0], list):
        return [cmd if isinstance(cmd, list) else [cmd] for cmd in chain]
    return [[cmd] for cmd in chain]


def apply_turn(state: TerminalState, action: dict[str, Any], *, board_size: int = 10) -> list[dict[str, int]]:
    """Apply farmer + hand unit chains; return per-actor deposit deltas this turn."""
    from .terminal_search import PRODUCTS

    n_actors = max(1, len(state.inventories()))
    deposited = [{item: 0 for item in PRODUCTS} for _ in range(n_actors)]

    farmer_chain = _normalize_commands(action.get("farmer") or [])
    for cmd in farmer_chain:
        pre = dict(state.private.get("shed") or {})
        apply_unit_command(state, 0, cmd, board_size=board_size)
        post = dict(state.private.get("shed") or {})
        for item in PRODUCTS:
            delta = int(post.get(item, 0) or 0) - int(pre.get(item, 0) or 0)
            if delta > 0:
                deposited[0][item] += delta

    hands = action.get("hands") or []
    for h_idx, hand_chain in enumerate(hands):
        actor = h_idx + 1
        for cmd in _normalize_commands(hand_chain or []):
            pre = dict(state.private.get("shed") or {})
            apply_unit_command(state, actor, cmd, board_size=board_size)
            post = dict(state.private.get("shed") or {})
            if actor < len(deposited):
                for item in PRODUCTS:
                    delta = int(post.get(item, 0) or 0) - int(pre.get(item, 0) or 0)
                    if delta > 0:
                        deposited[actor][item] += delta
    return deposited
