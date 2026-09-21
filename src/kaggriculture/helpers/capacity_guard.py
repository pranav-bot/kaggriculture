"""Shed capacity projection, sell clamping, and day-close room guard (P0)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kaggriculture.actions.actions import ANIMALS, Actions
from kaggriculture.env.items import (
    MAX_MARKET_ORDERS_PER_TURN,
    PRODUCTS_LIST,
    SHED_CAPACITY,
    TURNS_PER_DAY,
)

_ANIMAL_NAMES = frozenset(ANIMALS.keys())
_TARGET_OCCUPANCY = SHED_CAPACITY - 1  # 99


def planned_drop_inventory(
    farm: Mapping[str, Any],
    private: Mapping[str, Any],
    unit_actions: Sequence[Sequence[Any]],
    board_size: int,
) -> dict[str, int]:
    """Inventory expected to enter the shed from same-turn DROP beside shed tiles."""
    planned: dict[str, int] = {}
    positions = [farm.get("farmer", [0, 0]), *(farm.get("hands") or [])]
    inventories = private.get("inventories") or []
    for idx, action in enumerate(unit_actions):
        if idx >= len(positions) or not action or action[0] != "DROP":
            continue
        pos = (int(positions[idx][0]), int(positions[idx][1]))
        if not Actions.is_shed_adjacent(pos, board_size):
            continue
        inv = inventories[idx] if idx < len(inventories) else {}
        for item, amount in inv.items():
            planned[str(item)] = planned.get(str(item), 0) + int(amount)
    return planned


def projected_shed_from_action(
    shed: Mapping[str, Any],
    unit_actions: Sequence[Sequence[Any]],
    positions: Sequence[Sequence[int]],
    inventories: Sequence[Mapping[str, Any]],
    board_size: int = Actions.BOARD_SIZE,
) -> dict[str, int]:
    """Project shed stock after same-turn shed-adjacent DROP, PLACE, and PICKUP."""
    stock = {str(item): max(0, int(quantity)) for item, quantity in shed.items()}
    total = sum(stock.values())
    room = max(0, SHED_CAPACITY - total)

    for worker in range(min(len(unit_actions), len(positions))):
        pos = (int(positions[worker][0]), int(positions[worker][1]))
        if not Actions.is_shed_adjacent(pos, board_size):
            continue
        work = unit_actions[worker] or ["PASS"]
        operation = work[0] if work else "PASS"
        inventory = inventories[worker] if worker < len(inventories) else {}

        if operation == "PICKUP" and len(work) >= 2:
            item = str(work[1])
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            taken = min(stock.get(item, 0), quantity)
            if taken > 0:
                stock[item] = stock.get(item, 0) - taken
                total -= taken
        elif operation == "DROP":
            for item, held in inventory.items():
                added = min(max(0, int(held)), room)
                if added > 0:
                    key = str(item)
                    stock[key] = stock.get(key, 0) + added
                    total += added
                    room -= added
        elif operation == "PLACE" and len(work) >= 2 and str(work[1]) not in _ANIMAL_NAMES:
            item = str(work[1])
            quantity = max(0, int(work[2]) if len(work) >= 3 else 1)
            added = min(quantity, max(0, int(inventory.get(item, 0))), room)
            if added > 0:
                stock[item] = stock.get(item, 0) + added
                total += added
                room -= added
    return stock


def clamp_sells(projected_shed: Mapping[str, int], orders: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """Drop or shrink SELL orders that exceed projected on-hand quantity."""
    avail = {str(item): max(0, int(quantity)) for item, quantity in projected_shed.items()}
    kept: list[list[Any]] = []
    for order in orders:
        if not order:
            continue
        if str(order[0]).upper() == "SELL" and len(order) >= 3:
            item = str(order[1])
            have = avail.get(item, 0)
            quantity = min(max(0, int(order[2])), have)
            if quantity <= 0:
                continue
            avail[item] = have - quantity
            kept.append(["SELL", item, quantity])
        else:
            kept.append(list(order))
    return kept


def room_guard_99(
    obs: Mapping[str, Any],
    market_orders: Sequence[Sequence[Any]],
    unit_actions: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    """At hour 23, add or boost SELL orders so shed occupancy stays at or below 99."""
    step = int(obs.get("step", int(obs.get("day", 0)) * TURNS_PER_DAY + int(obs.get("hour", 0))))
    if step % TURNS_PER_DAY != TURNS_PER_DAY - 1:
        return [list(order) for order in market_orders]

    player = int(obs.get("player", 0))
    farms = obs.get("farms") or []
    farm = farms[player] if player < len(farms) else {}
    private = obs.get("private") or {}
    market = obs.get("market") or {}
    prices = market.get("prices") or {}
    shed = private.get("shed") or {}
    invs = private.get("inventories") or []
    tiles = farm.get("tiles") or []
    board = len(tiles) or Actions.BOARD_SIZE
    positions = [farm.get("farmer", [0, 0]), *(farm.get("hands") or [])]

    carried = sum(max(0, int(n)) for inv in invs for n in (inv or {}).values())
    produced = 0
    consumed = 0
    for worker in range(min(len(unit_actions), len(positions))):
        x, y = int(positions[worker][0]), int(positions[worker][1])
        if not (0 <= x < board and 0 <= y < board):
            continue
        tile = tiles[y][x]
        work = unit_actions[worker] or []
        if not work:
            continue
        op = work[0]
        if op == "HARVEST" and isinstance(tile, dict):
            produced += max(0, int(tile.get("yield_units", 0)))
        elif op == "COLLECT_FERTILIZER" and isinstance(tile, dict) and tile.get("fertilizer_available"):
            produced += 1
        elif op in ("FEED", "FERTILIZE"):
            consumed += 1
        elif op == "PLACE" and len(work) > 1 and str(work[1]) in _ANIMAL_NAMES:
            consumed += 1

    market = [list(order) for order in market_orders]
    planned_sells: dict[str, int] = {}
    planned_buys = 0
    for order in market:
        if not order:
            continue
        if order[0] == "SELL" and len(order) >= 3:
            item = str(order[1])
            planned_sells[item] = planned_sells.get(item, 0) + max(0, int(order[2]))
        elif order[0] in ("BUY_PRODUCT", "BUY_ANIMAL") and len(order) >= 3:
            planned_buys += max(0, int(order[2]))

    shed_total = sum(max(0, int(n)) for n in shed.values())
    actual_existing_sells = sum(
        min(max(0, int(shed.get(item, 0))), quantity) for item, quantity in planned_sells.items()
    )
    needed = shed_total + carried + produced - consumed + planned_buys - actual_existing_sells - _TARGET_OCCUPANCY
    if needed <= 0:
        return market[:MAX_MARKET_ORDERS_PER_TURN]

    priority = sorted(PRODUCTS_LIST, key=lambda item: (-int(prices.get(item, 0)), item))
    for item in priority:
        already = planned_sells.get(item, 0)
        available = max(0, int(shed.get(item, 0)) - already)
        quantity = min(needed, available)
        if quantity <= 0:
            continue
        slot = next(
            (index for index, order in enumerate(market) if order and order[0] == "SELL" and order[1] == item),
            -1,
        )
        if slot >= 0:
            market[slot][2] = max(0, int(market[slot][2])) + quantity
        elif len(market) < MAX_MARKET_ORDERS_PER_TURN:
            market.append(["SELL", item, quantity])
        else:
            continue
        planned_sells[item] = already + quantity
        needed -= quantity
        if needed <= 0:
            break

    return market[:MAX_MARKET_ORDERS_PER_TURN]
