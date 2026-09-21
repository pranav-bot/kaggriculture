"""Shed capacity projection, sell clamping, room guard, and dead-stock sells (Agent 3 / P0)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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

FutureSellsFn = Callable[[str, int], int]


def _resolve_future_sells(fn: FutureSellsFn | None, item: str, step: int) -> int:
    if fn is None:
        return 0
    return max(0, int(fn(item, step)))


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
    """Project shed stock after same-turn shed-adjacent DROP and PLACE (capacity-limited)."""
    proj = {str(item): max(0, int(quantity)) for item, quantity in shed.items()}
    room = SHED_CAPACITY - sum(proj.values())

    for worker in range(min(len(unit_actions), len(positions))):
        if room <= 0:
            break
        pos = (int(positions[worker][0]), int(positions[worker][1]))
        inv = inventories[worker] if worker < len(inventories) else {}
        if not inv or not Actions.is_shed_adjacent(pos, board_size):
            continue
        action = unit_actions[worker] or ["PASS"]
        if not action:
            continue
        if action[0] == "DROP":
            for item, amount in inv.items():
                take = min(max(0, int(amount)), room)
                if take > 0:
                    key = str(item)
                    proj[key] = proj.get(key, 0) + take
                    room -= take
        elif action[0] == "PLACE" and len(action) > 1 and str(action[1]) not in _ANIMAL_NAMES:
            item = str(action[1])
            take = min(
                max(0, int(action[2]) if len(action) > 2 else 1),
                max(0, int(inv.get(item, 0))),
                room,
            )
            if take > 0:
                proj[item] = proj.get(item, 0) + take
                room -= take
    return proj


# Improvement-plan name for the same projection used before clamp_sells.
projected_shed = projected_shed_from_action


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
            if have <= 0:
                continue
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
    *,
    future_sells_fn: FutureSellsFn | None = None,
) -> list[list[Any]]:
    """At hour 23, add or boost SELL orders so shed occupancy stays at or below 99."""
    step = int(obs.get("step", int(obs.get("day", 0)) * TURNS_PER_DAY + int(obs.get("hour", 0))))
    if step % TURNS_PER_DAY != TURNS_PER_DAY - 1:
        return [list(order) for order in market_orders]

    player = int(obs.get("player", 0))
    farms = obs.get("farms") or []
    farm = farms[player] if player < len(farms) else {}
    private = obs.get("private") or {}
    market_info = obs.get("market") or {}
    prices = market_info.get("prices") or {}
    shed = private.get("shed") or {}
    invs = private.get("inventories") or []
    tiles = farm.get("tiles") or []
    board = len(tiles) or Actions.BOARD_SIZE
    positions = [farm.get("farmer", [0, 0]), *(farm.get("hands") or [])]
    next_step = step + 1

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

    priority = sorted(
        PRODUCTS_LIST,
        key=lambda item: (
            _resolve_future_sells(future_sells_fn, item, next_step) > 0,
            -int(prices.get(item, 0)),
            item,
        ),
    )
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


def dead_stock_sells(
    projected_shed: Mapping[str, int],
    market_orders: Sequence[Sequence[Any]],
    obs: Mapping[str, Any],
    *,
    future_sells_fn: FutureSellsFn | None = None,
) -> list[list[Any]]:
    """Append SELL orders for surplus not already listed and not reserved by future route sells."""
    step = int(obs.get("step", int(obs.get("day", 0)) * TURNS_PER_DAY + int(obs.get("hour", 0))))
    day = int(obs.get("day", step // TURNS_PER_DAY))
    prices = (obs.get("market") or {}).get("prices") or {}
    next_step = step + 1

    planned: dict[str, int] = {}
    for order in market_orders:
        if order and order[0] == "SELL" and len(order) >= 3:
            item = str(order[1])
            planned[item] = planned.get(item, 0) + max(0, int(order[2]))

    extra: list[list[Any]] = []
    for item in PRODUCTS_LIST:
        have = max(0, int(projected_shed.get(item, 0))) - planned.get(item, 0)
        if have <= 0:
            continue
        if day >= 29:
            surplus = have
        else:
            surplus = have - _resolve_future_sells(future_sells_fn, item, next_step)
        if surplus <= 0:
            continue
        if int(prices.get(item, 0)) <= 1:
            continue
        extra.append(["SELL", item, surplus])

    extra.sort(key=lambda order: -int(prices.get(order[1], 0)) * int(order[2]))
    market = [list(order) for order in market_orders]
    return (market + extra)[:MAX_MARKET_ORDERS_PER_TURN]
