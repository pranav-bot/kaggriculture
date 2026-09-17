"""Melon Surge: explicit full-board throughput scheduler.

Unlike the generic controller, this agent treats each turn as a bounded matching
problem. Every available unit receives one unique tile task, while market orders
reinvest harvested cash into land, seeds, and the next daily labor burst.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from kaggriculture import Actions, CROPS


MAX_HANDS = 8
LIQUIDITY_BUFFER = 100.0
LAST_INVESTMENT_DAY = 25


def _distance(a: Sequence[int], b: Tuple[int, int]) -> int:
    return abs(int(a[0]) - b[0]) + abs(int(a[1]) - b[1])


def _farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    return obs["farms"][int(obs.get("player", 0))]


def _tasks(obs: Mapping[str, Any]) -> List[Tuple[int, Tuple[int, int], List[str]]]:
    farm = _farm(obs)
    day = int(obs.get("day", 0))
    tiles = farm.get("tiles", [])
    tasks: List[Tuple[int, Tuple[int, int], List[str]]] = []
    seeds = int(obs.get("private", {}).get("seeds", {}).get("MELON", 0))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            pos = (x, y)
            if tile == "LOCKED":
                continue
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                if tile.get("yield_units", 0) > 0 and day - tile.get("planted_day", 0) >= CROPS[tile["crop"]].first_yield_day:
                    tasks.append((0, pos, Actions.harvest()))
                elif not tile.get("watered_today", False):
                    tasks.append((1, pos, Actions.water()))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append((2, pos, Actions.dig()))
            elif tile is None and day <= LAST_INVESTMENT_DAY and seeds > 0:
                tasks.append((3, pos, Actions.plant("MELON")))
                seeds -= 1
    return tasks


def _unit_actions(obs: Mapping[str, Any]) -> Tuple[List[Any], List[List[Any]]]:
    farm = _farm(obs)
    positions = [farm.get("farmer", [4, 4]), *farm.get("hands", [])]
    tasks = _tasks(obs)
    assigned: set[Tuple[int, int]] = set()
    actions: List[List[Any]] = []
    for position in positions:
        candidates = [task for task in tasks if task[1] not in assigned]
        if not candidates:
            actions.append(Actions.pass_action())
            continue
        _, target, action = min(candidates, key=lambda task: (task[0], _distance(position, task[1])))
        assigned.add(target)
        if tuple(position) == target:
            actions.append(action)
        else:
            dx, dy = target[0] - int(position[0]), target[1] - int(position[1])
            actions.append(Actions.move(
                "EAST" if abs(dx) >= abs(dy) and dx > 0 else
                "WEST" if abs(dx) >= abs(dy) else
                "SOUTH" if dy > 0 else "NORTH"
            ))
    return actions[0], actions[1:]


def _market_orders(obs: Mapping[str, Any]) -> List[List[Any]]:
    day = int(obs.get("day", 0))
    step = int(obs.get("step", 0))
    farm = _farm(obs)
    private = obs.get("private", {})
    shed = private.get("shed", {})
    prices = obs.get("market", {}).get("prices", {})
    money = float(farm.get("money", 0))
    orders: List[List[Any]] = []

    # Selling first makes the same turn's reinvestment cash available.
    melon_count = int(shed.get("MELON", 0))
    if melon_count > 0:
        orders.append(Actions.sell("MELON", melon_count))
        money += melon_count * float(prices.get("MELON", 250))

    unlocked = farm.get("unlocked_quadrants", ["NW"])
    if day <= LAST_INVESTMENT_DAY:
        next_land = Actions.land_cost(unlocked)
        if next_land is not None and money >= next_land + LIQUIDITY_BUFFER:
            orders.append(Actions.buy_land())
            money -= next_land

        vacant = sum(
            tile is None
            for row in farm.get("tiles", [])
            for tile in row
        )
        seed_cost = CROPS["MELON"].seed_cost
        current_seeds = int(private.get("seeds", {}).get("MELON", 0))
        affordable = int(max(0.0, money - LIQUIDITY_BUFFER) // seed_cost)
        quantity = min(max(0, vacant - current_seeds), affordable)
        if quantity > 0:
            orders.append(Actions.buy_seed("MELON", quantity))
            money -= quantity * seed_cost

        hires_today = int(farm.get("hires_today", 0))
        for index in range(hires_today, MAX_HANDS):
            if len(orders) >= Actions.MAX_MARKET_ORDERS_PER_TURN:
                break
            cost = Actions.hire_cost(index)
            if money - cost < LIQUIDITY_BUFFER:
                break
            orders.append(Actions.hire())
            money -= cost
    return orders[:Actions.MAX_MARKET_ORDERS_PER_TURN]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    farmer, hands = _unit_actions(obs)
    return {"farmer": farmer, "hands": hands, "market": _market_orders(obs)}
