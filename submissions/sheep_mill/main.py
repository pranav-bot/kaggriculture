"""Sheep Mill candidate: melon bootstrap followed by a feedable wool herd."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import Actions, ANIMALS, CROPS, MARKET_PARAMS


MAX_HANDS = 8
LAST_BUILD_DAY = 24
LIQUIDITY_BUFFER = 250.0


def _farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    return obs["farms"][int(obs.get("player", 0))]


def _dist(src: Sequence[int], target: Tuple[int, int]) -> int:
    return abs(int(src[0]) - target[0]) + abs(int(src[1]) - target[1])


def _move(src: Sequence[int], target: Tuple[int, int]) -> List[str]:
    dx, dy = target[0] - int(src[0]), target[1] - int(src[1])
    if abs(dx) >= abs(dy):
        return Actions.move("EAST" if dx > 0 else "WEST")
    return Actions.move("SOUTH" if dy > 0 else "NORTH")


def _tasks(obs: Mapping[str, Any]) -> List[Tuple[int, Tuple[int, int], List[Any]]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    shed = private.get("shed", {})
    tasks: List[Tuple[int, Tuple[int, int], List[Any]]] = []
    empty_pastures = 0
    empty_tiles: List[Tuple[int, int]] = []

    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            pos = (x, y)
            if tile == "LOCKED":
                continue
            if tile is None:
                empty_tiles.append(pos)
                continue
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = CROPS[tile["crop"]]
                if tile.get("yield_units", 0) > 0 and day - tile.get("planted_day", 0) >= crop.first_yield_day:
                    tasks.append((0, pos, Actions.harvest()))
                elif not tile.get("watered_today", False):
                    tasks.append((1, pos, Actions.water()))
            elif tile.get("kind") == "WEED":
                tasks.append((2, pos, Actions.dig()))
            elif tile.get("kind") == "PASTURE":
                if "animal" not in tile:
                    empty_pastures += 1
                elif tile.get("yield_units", 0) > 0:
                    tasks.append((2, pos, Actions.harvest()))
                if "animal" in tile and not tile.get("fed_today", False):
                    tasks.append((3, pos, Actions.feed()))
                if "animal" in tile and not tile.get("cared_today", False):
                    tasks.append((4, pos, Actions.care()))

    inventories = private.get("inventories", [])
    sheep_in_shed = int(shed.get("SHEEP", 0))
    wheat_in_shed = int(shed.get("WHEAT", 0))
    for index, inv in enumerate(inventories):
        if index == 0:
            continue
        hand_positions = farm.get("hands", [])
        pickup_pos = tuple(hand_positions[index - 1]) if index - 1 < len(hand_positions) else (5, 4)
        if sheep_in_shed > 0 and not inv.get("SHEEP", 0):
            # Hands begin each day on shed-access tiles, so pickup is cheap.
            tasks.append((-1, pickup_pos, Actions.pickup("SHEEP", 1)))
            sheep_in_shed -= 1
        elif wheat_in_shed > 0 and not inv.get("WHEAT", 0):
            tasks.append((-1, pickup_pos, Actions.pickup("WHEAT", 12)))
            wheat_in_shed -= 12

    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile:
                if int(shed.get("SHEEP", 0)) > 0:
                    tasks.append((1, (x, y), Actions.place("SHEEP", 1)))

    if 10 <= day <= LAST_BUILD_DAY:
        for pos in empty_tiles:
            tasks.append((5, pos, Actions.build_pasture()))
            if len(tasks) > 180:
                break

    if day <= 9:
        seeds = int(private.get("seeds", {}).get("MELON", 0))
        for pos in empty_tiles:
            if seeds <= 0:
                break
            tasks.append((6, pos, Actions.plant("MELON")))
            seeds -= 1
    return tasks


def _unit_actions(obs: Mapping[str, Any]) -> Tuple[List[Any], List[List[Any]]]:
    farm = _farm(obs)
    positions = [farm.get("farmer", [4, 4]), *farm.get("hands", [])]
    inventories = obs.get("private", {}).get("inventories", [])
    tasks = _tasks(obs)
    used: set[Tuple[int, int]] = set()
    result: List[List[Any]] = []
    for index, position in enumerate(positions):
        inv = inventories[index] if index < len(inventories) else {}
        candidates = []
        for task in tasks:
            priority, target, action = task
            if target in used:
                continue
            if action[0] == "PICKUP" and index == 0:
                continue
            if action[0] == "PLACE" and inv.get("SHEEP", 0) <= 0:
                continue
            if action[0] == "FEED" and inv.get("WHEAT", 0) <= 0:
                continue
            if action[0] == "PLANT" and index > 0 and not inv.get("_planting", True):
                continue
            candidates.append((priority, _dist(position, target), target, action))
        if not candidates:
            result.append(Actions.pass_action())
            continue
        _, _, target, action = min(candidates, key=lambda item: (item[0], item[1]))
        used.add(target)
        result.append(action if tuple(position) == target else _move(position, target))
    return result[0], result[1:]


def _market_orders(obs: Mapping[str, Any]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    money = float(farm.get("money", 0))
    shed = private.get("shed", {})
    prices = obs.get("market", {}).get("prices", {})
    orders: List[List[Any]] = []

    for item, count in shed.items():
        if count > 0 and item not in ANIMALS:
            orders.append(Actions.sell(item, count))
            money += count * float(prices.get(item, MARKET_PARAMS.get(item, {}).get("base", 0)))

    if day <= LAST_BUILD_DAY:
        unlocked = farm.get("unlocked_quadrants", ["NW"])
        land = Actions.land_cost(unlocked)
        if day >= 10 and land is not None and money >= land + LIQUIDITY_BUFFER:
            orders.append(Actions.buy_land())
            money -= land

        if day == 0 and private.get("seeds", {}).get("MELON", 0) == 0:
            quantity = min(20, int(money // CROPS["MELON"].seed_cost))
            if quantity:
                orders.append(Actions.buy_seed("MELON", quantity))
                money -= quantity * CROPS["MELON"].seed_cost

        empty_pastures = sum(
            isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile
            for row in farm.get("tiles", [])
            for tile in row
        )
        sheep_owned = int(shed.get("SHEEP", 0))
        sheep_to_buy = min(
            empty_pastures - sheep_owned,
            int(max(0, money - LIQUIDITY_BUFFER) // ANIMALS["SHEEP"].cost),
        ) if day >= 14 else 0
        if sheep_to_buy > 0:
            orders.append(Actions.buy_animal("SHEEP", sheep_to_buy))
            money -= sheep_to_buy * ANIMALS["SHEEP"].cost

        placed_sheep = sum(
            isinstance(tile, dict) and tile.get("animal") == "SHEEP"
            for row in farm.get("tiles", [])
            for tile in row
        )
        feed_qty = placed_sheep + int(shed.get("SHEEP", 0))
        if feed_qty and money >= feed_qty * float(prices.get("WHEAT", 25)):
            orders.append(Actions.buy_product("WHEAT", feed_qty))
            money -= feed_qty * float(prices.get("WHEAT", 25))

        hires_today = int(farm.get("hires_today", 0))
        for n in range(hires_today, MAX_HANDS):
            if len(orders) >= Actions.MAX_MARKET_ORDERS_PER_TURN:
                break
            cost = Actions.hire_cost(n)
            if money < cost:
                break
            orders.append(Actions.hire())
            money -= cost
    return orders[:Actions.MAX_MARKET_ORDERS_PER_TURN]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    farmer, hands = _unit_actions(obs)
    return {"farmer": farmer, "hands": hands, "market": _market_orders(obs)}
