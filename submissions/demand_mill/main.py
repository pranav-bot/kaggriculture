"""Shop-matched livestock mill.

Produce wool, milk, eggs, and fertilizer, then sell only about as fast as the
town is draining those goods. Hands reset at the shed each dawn, so structures
are built in a compact block around the shed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import Actions, ANIMALS


SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
PREMIUM = ("WOOL", "MILK", "EGG", "STRAWBERRY", "MELON", "FERTILIZER")
ANIMAL_ITEMS = {"SHEEP", "COW", "GOOSE"}
BASE = {"WOOL": 200, "MILK": 160, "EGG": 50, "FERTILIZER": 100, "WHEAT": 25, "STRAWBERRY": 120, "MELON": 250}
MAX_HANDS = 8
Pos = Tuple[int, int]


def _farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    return obs["farms"][int(obs.get("player", 0))]


def _dist(src: Sequence[int], target: Pos) -> int:
    return abs(int(src[0]) - target[0]) + abs(int(src[1]) - target[1])


def _shed_tiles(board: int) -> set[Pos]:
    half = board // 2
    return {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}


def _move(src: Sequence[int], target: Pos) -> List[str]:
    dx, dy = target[0] - int(src[0]), target[1] - int(src[1])
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _demand_per_day(shops: Sequence[str]) -> Dict[str, float]:
    per_tick: Dict[str, float] = {}
    for shop in shops:
        products = SHOPS.get(str(shop), ())
        weight = 2.0 if len(products) == 1 else 1.0
        for product in products:
            per_tick[product] = per_tick.get(product, 0.0) + weight
    demand = {item: ticks * 6.0 for item, ticks in per_tick.items()}
    for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"):
        demand[item] = demand.get(item, 0.0) + 1.0
    return demand


def _herd_plan(demand: Mapping[str, float], day: int) -> Dict[str, int]:
    """How many of each animal to own. Labor is the binding constraint."""
    wool = demand.get("WOOL", 0.0)
    milk = demand.get("MILK", 0.0)
    egg = demand.get("EGG", 0.0)
    # One sheep yields ~1 wool / 3 days without perfect care, ~1/day with care.
    # Care-boosted animals produce about one unit per interval day. Match the
    # town drain and leave labor headroom; extra animals only crash the book.
    sheep = int(min(12, max(0, round(wool * 2.5))))
    cows = int(min(8, max(0, round(milk * 1.5))))
    geese = int(min(6, max(0, round(egg))))
    if day < 3:
        return {"SHEEP": 4, "COW": 0, "GOOSE": 0}
    if sheep + cows + geese > 12:
        scale = 12 / (sheep + cows + geese)
        sheep, cows, geese = int(sheep * scale), int(cows * scale), int(geese * scale)
    return {"SHEEP": sheep, "COW": cows, "GOOSE": geese}


def _structures(animal: str) -> str:
    return "PASTURE" if animal in ("SHEEP", "COW") else "COOP"


def _scan(obs: Mapping[str, Any]) -> Dict[str, Any]:
    farm = _farm(obs)
    tiles = farm.get("tiles", [])
    board = len(tiles) or 10
    shed = _shed_tiles(board)
    animals = []
    empty_structures: Dict[str, List[Pos]] = {"PASTURE": [], "COOP": []}
    empties: List[Pos] = []
    weeds: List[Pos] = []
    counts = {"SHEEP": 0, "COW": 0, "GOOSE": 0}
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            pos = (x, y)
            if tile == "LOCKED":
                continue
            if tile is None:
                if pos not in shed:
                    empties.append(pos)
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                weeds.append(pos)
            elif kind in ("PASTURE", "COOP"):
                if "animal" not in tile:
                    empty_structures[kind].append(pos)
                else:
                    animal = str(tile["animal"])
                    counts[animal] = counts.get(animal, 0) + 1
                    animals.append((pos, tile))
    empties.sort(key=lambda p: (_dist((board // 2 - 1, board // 2 - 1), p), p[1], p[0]))
    return {
        "board": board,
        "shed": shed,
        "animals": animals,
        "empty_structures": empty_structures,
        "empties": empties,
        "weeds": weeds,
        "counts": counts,
    }


def _tasks(obs: Mapping[str, Any], scan: Mapping[str, Any], plan: Mapping[str, int]) -> List[Tuple[int, Pos, List[Any]]]:
    day = int(obs.get("day", 0))
    private = obs.get("private", {})
    shed_stock = private.get("shed", {})
    tasks: List[Tuple[int, Pos, List[Any]]] = []
    for pos, tile in scan["animals"]:
        missed = int(tile.get("consecutive_unfed", 0))
        if not tile.get("fed_today", False):
            tasks.append((0 if missed else 1, pos, ["FEED"]))
        if int(tile.get("yield_units", 0)) > 0:
            tasks.append((2, pos, ["HARVEST"]))
        if not tile.get("cared_today", False):
            tasks.append((4, pos, ["CARE"]))
    for kind, positions in scan["empty_structures"].items():
        animal = "SHEEP" if kind == "PASTURE" else "GOOSE"
        if kind == "PASTURE" and int(shed_stock.get("COW", 0)) and not int(shed_stock.get("SHEEP", 0)):
            animal = "COW"
        if int(shed_stock.get(animal, 0)) or any(
            int(inv.get(animal, 0)) for inv in private.get("inventories", [])
        ):
            for pos in positions:
                tasks.append((2, pos, ["PLACE", animal, 1]))
    if day < 26:
        wanted = {"PASTURE": plan["SHEEP"] + plan["COW"], "COOP": plan["GOOSE"]}
        have = {
            "PASTURE": scan["counts"]["SHEEP"] + scan["counts"]["COW"] + len(scan["empty_structures"]["PASTURE"]),
            "COOP": scan["counts"]["GOOSE"] + len(scan["empty_structures"]["COOP"]),
        }
        build = "PASTURE" if have["PASTURE"] < wanted["PASTURE"] else "COOP" if have["COOP"] < wanted["COOP"] else None
        if build:
            for pos in scan["empties"][:12]:
                tasks.append((6, pos, ["BUILD_PASTURE"] if build == "PASTURE" else ["BUILD_COOP"]))
    for pos in scan["weeds"]:
        tasks.append((7, pos, ["DIG"]))
    return tasks


def _needs_wheat(scan: Mapping[str, Any]) -> bool:
    return any(not tile.get("fed_today", False) for _pos, tile in scan["animals"])


def _unit_actions(obs: Mapping[str, Any], scan: Mapping[str, Any], plan: Mapping[str, int]) -> Tuple[List[Any], List[List[Any]]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    positions = [farm.get("farmer", [4, 4]), *farm.get("hands", [])]
    inventories = private.get("inventories", [])
    shed_stock = dict(private.get("shed", {}))
    tasks = _tasks(obs, scan, plan)
    hungry = _needs_wheat(scan)
    unfed_count = sum(1 for _pos, tile in scan["animals"] if not tile.get("fed_today", False))
    wheat_holders = sum(1 for inv in inventories if int(inv.get("WHEAT", 0)) > 0)
    claimed: set[Pos] = set()
    actions: List[List[Any]] = []
    fetchers = 0
    nearest_shed = lambda pos: min(scan["shed"], key=lambda tile: (_dist(pos, tile), tile[1], tile[0]))
    for index, position in enumerate(positions):
        pos = (int(position[0]), int(position[1]))
        inv = inventories[index] if index < len(inventories) else {}
        carried_goods = sum(int(v) for k, v in inv.items() if k not in ("WHEAT", *ANIMAL_ITEMS))
        # Bank harvested goods before they block the worker.
        if carried_goods >= 4:
            target = nearest_shed(pos)
            actions.append(["DROP"] if pos == target else _move(pos, target))
            continue
        # Feeding is impossible without wheat in hand. Fetch it before any other job.
        need_fetcher = hungry and (wheat_holders + fetchers) < max(1, unfed_count)
        if need_fetcher and int(inv.get("WHEAT", 0)) <= 0 and int(shed_stock.get("WHEAT", 0)) > 0:
            fetchers += 1
            if pos in scan["shed"]:
                qty = min(6, int(shed_stock.get("WHEAT", 0)))
                shed_stock["WHEAT"] = int(shed_stock["WHEAT"]) - qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue
        if pos in scan["shed"]:
            animal = next((item for item in ("SHEEP", "COW", "GOOSE") if int(shed_stock.get(item, 0)) > 0 and int(inv.get(item, 0)) == 0), None)
            if animal:
                shed_stock[animal] = int(shed_stock.get(animal, 0)) - 1
                actions.append(["PICKUP", animal, 1])
                continue
            if int(inv.get("WHEAT", 0)) < 4 and int(shed_stock.get("WHEAT", 0)) > 0:
                qty = min(6, int(shed_stock.get("WHEAT", 0)))
                shed_stock["WHEAT"] = int(shed_stock["WHEAT"]) - qty
                actions.append(["PICKUP", "WHEAT", qty])
                continue
        best: Optional[Tuple[int, int, Pos, List[Any]]] = None
        for priority, target, action in tasks:
            if target in claimed:
                continue
            op = action[0]
            if op == "FEED" and int(inv.get("WHEAT", 0)) <= 0:
                continue
            if op == "PLACE" and int(inv.get(action[1], 0)) <= 0:
                continue
            if op in ("BUILD_PASTURE", "BUILD_COOP", "DIG") and hungry:
                continue
            score = (priority, _dist(pos, target))
            if best is None or score < (best[0], best[1]):
                best = (priority, _dist(pos, target), target, action)
        if best is None:
            actions.append(["PASS"])
            continue
        _, _, target, action = best
        claimed.add(target)
        actions.append(action if pos == target else _move(pos, target))
    return actions[0], actions[1:]


def _sell_qty(item: str, have: int, demand: float, price: float, shed_total: int, step: int) -> int:
    if have <= 0 or item in ANIMAL_ITEMS or item == "FERTILIZER":
        return 0
    if item == "WHEAT":
        return max(0, have - 30) if shed_total >= 90 else 0
    # Fertilizer has no town drain; selling it only burns the price.
    rate = demand if demand > 0 else 0.5
    interval = max(1, int(round(24.0 / rate)))
    qty = 1
    if shed_total >= 90:
        interval = min(interval, 2)
        qty = min(have, 3)
    elif price < BASE.get(item, 1) * 0.7:
        interval = max(interval, 12)
    if step % interval != 0:
        return 0
    return min(have, qty)


def _market(obs: Mapping[str, Any], scan: Mapping[str, Any], plan: Mapping[str, int], demand: Mapping[str, float]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    step = int(obs.get("step", 0))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    prices = obs.get("market", {}).get("prices", {})
    orders: List[List[Any]] = []
    shed_total = sum(int(v) for v in shed.values())

    ranked = sorted(
        (item for item, count in shed.items() if int(count) > 0 and item not in ANIMAL_ITEMS),
        key=lambda item: -float(prices.get(item, 0)),
    )
    for item in ranked:
        qty = _sell_qty(item, int(shed.get(item, 0)), float(demand.get(item, 0)), float(prices.get(item, 0)), shed_total, step)
        if qty <= 0 or len(orders) >= 10:
            continue
        orders.append(["SELL", item, qty])
        money += qty * float(prices.get(item, BASE.get(item, 1)))
        shed_total -= qty

    hires = int(farm.get("hires_today", 0))
    while hires < MAX_HANDS and len(orders) < 10:
        cost = Actions.hire_cost(hires)
        if money < cost + 100:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    animals_live = sum(scan["counts"].values()) + sum(int(shed.get(a, 0)) for a in ANIMAL_ITEMS)
    wheat_have = int(shed.get("WHEAT", 0)) + sum(int(inv.get("WHEAT", 0)) for inv in private.get("inventories", []))
    wheat_floor = max(animals_live + 4, 16 if day <= 6 else animals_live + 2)
    wheat_need = max(0, wheat_floor - wheat_have)
    wheat_price = float(prices.get("WHEAT", 25))
    if wheat_need and money >= wheat_price and len(orders) < 10:
        qty = min(wheat_need, int((money - 200) // max(1.0, wheat_price)))
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price

    # Animals only when a free structure exists and feed is already on hand.
    if wheat_have + wheat_need >= animals_live + 1:
        for animal, target in (("SHEEP", plan["SHEEP"]), ("COW", plan["COW"]), ("GOOSE", plan["GOOSE"])):
            if day > 22 or len(orders) >= 10:
                break
            owned = scan["counts"].get(animal, 0) + int(shed.get(animal, 0))
            free = len(scan["empty_structures"][_structures(animal)])
            buy = min(max(0, target - owned), free)
            cost = ANIMALS[animal].cost
            if buy > 0 and money >= cost + 200 and len(orders) < 10:
                take = min(buy, max(1, int((money - 200) // cost)))
                orders.append(["BUY_ANIMAL", animal, take])
                money -= take * cost

    fed_ready = bool(scan["animals"]) and all(tile.get("fed_today", False) for _pos, tile in scan["animals"])
    if day >= 4 and fed_ready and len(farm.get("unlocked_quadrants", ["NW"])) < 3 and len(orders) < 10:
        cost = Actions.land_cost(farm.get("unlocked_quadrants", ["NW"]))
        if cost is not None and money >= cost + 500:
            orders.append(["BUY_LAND"])
            money -= cost
    return orders[:10]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    shops = obs.get("town", {}).get("unlocked_shops", []) or []
    demand = _demand_per_day(shops)
    plan = _herd_plan(demand, int(obs.get("day", 0)))
    scan = _scan(obs)
    farmer, hands = _unit_actions(obs, scan, plan)
    return {"farmer": farmer, "hands": hands, "market": _market(obs, scan, plan, demand)}
