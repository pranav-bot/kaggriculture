"""Supply the goods the open shops are draining, and sell at that drain rate.

The opening keeps its cash until the first shops appear. After that the herd
and any strawberry tiles follow milk, wool, egg, and strawberry demand.
Workers feed before they build, and only a few of them leave the shed to
fetch wheat so the rest can place animals and harvest.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import Actions


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
BASE = {
    "WOOL": 200, "MILK": 160, "EGG": 50, "STRAWBERRY": 120,
    "CARROT": 35, "WHEAT": 25, "TOMATO": 60, "MELON": 250, "FERTILIZER": 100,
}
ANIMALS = {"SHEEP": 500, "COW": 400, "GOOSE": 300}
ANIMAL_OF = {"SHEEP": "PASTURE", "COW": "PASTURE", "GOOSE": "COOP"}
MAX_HANDS = 8
Pos = Tuple[int, int]


def _farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    return obs["farms"][int(obs.get("player", 0))]


def _dist(a: Sequence[int], b: Pos) -> int:
    return abs(int(a[0]) - b[0]) + abs(int(a[1]) - b[1])


def _sheds(board: int) -> List[Pos]:
    half = board // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _move(src: Sequence[int], target: Pos) -> List[str]:
    dx, dy = target[0] - int(src[0]), target[1] - int(src[1])
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _demand(shops: Sequence[str]) -> Dict[str, float]:
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


def _plan(demand: Mapping[str, float], day: int) -> Dict[str, int]:
    if day < 3:
        return {"SHEEP": 0, "COW": 0, "GOOSE": 0, "STRAWBERRY": 0, "CARROT": 6}
    # One line, so labor is spent harvesting the good whose price is rising.
    scores = {
        "COW": demand.get("MILK", 0) * 160,
        "SHEEP": demand.get("WOOL", 0) * 200,
        "GOOSE": demand.get("EGG", 0) * 50,
    }
    focus = max(scores, key=scores.get)
    sheep = cows = geese = 0
    cap = 4 if day < 6 else 12
    if focus == "COW":
        cows = cap
    elif focus == "SHEEP":
        sheep = cap
    else:
        geese = cap
    while sheep + cows + geese > cap:
        if cows >= sheep and cows >= geese and cows:
            cows -= 1
        elif sheep >= geese and sheep:
            sheep -= 1
        elif geese:
            geese -= 1
    # Crops compete with feeding. Add them only after the herd exists.
    berries = 6 if day >= 12 and demand.get("STRAWBERRY", 0) >= 8 else 0
    carrots = 0
    return {"SHEEP": sheep, "COW": cows, "GOOSE": geese, "STRAWBERRY": berries, "CARROT": carrots}


def _scan(obs: Mapping[str, Any]) -> Dict[str, Any]:
    farm = _farm(obs)
    tiles = farm.get("tiles", [])
    board = len(tiles) or 10
    shed = set(_sheds(board))
    animals = []
    empty = {"PASTURE": [], "COOP": []}
    empties: List[Pos] = []
    weeds: List[Pos] = []
    crops: Dict[str, int] = {}
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
            elif kind == "PLANT":
                crops[tile["crop"]] = crops.get(tile["crop"], 0) + 1
                animals.append(("plant", pos, tile))
            elif kind in empty:
                if "animal" not in tile:
                    empty[kind].append(pos)
                else:
                    name = str(tile["animal"])
                    counts[name] = counts.get(name, 0) + 1
                    animals.append(("animal", pos, tile))
    center = (board // 2 - 1, board // 2 - 1)
    empties.sort(key=lambda p: (_dist(center, p), p))
    return {"shed": shed, "rows": animals, "empty": empty, "empties": empties, "weeds": weeds, "counts": counts, "crops": crops}


def _tasks(scan: Mapping[str, Any], plan: Mapping[str, int], seeds: Mapping[str, int], day: int) -> List[Tuple[int, Pos, List[Any]]]:
    tasks: List[Tuple[int, Pos, List[Any]]] = []
    for kind, pos, tile in scan["rows"]:
        if kind == "animal":
            missed = int(tile.get("consecutive_unfed", 0))
            if not tile.get("fed_today", False):
                tasks.append((0 if missed else 1, pos, ["FEED"]))
            if int(tile.get("yield_units", 0)) > 0:
                tasks.append((2, pos, ["HARVEST"]))
            if not tile.get("cared_today", False):
                tasks.append((3, pos, ["CARE"]))
        else:
            if int(tile.get("yield_units", 0)) > 0 and (day >= 29 or int(tile.get("yield_units", 0)) >= 2):
                tasks.append((2, pos, ["HARVEST"]))
            elif not tile.get("watered_today", False):
                danger = int(tile.get("consecutive_unwatered", 0)) >= 1
                tasks.append((0 if danger else 4, pos, ["WATER"]))
    for structure, positions in scan["empty"].items():
        for pos in positions:
            animal = "GOOSE" if structure == "COOP" else "COW"
            tasks.append((2, pos, ["PLACE", animal, 1]))
            tasks.append((2, pos, ["PLACE", "SHEEP", 1]))
    if day < 26:
        need_pasture = max(plan["SHEEP"] + plan["COW"], 8 if day < 8 else 0)
        have_pasture = scan["counts"]["SHEEP"] + scan["counts"]["COW"] + len(scan["empty"]["PASTURE"])
        need_coop = plan["GOOSE"]
        have_coop = scan["counts"]["GOOSE"] + len(scan["empty"]["COOP"])
        build = None
        if have_pasture < need_pasture:
            build = "BUILD_PASTURE"
        elif have_coop < need_coop:
            build = "BUILD_COOP"
        if build:
            for pos in scan["empties"][:8]:
                tasks.append((6, pos, [build]))
        berries = int(seeds.get("STRAWBERRY", 0))
        carrots = int(seeds.get("CARROT", 0))
        for pos in scan["empties"]:
            if berries and scan["crops"].get("STRAWBERRY", 0) < plan["STRAWBERRY"]:
                tasks.append((5, pos, ["PLANT", "STRAWBERRY"]))
                berries -= 1
            elif carrots and scan["crops"].get("CARROT", 0) < plan["CARROT"]:
                tasks.append((5, pos, ["PLANT", "CARROT"]))
                carrots -= 1
    for pos in scan["weeds"][:6]:
        tasks.append((8, pos, ["DIG"]))
    return tasks


def _units(obs: Mapping[str, Any], scan: Mapping[str, Any], plan: Mapping[str, int]) -> Tuple[List[Any], List[List[Any]]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    positions = [farm.get("farmer", [4, 4]), *farm.get("hands", [])]
    inventories = list(private.get("inventories", []))
    shed_stock = dict(private.get("shed", {}))
    seeds = dict(private.get("seeds", {}))
    tasks = _tasks(scan, plan, seeds, int(obs.get("day", 0)))
    unfed = sum(1 for kind, _p, tile in scan["rows"] if kind == "animal" and not tile.get("fed_today", False))
    holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)
    claimed: set[Pos] = set()
    planted = {"STRAWBERRY": 0, "CARROT": 0}
    actions: List[List[Any]] = []
    fetchers = 0

    def nearest(pos: Pos) -> Pos:
        return min(scan["shed"], key=lambda tile: (_dist(pos, tile), tile))

    for index, position in enumerate(positions):
        pos = (int(position[0]), int(position[1]))
        inv = inventories[index] if index < len(inventories) else {}
        goods = sum(int(v) for k, v in inv.items() if k not in ("WHEAT", "SHEEP", "COW", "GOOSE"))
        if goods >= 4:
            target = nearest(pos)
            actions.append(["DROP"] if pos == target else _move(pos, target))
            continue
        if unfed and holders + fetchers < unfed and int(inv.get("WHEAT", 0)) <= 0 and int(shed_stock.get("WHEAT", 0)) > 0:
            fetchers += 1
            if pos in scan["shed"]:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest(pos)))
            continue
        if pos in scan["shed"]:
            carried_animal = next((name for name in ("COW", "SHEEP", "GOOSE") if int(inv.get(name, 0)) > 0), None)
            if carried_animal is None:
                for name in ("COW", "SHEEP", "GOOSE"):
                    if int(shed_stock.get(name, 0)) > 0:
                        shed_stock[name] -= 1
                        actions.append(["PICKUP", name, 1])
                        break
                else:
                    carried_animal = ""
                if actions and len(actions) == index + 1:
                    continue
            if int(inv.get("WHEAT", 0)) < 2 and int(shed_stock.get("WHEAT", 0)) > 0 and unfed:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
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
            if op == "PLANT":
                crop = action[1]
                if int(seeds.get(crop, 0)) - planted[crop] <= 0:
                    continue
            if op in ("BUILD_PASTURE", "BUILD_COOP", "DIG") and unfed:
                continue
            score = (priority, _dist(pos, target))
            if best is None or score < (best[0], best[1]):
                best = (priority, score[1], target, action)
        if best is None:
            actions.append(["PASS"])
            continue
        _, _, target, action = best
        claimed.add(target)
        if action[0] == "PLANT":
            planted[action[1]] += 1
        actions.append(action if pos == target else _move(pos, target))
    return actions[0], actions[1:]


def _sell_qty(item: str, have: int, demand: float, price: float, step: int, shed_total: int) -> int:
    if have <= 0 or item in ANIMALS or item == "FERTILIZER":
        return 0
    if item == "WHEAT":
        return max(0, have - 24) if shed_total >= 92 else 0
    base = BASE.get(item, 1)
    rate = max(0.5, demand)
    # Above base, sell into the spike a little faster. Below 70% of base, wait
    # unless the shed is about to overflow.
    if price >= base * 1.25:
        rate *= 1.5
    elif price < base * 0.7 and shed_total < 92:
        rate = min(rate, 0.5)
    interval = max(1, int(round(24.0 / rate)))
    if shed_total >= 92:
        interval = 1
    if step % interval != 0:
        return 0
    qty = 2 if shed_total >= 92 or price >= base * 1.5 else 1
    return min(have, qty)


def _market(obs: Mapping[str, Any], scan: Mapping[str, Any], plan: Mapping[str, int], demand: Mapping[str, float]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    step = int(obs.get("step", int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    prices = obs.get("market", {}).get("prices", {})
    seeds = private.get("seeds", {})
    orders: List[List[Any]] = []
    shed_total = sum(int(v) for v in shed.values())

    for item, count in sorted(shed.items(), key=lambda kv: -float(prices.get(kv[0], 0))):
        qty = _sell_qty(str(item), int(count), float(demand.get(str(item), 0)), float(prices.get(item, 0)), step, shed_total)
        if qty > 0 and len(orders) < 10:
            orders.append(["SELL", item, qty])
            money += qty * float(prices.get(item, BASE.get(str(item), 1)))

    hires = int(farm.get("hires_today", 0))
    while hires < MAX_HANDS and len(orders) < 10:
        cost = Actions.hire_cost(hires)
        if money < cost + 50:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    animals_live = sum(scan["counts"].values())
    wheat_have = int(shed.get("WHEAT", 0)) + sum(int(inv.get("WHEAT", 0)) for inv in private.get("inventories", []))
    wheat_floor = 40 if animals_live == 0 else animals_live * 6
    wheat_need = max(0, wheat_floor - wheat_have)
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))
    if wheat_need and len(orders) < 10 and money >= wheat_price:
        qty = min(wheat_need, int(money // wheat_price))
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price

    if day >= 3:
        for animal in ("COW", "SHEEP", "GOOSE"):
            if len(orders) >= 10:
                break
            target = plan[animal]
            owned = scan["counts"][animal] + int(shed.get(animal, 0))
            free = len(scan["empty"][ANIMAL_OF[animal]])
            buy = min(max(0, target - owned), free, 1)
            cost = ANIMALS[animal]
            fed_stock = wheat_have >= animals_live + 4 or any(
                order[0] == "BUY_PRODUCT" and order[1] == "WHEAT" for order in orders
            )
            if buy > 0 and fed_stock and money >= cost + 100:
                orders.append(["BUY_ANIMAL", animal, buy])
                money -= cost * buy
        for crop, packet in (("STRAWBERRY", 100), ("CARROT", 20)):
            if len(orders) >= 10 or day > 18:
                break
            growing = scan["crops"].get(crop, 0) + int(seeds.get(crop, 0))
            need = plan[crop] - growing
            if need <= 0:
                continue
            unit = packet
            qty = min(need, int((money - 200) // unit))
            if qty > 0:
                orders.append(["BUY_SEED", crop, qty])
                money -= qty * unit

    if day >= 6 and animals_live >= 4 and len(orders) < 10:
        unlocked = farm.get("unlocked_quadrants", ["NW"])
        cost = Actions.land_cost(unlocked)
        if cost is not None and money >= cost + 400 and len(unlocked) < 3:
            orders.append(["BUY_LAND"])
    return orders[:10]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    plan = _plan(demand, int(obs.get("day", 0)))
    scan = _scan(obs)
    farmer, hands = _units(obs, scan, plan)
    return {"farmer": farmer, "hands": hands, "market": _market(obs, scan, plan, demand)}
