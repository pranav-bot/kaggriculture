"""care_mill labor. Once a day, a table picks the milk inventory floor.

Actions are hold (floor 40), the care_mill floor (20), and a tight floor
(8). Greedy play uses the learned advantages in q_values.json on top of a
prior that keeps 20 while the quote is above $210 and holds below that.
scripts/train_sell_q.py fills the table from paired episodes.
"""

from __future__ import annotations

import json
import os
import random
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
# Cared output per day, times a curve weight. Wool gluts are punished: the
# scarcity curve is only a log, and oversupply is a steep square. Milk's
# scarcity curve is a square root, so unmet milk drain stays expensive.
SPECIES = {
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5, "curve": 1.7},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 4 / 3, "curve": 0.85},
    "GOOSE": {"structure": "COOP", "product": "EGG", "cost": 300, "per_day": 2.0, "curve": 1.05},
}
MAX_HANDS = 8
WHEAT_PLOTS = 8
Pos = Tuple[int, int]

# Locked on the first decision day so a later shop does not abandon the herd.
_LOCK: Dict[str, Optional[str]] = {"animal": None}

# action 0 keeps 40, action 1 keeps 20, action 2 keeps 8.
# Q is the cash advantage, in thousands, versus greedy play on that seed.
MODE = "greedy"
EPSILON = 0.0
TRACE: List[Tuple[Tuple[int, int, int], int]] = []
Q: Dict[Tuple[Tuple[int, int, int], int], float] = {}
_DAY: Dict[str, int] = {"day": -1, "action": 1}
_TREND: Dict[str, Any] = {"dawn": None, "down": False}


def _load_q() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_values.json")
    if not os.path.isfile(path):
        return
    with open(path) as handle:
        rows = json.load(handle)
    for stage, band, trend, action, value in rows:
        Q[(int(stage), int(band), int(trend)), int(action)] = float(value)


def _note_milk(obs: Mapping[str, Any]) -> None:
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    if day == 0 and hour == 0:
        _TREND["dawn"] = None
        _TREND["down"] = False
    if hour != 0:
        return
    price = float(obs.get("market", {}).get("prices", {}).get("MILK", 160))
    prev = _TREND["dawn"]
    # Eight dollars is larger than the solo drift and smaller than a
    # contested dawn-to-dawn drop.
    if prev is not None and price < float(prev) - 8:
        _TREND["down"] = True
    elif prev is not None and price > float(prev) + 4:
        _TREND["down"] = False
    _TREND["dawn"] = price


def _state(day: int, price: float) -> Tuple[int, int, int]:
    if day < 14:
        stage = 0
    elif day < 27:
        stage = 1
    else:
        stage = 2
    if price < 160:
        band = 0
    elif price < 210:
        band = 1
    elif price < 250:
        band = 2
    else:
        band = 3
    trend = 1 if _TREND["down"] else 0
    return stage, band, trend


def _prior(state: Tuple[int, int, int], action: int) -> float:
    """Rich flat book keeps 20. A falling book sells down to 8. A thin book holds."""
    _stage, band, trend = state
    if band >= 2 and trend:
        return 1.0 if action == 2 else 0.0
    if band >= 2:
        return 1.0 if action == 1 else 0.0
    return 1.0 if action == 0 else 0.0


def _choose(day: int, price: float) -> int:
    if _DAY["day"] == day:
        return int(_DAY["action"])
    state = _state(day, price)
    best = max(range(3), key=lambda action: _prior(state, action) + Q.get((state, action), 0.0))
    if MODE == "train" and random.random() < EPSILON:
        action = random.randrange(3)
    else:
        action = best
    _DAY["day"] = day
    _DAY["action"] = action
    if MODE == "train":
        TRACE.append((state, action))
    return action


_load_q()


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


def _scores(demand: Mapping[str, float]) -> Dict[str, float]:
    """Shop drain only. The +1 town-center term is not a reason to start a herd."""
    scores: Dict[str, float] = {}
    for name, spec in SPECIES.items():
        shop = max(0.0, float(demand.get(spec["product"], 0.0)) - 1.0)
        scores[name] = shop * BASE[spec["product"]] * spec["per_day"] * spec["curve"]
    return scores


def _focus(demand: Mapping[str, float], day: int, owned: int) -> str:
    if int(day) == 0:
        _LOCK["animal"] = None
    scores = _scores(demand)
    best = max(scores, key=scores.get)
    # Pastures until two shops exist. Buying on the first shop locked sheep
    # into a farmers-market seed and floored wool at $1.
    # No animal shop yet: do not lock. Crop shops used to tie-break toward
    # sheep and then dump wool to $1.
    if day < 3 or scores[best] <= 0:
        return str(_LOCK["animal"] or "COW")
    # Eggs stay near base unless the deficit crosses a hinge far above what a
    # goose herd can build. Wait through day 15 in case milk or wool opens.
    if best == "GOOSE" and day < 15 and max(scores["COW"], scores["SHEEP"]) <= 0:
        return str(_LOCK["animal"] or "COW")
    locked = _LOCK["animal"]
    if locked in SPECIES and owned > 4:
        return str(locked)
    if locked in SPECIES and owned > 0 and scores[best] < scores[str(locked)] * 1.35:
        return str(locked)
    _LOCK["animal"] = best
    return best


def _wanted(animal: str, demand: Mapping[str, float]) -> int:
    """How many cared animals match current shop drain, before the tile cap."""
    spec = SPECIES[animal]
    shop = max(0.0, float(demand.get(spec["product"], 0.0)) - 1.0)
    if shop <= 0 or spec["per_day"] <= 0:
        return 0
    return max(4, int(round(shop / spec["per_day"])) + 2)


def _drain_cap(animal: str, demand: Mapping[str, float], quadrants: int = 1) -> int:
    """Opening quadrant holds 18. A second quadrant is only bought when drain wants more."""
    room = 18 if quadrants < 2 else 24
    return min(_wanted(animal, demand), room)


def _target(day: int, quadrants: int = 1) -> int:
    """The opening quadrant has room for 18 pastures. Land is a longer walk.

    Twelve cows on a milk-heavy board still left the price near $320, so the
    drain can absorb more. Grow past 12 once the first animals are fed.
    """
    del quadrants
    if day < 3:
        return 0
    if day < 5:
        return 4
    if day < 8:
        return 8
    if day < 11:
        return 14
    return 18


def _scan(obs: Mapping[str, Any]) -> Dict[str, Any]:
    farm = _farm(obs)
    tiles = farm.get("tiles", [])
    board = len(tiles) or 10
    shed = set(_sheds(board))
    rows: List[Tuple[str, Pos, Mapping[str, Any]]] = []
    empty = {"PASTURE": [], "COOP": []}
    empties: List[Pos] = []
    weeds: List[Pos] = []
    crops: Dict[str, int] = {}
    counts = {"SHEEP": 0, "COW": 0, "GOOSE": 0}
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            pos = (x, y)
            if tile == "LOCKED" or pos in shed:
                continue
            if tile is None:
                empties.append(pos)
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                weeds.append(pos)
            elif kind == "PLANT":
                crops[str(tile.get("crop"))] = crops.get(str(tile.get("crop")), 0) + 1
                rows.append(("plant", pos, tile))
            elif kind in empty:
                if "animal" not in tile:
                    empty[kind].append(pos)
                else:
                    name = str(tile["animal"])
                    counts[name] = counts.get(name, 0) + 1
                    rows.append(("animal", pos, tile))
    center = (board // 2 - 1, board // 2 - 1)
    empties.sort(key=lambda p: (_dist(center, p), p))
    return {
        "shed": shed, "rows": rows, "empty": empty, "empties": empties,
        "weeds": weeds, "counts": counts, "crops": crops,
    }


def _sell_qty(
    item: str, have: int, price: float, shed_total: int, fund: bool, day: int,
    save_land: bool, need_cash: bool, hour: int = 0, herd: int = 0,
    drain: float = 0.0, per_day: float = 1.5,
) -> int:
    """Milk path from the scipy grid. Other goods keep the hold rule."""
    if have <= 0 or item in SPECIES:
        return 0
    if item == "WHEAT":
        return max(0, have - 12) if shed_total >= 92 else 0
    if item == "FERTILIZER":
        if price < 60:
            return 0
        if fund or save_land:
            return min(have, 4 if save_land else 2)
        return min(have, 4) if shed_total >= 85 else 0
    base = BASE.get(item, 1)
    if price < base:
        return 0
    if need_cash:
        return min(have, 2)
    if shed_total >= 92:
        return min(have, 4)
    # Wool and eggs were not in the milk fit. Hold them the old way.
    if item != "MILK":
        if day < 28 and shed_total < 78 and have < 20:
            return 0
        pace = 4 if price < base * 1.15 else (8 if day >= 28 or shed_total >= 90 else 3)
        return min(have, pace)
    # Last two days: bank anything still above the base. Before that the
    # day's action is an inventory floor. Selling stops at the floor so a
    # single turn cannot walk the sqrt curve down.
    if day >= 28 or shed_total >= 92:
        return min(have, 4)
    action = _choose(day, price)
    floor = (40, 20, 8)[action]
    if have <= floor:
        return 0
    return min(have - floor, 4 if price >= 240 else 3)


def _market(obs: Mapping[str, Any], scan: Mapping[str, Any], animal: str, demand: Mapping[str, float]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    prices = obs.get("market", {}).get("prices", {})
    seeds = dict(private.get("seeds", {}))
    orders: List[List[Any]] = []
    spec = SPECIES[animal]
    live = sum(scan["counts"].values())
    carried = sum(int((inv or {}).get(animal, 0)) for inv in private.get("inventories", []))
    focus_owned = scan["counts"][animal] + int(shed.get(animal, 0)) + carried
    others = live - scan["counts"][animal]
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    target = min(
        _target(day, len(quadrants)),
        _drain_cap(animal, demand, len(quadrants)),
        max(0, (18 if len(quadrants) < 2 else 24) - others),
    )
    owned = focus_owned
    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    save_land = False
    fund = owned < target or money < 1500
    shed_total = sum(int(v) for v in shed.values())

    need_cash = False
    product = spec["product"]
    drain = max(0.0, float(demand.get(product, 0.0)) - 1.0)
    for item, count in sorted(shed.items(), key=lambda kv: -float(prices.get(kv[0], 0))):
        qty = _sell_qty(
            str(item), int(count), float(prices.get(item, 0)),
            shed_total, fund, day, save_land, need_cash,
            hour=hour, herd=int(scan["counts"].get(animal, 0)),
            drain=drain, per_day=float(spec["per_day"]),
        )
        if qty > 0 and len(orders) < 10:
            orders.append(["SELL", item, qty])
            money += qty * float(prices.get(item, BASE.get(str(item), 1)))

    wheat_floor = 16 if live == 0 else live * 3
    wheat_need = max(0, wheat_floor - wheat_have)
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))
    if wheat_need and len(orders) < 10 and money >= wheat_price:
        qty = min(wheat_need, int(money // wheat_price))
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price

    growing = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    if day == 0 and growing < WHEAT_PLOTS and len(orders) < 10:
        qty = min(WHEAT_PLOTS - int(seeds.get("WHEAT", 0)), int(money // 10))
        if qty > 0:
            orders.append(["BUY_SEED", "WHEAT", qty])
            money -= qty * 10

    if day >= 3 and _scores(demand).get(animal, 0) > 0 and owned < target and len(orders) < 10:
        free = len(scan["empty"][spec["structure"]])
        # Two days of wheat is enough: more is bought every turn, and a
        # four-day gate froze the herd once the floor was lowered.
        room = 2 if money >= spec["cost"] * 2 + 200 else 1
        buy = min(room, free, target - owned)
        if buy > 0 and wheat_have >= live + buy + 2 and money >= spec["cost"] * buy + 80:
            orders.append(["BUY_ANIMAL", animal, buy])
            money -= spec["cost"] * buy

    hires = int(farm.get("hires_today", 0))
    while hires < MAX_HANDS and len(orders) < 10:
        cost = Actions.hire_cost(hires)
        if money < cost + 40:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1
    return orders[:10]


def _units(
    obs: Mapping[str, Any], scan: Mapping[str, Any], animal: str, demand: Mapping[str, float],
) -> Tuple[List[Any], List[List[Any]]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    spec = SPECIES[animal]
    positions = [farm.get("farmer", [4, 4]), *farm.get("hands", [])]
    inventories = list(private.get("inventories", []))
    shed_stock = dict(private.get("shed", {}))
    seeds = dict(private.get("seeds", {}))
    claimed: set[Pos] = set()
    actions: List[List[Any]] = []
    fetchers = 0
    planted = 0
    money = float(farm.get("money", 0))
    carried_animals = sum(int((inv or {}).get(animal, 0)) for inv in inventories)
    owned = sum(scan["counts"].values()) + int(shed_stock.get(animal, 0)) + carried_animals
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    want_fert = owned < _target(day, len(quadrants)) or money < 1500

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]
    unfed_left = sum(1 for _p, tile in animals if not tile.get("fed_today", False))
    holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)

    def nearest_shed(pos: Pos) -> Pos:
        return min(scan["shed"], key=lambda tile: (_dist(pos, tile), tile))

    def take_nearest(pos: Pos, pool: List[Pos]) -> Optional[Pos]:
        open_tiles = [tile for tile in pool if tile not in claimed]
        if not open_tiles:
            return None
        choice = min(open_tiles, key=lambda tile: (_dist(pos, tile), tile))
        claimed.add(choice)
        return choice

    for index, position in enumerate(positions):
        pos = (int(position[0]), int(position[1]))
        inv = inventories[index] if index < len(inventories) else {}
        wheat = int(inv.get("WHEAT", 0))
        carrying_animal = int(inv.get(animal, 0)) > 0
        here = animal_at.get(pos)
        # Finish the animal underfoot before walking. A second trip from the
        # shed is what was leaving yield unharvested at dusk.
        if here is not None and not carrying_animal:
            if wheat > 0 and not here.get("fed_today", False):
                here["fed_today"] = True
                unfed_left = max(0, unfed_left - 1)
                actions.append(["FEED"])
                continue
            if not here.get("cared_today", False):
                here["cared_today"] = True
                actions.append(["CARE"])
                continue
            if int(here.get("yield_units", 0)) > 0:
                here["yield_units"] = 0
                actions.append(["HARVEST"])
                continue
            if want_fert and here.get("fertilizer_available", False):
                here["fertilizer_available"] = False
                actions.append(["COLLECT_FERTILIZER"])
                continue

        if unfed_left and wheat > 0:
            target = take_nearest(pos, [p for p, tile in animals if not tile.get("fed_today", False)])
            if target is not None:
                actions.append(["FEED"] if pos == target else _move(pos, target))
                continue
        if unfed_left and wheat <= 0 and int(shed_stock.get("WHEAT", 0)) > 0 and holders + fetchers < unfed_left:
            fetchers += 1
            if pos in scan["shed"]:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        if carrying_animal:
            target = take_nearest(pos, list(scan["empty"][spec["structure"]]))
            if target is not None:
                actions.append(["PLACE", animal, 1] if pos == target else _move(pos, target))
                continue

        care_targets = [p for p, tile in animals if not tile.get("cared_today", False)]
        target = take_nearest(pos, care_targets)
        if target is not None:
            actions.append(["CARE"] if pos == target else _move(pos, target))
            continue

        harvest_animals = [p for p, tile in animals if int(tile.get("yield_units", 0)) > 0]
        target = take_nearest(pos, harvest_animals)
        if target is not None:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        if day >= 4:
            ripe = [p for p, tile in plants if str(tile.get("crop")) == "WHEAT" and int(tile.get("yield_units", 0)) > 0]
            target = take_nearest(pos, ripe)
            if target is not None:
                actions.append(["HARVEST"] if pos == target else _move(pos, target))
                continue

        if want_fert:
            fert = [p for p, tile in animals if tile.get("fertilizer_available", False)]
            target = take_nearest(pos, fert)
            if target is not None:
                actions.append(["COLLECT_FERTILIZER"] if pos == target else _move(pos, target))
                continue

        thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
        # Dying plants outrank building. A missed water day kills the feed crop.
        danger = [
            p for p, tile in plants
            if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1
        ]
        target = take_nearest(pos, danger or thirsty)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        if (
            hour < 20
            and pos in scan["shed"]
            and int(shed_stock.get(animal, 0)) > 0
            and scan["empty"][spec["structure"]]
        ):
            shed_stock[animal] = int(shed_stock.get(animal, 0)) - 1
            actions.append(["PICKUP", animal, 1])
            continue

        structure = spec["structure"]
        have_struct = scan["counts"][animal] + len(scan["empty"][structure])
        need_struct = min(_target(day, len(quadrants)), _drain_cap(animal, demand, len(quadrants))) if day >= 3 else 0
        if day >= 3 and day < 26 and have_struct < max(need_struct, owned):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                build = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
                actions.append([build] if pos == target else _move(pos, target))
                continue

        if day < 2 and int(seeds.get("WHEAT", 0)) - planted > 0 and scan["crops"].get("WHEAT", 0) + planted < WHEAT_PLOTS:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # Park leftover goods in the shed so the next market tick can sell them.
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue
        actions.append(["PASS"])

    if not actions:
        return ["PASS"], []
    return actions[0], actions[1:]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    day = int(obs.get("day", 0))
    if day == 0 and int(obs.get("hour", 0)) == 0:
        _LOCK["animal"] = None
        _DAY["day"] = -1
    _note_milk(obs)
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    scan = _scan(obs)
    shed = obs.get("private", {}).get("shed", {})
    owned_any = sum(scan["counts"].values()) + sum(int(shed.get(name, 0)) for name in SPECIES)
    animal = _focus(demand, day, owned_any)
    farmer, hands = _units(obs, scan, animal, demand)
    return {"farmer": farmer, "hands": hands, "market": _market(obs, scan, animal, demand)}
