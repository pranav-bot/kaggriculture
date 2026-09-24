"""velocity_mill — cash-flow driven herd scaling with optimized sell timing.

Key improvements over rl_fert_mill (from replay gap analysis):
 1. Cash-flow herd controller: buy animals ASAP when cash allows, not on fixed days
 2. Proportional feed: 2-day wheat buffer per animal, bought in priority slots
 3. Aggressive early ramp: target 6 animals by day 3, 12 by day 7, 18 by day 14
 4. Worker PASS elimination: deterministic task fallback chain, target <5% PASS
 5. Terminal liquidation: dump all shed inventory days 28-29, return workers to shed
 6. Market order slot budgeting: survival feed + hire first, then sells
 7. Fertilizer: sell everything above $1 floor (scipy optimal)
 8. Continuous sell policy: sell premium goods when price >= base, pace by shed pressure
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
SPECIES = {
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5, "curve": 1.7},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 4 / 3, "curve": 0.85},
    "GOOSE": {"structure": "COOP", "product": "EGG", "cost": 300, "per_day": 2.0, "curve": 1.05},
}
MAX_HANDS = 8
WHEAT_PLOTS = 8
OPERATING_RESERVE = 100  # Never let cash go below this
Pos = Tuple[int, int]

_LOCK: Dict[str, Optional[str]] = {"animal": None}


# ---------------------------------------------------------------------------
# Board scan
# ---------------------------------------------------------------------------
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
    if scores.get("SHEEP", 0) > scores.get("COW", 0) * 1.1:
        best = "SHEEP"
    elif scores.get("COW", 0) >= scores.get("SHEEP", 0):
        best = "COW"
    if day < 3 or scores[best] <= 0:
        return str(_LOCK["animal"] or ("SHEEP" if scores.get("SHEEP", 0) > scores.get("COW", 0) else "COW"))
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
    spec = SPECIES[animal]
    shop = max(0.0, float(demand.get(spec["product"], 0.0)) - 1.0)
    if shop <= 0 or spec["per_day"] <= 0:
        return 0
    return max(4, int(round(shop / spec["per_day"])) + 2)


def _drain_cap(animal: str, demand: Mapping[str, float], quadrants: int = 1) -> int:
    room = 18 if quadrants < 2 else 24
    return min(_wanted(animal, demand), room)


def _target_herd(day: int, quadrants: int = 1) -> int:
    """AGGRESSIVE ramp: 6 on day 3, 10 on day 5, 14 on day 7, 18 on day 10, 24 in quad 2."""
    if day < 3:
        return 0
    if day < 5:
        return 6
    if day < 7:
        return 10
    if day < 10:
        return 14
    if day < 14:
        return 18
    return 18 if quadrants < 2 else 24


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


# ---------------------------------------------------------------------------
# Sell policy — continuous with pressure-aware pacing
# ---------------------------------------------------------------------------
def _sell_qty(
    item: str, have: int, price: float, shed_total: int, day: int,
    need_cash: bool, is_endgame: bool,
) -> int:
    if have <= 0 or item in SPECIES:
        return 0
    if item == "WHEAT":
        # Only sell wheat to relieve shed pressure or at endgame
        if is_endgame:
            return max(0, have - 4)
        return max(0, have - 12) if shed_total >= 90 else 0

    if item == "FERTILIZER":
        # Sell all fertilizer — scipy confirms floor doesn't matter
        if price < 1:
            return 0
        return min(have, 4)

    base = BASE.get(item, 1)

    # Endgame: aggressive dump from day 27
    if day >= 27:
        return min(have, 10)

    # Never sell under base
    if price < base:
        return 0

    # Need cash urgently: sell to fund expansion
    if need_cash:
        return min(have, 3)

    # Shed pressure: sell faster
    if shed_total >= 85:
        return min(have, 8)
    if shed_total >= 75:
        return min(have, 4)

    # Hold early to ride premium, then sell continuously
    # rl_fert_mill Q-learning found: start selling around day 14-16
    if day < 14 and shed_total < 70:
        return 0
    # From day 14: sell continuously, faster when quote is high
    if day < 22:
        if price >= base * 1.35:
            pace = 5 if shed_total >= 60 else 4
        elif price >= base * 1.15:
            pace = 4 if shed_total >= 70 else 3
        else:
            pace = 2
        return min(have, pace)
    # Days 22-26: sell faster to avoid terminal glut
    if price >= base * 1.3:
        pace = 6 if shed_total >= 60 else 5
    else:
        pace = 4 if shed_total >= 70 else 3
    return min(have, pace)


# ---------------------------------------------------------------------------
# Market orders — slot-budgeted
# ---------------------------------------------------------------------------
def _market(obs: Mapping[str, Any], scan: Mapping[str, Any], animal: str, demand: Mapping[str, float]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
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
        _target_herd(day, len(quadrants)),
        max(0, (18 if len(quadrants) < 2 else 24) - others),
    )
    owned = focus_owned
    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    shed_total = sum(int(v) for v in shed.values())
    is_endgame = day >= 28
    need_cash = money < spec["cost"] + OPERATING_RESERVE and owned < target

    # === PRIORITY 1: Survival wheat (feed) — always first ===
    wheat_floor = 16 if (live <= 3 and day < 4) else live + 4  # lean buffer: herd + 4
    wheat_need = max(0, wheat_floor - wheat_have)
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))
    if wheat_need and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 16)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === PRIORITY 2: Land Expansion (NE Quadrant) ===
    if len(quadrants) == 1 and 7 <= day <= 12 and money >= 1000 + OPERATING_RESERVE and len(orders) < 8:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000

    # === PRIORITY 3: Hire workers (scaled to herd demand) ===
    if live < 6:
        target_hands = 3
    elif live < 12:
        target_hands = 5
    elif live < 18:
        target_hands = 7
    else:
        target_hands = MAX_HANDS

    hires = int(farm.get("hires_today", 0))
    while hires < target_hands and len(orders) < 8:
        cost = Actions.hire_cost(hires)
        if money < cost + OPERATING_RESERVE:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    # === PRIORITY 4: Buy animals (cash-flow driven, cut-off at day 18) ===
    if 3 <= day <= 18 and owned < target and len(orders) < 9:
        free = len(scan["empty"][spec["structure"]])
        # Buy as many as we can afford, keeping operating reserve
        affordable = int((money - OPERATING_RESERVE - wheat_floor * wheat_price) // spec["cost"])
        affordable = max(0, affordable)
        buy = min(affordable, free, target - owned, 3)  # Max 3 per turn to leave order slots
        if buy > 0 and wheat_have >= live + buy:
            orders.append(["BUY_ANIMAL", animal, buy])
            money -= spec["cost"] * buy

    # === PRIORITY 5: Seeds (wheat crop on Day 0) ===
    growing = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    if day == 0 and growing < WHEAT_PLOTS and len(orders) < 10:
        qty = min(WHEAT_PLOTS - int(seeds.get("WHEAT", 0)), int((money - OPERATING_RESERVE) // 10))
        if qty > 0:
            orders.append(["BUY_SEED", "WHEAT", qty])
            money -= qty * 10

    # === PRIORITY 5: Sells — use remaining slots ===
    # Sort: fertilizer first (funds cows), then by price descending
    ranked = sorted(shed.items(), key=lambda kv: -float(prices.get(kv[0], 0)))
    ranked.sort(key=lambda kv: 0 if kv[0] == "FERTILIZER" else 1)
    for item, count in ranked:
        if len(orders) >= 10:
            break
        qty = _sell_qty(
            str(item), int(count), float(prices.get(item, 0)),
            shed_total, day, need_cash, is_endgame,
        )
        if qty > 0:
            orders.append(["SELL", item, qty])
            money += qty * float(prices.get(item, BASE.get(str(item), 1)))

    return orders[:10]


# ---------------------------------------------------------------------------
# Worker actions — eliminate PASS
# ---------------------------------------------------------------------------
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
    planted_wheat = 0
    planted_straw = 0
    money = float(farm.get("money", 0))
    carried_animals = sum(int((inv or {}).get(animal, 0)) for inv in inventories)
    owned = sum(scan["counts"].values()) + int(shed_stock.get(animal, 0)) + carried_animals
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 13

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

        # Terminal return: go to shed and drop everything
        if is_endgame:
            carried = sum(int(v) for k, v in inv.items())
            if carried > 0:
                dest = nearest_shed(pos)
                actions.append(["DROP"] if pos == dest else _move(pos, dest))
                continue
            # If near a ripe animal/crop, harvest on the way
            if here is not None and int(here.get("yield_units", 0)) > 0:
                actions.append(["HARVEST"])
                continue
            actions.append(["PASS"])
            continue

        # === Priority 1: Act on animal underfoot ===
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
            if here.get("fertilizer_available", False):
                here["fertilizer_available"] = False
                actions.append(["COLLECT_FERTILIZER"])
                continue

        # === Priority 2: Feed hungry animals ===
        if unfed_left and wheat > 0:
            target = take_nearest(pos, [p for p, tile in animals if not tile.get("fed_today", False)])
            if target is not None:
                actions.append(["FEED"] if pos == target else _move(pos, target))
                continue

        # === Priority 3: Get wheat to feed ===
        if unfed_left and wheat <= 0 and int(shed_stock.get("WHEAT", 0)) > 0 and holders + fetchers < unfed_left:
            fetchers += 1
            if pos in scan["shed"]:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        # === Priority 4: Place carried animals ===
        if carrying_animal:
            target = take_nearest(pos, list(scan["empty"][spec["structure"]]))
            if target is not None:
                actions.append(["PLACE", animal, 1] if pos == target else _move(pos, target))
                continue

        # === Priority 5: Care uncared animals ===
        care_targets = [p for p, tile in animals if not tile.get("cared_today", False)]
        target = take_nearest(pos, care_targets)
        if target is not None:
            actions.append(["CARE"] if pos == target else _move(pos, target))
            continue

        # === Priority 6: Harvest animals ===
        harvest_animals = [p for p, tile in animals if int(tile.get("yield_units", 0)) > 0]
        target = take_nearest(pos, harvest_animals)
        if target is not None:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # === Priority 7: Harvest ripe crops (Wheat & Strawberry) ===
        if day >= 2:
            def _crop_ripe(tile: Mapping[str, Any]) -> bool:
                c = str(tile.get("crop"))
                p_day = int(tile.get("planted_day", 0))
                if int(tile.get("yield_units", 0)) <= 0:
                    return False
                m_age = 3 if c == "WHEAT" else (10 if c == "STRAWBERRY" else 99)
                return (day - p_day) >= m_age

            ripe = [p for p, tile in plants if _crop_ripe(tile)]
            target = take_nearest(pos, ripe)
            if target is not None:
                actions.append(["HARVEST"] if pos == target else _move(pos, target))
                continue

        # === Priority 8: Collect fertilizer ===
        fert = [p for p, tile in animals if tile.get("fertilizer_available", False)]
        target = take_nearest(pos, fert)
        if target is not None:
            actions.append(["COLLECT_FERTILIZER"] if pos == target else _move(pos, target))
            continue

        # === Priority 9: Water thirsty plants (danger first) ===
        thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
        danger = [
            p for p, tile in plants
            if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1
        ]
        target = take_nearest(pos, danger or thirsty)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # === Priority 10: Pick up animals from shed ===
        if (
            hour < 20
            and pos in scan["shed"]
            and int(shed_stock.get(animal, 0)) > 0
            and scan["empty"][spec["structure"]]
        ):
            shed_stock[animal] = int(shed_stock.get(animal, 0)) - 1
            actions.append(["PICKUP", animal, 1])
            continue

        # === Priority 11: Build structures ===
        structure = spec["structure"]
        have_struct = scan["counts"][animal] + len(scan["empty"][structure])
        need_struct = _target_herd(day, len(quadrants))
        if day < 26 and have_struct < max(need_struct, owned):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                build = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
                actions.append([build] if pos == target else _move(pos, target))
                continue

        # === Priority 12: Plant wheat ===
        avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
        if day < 3 and avail_wheat > 0 and scan["crops"].get("WHEAT", 0) + planted_wheat < WHEAT_PLOTS:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_wheat += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        # === Priority 13: Clear weeds ===
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # === Priority 14: Drop carried goods at shed ===
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 15: Move toward staging position (near shed) ===
        # Instead of PASS, move toward a useful position
        if pos not in scan["shed"]:
            dest = nearest_shed(pos)
            if _dist(pos, dest) > 1:
                actions.append(_move(pos, dest))
                continue

        actions.append(["PASS"])

    if not actions:
        return ["PASS"], []
    return actions[0], actions[1:]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    day = int(obs.get("day", 0))
    if day == 0 and int(obs.get("hour", 0)) == 0:
        _LOCK["animal"] = None
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    scan = _scan(obs)
    shed = obs.get("private", {}).get("shed", {})
    owned_any = sum(scan["counts"].values()) + sum(int(shed.get(name, 0)) for name in SPECIES)
    animal = _focus(demand, day, owned_any)
    farmer, hands = _units(obs, scan, animal, demand)
    return {"farmer": farmer, "hands": hands, "market": _market(obs, scan, animal, demand)}
