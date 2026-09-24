"""apex_sovereign — Multi-species compound expansion with Day 0 kickstart.

Key Architectural Pillars:
 1. Day 0 Capital Deployment: 2 Cows + 2 Sheep + 16 Wheat feed + 8 Wheat seeds.
 2. Multi-Species Hedge: 12 Cows + 11 Sheep balanced portfolio prevents market quote collapse.
 3. Dawn Labor Pulse: Batch hire 4-8 hands at Hour 0/1 scaled strictly to herd size.
 4. Progressive Land Expansion: BUY_LAND (NE) on Days 6-10 at $1,000 for 50-tile footprint.
 5. Quadrant 2 Strawberry Pipeline: 10-12 ongoing Strawberry plots yielding $1,200+/day.
 6. Homegrown Feed Self-Sufficiency: 8 wheat plots eliminating feed purchase drag.
 7. Active Shed Delivery: Workers proactively fetch animals from shed to place in pastures.
 8. Priority Crop Sowing: Wheat on Day 0 and Strawberry on Day 7+ planted before secondary pasture building.
 9. Pressure-Aware Continuous Liquidation: Sell above base price, merge and dump Days 28-29.
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
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 1.333},
}
MAX_HANDS = 8
WHEAT_PLOTS = 8
STRAWBERRY_PLOTS = 10
OPERATING_RESERVE = 100
Pos = Tuple[int, int]


# ---------------------------------------------------------------------------
# Board Scan & Spatial Utilities
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


def _scan(obs: Mapping[str, Any]) -> Dict[str, Any]:
    farm = _farm(obs)
    tiles = farm.get("tiles", [])
    board = len(tiles) or 10
    shed = set(_sheds(board))
    rows: List[Tuple[str, Pos, Mapping[str, Any]]] = []
    empty_pastures: List[Pos] = []
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
                crop = str(tile.get("crop"))
                crops[crop] = crops.get(crop, 0) + 1
                rows.append(("plant", pos, tile))
            elif kind == "PASTURE":
                if "animal" not in tile:
                    empty_pastures.append(pos)
                else:
                    name = str(tile["animal"])
                    counts[name] = counts.get(name, 0) + 1
                    rows.append(("animal", pos, tile))

    center = (board // 2 - 1, board // 2 - 1)
    empties.sort(key=lambda p: (_dist(center, p), p))
    return {
        "shed": shed, "rows": rows, "empty_pastures": empty_pastures,
        "empties": empties, "weeds": weeds, "counts": counts, "crops": crops,
    }


# ---------------------------------------------------------------------------
# Target Herd Ramp (Multi-Species)
# ---------------------------------------------------------------------------
def _target_herd(day: int, quadrants: int = 1) -> Tuple[int, int]:
    """Target counts for (COW, SHEEP) across game days."""
    if day == 0:
        return (2, 2)    # 4 animals opening
    if day < 3:
        return (2, 2)
    if day < 5:
        return (4, 3)    # 7 animals
    if day < 7:
        return (6, 5)    # 11 animals
    if day < 10:
        return (8, 7)    # 15 animals
    if day < 14:
        return (10, 8)   # 18 animals
    if quadrants < 2:
        return (10, 8)
    return (12, 11)      # 23 animals full capacity in 2 quadrants


# ---------------------------------------------------------------------------
# Sell Policy — Continuous with Pressure-Aware Pacing
# ---------------------------------------------------------------------------
def _sell_qty(
    item: str, have: int, price: float, shed_total: int, day: int,
    need_cash: bool, is_endgame: bool,
) -> int:
    if have <= 0 or item in SPECIES:
        return 0
    if item == "WHEAT":
        if is_endgame:
            return max(0, have - 4)
        return max(0, have - 12) if shed_total >= 90 else 0

    if item == "FERTILIZER":
        return min(have, 6) if price >= 1 else 0

    base = BASE.get(item, 1)

    # Endgame: aggressive dump from day 27
    if day >= 27:
        return min(have, 20)

    # Never sell under base quote during growth unless shed is overflowing
    if price < base and shed_total < 85:
        return 0

    # Need cash urgently for land or animal purchase
    if need_cash:
        return min(have, 4)

    # Shed pressure relief
    if shed_total >= 85:
        return min(have, 10)
    if shed_total >= 70:
        return min(have, 6)

    # Normal selling: hold before day 12 unless high premium
    if day < 12:
        return min(have, 3) if price >= base * 1.25 else 0

    # Days 12-21: continuous selling, pace by inventory size and price
    if day < 22:
        if price >= base * 1.2:
            return min(have, 6 if have >= 10 else 4)
        elif price >= base * 1.05:
            return min(have, 5 if have >= 10 else 3)
        else:
            return min(have, 3)

    # Days 22-26: accelerated liquidation
    return min(have, 8 if have >= 12 else 5)


# ---------------------------------------------------------------------------
# Market Decisions (Priority-Ordered Queue)
# ---------------------------------------------------------------------------
def _market(obs: Mapping[str, Any], scan: Mapping[str, Any]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    prices = obs.get("market", {}).get("prices", {})
    seeds = dict(private.get("seeds", {}))
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    orders: List[List[Any]] = []

    live = sum(scan["counts"].values())
    carried_cows = sum(int((inv or {}).get("COW", 0)) for inv in private.get("inventories", []))
    carried_sheep = sum(int((inv or {}).get("SHEEP", 0)) for inv in private.get("inventories", []))
    cows_owned = scan["counts"]["COW"] + int(shed.get("COW", 0)) + carried_cows
    sheep_owned = scan["counts"]["SHEEP"] + int(shed.get("SHEEP", 0)) + carried_sheep
    total_owned = cows_owned + sheep_owned

    target_cows, target_sheep = _target_herd(day, len(quadrants))
    total_target = target_cows + target_sheep

    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))
    shed_total = sum(int(v) for v in shed.values())
    is_endgame = day >= 28
    need_cash = money < 500 + OPERATING_RESERVE and total_owned < total_target

    # === 1. SURVIVAL FEED BUFFER (Absolute Top Priority) ===
    if day < 4 and live <= 4:
        wheat_floor = 16
    else:
        wheat_floor = live + 6
    wheat_need = max(0, wheat_floor - wheat_have)
    if wheat_need > 0 and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 10)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === 2. DAWN LABOR PULSE (Hours 0-1) ===
    if live < 5:
        target_hands = 4
    elif live < 11:
        target_hands = 5
    elif live < 17:
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

    # === 3. PROGRESSIVE LAND EXPANSION (NE Quadrant) ===
    if len(quadrants) == 1 and 6 <= day <= 12 and money >= 1000 + OPERATING_RESERVE and len(orders) < 8:
        orders.append(["BUY_LAND"])
        money -= 1000

    # === 4. DAY 0 CAPITAL DEPLOYMENT (Opening Crops & Animals) ===
    if day == 0:
        # Wheat seeds (8 plots for feed)
        cur_wheat_seeds = int(seeds.get("WHEAT", 0))
        if cur_wheat_seeds < WHEAT_PLOTS and money >= 10 + OPERATING_RESERVE and len(orders) < 9:
            qty = min(WHEAT_PLOTS - cur_wheat_seeds, int((money - OPERATING_RESERVE) // 10))
            if qty > 0:
                orders.append(["BUY_SEED", "WHEAT", qty])
                money -= qty * 10

        # Opening Animals: 2 Cows + 2 Sheep
        if cows_owned < 2 and money >= 400 + OPERATING_RESERVE and len(orders) < 9:
            orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_owned += 1
        if sheep_owned < 2 and money >= 500 + OPERATING_RESERVE and len(orders) < 9:
            orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_owned += 1
        if cows_owned < 2 and money >= 400 + OPERATING_RESERVE and len(orders) < 9:
            orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_owned += 1
        if sheep_owned < 2 and money >= 500 + OPERATING_RESERVE and len(orders) < 9:
            orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_owned += 1

    # === 5. CASH-FLOW HERD SCALING (Days 1 to 18) ===
    # Allow buying if we have free pastures OR free land tiles to build pastures
    free_capacity = len(scan["empty_pastures"]) + (1 if len(scan["empties"]) > 2 else 0)
    if 1 <= day <= 18 and total_owned < total_target and len(orders) < 9:
        need_cow = max(0, target_cows - cows_owned)
        need_sheep = max(0, target_sheep - sheep_owned)

        chosen_species = "COW" if need_cow >= need_sheep else "SHEEP"
        cost = SPECIES[chosen_species]["cost"]

        affordable = int((money - OPERATING_RESERVE - 20) // cost)
        if affordable > 0 and free_capacity > 0 and wheat_have >= live + 1:
            buy_qty = min(affordable, free_capacity, 2)
            if buy_qty > 0:
                orders.append(["BUY_ANIMAL", chosen_species, buy_qty])
                money -= cost * buy_qty

    # === 6. QUADRANT 2 STRAWBERRY CASH PIPELINE ===
    if len(quadrants) >= 2 and 7 <= day <= 16 and live >= 10 and len(orders) < 8:
        cur_straw = int(seeds.get("STRAWBERRY", 0)) + int(scan["crops"].get("STRAWBERRY", 0))
        if cur_straw < STRAWBERRY_PLOTS and money >= 100 + OPERATING_RESERVE:
            qty = min(STRAWBERRY_PLOTS - cur_straw, int((money - OPERATING_RESERVE) // 100), 3)
            if qty > 0:
                orders.append(["BUY_SEED", "STRAWBERRY", qty])
                money -= qty * 100

    # === 7. WHEAT FEED REPLENISHMENT SEEDS ===
    if len(quadrants) >= 2 and 8 <= day <= 20 and len(orders) < 8:
        cur_wheat = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
        if cur_wheat < WHEAT_PLOTS and money >= 10 + OPERATING_RESERVE:
            qty = min(WHEAT_PLOTS - cur_wheat, int((money - OPERATING_RESERVE) // 10), 4)
            if qty > 0:
                orders.append(["BUY_SEED", "WHEAT", qty])
                money -= qty * 10

    # === 8. SELLS — Continuous & Pressure-Aware ===
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
# Worker Planning — Zero-PASS Fallback Engine
# ---------------------------------------------------------------------------
def _units(
    obs: Mapping[str, Any], scan: Mapping[str, Any],
) -> Tuple[List[Any], List[List[Any]]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    positions = [farm.get("farmer", [4, 4]), *farm.get("hands", [])]
    inventories = list(private.get("inventories", []))
    shed_stock = dict(private.get("shed", {}))
    seeds = dict(private.get("seeds", {}))
    claimed: set[Pos] = set()
    actions: List[List[Any]] = []
    fetchers = 0
    animal_fetchers = 0
    planted_wheat = 0
    planted_straw = 0
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 13

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]
    unfed_left = sum(1 for _p, tile in animals if not tile.get("fed_today", False))
    holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)

    target_cows, target_sheep = _target_herd(day, len(quadrants))
    total_target = target_cows + target_sheep
    shed_animals = int(shed_stock.get("COW", 0)) + int(shed_stock.get("SHEEP", 0))
    total_owned = sum(scan["counts"].values()) + shed_animals
    need_pastures = max(total_target, total_owned)
    have_pastures = sum(scan["counts"].values()) + len(scan["empty_pastures"])

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

        # Check carried animal
        carried_species = None
        for sp in ("COW", "SHEEP", "GOOSE"):
            if int(inv.get(sp, 0)) > 0:
                carried_species = sp
                break

        here = animal_at.get(pos)

        # Terminal return: return to shed and drop everything
        if is_endgame:
            carried = sum(int(v) for k, v in inv.items())
            if carried > 0:
                dest = nearest_shed(pos)
                actions.append(["DROP"] if pos == dest else _move(pos, dest))
                continue
            if here is not None and int(here.get("yield_units", 0)) > 0:
                actions.append(["HARVEST"])
                continue
            actions.append(["PASS"])
            continue

        # === Priority 1: Act on animal underfoot ===
        if here is not None and carried_species is None:
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

        # === Priority 2: Place carried animals into empty pastures ===
        if carried_species is not None:
            target = take_nearest(pos, list(scan["empty_pastures"]))
            if target is not None:
                actions.append(["PLACE", carried_species, 1] if pos == target else _move(pos, target))
                continue

        # === Priority 3: Feed hungry animals ===
        if unfed_left and wheat > 0:
            target = take_nearest(pos, [p for p, tile in animals if not tile.get("fed_today", False)])
            if target is not None:
                actions.append(["FEED"] if pos == target else _move(pos, target))
                continue

        # === Priority 4: Fetch wheat to feed hungry animals ===
        if unfed_left and wheat <= 0 and int(shed_stock.get("WHEAT", 0)) > 0 and holders + fetchers < unfed_left:
            fetchers += 1
            if pos in scan["shed"]:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        # === Priority 5: Proactively FETCH ANIMALS from shed to place in pastures ===
        avail_shed_species = None
        if int(shed_stock.get("COW", 0)) > 0:
            avail_shed_species = "COW"
        elif int(shed_stock.get("SHEEP", 0)) > 0:
            avail_shed_species = "SHEEP"

        carrying_count = sum(1 for inv in inventories if any(int((inv or {}).get(sp, 0)) > 0 for sp in ("COW", "SHEEP", "GOOSE")))
        if (
            carried_species is None
            and avail_shed_species is not None
            and len(scan["empty_pastures"]) > animal_fetchers + carrying_count
        ):
            animal_fetchers += 1
            if pos in scan["shed"]:
                shed_stock[avail_shed_species] = int(shed_stock[avail_shed_species]) - 1
                actions.append(["PICKUP", avail_shed_species, 1])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        # === Priority 6: Care uncared animals ===
        care_targets = [p for p, tile in animals if not tile.get("cared_today", False)]
        target = take_nearest(pos, care_targets)
        if target is not None:
            actions.append(["CARE"] if pos == target else _move(pos, target))
            continue

        # === Priority 7: Harvest ripe livestock products ===
        harvest_animals = [p for p, tile in animals if int(tile.get("yield_units", 0)) > 0]
        target = take_nearest(pos, harvest_animals)
        if target is not None:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # === Priority 8: Harvest ripe crops (Wheat & Strawberry) ===
        ripe_crops = [
            p for p, tile in plants
            if int(tile.get("yield_units", 0)) > 0
            and (day - int(tile.get("planted_day", 0))) >= (3 if str(tile.get("crop")) == "WHEAT" else 10)
        ]
        target = take_nearest(pos, ripe_crops)
        if target is not None:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # === Priority 9: Collect fertilizer ===
        fert = [p for p, tile in animals if tile.get("fertilizer_available", False)]
        target = take_nearest(pos, fert)
        if target is not None:
            actions.append(["COLLECT_FERTILIZER"] if pos == target else _move(pos, target))
            continue

        # === Priority 10: Water thirsty plants (danger first) ===
        thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
        danger = [
            p for p, tile in plants
            if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1
        ]
        target = take_nearest(pos, danger or thirsty)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # === Priority 11: Plant Wheat (Day < 4 opening priority) ===
        avail_wheat_seeds = int(seeds.get("WHEAT", 0)) - planted_wheat
        if avail_wheat_seeds > 0 and (scan["crops"].get("WHEAT", 0) + planted_wheat < WHEAT_PLOTS):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_wheat += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        # === Priority 12: Plant Strawberry (Day 7+ Quadrant 2 priority) ===
        avail_straw_seeds = int(seeds.get("STRAWBERRY", 0)) - planted_straw
        if avail_straw_seeds > 0 and (scan["crops"].get("STRAWBERRY", 0) + planted_straw < STRAWBERRY_PLOTS):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_straw += 1
                actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                continue

        # === Priority 13: Build Pastures ===
        if day < 26 and have_pastures < need_pastures:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                have_pastures += 1
                actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                continue

        # === Priority 14: Clear weeds ===
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # === Priority 15: Drop carried goods at shed ===
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 16: Move toward staging position (near shed) ===
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
# Entry Point
# ---------------------------------------------------------------------------
def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    scan = _scan(obs)
    farmer, hands = _units(obs, scan)
    market = _market(obs, scan)
    return {"farmer": farmer, "hands": hands, "market": market}
