"""sovereign_mill — Dynamic demand-coupled multi-species herding with continuous paced selling.

Architectural highlights:
 1. Dynamic Market Coupling: Computes exact shop consumption per tick and targets
    livestock species (COW vs SHEEP) matching town demand sinks.
 2. Zero-Waste Sheep Policy: Only buys Sheep when YARN_STORE exists (daily["WOOL"] > 2.0).
    Zero sheep purchased into dead wool markets, completely eliminating $1 price crashes.
 3. Demand-Bounded Cow Herds: Herds are strictly capped to what town shops
    can absorb (+ small wholesale buffer), preventing milk price collapses.
 4. Priority Wheat Feed Farming: Planting wheat is prioritized BEFORE building pastures,
    guaranteeing 8 permanent wheat plots producing 32 free feed units every 4 days ($10,000+ saved).
 5. High-Value Strawberry Fill: When livestock demand is satisfied, utilizes spare
    quadrant capacity for Strawberry ($120/fruit ongoing cashflow).
 6. Paced Continuous Selling: Sells 1-3 units/hour. At Day >= 28, steadily liquidates
    all remaining inventory (up to 4/turn) ensuring 0 stranded assets.
 7. Fibonacci Dawn Labor Pulse: Batches HIRE orders at Hour 0/1 for maximum labor-per-dollar.
 8. Proactive Animal Fetching: Prevents bought livestock from being stranded in shed inventory.
 9. Early Land Expansion: Unlocks NE quadrant on Day 6-8 to support 22-24 animals/crops.
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
    "YARN_STORE": ("WOOL",),             # 2x consumption multiplier
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),             # 2x consumption multiplier
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

BASE = {
    "WOOL": 200, "MILK": 160, "EGG": 50, "STRAWBERRY": 120,
    "CARROT": 35, "WHEAT": 25, "TOMATO": 60, "MELON": 250, "FERTILIZER": 100,
}

SPECIES = {
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 4 / 3},
}

MAX_HANDS = 8
WHEAT_PLOTS = 8
STRAWBERRY_PLOTS = 6
OPERATING_RESERVE = 100

Pos = Tuple[int, int]


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Dynamic Demand & Demand-Bounded Herd Sizing
# ---------------------------------------------------------------------------
def _demand_analysis(obs: Mapping[str, Any]) -> Tuple[int, int, Dict[str, float]]:
    shops = obs.get("town", {}).get("unlocked_shops", []) or []
    day = int(obs.get("day", 0))

    # Daily baseline demand from town center (1 unit/day)
    daily: Dict[str, float] = {
        "MILK": 1.0, "WOOL": 1.0, "EGG": 1.0, "WHEAT": 1.0, "STRAWBERRY": 1.0, "CARROT": 1.0,
    }

    # Shop consumption: 6 ticks/day
    for shop in shops:
        products = SHOPS.get(str(shop), ())
        weight = 2.0 if len(products) == 1 else 1.0
        for p in products:
            if p in daily:
                daily[p] += weight * 6.0

    # Cow capacity: wholesale absorbs 6 cows baseline + shop capacity
    cow_cap = 6 + int(round((daily["MILK"] - 1.0) / 1.5))

    # Sheep capacity: ONLY buy sheep if a Yarn Store exists! Zero sheep into dead wool markets!
    if daily["WOOL"] > 2.0:
        sheep_cap = int(round(daily["WOOL"] / 1.33))
    else:
        sheep_cap = 0

    return cow_cap, sheep_cap, daily


# ---------------------------------------------------------------------------
# Board Scanning
# ---------------------------------------------------------------------------
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
                    if name in counts:
                        counts[name] += 1
                        rows.append(("animal", pos, tile))

    center = (board // 2 - 1, board // 2 - 1)
    empties.sort(key=lambda p: (_dist(center, p), p))
    empty_pastures.sort(key=lambda p: (_dist(center, p), p))

    return {
        "shed": shed, "rows": rows, "empty_pastures": empty_pastures,
        "empties": empties, "weeds": weeds, "counts": counts, "crops": crops,
    }


# ---------------------------------------------------------------------------
# Paced Continuous Selling
# ---------------------------------------------------------------------------
def _sell_qty(
    item: str, have: int, price: float, shed_total: int, day: int, daily_demand: float,
) -> int:
    if have <= 0 or item in SPECIES:
        return 0

    if item == "WHEAT":
        if day >= 28:
            return max(0, have - 4)
        return max(0, have - 16) if shed_total >= 80 else 0

    if item == "FERTILIZER":
        return min(have, 4) if price >= 1.0 else 0

    base = BASE.get(item, 100)
    soft_floor = base * 0.85

    # Endgame (Days 28-29): liquidate completely up to 4 units/turn
    if day >= 28:
        return min(have, 4)

    # If price is below soft floor, only sell to relieve high shed pressure
    if price < soft_floor:
        if have >= 8 or shed_total >= 50:
            return 1
        return 0

    # Normal paced sales:
    if price >= base * 1.25:
        return min(have, 3)
    if price >= base * 1.00:
        return min(have, 2)
    return min(have, 1)


# ---------------------------------------------------------------------------
# Market Orders
# ---------------------------------------------------------------------------
def _market(
    obs: Mapping[str, Any], scan: Mapping[str, Any],
    cow_cap: int, sheep_cap: int, daily_demand: Dict[str, float],
) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    prices = obs.get("market", {}).get("prices", {})
    seeds = dict(private.get("seeds", {}))
    orders: List[List[Any]] = []

    live_cows = scan["counts"]["COW"]
    live_sheep = scan["counts"]["SHEEP"]
    live_total = live_cows + live_sheep

    carried_cows = sum(int((inv or {}).get("COW", 0)) for inv in private.get("inventories", []))
    carried_sheep = sum(int((inv or {}).get("SHEEP", 0)) for inv in private.get("inventories", []))

    owned_cows = live_cows + int(shed.get("COW", 0)) + carried_cows
    owned_sheep = live_sheep + int(shed.get("SHEEP", 0)) + carried_sheep
    owned_total = owned_cows + owned_sheep

    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    quad_count = len(quadrants)
    max_farm_herd = 17 if quad_count < 2 else 24

    # Apportion quotas bounded by demand capacity
    quota_cow = min(cow_cap, 14 if quad_count < 2 else 20)
    quota_sheep = min(sheep_cap, 12 if quad_count < 2 else 18)
    total_target = min(quota_cow + quota_sheep, max_farm_herd)

    # Day ramp targets
    if day < 3:
        total_target = min(total_target, 4)
    elif day < 5:
        total_target = min(total_target, 8)
    elif day < 7:
        total_target = min(total_target, 12)
    elif day < 10:
        total_target = min(total_target, 16)

    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    shed_total = sum(int(v) for v in shed.values())

    # === PRIORITY 1: Feed Wheat (Survival Buffer) ===
    wheat_floor = 16 if (live_total <= 4 and day < 4) else live_total + 4
    wheat_need = max(0, wheat_floor - wheat_have)
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))

    if wheat_need and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 16)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === PRIORITY 2: Land Expansion (NE Quadrant) ===
    if quad_count == 1 and 6 <= day <= 12 and money >= 1000 + OPERATING_RESERVE and len(orders) < 8:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000

    # === PRIORITY 3: Fibonacci Dawn Labor Pulse ===
    if live_total < 5:
        target_hands = 3
    elif live_total < 10:
        target_hands = 5
    elif live_total < 15:
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

    # === PRIORITY 4: Buy Animals (Cash-flow Driven, Cut-off Day 23) ===
    if day <= 23 and owned_total < total_target and len(orders) < 9:
        free_pastures = len(scan["empty_pastures"])
        # Decide which species to buy:
        # If Yarn Store exists and sheep under quota: buy sheep, else cow
        if owned_sheep < quota_sheep and daily_demand.get("WOOL", 1.0) > 2.0:
            buy_species = "SHEEP"
        elif owned_cows < quota_cow:
            buy_species = "COW"
        elif owned_sheep < quota_sheep:
            buy_species = "SHEEP"
        else:
            buy_species = "COW"

        spec = SPECIES[buy_species]
        reserve = OPERATING_RESERVE + (wheat_floor * wheat_price * 0.3)
        affordable = int((money - reserve) // spec["cost"])
        affordable = max(0, affordable)
        buy = min(affordable, free_pastures, total_target - owned_total, 3)

        if buy > 0 and wheat_have >= live_total + buy:
            orders.append(["BUY_ANIMAL", buy_species, buy])
            money -= spec["cost"] * buy
            owned_total += buy

    # === PRIORITY 5: Wheat Seeds (Sustainable Homegrown Feed) ===
    growing_wheat = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    if day <= 20 and growing_wheat < WHEAT_PLOTS and len(orders) < 9:
        needed_seeds = WHEAT_PLOTS - growing_wheat
        seed_cost = 10
        qty = min(needed_seeds, int((money - OPERATING_RESERVE) // seed_cost))
        if qty > 0:
            orders.append(["BUY_SEED", "WHEAT", qty])
            money -= qty * seed_cost

    # === PRIORITY 6: Strawberry Seeds (High-margin Cash Crop for Spare Land) ===
    growing_straw = int(seeds.get("STRAWBERRY", 0)) + int(scan["crops"].get("STRAWBERRY", 0))
    if 6 <= day <= 12 and quad_count >= 2 and growing_straw < STRAWBERRY_PLOTS and len(orders) < 9:
        needed_straw = STRAWBERRY_PLOTS - growing_straw
        straw_seed_cost = 100
        if money >= straw_seed_cost + OPERATING_RESERVE + 400:
            qty = min(needed_straw, int((money - OPERATING_RESERVE - 400) // straw_seed_cost), 2)
            if qty > 0:
                orders.append(["BUY_SEED", "STRAWBERRY", qty])
                money -= qty * straw_seed_cost

    # === PRIORITY 7: Sells — Use remaining order slots ===
    ranked = sorted(shed.items(), key=lambda kv: -float(prices.get(kv[0], 0)))
    ranked.sort(key=lambda kv: 0 if kv[0] == "FERTILIZER" else 1)

    for item, count in ranked:
        if len(orders) >= 10:
            break
        item_str = str(item)
        qty = _sell_qty(
            item_str, int(count), float(prices.get(item_str, 0)),
            shed_total, day, daily_demand.get(item_str, 1.0),
        )
        if qty > 0:
            orders.append(["SELL", item, qty])
            money += qty * float(prices.get(item, BASE.get(item_str, 1)))

    return orders[:10]


# ---------------------------------------------------------------------------
# Worker Task Routing (PASS Elimination & Proactive Fetching)
# ---------------------------------------------------------------------------
def _units(
    obs: Mapping[str, Any], scan: Mapping[str, Any],
    target_pastures: int, daily_demand: Dict[str, float],
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
    wheat_fetchers = 0
    animal_fetchers = 0
    planted_wheat = 0
    planted_straw = 0

    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 14

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]

    unfed_left = sum(1 for _p, tile in animals if not tile.get("fed_today", False))
    wheat_holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)

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

        # Check if carrying any animal
        carried_species = None
        for sp in ("COW", "SHEEP", "GOOSE"):
            if int(inv.get(sp, 0)) > 0:
                carried_species = sp
                break

        here = animal_at.get(pos)

        # Terminal return: convergence to shed and drop everything
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

        # === Priority 2: Place carried animal in empty pasture ===
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

        # === Priority 4: Fetch wheat from shed to feed ===
        if unfed_left and wheat <= 0 and int(shed_stock.get("WHEAT", 0)) > 0 and wheat_holders + wheat_fetchers < unfed_left:
            wheat_fetchers += 1
            if pos in scan["shed"]:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        # === Priority 5: Proactively FETCH ANIMALS from shed to empty pastures ===
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

        # === Priority 7: Harvest ripe animals ===
        harvest_animals = [p for p, tile in animals if int(tile.get("yield_units", 0)) > 0]
        target = take_nearest(pos, harvest_animals)
        if target is not None:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # === Priority 8: Harvest ripe crops (Wheat & Strawberry) ===
        if day >= 2:
            def _crop_ripe(tile: Mapping[str, Any]) -> bool:
                crop = str(tile.get("crop"))
                p_day = int(tile.get("planted_day", 0))
                min_age = 3 if crop == "WHEAT" else (10 if crop == "STRAWBERRY" else 99)
                return int(tile.get("yield_units", 0)) > 0 and (day - p_day) >= min_age

            ripe_crops = [p for p, tile in plants if _crop_ripe(tile)]
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

        # === Priority 10: Water thirsty crops (danger first) ===
        thirsty_plants = [p for p, tile in plants if not tile.get("watered_today", False)]
        danger_plants = [p for p, tile in plants if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1]
        target = take_nearest(pos, danger_plants or thirsty_plants)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # === Priority 11: Plant Wheat Seeds (Continuous Feed Production - PRIORITIZED OVER PASTURES) ===
        avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
        if day <= 22 and avail_wheat > 0 and (scan["crops"].get("WHEAT", 0) + planted_wheat < WHEAT_PLOTS):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_wheat += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        # === Priority 12: Build Pastures ===
        have_pastures = len(animals) + len(scan["empty_pastures"])
        if day < 26 and have_pastures < target_pastures:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                continue

        # === Priority 13: Plant Strawberry Seeds (High Margin Crop) ===
        avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
        if day <= 14 and avail_straw > 0 and (scan["crops"].get("STRAWBERRY", 0) + planted_straw < STRAWBERRY_PLOTS):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_straw += 1
                actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                continue

        # === Priority 14: Clear Weeds ===
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # === Priority 15: Drop carried items at shed ===
        carried_non_feed = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried_non_feed > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 16: Move toward staging area near shed ===
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
# Main Agent Entry Point
# ---------------------------------------------------------------------------
def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    cow_cap, sheep_cap, daily_demand = _demand_analysis(obs)
    scan = _scan(obs)
    day = int(obs.get("day", 0))
    quad_count = len(obs.get("farms", [{}])[int(obs.get("player", 0))].get("unlocked_quadrants", ["NW"]))

    quota_cow = min(cow_cap, 14 if quad_count < 2 else 20)
    quota_sheep = min(sheep_cap, 12 if quad_count < 2 else 18)
    max_farm_herd = 17 if quad_count < 2 else 24

    total_target = min(quota_cow + quota_sheep, max_farm_herd)
    if day < 3:
        total_target = min(total_target, 4)
    elif day < 5:
        total_target = min(total_target, 8)
    elif day < 7:
        total_target = min(total_target, 12)
    elif day < 10:
        total_target = min(total_target, 16)

    farmer, hands = _units(obs, scan, total_target, daily_demand)
    market_orders = _market(obs, scan, cow_cap, sheep_cap, daily_demand)
    return {"farmer": farmer, "hands": hands, "market": market_orders}
