"""apex_mill — Compound-expansion macroeconomic agent modeled on Rank 1–3 elite replays.

Key Elite Innovations:
 1. Dawn Labor Pulse: Batch-hire 6–8 hands at Hour 0/1 daily for full 23-hour labor.
 2. Melon Capital Catalyst: 10 Melons planted on Day 0-1 for a $15,000+ Day 10 windfall.
 3. Feed & Cash Crop Rotation: 10 Wheat on Day 0 -> harvested on Day 4 -> replanted with Strawberry.
 4. Progressive Land Expansion:
    - Day 6-7: BUY_LAND NE ($1,000) -> expands to 50 tiles.
    - Day 10-12: BUY_LAND SW ($2,000) funded by Melon windfall -> expands to 75 tiles.
 5. Mixed Livestock Portfolio: Both Cows (Milk, sqrt scarcity) and Sheep (Wool, log scarcity).
 6. 15-Level Task Fallback Chain: Zero idle PASS turns; workers always produce or stage.
 7. Priority-Queued Market Orders: HIRE -> BUY_LAND -> Feed -> Animals -> Sells.
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
BASE = {
    "WOOL": 200, "MILK": 160, "EGG": 50, "STRAWBERRY": 120,
    "CARROT": 35, "WHEAT": 25, "TOMATO": 60, "MELON": 250, "FERTILIZER": 100,
}
SPECIES = {
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 4 / 3},
}
OPERATING_RESERVE = 80
Pos = Tuple[int, int]

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


def _best_animal(demand: Mapping[str, float]) -> str:
    milk_d = demand.get("MILK", 1.0)
    wool_d = demand.get("WOOL", 1.0)
    # Wool has a log scarcity curve that crashes catastrophically under glut.
    # Milk has a sqrt scarcity curve that holds value strongly.
    # Only pick sheep if wool drain is significantly higher than milk drain.
    if wool_d >= 12.0 and milk_d <= 2.0:
        return "SHEEP"
    return "COW"

_STATE = {
    "day": -1,
    "melon_planted": False,
    "initial_wheat_planted": False,
}


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

    live_animals = sum(scan["counts"].values())
    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))

    # === 1. DAWN LABOR PULSE (Hours 0-1) ===
    # Hire target hands immediately at start of day
    target_hands = 4 if day == 0 else (6 if day < 7 else 8)
    hires_today = int(farm.get("hires_today", 0))
    if hour <= 1 and hires_today < target_hands:
        while hires_today < target_hands and len(orders) < 8:
            cost = Actions.hire_cost(hires_today)
            if money < cost + 10:
                break
            orders.append(["HIRE"])
            money -= cost
            hires_today += 1

    # === 2. PROGRESSIVE LAND EXPANSION ===
    # Day 6-9: Unlock NE ($1000) when affordable
    if len(quadrants) == 1 and day >= 6 and money >= 1000 + OPERATING_RESERVE and len(orders) < 9:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000
    # Day 10+: Unlock SW ($2000) once Melons liquidate
    elif len(quadrants) == 2 and day >= 10 and money >= 2000 + OPERATING_RESERVE and len(orders) < 9:
        orders.append(["BUY_LAND", "SW"])
        money -= 2000
    # Day 12+: Unlock SE ($4000) when cash allows
    elif len(quadrants) == 3 and day >= 12 and money >= 4000 + OPERATING_RESERVE and len(orders) < 9:
        orders.append(["BUY_LAND", "SE"])
        money -= 4000

    # === 3. SURVIVAL FEED BUFFER ===
    # Maintain lean feed buffer (live + 4)
    wheat_floor = 6 if live_animals == 0 else live_animals + 3
    wheat_need = max(0, wheat_floor - wheat_have)
    if wheat_need > 0 and money >= wheat_price + OPERATING_RESERVE and len(orders) < 8:
        qty = min(wheat_need, int((money - OPERATING_RESERVE) // wheat_price), 10)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price

    # === 4. DAY 0 CAPITAL DEPLOYMENT (Compound Opening) ===
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    choice_animal = _best_animal(demand)

    if day == 0:
        # Buy 8-10 Melon seeds + 8 Wheat seeds
        cur_melon_seeds = int(seeds.get("MELON", 0))
        if cur_melon_seeds < 10 and money >= 80 + OPERATING_RESERVE and len(orders) < 9:
            qty = min(10 - cur_melon_seeds, int((money - OPERATING_RESERVE) // 80))
            if qty > 0:
                orders.append(["BUY_SEED", "MELON", qty])
                money -= qty * 80

        cur_wheat_seeds = int(seeds.get("WHEAT", 0))
        if cur_wheat_seeds < 10 and money >= 10 + OPERATING_RESERVE and len(orders) < 9:
            qty = min(10 - cur_wheat_seeds, int((money - OPERATING_RESERVE) // 10))
            if qty > 0:
                orders.append(["BUY_SEED", "WHEAT", qty])
                money -= qty * 10

        # Buy opening livestock targeted to town demand
        target_cost = SPECIES[choice_animal]["cost"]
        owned_target = scan["counts"][choice_animal] + int(shed.get(choice_animal, 0))
        if owned_target < 3 and money >= target_cost + OPERATING_RESERVE and len(orders) < 9:
            orders.append(["BUY_ANIMAL", choice_animal, 1])
            money -= target_cost

    # === 5. MID-GAME CROPS (Strawberry cash + Wheat feed) ===
    # On Days 4-7 with 1 quadrant, save money for BUY_LAND NE ($1000)
    can_spend_crops = len(quadrants) > 1 or day > 7 or money >= 1200 + OPERATING_RESERVE
    if day >= 4 and can_spend_crops and len(orders) < 8:
        # Strawberry seeds
        cur_straw_seeds = int(seeds.get("STRAWBERRY", 0))
        straw_crops = scan["crops"].get("STRAWBERRY", 0)
        max_straw = 10 if len(quadrants) == 1 else (20 if len(quadrants) == 2 else 30)
        if straw_crops + cur_straw_seeds < max_straw and money >= 100 + OPERATING_RESERVE:
            qty = min(max_straw - (straw_crops + cur_straw_seeds), int((money - OPERATING_RESERVE) // 100), 4)
            if qty > 0:
                orders.append(["BUY_SEED", "STRAWBERRY", qty])
                money -= qty * 100

        # Ongoing wheat seeds for feed independence
        cur_wheat_seeds = int(seeds.get("WHEAT", 0))
        wheat_crops = scan["crops"].get("WHEAT", 0)
        max_wheat = 8 if len(quadrants) == 1 else (16 if len(quadrants) == 2 else 24)
        if day >= 6 and wheat_crops + cur_wheat_seeds < max_wheat and money >= 10 + OPERATING_RESERVE and len(orders) < 8:
            qty = min(max_wheat - (wheat_crops + cur_wheat_seeds), int((money - OPERATING_RESERVE) // 10), 6)
            if qty > 0:
                orders.append(["BUY_SEED", "WHEAT", qty])
                money -= qty * 10

    # === 6. LIVESTOCK EXPANSION ===
    shed_animals = int(shed.get("COW", 0)) + int(shed.get("SHEEP", 0))
    carried_animals = sum(int((inv or {}).get("COW", 0)) + int((inv or {}).get("SHEEP", 0)) for inv in private.get("inventories", []))
    total_owned = live_animals + shed_animals + carried_animals
    max_herd = 8 if len(quadrants) == 1 else (16 if len(quadrants) == 2 else 24)
    free_stalls = len(scan["empty_pastures"])

    # CRITICAL QUANT RULE: NEVER BUY ANIMALS AFTER DAY 18 (animals bought after day 18 have negative ROI)
    can_buy_animals = len(quadrants) > 1 or day > 7 or money >= 1400 + OPERATING_RESERVE
    if 3 <= day <= 18 and can_buy_animals and total_owned < max_herd and shed_animals == 0 and free_stalls > 0 and len(orders) < 9:
        cost = SPECIES[choice_animal]["cost"]
        if money >= cost + OPERATING_RESERVE and wheat_have >= live_animals + 2:
            orders.append(["BUY_ANIMAL", choice_animal, 1])
            money -= cost

    # === 7. SELLING PRODUCTS ===
    shed_total = sum(int(v) for v in shed.values())
    is_endgame = day >= 27

    # Sort sales: Fertilizer first, then Melons, then Milk/Wool/Strawberries
    for item, count in sorted(shed.items(), key=lambda kv: 0 if kv[0] == "FERTILIZER" else (1 if kv[0] == "MELON" else 2)):
        if len(orders) >= 10:
            break
        have = int(count)
        if have <= 0 or item in SPECIES:
            continue
        p = float(prices.get(item, BASE.get(item, 1)))

        # Fertilizer: sell immediately
        if item == "FERTILIZER":
            qty = min(have, 4)
            orders.append(["SELL", item, qty])
            continue

        # Melon: massive harvest windfall, liquidate cleanly
        if item == "MELON":
            qty = min(have, 8 if day < 27 else have)
            orders.append(["SELL", item, qty])
            continue

        # Wheat: sell only under high shed pressure or endgame
        if item == "WHEAT":
            if is_endgame:
                qty = max(0, have - 3)
                if qty > 0: orders.append(["SELL", item, min(qty, 10)])
            elif shed_total >= 85:
                orders.append(["SELL", item, min(have, 8)])
            continue

        # Premium Goods: MILK, WOOL, STRAWBERRY
        base_p = BASE.get(item, 1)
        if is_endgame:
            orders.append(["SELL", item, min(have, 10)])
        elif p >= base_p:
            if p >= base_p * 1.35:
                pace = 6 if shed_total >= 60 else 4
            elif p >= base_p * 1.15:
                pace = 4 if shed_total >= 70 else 3
            else:
                pace = 3 if shed_total >= 80 else 2
            orders.append(["SELL", item, min(have, pace)])

    return orders[:10]


# ---------------------------------------------------------------------------
# Labor Coordination (Deterministic Fallback Chain)
# ---------------------------------------------------------------------------
def _units(
    obs: Mapping[str, Any], scan: Mapping[str, Any]
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

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]
    unfed_left = sum(1 for _p, tile in animals if not tile.get("fed_today", False))
    holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)
    fetchers = 0

    def nearest_shed(pos: Pos) -> Pos:
        return min(scan["shed"], key=lambda t: (_dist(pos, t), t))

    def take_nearest(pos: Pos, pool: List[Pos]) -> Optional[Pos]:
        open_tiles = [t for t in pool if t not in claimed]
        if not open_tiles:
            return None
        choice = min(open_tiles, key=lambda t: (_dist(pos, t), t))
        claimed.add(choice)
        return choice

    planted_crops: Dict[str, int] = {}
    pastures_planned = 0

    for idx, position in enumerate(positions):
        pos = (int(position[0]), int(position[1]))
        inv = inventories[idx] if idx < len(inventories) else {}
        wheat = int(inv.get("WHEAT", 0))
        carried_cow = int(inv.get("COW", 0)) > 0
        carried_sheep = int(inv.get("SHEEP", 0)) > 0
        here = animal_at.get(pos)

        # Terminal return
        if day >= 29 and hour >= 14:
            carried = sum(int(v) for k, v in inv.items())
            if carried > 0:
                dest = nearest_shed(pos)
                actions.append(["DROP"] if pos == dest else _move(pos, dest))
                continue

        # 1. Underfoot animal servicing
        if here is not None and not (carried_cow or carried_sheep):
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

        # 2. Feed hungry animals
        if unfed_left > 0 and wheat > 0:
            target = take_nearest(pos, [p for p, t in animals if not t.get("fed_today", False)])
            if target:
                actions.append(["FEED"] if pos == target else _move(pos, target))
                continue

        # 3. Fetch wheat for feeding
        if unfed_left > 0 and wheat <= 0 and int(shed_stock.get("WHEAT", 0)) > 0 and holders + fetchers < unfed_left:
            fetchers += 1
            if pos in scan["shed"]:
                qty = min(4, int(shed_stock["WHEAT"]))
                shed_stock["WHEAT"] -= qty
                actions.append(["PICKUP", "WHEAT", qty])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        # 4. Place carried animals
        if carried_cow or carried_sheep:
            animal_name = "COW" if carried_cow else "SHEEP"
            target = take_nearest(pos, list(scan["empty_pastures"]))
            if target:
                actions.append(["PLACE", animal_name, 1] if pos == target else _move(pos, target))
                continue

        # 5. Care uncared animals
        care_targets = [p for p, t in animals if not t.get("cared_today", False)]
        target = take_nearest(pos, care_targets)
        if target:
            actions.append(["CARE"] if pos == target else _move(pos, target))
            continue

        # 6. Harvest animals
        ripe_animals = [p for p, t in animals if int(t.get("yield_units", 0)) > 0]
        target = take_nearest(pos, ripe_animals)
        if target:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # 7. Harvest crops (Melon / Strawberry / Wheat)
        ripe_crops = [p for p, t in plants if int(t.get("yield_units", 0)) > 0]
        target = take_nearest(pos, ripe_crops)
        if target:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # 8. Collect fertilizer
        fert_tiles = [p for p, t in animals if t.get("fertilizer_available", False)]
        target = take_nearest(pos, fert_tiles)
        if target:
            actions.append(["COLLECT_FERTILIZER"] if pos == target else _move(pos, target))
            continue

        # 9. Water crops (danger first)
        danger_plants = [p for p, t in plants if not t.get("watered_today", False) and int(t.get("consecutive_unwatered", 0)) >= 1]
        thirsty_plants = [p for p, t in plants if not t.get("watered_today", False)]
        target = take_nearest(pos, danger_plants or thirsty_plants)
        if target:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # 10. Pickup animals from shed
        animal_to_pickup = None
        for a_type in ["COW", "SHEEP"]:
            if hour < 20 and int(shed_stock.get(a_type, 0)) > 0 and scan["empty_pastures"]:
                animal_to_pickup = a_type
                break

        if animal_to_pickup:
            if pos in scan["shed"]:
                shed_stock[animal_to_pickup] -= 1
                actions.append(["PICKUP", animal_to_pickup, 1])
            else:
                actions.append(_move(pos, nearest_shed(pos)))
            continue

        # 11. Plant seeds (Melon / Wheat / Strawberry) — Priority over excess pastures
        planted = False
        for crop in ["MELON", "STRAWBERRY", "WHEAT"]:
            avail_seeds = int(seeds.get(crop, 0)) - planted_crops.get(crop, 0)
            if avail_seeds > 0:
                target = take_nearest(pos, list(scan["empties"]))
                if target:
                    planted_crops[crop] = planted_crops.get(crop, 0) + 1
                    actions.append(["PLANT", crop] if pos == target else _move(pos, target))
                    planted = True
                    break
        if planted:
            continue

        # 12. Build Pastures (Quota: 8 in Quad 1, 16 in Quad 2, 24 in Quad 3)
        max_pastures = 8 if len(farm.get("unlocked_quadrants", [])) == 1 else (16 if len(farm.get("unlocked_quadrants", [])) == 2 else 24)
        total_pastures = sum(scan["counts"].values()) + len(scan["empty_pastures"]) + pastures_planned
        if total_pastures < max_pastures:
            target = take_nearest(pos, list(scan["empties"]))
            if target:
                pastures_planned += 1
                actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                continue

        # 13. Dig weeds
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # 14. Drop carried goods
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # 15. Proximity Staging (Never PASS idle)
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
# Entrypoint
# ---------------------------------------------------------------------------
def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    day = int(obs.get("day", 0))
    if day == 0 and int(obs.get("hour", 0)) == 0:
        _STATE["day"] = -1
    scan = _scan(obs)
    farmer, hands = _units(obs, scan)
    market_orders = _market(obs, scan)
    return {"farmer": farmer, "hands": hands, "market": market_orders}
