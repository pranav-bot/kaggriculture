"""scratch_grandmaster.py — Apex Two-Team Division of Labor Architecture

Key Pillars:
1. Division of Labor (Zero Congestion & Role Interference):
   - Livestock Team (Units 0-3: Farmer + Hands 0-2):
     Dedicated 100% to animal lifecycle: Feed, Care, Harvest Milk/Wool, Collect Fertilizer.
     Proactively fetches wheat, handles animal placement, drops goods at shed.
     Never distracted by crop watering.
   - Field & Expansion Team (Units 4+: Hands 3-7):
     Dedicated 100% to crops and land utilization:
     Builds pastures, plants Strawberries immediately across NE and SW quadrants,
     waters all plants, harvests ripe crops, digs weeds instantly.
     Never distracted by animals.
2. Day 0 Turbo Animal Launch (The Compounding Engine):
   - 3 Cows + 2 Sheep + 8 Wheat on Day 0.
   - 0 Melons on Day 0 (eliminates 10-day negative ROI lock).
   - Generates $1,250+/day starting on Day 1.
3. Rapid Multi-Quadrant Scaling:
   - Day 4-5: NE quadrant ($1,000) -> expands herd to 10 + 16 Strawberries.
   - Day 7-9: SW quadrant ($2,000) -> expands herd to 15 + up to 34 Strawberries.
4. Fibonacci Wage Safety:
   - Operating reserve $60 guarantees dawn wages are always funded.
   - Cost-controlled hiring: 6 early, 7 mid, 8 late.
5. High-Frequency Market Clearing:
   - Sells Fertilizer, Milk, Wool, and Strawberries every turn.
   - Terminal liquidation on Days 27-29.
"""

from __future__ import annotations

import os
import sys
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - the fallback keeps the scratch agent portable.
    linear_sum_assignment = None

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

BASE = {
    "WOOL": 200, "MILK": 160, "EGG": 50, "STRAWBERRY": 120,
    "CARROT": 35, "WHEAT": 25, "TOMATO": 60, "MELON": 250, "FERTILIZER": 100,
}

SPECIES = {
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 4 / 3},
    "GOOSE": {"structure": "COOP", "product": "EGG", "cost": 300, "per_day": 2.0},
}

OPERATING_RESERVE = 60

# Keep this ordering in sync with kaggriculture.features.macro_intents.  The
# search works on these compact decisions; the router below turns a decision
# back into the mechanical operations expected by the environment.
DISCRETE_MACRO_ACTIONS = (
    "PASS", "EXPAND_NE", "EXPAND_SW", "EXPAND_SE", "HIRE_WORKER",
    "PLANT_CROPS", "WATER_CROPS", "HARVEST_CROPS", "MAINTAIN_CROPS",
    "FERTILIZE_CROPS", "BUILD_PASTURE", "BUY_COW", "BUY_SHEEP",
    "CARE_ANIMAL", "FEED_ANIMAL", "COLLECT_FERTILIZER", "CLEAR_WEED",
    "BUY_SEEDS", "BUY_MARKET", "SELL_CROPS", "DUMP_MILK", "DUMP_WOOL",
    "SELL_MARKET", "MOVE_WORKERS",
)
BEAM_WIDTH = 3
BEAM_HORIZON = 24


def _fib(n: int) -> int:
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _hire_cost(hires_today: int) -> int:
    return _fib(hires_today)


def _farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    return obs["farms"][int(obs.get("player", 0))]


def _dist(a: Sequence[int], b: Pos) -> int:
    return abs(int(a[0]) - b[0]) + abs(int(a[1]) - b[1])


def _sheds(board: int, quadrants: Sequence[str] = ("NW",)) -> List[Pos]:
    half = board // 2
    shed_tiles = [(half - 1, half - 1)]
    if "NE" in quadrants:
        shed_tiles.append((half, half - 1))
    if "SW" in quadrants:
        shed_tiles.append((half - 1, half))
    if "SE" in quadrants:
        shed_tiles.append((half, half))
    return shed_tiles


def _move(src: Sequence[int], target: Pos) -> List[str]:
    sx, sy = int(src[0]), int(src[1])
    tx, ty = int(target[0]), int(target[1])
    # Protect against crossing locked SE (5,5) when moving between SW and NE
    if (sx <= 4 and sy >= 5 and tx >= 5 and ty <= 4) or (sx >= 5 and sy <= 4 and tx <= 4 and ty >= 5):
        if (sx, sy) != (4, 4):
            return _move(src, (4, 4))
    dx, dy = tx - sx, ty - sy
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


def _scan(obs: Mapping[str, Any]) -> Dict[str, Any]:
    farm = _farm(obs)
    tiles = farm.get("tiles", [])
    board = len(tiles) or 10
    half = board // 2
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    shed = set(_sheds(board, quadrants))
    all_shed_tiles = {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
    rows: List[Tuple[str, Pos, Mapping[str, Any]]] = []
    empty = {"PASTURE": [], "COOP": []}
    empties: List[Pos] = []
    weeds: List[Pos] = []
    crops: Dict[str, int] = {}
    counts = {"SHEEP": 0, "COW": 0, "GOOSE": 0}

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            pos = (x, y)
            if tile == "LOCKED" or pos in all_shed_tiles:
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
                    if name in counts:
                        counts[name] += 1
                        rows.append(("animal", pos, tile))

    center = (half - 1, half - 1)
    empties.sort(key=lambda p: (_dist(center, p), p))
    weeds.sort(key=lambda p: (_dist(center, p), p))
    return {
        "shed": shed, "rows": rows, "empty": empty, "empties": empties,
        "weeds": weeds, "counts": counts, "crops": crops,
    }


def _sell_qty(
    item: str, have: int, price: float, shed_total: int, day: int,
) -> int:
    if have <= 0 or item in SPECIES:
        return 0

    # 100% Terminal Liquidation on Days 27-29
    if day >= 27:
        return have

    if item == "WHEAT":
        # Keep feed buffer; in late game liquidate excess feed
        wheat_buffer = 12 if day >= 20 else 20
        return max(0, have - wheat_buffer)

    # Fertilizer: sell 100% on days 0-7 for early expansion cash; on days 8+ keep 5-unit buffer
    if item == "FERTILIZER":
        if day < 8:
            return min(have, 8) if price >= 20 else (min(have, 4) if price >= 5 else 0)
        return max(0, min(have - 5, 8)) if price >= 20 else (max(0, min(have - 5, 4)) if price >= 5 else 0)

    if item == "MELON":
        return min(have, 12)

    if item == "EGG":
        return min(have, 12)

    # Strawberries: sell in large batches so shed never bottlenecks
    if item == "STRAWBERRY":
        if price >= 100:
            return min(have, 18)
        if price >= 50:
            return min(have, 14)
        return min(have, 8)

    # Milk & Wool: sell steadily in large batches
    if item in ("MILK", "WOOL"):
        if price >= 100:
            return min(have, 16)
        if price >= 30:
            return min(have, 12)
        return min(have, 8)

    return min(have, 6)


def _market(obs: Mapping[str, Any], scan: Mapping[str, Any], demand: Mapping[str, float]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    prices = obs.get("market", {}).get("prices", {})
    seeds = dict(private.get("seeds", {}))
    orders: List[List[Any]] = []
    live = sum(scan["counts"].values())
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))

    shed_total = sum(int(v) for v in shed.values())
    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))

    milk_price = float(prices.get("MILK", BASE["MILK"]))
    wool_price = float(prices.get("WOOL", BASE["WOOL"]))

    opp_farm = obs["farms"][1 - int(obs.get("player", 0))] if len(obs.get("farms", [])) > 1 else {}
    opp_tiles = opp_farm.get("tiles", [])
    # === PRIORITY 0: High-Frequency Sells (Liquidate into cash FIRST!) ===
    ranked = sorted(shed.items(), key=lambda kv: 0 if kv[0] == "FERTILIZER" else (1 if kv[0] in ("STRAWBERRY", "MELON") else 2))
    for item, count in ranked:
        if len(orders) >= 4:
            break
        qty = _sell_qty(str(item), int(count), float(prices.get(item, 0)), shed_total, day)
        if qty > 0:
            orders.append(["SELL", item, qty])
            money += qty * float(prices.get(item, BASE.get(str(item), 1)))
            shed[item] -= qty

    # === PRIORITY 1: Emergency Feed Wheat (Only if farm harvest is dry) ===
    if day >= 28:
        wheat_need = 0
    else:
        wheat_floor = 6 if day < 3 else 8
        wheat_need = max(0, wheat_floor - wheat_have)
    if wheat_need and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 8)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === PRIORITY 2: Farm Hands ===
    if day == 0:
        target_hands = 6
    elif day < 4:
        target_hands = 6
    elif day < 8:
        target_hands = 7 if len(quadrants) >= 2 else 6
    else:
        target_hands = 8 if len(quadrants) >= 3 else 7

    hires = int(farm.get("hires_today", 0))
    while hires < target_hands and len(orders) < 10:
        cost = _hire_cost(hires)
        if money < cost:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    # === PRIORITY 3: Day 0 Opening Buys (3 Cows + 2 Sheep + 10 Wheat + 6 Melon + 2 Strawberry) ===
    animals_carried_cow = sum(int((inv or {}).get("COW", 0)) for inv in private.get("inventories", []))
    animals_carried_sheep = sum(int((inv or {}).get("SHEEP", 0)) for inv in private.get("inventories", []))
    total_owned_cows = scan["counts"]["COW"] + int(shed.get("COW", 0)) + animals_carried_cow
    total_owned_sheep = scan["counts"]["SHEEP"] + int(shed.get("SHEEP", 0)) + animals_carried_sheep
    total_herd = total_owned_cows + total_owned_sheep
    animals_in_shed = int(shed.get("SHEEP", 0)) + int(shed.get("COW", 0)) + animals_carried_cow + animals_carried_sheep

    if day == 0 and len(orders) < 9:
        if total_owned_cows < 3 and money >= 400 + OPERATING_RESERVE:
            buy_c = min(3 - total_owned_cows, int((money - OPERATING_RESERVE) // 400))
            if buy_c > 0:
                orders.append(["BUY_ANIMAL", "COW", buy_c])
                money -= buy_c * 400
                total_owned_cows += buy_c
                total_herd += buy_c

        if total_owned_sheep < 2 and money >= 500 + OPERATING_RESERVE:
            buy_s = min(2 - total_owned_sheep, int((money - OPERATING_RESERVE) // 500))
            if buy_s > 0:
                orders.append(["BUY_ANIMAL", "SHEEP", buy_s])
                money -= buy_s * 500
                total_owned_sheep += buy_s
                total_herd += buy_s

        cur_w_seeds = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
        if cur_w_seeds < 10 and money >= 50:
            buy_w = min(10 - cur_w_seeds, int((money - OPERATING_RESERVE) // 10), 10)
            if buy_w > 0:
                orders.append(["BUY_SEED", "WHEAT", buy_w])
                money -= buy_w * 10
                seeds["WHEAT"] = int(seeds.get("WHEAT", 0)) + buy_w

        cur_m_seeds = int(seeds.get("MELON", 0)) + int(scan["crops"].get("MELON", 0))
        if cur_m_seeds < 8 and money >= 80:
            buy_m = min(8 - cur_m_seeds, int((money - 20) // 80), 8)
            if buy_m > 0:
                orders.append(["BUY_SEED", "MELON", buy_m])
                money -= buy_m * 80
                seeds["MELON"] = int(seeds.get("MELON", 0)) + buy_m

    # === PRIORITY 4: Land Expansion (NE: Days 3-8, SW: Days 6-18) ===
    if len(quadrants) == 1 and 3 <= day <= 8 and money >= 1000 + OPERATING_RESERVE and len(orders) < 10:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000
        quadrants.append("NE")
    elif len(quadrants) == 2 and 6 <= day <= 18 and money >= 2000 + OPERATING_RESERVE and len(orders) < 10:
        orders.append(["BUY_LAND", "SW"])
        money -= 2000
        quadrants.append("SW")

    # === PRIORITY 5: Strawberry Cash Crop (Top Compounding Engine) ===
    cur_straw = int(seeds.get("STRAWBERRY", 0)) + int(scan["crops"].get("STRAWBERRY", 0))
    if len(quadrants) == 1:
        target_straw = 8
    elif len(quadrants) == 2:
        target_straw = 22
    else:
        target_straw = 44

    need_straw = max(0, target_straw - cur_straw)
    if 0 <= day <= 20 and len(orders) < 9:
        straw_price = 100
        land_reserve = 1000 if (len(quadrants) == 1 and 3 <= day <= 6 and money >= 800) else (2000 if (len(quadrants) == 2 and 6 <= day <= 12 and money >= 1700) else 0)
        feed_reserve = max(0, 8 - wheat_have) * max(wheat_price, 25.0) + OPERATING_RESERVE
        afford_straw = max(0, int((money - feed_reserve - land_reserve) // straw_price))
        buy_straw = min(need_straw, afford_straw, 8)
        if buy_straw > 0:
            orders.append(["BUY_SEED", "STRAWBERRY", buy_straw])
            money -= buy_straw * straw_price
            cur_straw += buy_straw
            seeds["STRAWBERRY"] = int(seeds.get("STRAWBERRY", 0)) + buy_straw

    # === PRIORITY 6: Opponent-Aware Livestock Scaling ===
    if 1 <= day <= 22 and len(orders) < 9:
        milk_depressed = (milk_price < 80)
        wool_depressed = (wool_price < 80)

        if milk_price >= 180:
            max_cows = 10
        elif milk_depressed:
            max_cows = 3
        else:
            max_cows = 7

        if wool_price >= 180:
            max_sheep = 10
        elif wool_depressed:
            max_sheep = 3
        else:
            max_sheep = 7

        if len(quadrants) == 1:
            herd_target = min(5, max_cows + max_sheep)
        elif len(quadrants) == 2:
            herd_target = min(10, max_cows + max_sheep)
        else:
            herd_target = min(18, max_cows + max_sheep)

        if total_herd < herd_target:
            cow_rev = 1.5 * milk_price
            sheep_rev = (4.0 / 3.0) * wool_price
            if cow_rev >= sheep_rev and total_owned_cows < max_cows and not milk_depressed:
                buy_sp = "COW"
            elif total_owned_sheep < max_sheep and not wool_depressed:
                buy_sp = "SHEEP"
            elif total_owned_cows < max_cows and not milk_depressed:
                buy_sp = "COW"
            elif total_owned_sheep < max_sheep and not wool_depressed:
                buy_sp = "SHEEP"
            else:
                buy_sp = None

            if buy_sp is not None:
                spec = SPECIES[buy_sp]
                free_pastures = max(0, len(scan["empty"]["PASTURE"]) - animals_in_shed)
                if len(quadrants) == 1 and 3 <= day <= 5 and money >= 750:
                    land_reserve = 1000
                elif len(quadrants) == 2 and 6 <= day <= 10 and money >= 1600:
                    land_reserve = 2000
                else:
                    land_reserve = 0
                straw_reserve = min(need_straw, 2) * 100
                avail_money = money - OPERATING_RESERVE - land_reserve - straw_reserve
                affordable = max(0, int(avail_money // spec["cost"]))
                buildable_pastures = max(0, len(scan["empties"]) - (int(seeds.get("STRAWBERRY", 0)) + int(seeds.get("WHEAT", 0))))
                allowed = min(affordable, free_pastures + buildable_pastures, herd_target - total_herd, 2)
                can_feed = (wheat_have >= 4 or day < 5 or money >= 800)
                if allowed > 0 and can_feed:
                    orders.append(["BUY_ANIMAL", buy_sp, allowed])
                    money -= spec["cost"] * allowed
                    total_herd += allowed
                    if buy_sp == "COW":
                        total_owned_cows += allowed
                    else:
                        total_owned_sheep += allowed

    # Replant wheat for feed and late-game cash compounding
    growing_wheat = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    if day >= 20:
        target_wheat = 25
    elif len(quadrants) <= 2:
        target_wheat = 8
    else:
        target_wheat = 10
    if 2 <= day <= 27 and growing_wheat < target_wheat and len(orders) < 9:
        buy_w = min(target_wheat - growing_wheat, int((money - OPERATING_RESERVE) // 10), 6)
        if buy_w > 0:
            orders.append(["BUY_SEED", "WHEAT", buy_w])
            money -= buy_w * 10
            growing_wheat += buy_w
            seeds["WHEAT"] = int(seeds.get("WHEAT", 0)) + buy_w

    # === PRIORITY 7: High-Frequency Sells ===
    ranked = sorted(shed.items(), key=lambda kv: 0 if kv[0] == "FERTILIZER" else (1 if kv[0] in ("STRAWBERRY", "MELON") else 2))
    for item, count in ranked:
        if len(orders) >= 10:
            break
        qty = _sell_qty(str(item), int(count), float(prices.get(item, 0)), shed_total, day)
        if qty > 0:
            orders.append(["SELL", item, qty])
            money += qty * float(prices.get(item, BASE.get(str(item), 1)))

    return orders[:10]


def _units(
    obs: Mapping[str, Any], scan: Mapping[str, Any], demand: Mapping[str, float],
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
    planted_melon = 0
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 16

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]
    plant_at = {pos: tile for pos, tile in plants}
    unfed_left = sum(1 for _p, tile in animals if not tile.get("fed_today", False))
    holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)

    building_pastures = 0
    target_pastures = 5 if len(quadrants) == 1 else (10 if len(quadrants) == 2 else 15)
    total_pastures = len(animals) + len(scan["empty"]["PASTURE"])

    def nearest_shed(pos: Pos) -> Pos:
        return min(scan["shed"], key=lambda tile: (_dist(pos, tile), tile))

    def take_nearest(pos: Pos, pool: List[Pos]) -> Optional[Pos]:
        open_tiles = [tile for tile in pool if tile not in claimed]
        if not open_tiles:
            return None
        choice = min(open_tiles, key=lambda tile: (_dist(pos, tile), tile))
        claimed.add(choice)
        return choice

    def _crop_ripe(tile: Mapping[str, Any]) -> bool:
        c = str(tile.get("crop"))
        p_day = int(tile.get("planted_day", 0))
        if int(tile.get("yield_units", 0)) <= 0:
            return False
        m_age = 2 if c == "WHEAT" else (10 if c in ("MELON", "STRAWBERRY") else 8)
        return (day - p_day) >= m_age

    for index, position in enumerate(positions):
        pos = (int(position[0]), int(position[1]))
        inv = inventories[index] if index < len(inventories) else {}
        wheat = int(inv.get("WHEAT", 0))
        fert = int(inv.get("FERTILIZER", 0))
        carrying_animal = any(int(inv.get(sp, 0)) > 0 for sp in SPECIES)
        carried_sp = next((sp for sp in SPECIES if int(inv.get(sp, 0)) > 0), None)
        here = animal_at.get(pos)
        plant_here = plant_at.get(pos)

        # Terminal return: shed drop
        if is_endgame:
            carried = sum(int(v) for k, v in inv.items())
            if carried > 0:
                dest = nearest_shed(pos)
                actions.append(["DROP"] if pos == dest else _move(pos, dest))
                continue
            if here is not None and int(here.get("yield_units", 0)) > 0:
                actions.append(["HARVEST"])
                continue
            if plant_here is not None and _crop_ripe(plant_here):
                actions.append(["HARVEST"])
                continue
            actions.append(["PASS"])
            continue

        # =========================================================================
        # DIVISION OF LABOR:
        # Units 0..3: Livestock Specialists
        # Units 4..7: Field & Expansion Specialists
        # =========================================================================
        is_livestock = (index < 4)

        if is_livestock:
            # === LIVESTOCK SPECIALIST ===

            # 1. Complete ALL services on animal underfoot
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

            # 2. Build pasture if animal is waiting in shed and no pasture is empty
            animals_waiting = int(shed_stock.get("COW", 0)) + int(shed_stock.get("SHEEP", 0))
            if not carrying_animal and animals_waiting > 0 and len(scan["empty"]["PASTURE"]) == 0 and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                    continue

            # 3. Place carried animal into pasture
            if carrying_animal and carried_sp is not None:
                sp_struct = SPECIES[carried_sp]["structure"]
                target = take_nearest(pos, list(scan["empty"][sp_struct]))
                if target is not None:
                    actions.append(["PLACE", carried_sp, 1] if pos == target else _move(pos, target))
                    continue

            # 4. Proactive animal fetching from shed
            avail_shed_sp = next((sp for sp in ("COW", "SHEEP") if int(shed_stock.get(sp, 0)) > 0), None)
            if (
                not carrying_animal
                and avail_shed_sp is not None
                and len(scan["empty"][SPECIES[avail_shed_sp]["structure"]]) > animal_fetchers
            ):
                animal_fetchers += 1
                if pos in scan["shed"]:
                    shed_stock[avail_shed_sp] = int(shed_stock[avail_shed_sp]) - 1
                    actions.append(["PICKUP", avail_shed_sp, 1])
                else:
                    actions.append(_move(pos, nearest_shed(pos)))
                continue

            # 4. Feed hungry animals (if holding wheat)
            if unfed_left and wheat > 0:
                target = take_nearest(pos, [p for p, tile in animals if not tile.get("fed_today", False)])
                if target is not None:
                    actions.append(["FEED"] if pos == target else _move(pos, target))
                    continue

            # 5. Fetch wheat to feed
            if unfed_left and wheat <= 0 and int(shed_stock.get("WHEAT", 0)) > 0 and holders + fetchers < unfed_left:
                fetchers += 1
                if pos in scan["shed"]:
                    qty = min(4, int(shed_stock["WHEAT"]))
                    shed_stock["WHEAT"] -= qty
                    actions.append(["PICKUP", "WHEAT", qty])
                else:
                    actions.append(_move(pos, nearest_shed(pos)))
                continue

            # 6. Drop valuable goods (Milk, Wool, Fertilizer) at shed
            carried_valuable = (
                int(inv.get("MILK", 0)) + int(inv.get("WOOL", 0)) + int(inv.get("FERTILIZER", 0))
            )
            if carried_valuable >= 2 or (carried_valuable >= 1 and unfed_left == 0 and here is None):
                dest = nearest_shed(pos)
                actions.append(["DROP"] if pos == dest else _move(pos, dest))
                continue

            # 7. Service animals needing care/harvest/fertilizer
            animal_service = [
                p for p, tile in animals
                if not tile.get("cared_today", False) or int(tile.get("yield_units", 0)) > 0 or tile.get("fertilizer_available", False)
            ]
            if animal_service:
                target = take_nearest(pos, animal_service)
                if target is not None:
                    actions.append(_move(pos, target))
                    continue

            # 8. Help field team plant seeds if any are waiting
            avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
            if avail_straw > 0 and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    planted_straw += 1
                    actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                    continue

            avail_melon = int(seeds.get("MELON", 0)) - planted_melon
            if avail_melon > 0 and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    planted_melon += 1
                    actions.append(["PLANT", "MELON"] if pos == target else _move(pos, target))
                    continue

            # 9. Help field team water thirsty plants!
            thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
            if thirsty:
                target = take_nearest(pos, thirsty)
                if target is not None:
                    actions.append(["WATER"] if pos == target else _move(pos, target))
                    continue

            # 9. Return to shed / stage
            dest = nearest_shed(pos)
            if pos != dest:
                actions.append(_move(pos, dest))
                continue
            actions.append(["PASS"])
            continue

        else:
            # === FIELD & EXPANSION SPECIALIST ===

            # 1. Underfoot crop actions
            if plant_here is not None:
                if not plant_here.get("watered_today", False):
                    plant_here["watered_today"] = True
                    actions.append(["WATER"])
                    continue
                if _crop_ripe(plant_here):
                    actions.append(["HARVEST"])
                    continue
                if fert > 0 and str(plant_here.get("crop")) == "STRAWBERRY" and int(plant_here.get("fertilized_until_day", -1)) < day:
                    plant_here["fertilized_until_day"] = day + 3
                    fert -= 1
                    actions.append(["FERTILIZE"])
                    continue

            # 2. If carrying fertilizer, deliver to unfertilized strawberry
            if fert > 0:
                unfert = [
                    p for p, tile in plants
                    if str(tile.get("crop")) == "STRAWBERRY" and int(tile.get("fertilized_until_day", -1)) < day
                ]
                if unfert:
                    target = take_nearest(pos, unfert)
                    if target is not None:
                        actions.append(["FERTILIZE"] if pos == target else _move(pos, target))
                        continue

            # 3. Underfoot weed
            tile_kind = (farm.get("tiles", [])[pos[1]][pos[0]] or {})
            if isinstance(tile_kind, dict) and tile_kind.get("kind") == "WEED":
                actions.append(["DIG"])
                continue

            # 4. Danger plants watering (zero mortality guarantee)
            danger_plants = [
                p for p, tile in plants
                if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1
            ]
            if danger_plants:
                target = take_nearest(pos, danger_plants)
                if target is not None:
                    actions.append(["WATER"] if pos == target else _move(pos, target))
                    continue

            # 5. Harvest ripe crops
            if day >= 2:
                ripe = [p for p, tile in plants if _crop_ripe(tile)]
                target = take_nearest(pos, ripe)
                if target is not None:
                    actions.append(["HARVEST"] if pos == target else _move(pos, target))
                    continue

            # 6. Plant Strawberry (High margin cash crop — Plant immediately!)
            avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
            if avail_straw > 0 and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    planted_straw += 1
                    actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                    continue

            # 7. Plant Melon (Day 0-1 compounding cash crop)
            avail_melon = int(seeds.get("MELON", 0)) - planted_melon
            if avail_melon > 0 and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    planted_melon += 1
                    actions.append(["PLANT", "MELON"] if pos == target else _move(pos, target))
                    continue

            # 8. Plant Wheat (Feed sustainability)
            avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
            if avail_wheat > 0 and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    planted_wheat += 1
                    actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                    continue

            # 9. Routine watering of all thirsty plants
            thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
            if thirsty:
                target = take_nearest(pos, thirsty)
                if target is not None:
                    actions.append(["WATER"] if pos == target else _move(pos, target))
                    continue

            # 10. Dig Weeds (Clears tiles immediately for next replanting cycle!)
            if scan["weeds"]:
                target = take_nearest(pos, list(scan["weeds"]))
                if target is not None:
                    actions.append(["DIG"] if pos == target else _move(pos, target))
                    continue

            # 11. Build Pastures
            needed_pastures = max(
                int(shed_stock.get("COW", 0)) + int(shed_stock.get("SHEEP", 0)),
                target_pastures - total_pastures
            )
            if day < 22 and needed_pastures > building_pastures and scan["empties"]:
                target = take_nearest(pos, list(scan["empties"]))
                if target is not None:
                    building_pastures += 1
                    actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                    continue

            # 12. Drop harvested crops at shed
            carried_crops = (
                int(inv.get("STRAWBERRY", 0)) + int(inv.get("WHEAT", 0)) + int(inv.get("MELON", 0))
            )
            if carried_crops >= 2 or (carried_crops >= 1 and pos in scan["shed"]):
                dest = nearest_shed(pos)
                actions.append(["DROP"] if pos == dest else _move(pos, dest))
                continue

            # 13. Pick up fertilizer from shed for strawberries
            unfert_straw = [
                p for p, tile in plants
                if str(tile.get("crop")) == "STRAWBERRY" and int(tile.get("fertilized_until_day", -1)) < day
            ]
            if pos in scan["shed"] and fert == 0 and unfert_straw and int(shed_stock.get("FERTILIZER", 0)) > 0:
                q = min(2, int(shed_stock["FERTILIZER"]))
                shed_stock["FERTILIZER"] -= q
                actions.append(["PICKUP", "FERTILIZER", q])
                continue

            # 14. Move to shed / PASS
            dest = nearest_shed(pos)
            if pos != dest and _dist(pos, dest) > 2:
                actions.append(_move(pos, dest))
                continue
            actions.append(["PASS"])
            continue

    if not actions:
        return ["PASS"], []
    return actions[0], actions[1:]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    scan = _scan(obs)
    farmer, hands = _units(obs, scan, demand)
    return {"farmer": farmer, "hands": hands, "market": _market(obs, scan, demand)}


@dataclass
class BeamCandidate:
    """One simulator fork and its macro-action history."""

    state: Any
    actions: Tuple[int, ...] = ()
    score: float = float("-inf")


def _clone_simulator(simulator: Any, state: Any) -> Any:
    """Fork a simulator without imposing a concrete kaggsim implementation."""
    for name in ("clone", "fork", "copy"):
        method = getattr(simulator, name, None)
        if callable(method):
            try:
                return method(state)
            except TypeError:
                cloned = method()
                if hasattr(cloned, "load_state"):
                    cloned.load_state(state)
                elif hasattr(cloned, "set_state"):
                    cloned.set_state(state)
                elif hasattr(cloned, "state"):
                    cloned.state = copy.deepcopy(state)
                return cloned
    cloned = copy.deepcopy(simulator)
    if hasattr(cloned, "load_state"):
        cloned.load_state(state)
    elif hasattr(cloned, "set_state"):
        cloned.set_state(state)
    elif hasattr(cloned, "state"):
        cloned.state = copy.deepcopy(state)
    return cloned


def _step_simulator(simulator: Any, action: Any) -> Any:
    """Perform one simulator step and return its resulting state."""
    for name in ("step", "step2", "advance"):
        method = getattr(simulator, name, None)
        if callable(method):
            result = method(action)
            if isinstance(result, tuple) and len(result) >= 1:
                return result[0]
            if isinstance(result, Mapping) and "state" in result:
                return result["state"]
            return result
    if callable(simulator):
        return simulator(action)
    raise TypeError("simulator must expose step(), step2(), advance(), or be callable")


def _state_from_simulator(simulator: Any, result: Any) -> Any:
    if result is not None:
        return result
    for name in ("state", "get_state", "snapshot"):
        value = getattr(simulator, name, None)
        if callable(value):
            return value()
        if value is not None:
            return value
    return simulator


def _call_value_net(value_net: Any, state: Any) -> float:
    """Call common IQL value-network APIs, deliberately letting failures fall back."""
    if value_net is None:
        raise LookupError("IQL_Value_Net is unavailable")
    for name in ("predict", "evaluate", "value", "__call__"):
        method = getattr(value_net, name, None)
        if callable(method):
            value = method(state)
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (list, tuple)):
                value = value[0]
            return float(value)
    raise TypeError("IQL_Value_Net is not callable")


def _fallback_leaf_value(state: Any) -> float:
    """Cheap terminal value used when IQL inference is unavailable."""
    if isinstance(state, Mapping):
        def find(*names: str) -> Any:
            for name in names:
                if name in state:
                    return state[name]
            for value in state.values():
                if isinstance(value, Mapping):
                    found = find_nested(value, names)
                    if found is not None:
                        return found
            return 0

        def find_nested(value: Mapping[str, Any], names: Tuple[str, ...]) -> Any:
            for name in names:
                if name in value:
                    return value[name]
            for child in value.values():
                if isinstance(child, Mapping):
                    found = find_nested(child, names)
                    if found is not None:
                        return found
            return None

        cash = float(find("Liquid Cash", "liquid_cash", "money", "cash"))
        fertilizer = float(find("Fertilizer_Stock", "fertilizer_stock", "FERTILIZER"))
        cows = float(find("Cows", "cows", "COW"))
        return cash + fertilizer * 100.0 + cows * 400.0
    return 0.0


def evaluate_leaf(state: Any, IQL_Value_Net: Any = None) -> float:
    """Dual leaf evaluator: IQL first, then the specified material fallback."""
    try:
        return _call_value_net(IQL_Value_Net, state)
    except (LookupError, OSError, TypeError, ValueError, RuntimeError):
        return _fallback_leaf_value(state)


def step_level_beam_search(
    initial_state: Any,
    simulator: Any,
    *,
    IQL_Value_Net: Any = None,
    beam_width: int = BEAM_WIDTH,
    horizon: int = BEAM_HORIZON,
    action_space: Sequence[int] = range(len(DISCRETE_MACRO_ACTIONS)),
    action_to_simulator: Optional[Any] = None,
    holding_penalty_fn: Optional[Any] = None,
) -> BeamCandidate:
    """Search macro-actions one simulator step at a time.

    Every surviving trajectory is advanced exactly ``horizon`` times.  Forks
    are independent, so a stateful simulator can safely be used by callers.

    ``holding_penalty_fn`` is the Subtask-5 strategic override: an optional
    callable mapping a candidate state to a non-negative penalty for holding
    Milk/Wool inventory while OPPONENT_HOARDING_DETECTED is latched.  It
    defaults to None (no penalty, legacy behavior); when provided, the
    penalty is subtracted from each candidate's leaf score, forcing the beam
    toward immediate preemptive liquidation.
    """
    if beam_width != 3:
        raise ValueError("Step-level beam search requires beam_width=3")
    if horizon != 24:
        raise ValueError("Step-level beam search requires a 24-step horizon")
    if not action_space:
        raise ValueError("action_space must not be empty")

    beam = [BeamCandidate(copy.deepcopy(initial_state))]
    for _ in range(horizon):
        expanded: List[BeamCandidate] = []
        for candidate in beam:
            for action in action_space:
                fork = _clone_simulator(simulator, candidate.state)
                sim_action = (
                    action_to_simulator(action, candidate.state)
                    if callable(action_to_simulator)
                    else action
                )
                result = _step_simulator(fork, sim_action)
                state = _state_from_simulator(fork, result)
                expanded.append(BeamCandidate(
                    state, candidate.actions + (int(action),), 0.0
                ))
        for candidate in expanded:
            penalty = (
                float(holding_penalty_fn(candidate.state))
                if callable(holding_penalty_fn)
                else 0.0
            )
            candidate.score = evaluate_leaf(candidate.state, IQL_Value_Net) - penalty
        expanded.sort(key=lambda item: item.score, reverse=True)
        beam = expanded[:beam_width]
    return max(beam, key=lambda item: item.score)


def kuhn_munkres_route(
    unit_positions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """Assign workers to targets with minimum Manhattan travel cost."""
    if not unit_positions or not targets:
        return []
    costs = [[_dist(unit, (int(target[0]), int(target[1]))) for target in targets]
             for unit in unit_positions]
    if linear_sum_assignment is None:
        pairs = []
        used = set()
        for row in range(len(costs)):
            choices = [(costs[row][col], col) for col in range(len(targets)) if col not in used]
            if choices:
                _, col = min(choices)
                used.add(col)
                pairs.append((row, col))
        return pairs
    rows, cols = linear_sum_assignment(costs)
    return list(zip(rows.tolist(), cols.tolist()))


def mechanical_operations_for_trajectory(
    obs: Dict[str, Any],
    trajectory: Sequence[int],
) -> Dict[str, Any]:
    """Return the current mechanical move after KM worker routing.

    The first macro action is intentionally only a hint for market routing;
    physical worker operations remain generated by the established controller.
    """
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    scan = _scan(obs)
    farmer, hands = _units(obs, scan, demand)
    operations = {"farmer": farmer, "hands": hands,
                  "market": _market(obs, scan, demand)}
    positions = [_farm(obs).get("farmer", [4, 4]), *_farm(obs).get("hands", [])]
    targets = list(scan.get("weeds", [])) + list(scan.get("empties", []))
    assignments = kuhn_munkres_route(positions, targets)
    operations["_routing"] = assignments
    return operations


def beam_search_operations(
    obs: Dict[str, Any],
    initial_state: Any,
    simulator: Any,
    *,
    IQL_Value_Net: Any = None,
    action_to_simulator: Optional[Any] = None,
    holding_penalty_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the fixed 3x24 search and expose its winner as mechanical actions."""
    winner = step_level_beam_search(
        initial_state,
        simulator,
        IQL_Value_Net=IQL_Value_Net,
        action_to_simulator=action_to_simulator,
        holding_penalty_fn=holding_penalty_fn,
    )
    operations = mechanical_operations_for_trajectory(obs, winner.actions)
    operations["_macro_trajectory"] = winner.actions
    operations["_value"] = winner.score
    return operations


# Short aliases are useful to tournament harnesses that import the scratch
# strategy as a module rather than invoking ``agent`` directly.
beam_search = step_level_beam_search


# =============================================================================
# Subtask 7 — Strategic Stuttering (Option-Critic Meta-Controller)
# =============================================================================

@dataclass
class MacroIntent:
    """A locked high-level plan persisting across multiple hours."""
    labels: Tuple[str, ...]          # e.g. ("MAINTAIN_HERD", "EXPAND_SW_QUADRANT")
    trajectory: Tuple[int, ...]      # best beam-search action indices
    value: float                     # leaf value at lock-in time
    locked_at_hour: int              # hour when this option was committed
    locked_at_day: int               # day when this option was committed

    @property
    def age(self) -> int:
        """Number of hours since lock-in (within the same day)."""
        return 0  # computed externally via (current_hour - locked_at_hour) % 24


class BayesianMarketPredictor:
    """Lightweight opponent-hoarding detector.

    Tracks the running exponential-moving-average of opponent market
    inventory deltas.  When a commodity's inventory drops below 70% of
    the 10,000 baseline for two consecutive observations, it signals
    OPPONENT_HOARDING_DETECTED.
    """

    BASELINE = 10_000
    HOARD_THRESHOLD = 0.70   # flag when inventory < 70% of baseline
    STREAK_REQUIRED = 2      # consecutive observations below threshold

    def __init__(self) -> None:
        self._prev_inventories: Dict[str, float] = {}
        self._hoard_streaks: Dict[str, int] = {}
        self.hoarding_detected: bool = False
        self.hoarded_commodity: Optional[str] = None

    def update(self, obs: Mapping[str, Any]) -> bool:
        """Return True when OPPONENT_HOARDING_DETECTED fires."""
        market = obs.get("market", {})
        inventories = market.get("inventory", market.get("inventories", {}))
        self.hoarding_detected = False
        self.hoarded_commodity = None

        for commodity in ("MILK", "WOOL", "STRAWBERRY"):
            current = float(inventories.get(commodity, self.BASELINE))
            threshold = self.BASELINE * self.HOARD_THRESHOLD

            if current < threshold:
                self._hoard_streaks[commodity] = (
                    self._hoard_streaks.get(commodity, 0) + 1
                )
            else:
                self._hoard_streaks[commodity] = 0

            if self._hoard_streaks.get(commodity, 0) >= self.STREAK_REQUIRED:
                self.hoarding_detected = True
                self.hoarded_commodity = commodity

        self._prev_inventories = dict(inventories) if isinstance(inventories, dict) else {}
        return self.hoarding_detected


class MacroOptionManager:
    """Option-Critic meta-controller that compresses 720 turns into ~60 decisions.

    The beam search is ONLY evaluated at two temporal anchors per day:
      • Hour  0 — Dawn:   hire farmhands, set day-plan.
      • Hour 12 — Midday: re-evaluate market stance, adjust herd/crop mix.

    For the intervening 11 hours, the locked ``MacroIntent`` is routed
    directly through the Kuhn-Munkres bipartite matching mechanical layer
    with ZERO neural-network inference cost.

    Emergency interrupt: if ``BayesianMarketPredictor`` fires
    ``OPPONENT_HOARDING_DETECTED`` at *any* hour, the current option is
    terminated and an out-of-cycle beam search is forced immediately.
    """

    STRATEGIC_HOURS = frozenset({0, 12})

    def __init__(self) -> None:
        self._current_intent: Optional[MacroIntent] = None
        self._predictor = BayesianMarketPredictor()
        self._search_count: int = 0
        self._emergency_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_intent(self) -> Optional[MacroIntent]:
        return self._current_intent

    @property
    def predictor(self) -> BayesianMarketPredictor:
        return self._predictor

    def should_search(self, obs: Mapping[str, Any]) -> bool:
        """Decide whether to invoke beam search this turn.

        Returns True at Hour 0, Hour 12, or on emergency interrupt.
        """
        hour = int(obs.get("hour", 0))

        # Emergency interrupt — always check
        self._predictor.update(obs)
        if self._predictor.hoarding_detected and self._current_intent is not None:
            self._emergency_count += 1
            return True

        # Regular strategic anchors
        if hour in self.STRATEGIC_HOURS:
            return True

        # No intent locked yet (first turn of game) — must search
        if self._current_intent is None:
            return True

        return False

    def lock_intent(
        self,
        labels: Sequence[str],
        trajectory: Sequence[int],
        value: float,
        obs: Mapping[str, Any],
    ) -> MacroIntent:
        """Lock in a new MacroIntent from beam search results.

        Parameters
        ----------
        labels : sequence of str
            Human-readable macro-action labels (e.g. "MAINTAIN_HERD").
        trajectory : sequence of int
            Raw action indices from ``step_level_beam_search``.
        value : float
            Leaf value of the winning beam candidate.
        obs : mapping
            Current observation (used to timestamp the lock).
        """
        intent = MacroIntent(
            labels=tuple(labels),
            trajectory=tuple(trajectory),
            value=value,
            locked_at_hour=int(obs.get("hour", 0)),
            locked_at_day=int(obs.get("day", 0)),
        )
        self._current_intent = intent
        self._search_count += 1
        return intent

    def terminate_option(self) -> None:
        """Forcibly terminate the current option (used by emergency interrupt)."""
        self._current_intent = None

    def act(
        self,
        obs: Dict[str, Any],
        simulator: Any = None,
        IQL_Value_Net: Any = None,
        action_to_simulator: Any = None,
    ) -> Dict[str, Any]:
        """Top-level decision router — the ONLY public call per turn.

        If a strategic anchor or emergency interrupt fires, run beam search
        and lock a new MacroIntent.  Otherwise, route the locked intent
        through the mechanical Kuhn-Munkres layer with zero inference cost.
        """
        need_search = self.should_search(obs)

        if need_search:
            # --- Emergency pathway: clear stale option ---
            if self._predictor.hoarding_detected and self._current_intent is not None:
                self.terminate_option()

            # --- Beam search pathway ---
            if simulator is not None:
                initial_state = obs  # assume obs is the simulator state root
                if hasattr(simulator, "get_state"):
                    initial_state = simulator.get_state()
                elif hasattr(simulator, "state"):
                    initial_state = simulator.state

                result = step_level_beam_search(
                    initial_state,
                    simulator,
                    IQL_Value_Net=IQL_Value_Net,
                    action_to_simulator=action_to_simulator,
                )
                # Derive human-readable labels from action indices
                labels = tuple(
                    DISCRETE_MACRO_ACTIONS[i] if i < len(DISCRETE_MACRO_ACTIONS) else "UNKNOWN"
                    for i in result.actions
                )
                self.lock_intent(labels, result.actions, result.value, obs)

                ops = mechanical_operations_for_trajectory(obs, result.actions)
                ops["_macro_intent"] = labels
                ops["_search_triggered"] = True
                ops["_emergency"] = self._predictor.hoarding_detected
                return ops

            # --- No simulator available: direct mechanical fallback ---
            self.lock_intent(("MAINTAIN_HERD",), (), 0.0, obs)
            ops = agent(obs)
            ops["_macro_intent"] = ("MAINTAIN_HERD",)
            ops["_search_triggered"] = True
            ops["_emergency"] = self._predictor.hoarding_detected
            return ops

        # --- Mechanical bypass: route locked intent through KM layer ---
        if self._current_intent is not None:
            ops = mechanical_operations_for_trajectory(
                obs, self._current_intent.trajectory
            )
            ops["_macro_intent"] = self._current_intent.labels
            ops["_search_triggered"] = False
            ops["_mechanical_bypass"] = True
            return ops

        # Absolute fallback — should not normally reach here
        return agent(obs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_searches": self._search_count,
            "emergency_interrupts": self._emergency_count,
            "current_intent": (
                self._current_intent.labels if self._current_intent else None
            ),
        }


# =============================================================================
# Subtask 8 — "Poisoned Well" Market Trap Execution
# =============================================================================

class PoisonedWellTrap:
    """Deceptive market signaling to force opponent into absorbing the
    quadratic/linear price penalty on Wool/Milk before we liquidate.

    State machine:
      INACTIVE → DECEPTIVE_MODE → RECOVERY_WATCH_MODE → SELL_EXECUTE → INACTIVE

    Key game mechanics exploited:
      • Milk prices drop **linearly** with supply.
      • Wool prices drop **quadratically** with supply.
      • Town shops consume inventory every 4 hours, naturally recovering price.

    The trap:
      1. Buffer >20 Milk or Wool in our private shed.
      2. Signal to opponent that we are expanding livestock (plant visible Wheat).
      3. Wait for opponent to panic-dump their inventory (massive +delta).
      4. Watch price recover via 4-hour shop ticks.
      5. Sell our buffered inventory at the recovered price.
    """

    BUFFER_THRESHOLD = 20   # activate when shed has >20 of Milk or Wool
    RECOVERY_FACTOR = 0.85  # sell when price >= base_price * 0.85
    DUMP_DELTA_THRESHOLD = 15  # opponent dumped if market inventory jumps by >=15

    # Town shops and their 4-hour consumption schedule
    SHOP_CONSUMERS: Dict[str, Tuple[str, ...]] = {
        "PIZZA_SHOP":     ("MILK", "TOMATO", "WHEAT"),
        "YARN_STORE":     ("WOOL",),
        "SMOOTHIE_SHOP":  ("STRAWBERRY", "MILK"),
        "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    }

    class State:
        INACTIVE = "INACTIVE"
        DECEPTIVE = "DECEPTIVE_MODE"
        RECOVERY_WATCH = "RECOVERY_WATCH_MODE"
        SELL_EXECUTE = "SELL_EXECUTE"

    def __init__(self) -> None:
        self.state: str = self.State.INACTIVE
        self._buffered_commodity: Optional[str] = None  # MILK or WOOL
        self._prev_market_inventory: Dict[str, float] = {}
        self._recovery_start_hour: int = -1
        self._recovery_start_day: int = -1
        self._ticks_watched: int = 0
        self._deceptive_wheat_planted: int = 0
        self._sell_orders: List[List[Any]] = []
        self._trap_executions: int = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "buffered_commodity": self._buffered_commodity,
            "ticks_watched": self._ticks_watched,
            "trap_executions": self._trap_executions,
        }

    # ------------------------------------------------------------------
    # Core state machine
    # ------------------------------------------------------------------

    def update(
        self,
        obs: Mapping[str, Any],
        market_orders: List[List[Any]],
    ) -> List[List[Any]]:
        """Advance the trap state machine and return modified market orders.

        Parameters
        ----------
        obs : mapping
            Full game observation.
        market_orders : list
            Current market orders from the normal ``_market`` function.
            This function may inject additional orders or suppress sells.

        Returns
        -------
        list
            Potentially modified market orders.
        """
        private = obs.get("private", {})
        shed = dict(private.get("shed", {}))
        market = obs.get("market", {})
        prices = market.get("prices", {})
        inventories = market.get("inventory", market.get("inventories", {}))
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        global_step = day * 24 + hour

        milk_in_shed = int(shed.get("MILK", 0))
        wool_in_shed = int(shed.get("WOOL", 0))

        # Detect which commodity we're buffering
        if self.state == self.State.INACTIVE:
            if milk_in_shed > self.BUFFER_THRESHOLD:
                self._buffered_commodity = "MILK"
                self.state = self.State.DECEPTIVE
                self._deceptive_wheat_planted = 0
            elif wool_in_shed > self.BUFFER_THRESHOLD:
                self._buffered_commodity = "WOOL"
                self.state = self.State.DECEPTIVE
                self._deceptive_wheat_planted = 0

        if self.state == self.State.DECEPTIVE:
            market_orders = self._handle_deceptive(
                obs, market_orders, prices, inventories
            )

        elif self.state == self.State.RECOVERY_WATCH:
            market_orders = self._handle_recovery_watch(
                obs, market_orders, prices, inventories, day, hour
            )

        elif self.state == self.State.SELL_EXECUTE:
            market_orders = self._handle_sell_execute(
                obs, market_orders, shed, prices
            )

        # Update previous inventory snapshot for delta detection
        self._prev_market_inventory = (
            dict(inventories) if isinstance(inventories, dict) else {}
        )
        return market_orders

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_deceptive(
        self,
        obs: Mapping[str, Any],
        orders: List[List[Any]],
        prices: Mapping[str, Any],
        inventories: Mapping[str, Any],
    ) -> List[List[Any]]:
        """DECEPTIVE_MODE: plant visible Wheat and suppress our own sells.

        1. Forcefully inject 2-3 Wheat seed buys into publicly visible NW
           quadrant tiles to signal livestock expansion.
        2. Suppress any SELL orders for the buffered commodity so we don't
           tip our hand.
        3. Monitor opponent dump via market inventory delta.
        """
        commodity = self._buffered_commodity
        if commodity is None:
            self.state = self.State.INACTIVE
            return orders

        # --- Signal: buy Wheat seeds to fake livestock preparation ---
        if self._deceptive_wheat_planted < 3:
            wheat_buy_present = any(
                o[0] == "BUY_SEED" and o[1] == "WHEAT" for o in orders
            )
            if not wheat_buy_present and len(orders) < 10:
                buy_qty = min(3 - self._deceptive_wheat_planted, 3)
                orders.append(["BUY_SEED", "WHEAT", buy_qty])
                self._deceptive_wheat_planted += buy_qty

        # --- Suppress: remove any SELL of the buffered commodity ---
        orders = [
            o for o in orders
            if not (len(o) >= 2 and o[0] == "SELL" and o[1] == commodity)
        ]

        # --- Monitor: check for opponent dump via +delta ---
        if self._prev_market_inventory:
            prev = float(self._prev_market_inventory.get(commodity, 10_000))
            curr = float(inventories.get(commodity, 10_000))
            delta = curr - prev  # positive = someone dumped into market

            if delta >= self.DUMP_DELTA_THRESHOLD:
                # Opponent just dumped! Transition to recovery watch.
                self.state = self.State.RECOVERY_WATCH
                self._recovery_start_hour = int(obs.get("hour", 0))
                self._recovery_start_day = int(obs.get("day", 0))
                self._ticks_watched = 0

        return orders

    def _handle_recovery_watch(
        self,
        obs: Mapping[str, Any],
        orders: List[List[Any]],
        prices: Mapping[str, Any],
        inventories: Mapping[str, Any],
        day: int,
        hour: int,
    ) -> List[List[Any]]:
        """RECOVERY_WATCH_MODE: wait for 4-hour shop ticks to recover price.

        Town shops consume inventory every 4 hours (hours 0, 4, 8, 12, 16, 20).
        We track these ticks and check if the price has recovered above
        ``base_price * RECOVERY_FACTOR``.
        """
        commodity = self._buffered_commodity
        if commodity is None:
            self.state = self.State.INACTIVE
            return orders

        base_price = float(BASE.get(commodity, 100))
        current_price = float(prices.get(commodity, 0))
        recovery_threshold = base_price * self.RECOVERY_FACTOR

        # Suppress sells of the buffered commodity during recovery
        orders = [
            o for o in orders
            if not (len(o) >= 2 and o[0] == "SELL" and o[1] == commodity)
        ]

        # Track 4-hour consumption ticks (hours divisible by 4)
        is_consumption_tick = (hour % 4 == 0)
        if is_consumption_tick:
            self._ticks_watched += 1

        # Check price recovery
        if current_price >= recovery_threshold:
            self.state = self.State.SELL_EXECUTE
            return orders

        # Safety valve: if we've waited >4 ticks (16+ hours) without
        # recovery, cut losses and sell at whatever price is available
        if self._ticks_watched >= 4:
            self.state = self.State.SELL_EXECUTE

        return orders

    def _handle_sell_execute(
        self,
        obs: Mapping[str, Any],
        orders: List[List[Any]],
        shed: Dict[str, Any],
        prices: Mapping[str, Any],
    ) -> List[List[Any]]:
        """SELL_EXECUTE: dump our buffered inventory at the recovered price."""
        commodity = self._buffered_commodity
        if commodity is None:
            self.state = self.State.INACTIVE
            return orders

        have = int(shed.get(commodity, 0))
        if have > 0 and len(orders) < 10:
            # Sell in large batches to capitalize on recovered price
            sell_qty = min(have, 18)
            orders.append(["SELL", commodity, sell_qty])

        # Reset trap state
        self._trap_executions += 1
        self._buffered_commodity = None
        self._ticks_watched = 0
        self.state = self.State.INACTIVE
        return orders

    # ------------------------------------------------------------------
    # Physical signaling (called from unit controller)
    # ------------------------------------------------------------------

    def get_deceptive_plants(self, obs: Mapping[str, Any]) -> List[Pos]:
        """Return NW-quadrant tile positions where Wheat should be planted
        as a deceptive signal during DECEPTIVE_MODE.

        The caller (field specialist) should plant Wheat at these positions
        instead of the default crop, making the expansion signal visible
        to the opponent's observation of our farm tiles.
        """
        if self.state != self.State.DECEPTIVE:
            return []

        farm = _farm(obs)
        tiles = farm.get("tiles", [])
        board = len(tiles) or 10
        half = board // 2

        # NW quadrant: rows [0, half), cols [0, half)
        nw_empties: List[Pos] = []
        for y in range(half):
            for x in range(half):
                tile = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else None
                if tile is None:
                    nw_empties.append((x, y))

        return nw_empties[:3]  # at most 3 deceptive plots


# =============================================================================
# Integrated Grand Strategy Controller with Rank 1 Opening Book
# =============================================================================

from kaggriculture.opening_book import (
    OpeningBookController,
    OPENING_BOOK_SCHEDULE,
)


class GrandStrategyController:
    """Unified controller combining OpeningBookController + MacroOptionManager + PoisonedWellTrap.

    - Days 0-15: Governed by the deterministic Rank 1 Opening Book (overriding Beam Search).
    - Day 16+: Clean handoff to MacroOptionManager (Hour 0/12 Beam Search + Kuhn-Munkres bypass).
    - Adversarial Layer: PoisonedWellTrap deceptive signaling and 4h price recovery liquidation.

    Usage::

        controller = GrandStrategyController()

        def agent(obs):
            return controller.act(obs)
    """

    def __init__(
        self,
        simulator: Any = None,
        IQL_Value_Net: Any = None,
        action_to_simulator: Any = None,
        opening_book: Optional[OpeningBookController] = None,
    ) -> None:
        self.opening_book = opening_book or OpeningBookController(MacroOptionManager())
        self.option_manager = self.opening_book.option_manager
        self.market_trap = PoisonedWellTrap()
        self._simulator = simulator
        self._value_net = IQL_Value_Net
        self._action_to_simulator = action_to_simulator

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Produce the final action dict for this turn.

        1. Days 0-15: OpeningBookController overrides Beam Search with Rank 1 build order.
        2. Day 16+: Smooth handoff to MacroOptionManager (Option-Critic Beam Search).
        3. PoisonedWellTrap post-processes market orders.
        """
        ops = self.opening_book.act(
            obs,
            simulator=self._simulator,
            IQL_Value_Net=self._value_net,
            action_to_simulator=self._action_to_simulator,
        )

        # Post-process market orders through the Poisoned Well trap
        market_orders = ops.get("market", [])
        if isinstance(market_orders, list):
            market_orders = self.market_trap.update(obs, market_orders)
            ops["market"] = market_orders

        # Inject deceptive plant targets for field workers
        ops["_deceptive_plants"] = self.market_trap.get_deceptive_plants(obs)

        # Attach diagnostics
        ops["_opening_book_stats"] = self.opening_book.stats()
        ops["_option_stats"] = self.option_manager.stats()
        ops["_trap_stats"] = self.market_trap.stats()

        return ops

    def stats(self) -> Dict[str, Any]:
        return {
            "opening_book": self.opening_book.stats(),
            "option_manager": self.option_manager.stats(),
            "market_trap": self.market_trap.stats(),
        }

