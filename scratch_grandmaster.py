"""scratch_grandmaster.py — Apex Grandmaster Policy for Kaggriculture

Architectural Principles:
1. Day 0 Animal Turbo Kickstart:
   - 3 Cows + 2 Sheep + 8 Wheat on Day 0.
   - Concurrently builds 5 pastures and places all 5 animals on Day 0.
   - 0 Melons on Day 0 (avoids 10-day tile freeze and negative ROI).
   - 6 Hands hired on Day 0 (cost $20).
2. Continuous Fertilizer & Commodity Cash Engine:
   - Collects fertilizer from animals and sells immediately at $100/unit every turn.
   - Generates $1,250+/day starting on Day 1.
3. Rapid Multi-Quadrant Expansion:
   - Day 4-5: NE quadrant ($1,000) -> 4 more pastures + 16 Strawberries.
   - Day 7-9: SW quadrant ($2,000) -> 4-6 more pastures + up to 32-36 Strawberries.
   - 14-16 total animals (6 Cows + 8-10 Sheep) + 32-36 Strawberries + 8 Wheat plots.
4. Optimal Fibonacci Labor (Cost-Controlled):
   - Day 0-3: 6 hands ($20/day)
   - Day 4-7: 7 hands ($33/day)
   - Day 8+: 8 hands ($54/day) — never over-hires to avoid wage bleed.
5. High-Frequency Market Clearing:
   - Sells Fertilizer, Milk, Wool, and Strawberries every turn in efficient batch sizes.
   - Terminal liquidation on Days 27-29.
6. Safe Quadrant Navigation:
   - Routes via (4,4) when crossing between SW and NE to avoid locked SE (5,5).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

OPERATING_RESERVE = 30


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
        # Keep 10 feed for cows and sheep
        return max(0, have - 10)

    # Fertilizer: sell immediately for $100 cash!
    if item == "FERTILIZER":
        return min(have, 6) if price >= 20 else (min(have, 4) if price >= 5 else 0)

    if item == "MELON":
        return min(have, 8)

    # Strawberries: sell in large chunks so shed doesn't clog
    if item == "STRAWBERRY":
        if price >= 100:
            return min(have, 8)
        if price >= 50:
            return min(have, 6)
        return min(have, 4)

    # Milk & Wool: sell steadily
    if item in ("MILK", "WOOL"):
        if price >= 100:
            return min(have, 6)
        if price >= 30:
            return min(have, 4)
        return min(have, 2)

    return min(have, 4)


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
    opp_cows = sum(1 for row in opp_tiles for t in row if isinstance(t, dict) and t.get("animal") == "COW")
    opp_sheep = sum(1 for row in opp_tiles for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")

    # === PRIORITY 1: Feed Wheat (Survival) ===
    if day >= 28:
        wheat_need = 0
    else:
        wheat_floor = 10 if day < 3 else (live + 4)
        wheat_need = max(0, wheat_floor - wheat_have)
    if wheat_need and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 10)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === PRIORITY 2: Farm Hands ===
    # Cost-controlled Fibonacci hiring: 6 early, 7 mid, 8 late
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
        if money < cost + 5:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    # === PRIORITY 3: Day 0 Opening Buys (3 Cows + 2 Sheep + 8 Wheat Seeds) ===
    animals_in_shed = int(shed.get("SHEEP", 0)) + int(shed.get("COW", 0))
    total_owned_cows = scan["counts"]["COW"] + int(shed.get("COW", 0))
    total_owned_sheep = scan["counts"]["SHEEP"] + int(shed.get("SHEEP", 0))
    total_herd = total_owned_cows + total_owned_sheep

    if day == 0 and len(orders) < 9:
        # Buy up to 3 Cows
        if total_owned_cows < 3 and money >= 400 + OPERATING_RESERVE:
            buy_c = min(3 - total_owned_cows, int((money - OPERATING_RESERVE) // 400))
            if buy_c > 0:
                orders.append(["BUY_ANIMAL", "COW", buy_c])
                money -= buy_c * 400
                total_owned_cows += buy_c
                total_herd += buy_c

        # Buy up to 2 Sheep
        if total_owned_sheep < 2 and money >= 500 + OPERATING_RESERVE:
            buy_s = min(2 - total_owned_sheep, int((money - OPERATING_RESERVE) // 500))
            if buy_s > 0:
                orders.append(["BUY_ANIMAL", "SHEEP", buy_s])
                money -= buy_s * 500
                total_owned_sheep += buy_s
                total_herd += buy_s

        # Buy 8 Wheat seeds
        cur_w_seeds = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
        if cur_w_seeds < 8 and money >= 80:
            buy_w = min(8 - cur_w_seeds, int((money - OPERATING_RESERVE) // 10), 8)
            if buy_w > 0:
                orders.append(["BUY_SEED", "WHEAT", buy_w])
                money -= buy_w * 10
                seeds["WHEAT"] = int(seeds.get("WHEAT", 0)) + buy_w

    # === PRIORITY 4: Land Expansion ===
    # Day 4-8: NE quadrant ($1,000)
    # Day 7-14: SW quadrant ($2,000)
    if len(quadrants) == 1 and 4 <= day <= 8 and money >= 1000 + OPERATING_RESERVE and len(orders) < 10:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000
        quadrants.append("NE")
    elif len(quadrants) == 2 and 7 <= day <= 15 and money >= 2000 + OPERATING_RESERVE and len(orders) < 10:
        orders.append(["BUY_LAND", "SW"])
        money -= 2000
        quadrants.append("SW")

    # === PRIORITY 5: Midgame Animal Scaling ===
    if 1 <= day <= 22 and len(orders) < 9:
        if len(quadrants) == 1:
            herd_target = 5
        elif len(quadrants) == 2:
            herd_target = 10
        else:
            herd_target = 15

        max_cows = 4 if (opp_cows >= 4 or milk_price < 130) else 6
        if total_herd < herd_target:
            if total_owned_cows < max_cows and milk_price >= 135:
                buy_sp = "COW"
            else:
                buy_sp = "SHEEP"

            spec = SPECIES[buy_sp]
            free_pastures = max(0, len(scan["empty"]["PASTURE"]) - animals_in_shed)
            land_reserve = 1000 if (len(quadrants) == 1 and day <= 5) else (2000 if (len(quadrants) == 2 and day <= 8) else 0)
            avail_money = money - OPERATING_RESERVE - land_reserve
            affordable = max(0, int(avail_money // spec["cost"]))
            # Allow buying if free pasture available or empty tiles available to build pasture
            buildable_pastures = max(0, len(scan["empties"]) - (int(seeds.get("STRAWBERRY", 0)) + int(seeds.get("WHEAT", 0))))
            allowed = min(affordable, free_pastures + buildable_pastures, herd_target - total_herd, 2)
            if allowed > 0 and (wheat_have >= live + allowed or day < 3):
                orders.append(["BUY_ANIMAL", buy_sp, allowed])
                money -= spec["cost"] * allowed
                total_herd += allowed
                if buy_sp == "COW":
                    total_owned_cows += allowed
                else:
                    total_owned_sheep += allowed

    # === PRIORITY 6: Strawberry Cash Crop (Days 3-22) ===
    cur_straw = int(seeds.get("STRAWBERRY", 0)) + int(scan["crops"].get("STRAWBERRY", 0))
    if len(quadrants) == 1:
        target_straw = 4
    elif len(quadrants) == 2:
        target_straw = 18
    else:
        target_straw = 34

    if 3 <= day <= 22 and len(orders) < 9:
        need_straw = max(0, target_straw - cur_straw)
        straw_price = 100
        land_reserve = 1000 if (len(quadrants) == 1 and day <= 5) else (2000 if (len(quadrants) == 2 and day <= 8) else 0)
        feed_reserve = max(0, live * 2 - wheat_have) * max(wheat_price, 30.0) + OPERATING_RESERVE
        afford_straw = max(0, int((money - feed_reserve - land_reserve) // straw_price))
        buy_straw = min(need_straw, afford_straw, 6)
        if buy_straw > 0:
            orders.append(["BUY_SEED", "STRAWBERRY", buy_straw])
            money -= buy_straw * straw_price
            cur_straw += buy_straw
            seeds["STRAWBERRY"] = int(seeds.get("STRAWBERRY", 0)) + buy_straw

    # Replant wheat for feed
    growing_wheat = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    target_wheat = 8 if len(quadrants) <= 2 else 10
    if 2 <= day <= 24 and growing_wheat < target_wheat and len(orders) < 9:
        buy_w = min(target_wheat - growing_wheat, int((money - OPERATING_RESERVE) // 10), 4)
        if buy_w > 0:
            orders.append(["BUY_SEED", "WHEAT", buy_w])
            money -= buy_w * 10
            growing_wheat += buy_w
            seeds["WHEAT"] = int(seeds.get("WHEAT", 0)) + buy_w

    # === PRIORITY 7: High-Frequency Sells ===
    ranked = sorted(shed.items(), key=lambda kv: 0 if kv[0] == "FERTILIZER" else (1 if kv[0] == "STRAWBERRY" else 2))
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
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 16

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]
    plant_at = {pos: tile for pos, tile in plants}
    unfed_left = sum(1 for _p, tile in animals if not tile.get("fed_today", False))
    holders = sum(1 for inv in inventories if int((inv or {}).get("WHEAT", 0)) > 0)

    # Track how many pastures are scheduled to be built this step
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

        # === Priority 1: Act on tile underfoot (Full unified service) ===
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

        danger_plants = [
            p for p, tile in plants
            if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1
        ]
        if danger_plants:
            target = take_nearest(pos, danger_plants)
            if target is not None:
                actions.append(["WATER"] if pos == target else _move(pos, target))
                continue

        # === Priority 2: Feed hungry animals ===
        if unfed_left and wheat > 0:
            target = take_nearest(pos, [p for p, tile in animals if not tile.get("fed_today", False)])
            if target is not None:
                actions.append(["FEED"] if pos == target else _move(pos, target))
                continue

        # === Priority 3: Fetch wheat to feed ===
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
        if carrying_animal and carried_sp is not None:
            sp_struct = SPECIES[carried_sp]["structure"]
            target = take_nearest(pos, list(scan["empty"][sp_struct]))
            if target is not None:
                actions.append(["PLACE", carried_sp, 1] if pos == target else _move(pos, target))
                continue

        # === Priority 5: Proactive animal fetching from shed ===
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

        # === Priority 6: Fast Shed Drop (Valuable goods: Fertilizer, Milk, Wool, Strawberry) ===
        carried_valuable = (
            int(inv.get("MILK", 0)) + int(inv.get("WOOL", 0)) +
            int(inv.get("FERTILIZER", 0)) + int(inv.get("STRAWBERRY", 0)) + int(inv.get("MELON", 0))
        )
        if carried_valuable >= 2 or (carried_valuable >= 1 and here is None and plant_here is None):
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 7: Unified Animal Service (Care, Harvest, Collect Fertilizer) ===
        animal_service = [
            p for p, tile in animals
            if not tile.get("cared_today", False) or int(tile.get("yield_units", 0)) > 0 or tile.get("fertilizer_available", False)
        ]
        if animal_service and wheat <= 0:
            target = take_nearest(pos, animal_service)
            if target is not None:
                actions.append(_move(pos, target))
                continue

        # === Priority 8: Harvest ripe crops ===
        if day >= 2:
            ripe = [p for p, tile in plants if _crop_ripe(tile)]
            target = take_nearest(pos, ripe)
            if target is not None:
                actions.append(["HARVEST"] if pos == target else _move(pos, target))
                continue

        # === Priority 9: Weed Clearing (Keep farm clean and prevent weed spread) ===
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # === Priority 10: Plant Strawberry (Top Cash Crop) ===
        avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
        if avail_straw > 0 and scan["empties"]:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_straw += 1
                actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                continue

        # === Priority 11: BUILD PASTURES (Concurrent & Reliable) ===
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

        # === Priority 12: Routine watering of all thirsty plants ===
        thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
        target = take_nearest(pos, thirsty)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # === Priority 15: Plant Wheat (Feed & Weed Prevention) ===
        avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
        if avail_wheat > 0 and scan["empties"]:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_wheat += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        # === Priority 16: Drop carried goods at shed ===
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 17: Move toward shed ===
        if pos not in scan["shed"]:
            dest = nearest_shed(pos)
            if _dist(pos, dest) > 1:
                actions.append(_move(pos, dest))
                continue

        actions.append(["PASS"])

    if not actions:
        return ["PASS"], []
    return actions[0], actions[1:]


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    demand = _demand(obs.get("town", {}).get("unlocked_shops", []) or [])
    scan = _scan(obs)
    farmer, hands = _units(obs, scan, demand)
    return {"farmer": farmer, "hands": hands, "market": _market(obs, scan, demand)}
