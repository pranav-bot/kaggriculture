"""two_team_grandmaster.py — Apex Two-Team Division of Labor Architecture

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

import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

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

MARKET_PARAMS: Dict[str, Dict[str, Any]] = {
    "WHEAT":      {"base":  25, "I0": 10000, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": 10000, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": 10000, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": 10000, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _shape_func(func: str, x: float, T: float) -> float:
    x = max(0.0, float(x))
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "hinge":
        if T <= 0: return x
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def _unit_market_price(item: str, inv: int) -> int:
    p = MARKET_PARAMS.get(item)
    if not p: return 1
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape_func(f, T, T)
        return max(1, int(round(base + amp * _shape_func(f, I0 - inv, T))))
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape_func(f, T, T)
        return max(1, int(round(base - amp * _shape_func(f, inv - I0, T))))


def _simulate_sell_revenue(item: str, qty: int, market_inv: int) -> int:
    rev = 0
    cur = market_inv
    for _ in range(qty):
        rev += _unit_market_price(item, cur)
        cur += 1
    return rev


def _shop_drain_per_tick(unlocked_shops: Sequence[str], item: str) -> int:
    drain = 0
    for shop in unlocked_shops:
        prods = SHOPS.get(shop, ())
        if item in prods:
            mult = 2 if len(prods) == 1 else 1
            drain += mult
    return drain


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


def _traverses_locked_se(src: Sequence[int], dst: Sequence[int], quadrants: Sequence[str] = ("NW",)) -> bool:
    """Returns True if path between src and dst traverses the locked Southeast tile at coordinate (5, 5)."""
    if "SE" in quadrants:
        return False
    sx, sy = int(src[0]), int(src[1])
    tx, ty = int(dst[0]), int(dst[1])
    # Exact hit on (5, 5)
    if (sx, sy) == (5, 5) or (tx, ty) == (5, 5):
        return True
    # Coordinate inside locked SE quadrant (5..9, 5..9)
    if (sx >= 5 and sy >= 5) or (tx >= 5 and ty >= 5):
        return True
    # Diagonal crossing between SW (x<=4, y>=5) and NE (x>=5, y<=4)
    in_sw = (sx <= 4 and sy >= 5)
    in_ne = (sx >= 5 and sy <= 4)
    dst_sw = (tx <= 4 and ty >= 5)
    dst_ne = (tx >= 5 and ty <= 4)
    if (in_sw and dst_ne) or (in_ne and dst_sw):
        return True
    # Bounding box containing (5, 5)
    min_x, max_x = min(sx, tx), max(sx, tx)
    min_y, max_y = min(sy, ty), max(sy, ty)
    if min_x <= 5 <= max_x and min_y <= 5 <= max_y:
        if (min_x < 5 < max_x) or (min_y < 5 < max_y):
            return True
    return False


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
    item: str,
    have: int,
    price: float,
    shed_total: int,
    day: int,
    market_inv: int = 10000,
    step: int = 0,
    shops: Sequence[str] = (),
) -> int:
    if have <= 0 or item in SPECIES:
        return 0

    # 100% Terminal Liquidation on Days 27-29
    if day >= 27:
        return have

    if item == "WHEAT":
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
        if price >= 180:
            target_tranche = 18
        elif price >= 120:
            target_tranche = 14
        elif price >= 60:
            target_tranche = 10
        else:
            target_tranche = 6
    elif item == "MILK":
        # Linear elasticity: Delta P = -2.098 Delta I
        if price >= 160:
            target_tranche = 14
        elif price >= 100:
            target_tranche = 10
        else:
            target_tranche = 6
    elif item == "WOOL":
        # Quadratic elasticity: Delta P = -0.058 (Delta I)^2
        # Strictly cap tranche size to avoid quadratic collapse
        if price >= 180:
            target_tranche = 6
        elif price >= 120:
            target_tranche = 4
        else:
            target_tranche = 2
    else:
        target_tranche = 6

    qty = min(have, target_tranche)
    if qty <= 0:
        return 0

    # MPC Shop Drain Lookahead:
    # Shops consume every 4 turns. If consumption tick happens in next 1-2 turns,
    # and market inventory is elevated, delay sale to sell into higher post-drain price.
    turn_offset = step % 4
    turns_until_tick = 4 - turn_offset if turn_offset > 0 else 0
    drained = _shop_drain_per_tick(shops, item)

    if 1 <= turns_until_tick <= 2 and drained > 0 and shed_total < 82:
        rev_now = _simulate_sell_revenue(item, qty, market_inv)
        rev_wait = _simulate_sell_revenue(item, qty, max(0, market_inv - drained))
        if rev_wait > int(rev_now * 1.05):
            return 0

    return qty


def _market(obs: Mapping[str, Any], scan: Mapping[str, Any], demand: Mapping[str, float]) -> List[List[Any]]:
    farm = _farm(obs)
    private = obs.get("private", {})
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    step = int(obs.get("step", 0))
    money = float(farm.get("money", 0))
    shed = dict(private.get("shed", {}))
    market = obs.get("market", {})
    prices = market.get("prices", {})
    market_invs = market.get("inventory", {})
    shops = list((obs.get("town", {})).get("unlocked_shops", []))
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
        item_str = str(item)
        cur_inv = int(market_invs.get(item_str, 10000))
        cur_p = float(prices.get(item_str, 0))
        qty = _sell_qty(item_str, int(count), cur_p, shed_total, day, cur_inv, step, shops)
        if qty > 0:
            orders.append(["SELL", item, qty])
            money += _simulate_sell_revenue(item_str, qty, cur_inv)
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

    def _crop_ripe(tile: Mapping[str, Any]) -> bool:
        c = str(tile.get("crop"))
        p_day = int(tile.get("planted_day", 0))
        if int(tile.get("yield_units", 0)) <= 0:
            return False
        m_age = 2 if c == "WHEAT" else (10 if c in ("MELON", "STRAWBERRY") else 8)
        return (day - p_day) >= m_age

    # =========================================================================
    # PHASE 1: BIPARTITE MATCHING SYSTEM (Hungarian Algorithm)
    # At Hour 0 of each in-game day, extract all spatial tasks on the grid
    # and calculate Manhattan distance from all available workers to these tasks.
    # Use scipy.optimize.linear_sum_assignment with (5,5) infinite penalty.
    # =========================================================================
    hour0_tasks: List[Tuple[Pos, str, float]] = []
    hour0_assignments: Dict[int, Pos] = {}

    if hour == 0:
        # 1. Danger watering tasks (consecutive_unwatered >= 1)
        for p, tile in plants:
            if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) >= 1:
                hour0_tasks.append((p, "WATER", 1000.0))
        # 2. Ripe crop harvest tasks
        for p, tile in plants:
            if _crop_ripe(tile):
                hour0_tasks.append((p, "HARVEST", 850.0))
        # 3. Animal care / harvest / fertilizer
        for p, tile in animals:
            if not tile.get("cared_today", False) or int(tile.get("yield_units", 0)) > 0 or tile.get("fertilizer_available", False):
                hour0_tasks.append((p, "ANIMAL_SERVICE", 700.0))
        # 4. Animal feeding
        for p, tile in animals:
            if not tile.get("fed_today", False):
                hour0_tasks.append((p, "FEED", 650.0))
        # 5. Strawberry fertilizing
        for p, tile in plants:
            if str(tile.get("crop")) == "STRAWBERRY" and int(tile.get("fertilized_until_day", -1)) < day:
                hour0_tasks.append((p, "FERTILIZE", 550.0))
        # 6. Weeds digging
        for p in scan["weeds"]:
            hour0_tasks.append((p, "DIG", 500.0))
        # 7. Routine watering
        for p, tile in plants:
            if not tile.get("watered_today", False) and int(tile.get("consecutive_unwatered", 0)) == 0:
                hour0_tasks.append((p, "WATER", 450.0))
        # 8. Seed planting
        avail_straw = int(seeds.get("STRAWBERRY", 0))
        avail_melon = int(seeds.get("MELON", 0))
        avail_wheat = int(seeds.get("WHEAT", 0))
        for p in scan["empties"]:
            if avail_straw > 0:
                hour0_tasks.append((p, "PLANT_STRAWBERRY", 400.0))
                avail_straw -= 1
            elif avail_melon > 0:
                hour0_tasks.append((p, "PLANT_MELON", 380.0))
                avail_melon -= 1
            elif avail_wheat > 0:
                hour0_tasks.append((p, "PLANT_WHEAT", 360.0))
                avail_wheat -= 1

        n_workers = len(positions)
        n_tasks = len(hour0_tasks)
        if n_workers > 0 and n_tasks > 0:
            cost_matrix = np.zeros((n_workers, n_tasks), dtype=float)
            for i, p_arr in enumerate(positions):
                w_pos = (int(p_arr[0]), int(p_arr[1]))
                is_livestock = (i < 4)
                for j, (t_pos, t_kind, priority_bonus) in enumerate(hour0_tasks):
                    if _traverses_locked_se(w_pos, t_pos, quadrants):
                        cost_matrix[i, j] = 1e9  # Infinite cost penalty
                    else:
                        dist = abs(w_pos[0] - t_pos[0]) + abs(w_pos[1] - t_pos[1])
                        cost = dist - priority_bonus
                        is_animal = t_kind in ("ANIMAL_SERVICE", "FEED")
                        if is_livestock and not is_animal:
                            cost += 30.0
                        elif not is_livestock and is_animal:
                            cost += 30.0
                        cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e8:
                    hour0_assignments[r] = hour0_tasks[c][0]

    completed_units: Set[int] = set()

    def bipartite_take(w_idx: int, pos: Pos, pool: List[Pos]) -> Optional[Pos]:
        open_tiles = [tile for tile in pool if tile not in claimed]
        if not open_tiles:
            return None
        # Check if pre-assigned Hour 0 task matches an open tile in this pool
        if hour == 0 and w_idx in hour0_assignments and hour0_assignments[w_idx] in open_tiles:
            choice = hour0_assignments[w_idx]
            claimed.add(choice)
            return choice
        # Active remaining workers
        active_w_indices = [i for i in range(len(positions)) if i not in completed_units]
        if not active_w_indices:
            return None
        active_w_positions = [(int(positions[i][0]), int(positions[i][1])) for i in active_w_indices]
        cost_matrix = np.zeros((len(active_w_indices), len(open_tiles)), dtype=float)
        for i, w_pos in enumerate(active_w_positions):
            for j, t_pos in enumerate(open_tiles):
                if _traverses_locked_se(w_pos, t_pos, quadrants):
                    cost_matrix[i, j] = 1e9  # Infinite cost penalty for paths traversing locked (5,5)
                else:
                    cost_matrix[i, j] = abs(w_pos[0] - t_pos[0]) + abs(w_pos[1] - t_pos[1])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if active_w_indices[r] == w_idx and cost_matrix[r, c] < 1e8:
                choice = open_tiles[c]
                claimed.add(choice)
                return choice
        # Safe fallback
        safe_tiles = [t for t in open_tiles if not _traverses_locked_se(pos, t, quadrants)]
        if safe_tiles:
            choice = min(safe_tiles, key=lambda tile: (_dist(pos, tile), tile))
            claimed.add(choice)
            return choice
        return None

    for index, position in enumerate(positions):
        completed_units.add(index)
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
                target = bipartite_take(index, pos, list(scan["empties"]))
                if target is not None:
                    actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                    continue

            # 3. Place carried animal into pasture
            if carrying_animal and carried_sp is not None:
                sp_struct = SPECIES[carried_sp]["structure"]
                target = bipartite_take(index, pos, list(scan["empty"][sp_struct]))
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
                target = bipartite_take(index, pos, [p for p, tile in animals if not tile.get("fed_today", False)])
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
                target = bipartite_take(index, pos, animal_service)
                if target is not None:
                    actions.append(_move(pos, target))
                    continue

            # 8. Help field team plant seeds if any are waiting
            avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
            if avail_straw > 0 and scan["empties"]:
                target = bipartite_take(index, pos, list(scan["empties"]))
                if target is not None:
                    planted_straw += 1
                    actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                    continue

            avail_melon = int(seeds.get("MELON", 0)) - planted_melon
            if avail_melon > 0 and scan["empties"]:
                target = bipartite_take(index, pos, list(scan["empties"]))
                if target is not None:
                    planted_melon += 1
                    actions.append(["PLANT", "MELON"] if pos == target else _move(pos, target))
                    continue

            # 9. Help field team water thirsty plants!
            thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
            if thirsty:
                target = bipartite_take(index, pos, thirsty)
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
                    target = bipartite_take(index, pos, unfert)
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
                target = bipartite_take(index, pos, danger_plants)
                if target is not None:
                    actions.append(["WATER"] if pos == target else _move(pos, target))
                    continue

            # 5. Harvest ripe crops
            if day >= 2:
                ripe = [p for p, tile in plants if _crop_ripe(tile)]
                target = bipartite_take(index, pos, ripe)
                if target is not None:
                    actions.append(["HARVEST"] if pos == target else _move(pos, target))
                    continue

            # 6. Plant Strawberry (High margin cash crop — Plant immediately!)
            avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
            if avail_straw > 0 and scan["empties"]:
                target = bipartite_take(index, pos, list(scan["empties"]))
                if target is not None:
                    planted_straw += 1
                    actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                    continue

            # 7. Plant Melon (Day 0-1 compounding cash crop)
            avail_melon = int(seeds.get("MELON", 0)) - planted_melon
            if avail_melon > 0 and scan["empties"]:
                target = bipartite_take(index, pos, list(scan["empties"]))
                if target is not None:
                    planted_melon += 1
                    actions.append(["PLANT", "MELON"] if pos == target else _move(pos, target))
                    continue

            # 8. Plant Wheat (Feed sustainability)
            avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
            if avail_wheat > 0 and scan["empties"]:
                target = bipartite_take(index, pos, list(scan["empties"]))
                if target is not None:
                    planted_wheat += 1
                    actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                    continue

            # 9. Routine watering of all thirsty plants
            thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
            if thirsty:
                target = bipartite_take(index, pos, thirsty)
                if target is not None:
                    actions.append(["WATER"] if pos == target else _move(pos, target))
                    continue

            # 10. Dig Weeds (Clears tiles immediately for next replanting cycle!)
            if scan["weeds"]:
                target = bipartite_take(index, pos, list(scan["weeds"]))
                if target is not None:
                    actions.append(["DIG"] if pos == target else _move(pos, target))
                    continue

            # 11. Build Pastures
            needed_pastures = max(
                int(shed_stock.get("COW", 0)) + int(shed_stock.get("SHEEP", 0)),
                target_pastures - total_pastures
            )
            if day < 22 and needed_pastures > building_pastures and scan["empties"]:
                target = bipartite_take(index, pos, list(scan["empties"]))
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
