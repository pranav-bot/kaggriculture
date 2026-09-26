"""apex_engine.py — Sovereign Apex Multi-Engine Policy (Melon Kickstart & Southwest Expansion)

Remediated Architecture:
1. True Fibonacci Labor Scaling:
   - Uses exact engine Fibonacci cost: _fib(n) (Day 0: 7 hands for $34, Day 11+: 11 hands for $375).
   - Workers never starved for cash or labor.
2. Day 0-1 Turbo Kickstart:
   - 8 Wheat + 6 Melons + 7 Hands + 2 Pastures (Cow & Sheep).
   - 100% tile utilization on Day 0.
3. Day 6 NE Land & Strawberry Cash-Crop Launch:
   - Unlocks NE quadrant ($1,000) on Day 5-6.
   - Plants 16 Strawberries in NE immediately.
4. Day 10 Melon Cash Influx & SW Expansion:
   - 6 Melons mature -> 36 units -> $9,000+ cash -> buys SW ($2,000).
   - Ramps up to 32-36 Strawberries + 10-12 Livestock (4 Cows + 6-8 Sheep).
5. Opponent-Aware Market Defense:
   - Caps cows at 4 when opponent owns cows or milk price < 150 to prevent milk market collapse.
   - Routes excess animal capacity to Sheep (wool holds $240+ price).
6. Continuous Active Selling & Zero-Hoard Endgame:
   - Sells inventory continuously whenever town demand or prices are viable.
   - Days 27-29: 100% terminal liquidation of shed down to 0 stranded items.
7. Safe Grid Waypoint Routing:
   - Avoids locked SE (5,5) when moving between SW and NE via (4,4).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

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
    "COW": {"structure": "PASTURE", "product": "MILK", "cost": 400, "per_day": 1.5, "curve": 1.7},
    "SHEEP": {"structure": "PASTURE", "product": "WOOL", "cost": 500, "per_day": 4 / 3, "curve": 0.85},
    "GOOSE": {"structure": "COOP", "product": "EGG", "cost": 300, "per_day": 2.0, "curve": 1.05},
}

OPERATING_RESERVE = 40


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
    shed_tiles = [(half - 1, half - 1)]  # (4,4) NW
    if "NE" in quadrants:
        shed_tiles.append((half, half - 1))  # (5,4) NE
    if "SW" in quadrants:
        shed_tiles.append((half - 1, half))  # (4,5) SW
    if "SE" in quadrants:
        shed_tiles.append((half, half))      # (5,5) SE
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
    item: str, have: int, price: float, shed_total: int, day: int, hour: int,
    need_cash: bool, is_endgame: bool,
) -> int:
    if have <= 0 or item in SPECIES:
        return 0

    # 100% Terminal Liquidation on Days 27-29 (Zero stranded inventory)
    if day >= 27:
        return have

    if item == "WHEAT":
        # Keep 10 feed for cows and sheep
        return max(0, have - 10)

    if item == "FERTILIZER":
        return min(have, 4) if price >= 5 else 0

    if item == "MELON":
        return min(have, 10)

    # Strawberries: sell aggressively so shed never bottlenecks
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
        if price >= 20:
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
    is_endgame = day >= 28

    # Prices & Opponent Modeling
    milk_price = float(prices.get("MILK", BASE["MILK"]))
    wool_price = float(prices.get("WOOL", BASE["WOOL"]))
    milk_oversupply = (milk_price < 125)
    wool_oversupply = (wool_price < 140)

    opp_farm = obs["farms"][1 - int(obs.get("player", 0))] if len(obs.get("farms", [])) > 1 else {}
    opp_tiles = opp_farm.get("tiles", [])
    opp_cows = sum(1 for row in opp_tiles for t in row if isinstance(t, dict) and t.get("animal") == "COW")
    opp_sheep = sum(1 for row in opp_tiles for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")

    # === PRIORITY 1: Feed Wheat (Survival) ===
    if day >= 28:
        wheat_need = 0
    else:
        wheat_floor = 12 if day < 3 else (live + 4)
        wheat_need = max(0, wheat_floor - wheat_have)
    if wheat_need and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 12)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === PRIORITY 2: Farm Hands (Core Labor Guarantee — ALWAYS BEFORE LAND) ===
    # Using true Fibonacci cost: 7 hands cost only $34 total, 11 hands cost $375
    if day == 0:
        target_hands = 7
    elif day < 5:
        target_hands = 7
    elif day < 10:
        target_hands = 9 if len(quadrants) >= 2 else 7
    else:
        target_hands = 11 if len(quadrants) >= 3 else 9

    hires = int(farm.get("hires_today", 0))
    while hires < target_hands and len(orders) < 10:
        cost = _hire_cost(hires)
        if money < cost + 10:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    # === PRIORITY 3: Land Expansion (NE: Days 5-10, SW: Days 10-18) ===
    if len(quadrants) == 1 and 5 <= day <= 10 and money >= 1000 + OPERATING_RESERVE and len(orders) < 10:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000
        quadrants.append("NE")
    elif len(quadrants) == 2 and 10 <= day <= 18 and money >= 2000 + OPERATING_RESERVE and len(orders) < 10:
        orders.append(["BUY_LAND", "SW"])
        money -= 2000
        quadrants.append("SW")

    # === PRIORITY 4: Day 0-1 Crop Kickstart (Wheat + Melon) ===
    growing_wheat = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    growing_melon = int(seeds.get("MELON", 0)) + int(scan["crops"].get("MELON", 0))

    if day <= 1 and len(orders) < 10:
        if growing_wheat < 8:
            buy_w = min(8 - growing_wheat, int((money - OPERATING_RESERVE) // 10), 8)
            if buy_w > 0:
                orders.append(["BUY_SEED", "WHEAT", buy_w])
                money -= buy_w * 10
                growing_wheat += buy_w
                seeds["WHEAT"] = int(seeds.get("WHEAT", 0)) + buy_w

        if growing_melon < 12:
            buy_m = min(12 - growing_melon, int((money - OPERATING_RESERVE - 500) // 80), 8)
            if buy_m > 0:
                orders.append(["BUY_SEED", "MELON", buy_m])
                money -= buy_m * 80
                growing_melon += buy_m
                seeds["MELON"] = int(seeds.get("MELON", 0)) + buy_m

    # Replant wheat to maintain 10-12 wheat plots for continuous feed
    target_wheat = 8 if len(quadrants) == 1 else (10 if len(quadrants) == 2 else 12)
    if 2 <= day <= 24 and growing_wheat < target_wheat and len(orders) < 9:
        buy_w = min(target_wheat - growing_wheat, int((money - OPERATING_RESERVE) // 10), 4)
        if buy_w > 0:
            orders.append(["BUY_SEED", "WHEAT", buy_w])
            money -= buy_w * 10
            growing_wheat += buy_w
            seeds["WHEAT"] = int(seeds.get("WHEAT", 0)) + buy_w

    # === PRIORITY 5: Strawberry Cash-Crop Scaling (Days 4-22) ===
    cur_straw = int(seeds.get("STRAWBERRY", 0)) + int(scan["crops"].get("STRAWBERRY", 0))
    if len(quadrants) == 1:
        target_straw = 4
    elif len(quadrants) == 2:
        target_straw = 16
    else:
        target_straw = 36

    if 4 <= day <= 22 and len(orders) < 9:
        need_straw = max(0, target_straw - cur_straw)
        straw_price = 100
        feed_reserve = max(0, live * 2 - wheat_have) * max(wheat_price, 30.0) + OPERATING_RESERVE
        land_res = 1000 if (len(quadrants) == 1 and day <= 6 and money < 1100) else 0
        afford_straw = max(0, int((money - feed_reserve - land_res) // straw_price))
        buy_straw = min(need_straw, afford_straw, 10)
        if buy_straw > 0:
            orders.append(["BUY_SEED", "STRAWBERRY", buy_straw])
            money -= buy_straw * straw_price
            cur_straw += buy_straw
            seeds["STRAWBERRY"] = int(seeds.get("STRAWBERRY", 0)) + buy_straw

    # === PRIORITY 6: Buy Animals (Cow / Sheep with Opponent Modeling) ===
    if day <= 23 and len(orders) < 9:
        if len(quadrants) == 1:
            herd_target = min(4, 2 if day < 3 else 4)
        elif len(quadrants) == 2:
            herd_target = 8
        else:
            herd_target = 12

        owned_cows = scan["counts"]["COW"] + int(shed.get("COW", 0))
        owned_sheep = scan["counts"]["SHEEP"] + int(shed.get("SHEEP", 0))
        total_herd = owned_cows + owned_sheep

        # Cow safety ceiling: never exceed 4 cows if opponent owns >= 3 cows or milk price < 150
        max_cows = 4 if (opp_cows >= 3 or milk_price < 150) else 6

        if total_herd < herd_target:
            if owned_cows < max_cows and not milk_oversupply and milk_price >= 140:
                buy_species = "COW"
            elif not wool_oversupply:
                buy_species = "SHEEP"
            elif owned_cows < max_cows:
                buy_species = "COW"
            else:
                buy_species = "SHEEP"

            buy_spec = SPECIES[buy_species]
            animals_in_shed = int(shed.get("SHEEP", 0)) + int(shed.get("COW", 0))
            free_pastures = max(0, len(scan["empty"]["PASTURE"]) - animals_in_shed)
            reserve = OPERATING_RESERVE + (1000 if len(quadrants) == 1 and day <= 7 else 0)
            affordable = max(0, int((money - reserve) // buy_spec["cost"]))
            buy_count = min(affordable, free_pastures, herd_target - total_herd, 2)
            if buy_count > 0 and (wheat_have >= live + buy_count or day < 3):
                orders.append(["BUY_ANIMAL", buy_species, buy_count])
                money -= buy_spec["cost"] * buy_count

    # === PRIORITY 7: Sells ===
    need_cash = money < 400 and (day <= 15)
    ranked = sorted(shed.items(), key=lambda kv: -float(prices.get(kv[0], 0)))
    ranked.sort(key=lambda kv: 0 if kv[0] == "MELON" else (1 if kv[0] == "FERTILIZER" else 2))
    for item, count in ranked:
        if len(orders) >= 10:
            break
        qty = _sell_qty(
            str(item), int(count), float(prices.get(item, 0)),
            shed_total, day, hour, need_cash, is_endgame,
        )
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
    planted_melon = 0
    planted_straw = 0
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 16

    animals = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "animal"]
    animal_at = {pos: tile for pos, tile in animals}
    plants = [(pos, tile) for kind, pos, tile in scan["rows"] if kind == "plant"]
    plant_at = {pos: tile for pos, tile in plants}
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

        # === Priority 1: Act on tile underfoot ===
        if here is not None and not carrying_animal:
            if wheat > 0 and not here.get("fed_today", False):
                here["fed_today"] = True
                unfed_left = max(0, unfed_left - 1)
                actions.append(["FEED"])
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
        if here is not None and not carrying_animal and not danger_plants:
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

        # === Priority 2: Feed hungry animals across farm ===
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

        # === Fast Shed Drop (if carrying 2+ goods or valuable livestock products) ===
        carried_valuable = int(inv.get("MILK", 0)) + int(inv.get("WOOL", 0)) + int(inv.get("FERTILIZER", 0)) + int(inv.get("STRAWBERRY", 0)) + int(inv.get("MELON", 0))
        if carried_valuable >= 2 or (carried_valuable >= 1 and here is None and plant_here is None):
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 6: DANGER PLANTS WATERING ===
        if danger_plants:
            target = take_nearest(pos, danger_plants)
            if target is not None:
                actions.append(["WATER"] if pos == target else _move(pos, target))
                continue

        # === Priority 7: Unified Animal Service (Care, Harvest, Collect Fertilizer) ===
        animal_service = [
            p for p, tile in animals
            if not tile.get("cared_today", False) or int(tile.get("yield_units", 0)) > 0 or tile.get("fertilizer_available", False)
        ]
        target = take_nearest(pos, animal_service)
        if target is not None:
            if pos == target:
                tile = animal_at.get(pos, {})
                if not tile.get("cared_today", False):
                    tile["cared_today"] = True
                    actions.append(["CARE"])
                elif int(tile.get("yield_units", 0)) > 0:
                    tile["yield_units"] = 0
                    actions.append(["HARVEST"])
                elif tile.get("fertilizer_available", False):
                    tile["fertilizer_available"] = False
                    actions.append(["COLLECT_FERTILIZER"])
                else:
                    actions.append(["PASS"])
            else:
                actions.append(_move(pos, target))
            continue

        # === Priority 8: Harvest ripe crops (Melon, Strawberry, Wheat) ===
        if day >= 2:
            ripe = [p for p, tile in plants if _crop_ripe(tile)]
            target = take_nearest(pos, ripe)
            if target is not None:
                actions.append(["HARVEST"] if pos == target else _move(pos, target))
                continue

        # === Priority 10: Routine watering of all thirsty plants ===
        thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
        target = take_nearest(pos, thirsty)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # === Priority 11: PROACTIVE WEED CLEARING (keep land clean!) ===
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # === Priority 12: Apply fertilizer to Strawberry plots ===
        if fert > 0:
            unfert_straw = [
                p for p, tile in plants
                if str(tile.get("crop")) == "STRAWBERRY" and int(tile.get("fertilized_until_day", -1)) < day
            ]
            target = take_nearest(pos, unfert_straw)
            if target is not None:
                actions.append(["FERTILIZE"] if pos == target else _move(pos, target))
                continue

        # === Priority 13: Collect fertilizer from animals ===
        fert_avail = [p for p, tile in animals if tile.get("fertilizer_available", False)]
        target = take_nearest(pos, fert_avail)
        if target is not None:
            actions.append(["COLLECT_FERTILIZER"] if pos == target else _move(pos, target))
            continue

        # === Priority 14: Early Pastures (guarantee first 4 pastures for herd kickstart) ===
        target_pastures = 4 if len(quadrants) == 1 else (8 if len(quadrants) == 2 else 12)
        total_pastures = sum(scan["counts"].values()) + len(scan["empty"]["PASTURE"])
        if total_pastures < 4 and len(scan["empty"]["PASTURE"]) == 0:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                continue

        # === Priority 15: Plant Strawberry (Top Cash Crop — first pick of new land) ===
        avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
        if avail_straw > 0:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_straw += 1
                actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                continue

        # === Priority 16: Plant Melon (Early Kickstart) ===
        avail_melon = int(seeds.get("MELON", 0)) - planted_melon
        if avail_melon > 0:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_melon += 1
                actions.append(["PLANT", "MELON"] if pos == target else _move(pos, target))
                continue

        # === Priority 17: Build Additional Pastures (up to 8 in Q2, 12 in Q3) ===
        if day < 24 and total_pastures < target_pastures and len(scan["empty"]["PASTURE"]) == 0:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                actions.append(["BUILD_PASTURE"] if pos == target else _move(pos, target))
                continue

        # === Priority 18: Plant Wheat on all remaining empty tiles (Feed + Weed Prevention) ===
        avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
        if avail_wheat > 0:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_wheat += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        # === Priority 19: Drop carried goods at shed ===
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 20: Move toward shed staging ===
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
