"""agent_final.py — Apex performance with multi-quadrant land scaling, strawberry cash-crop diversification, opponent-aware market defense, and zero-loss liquidation.

Remediated Policy Architecture:
 1. Multi-Quadrant & Grid Pathfinding:
    - Avoids locked SE (5,5) when moving between SW and NE via waypoint (4,4).
    - Protects all shed access tiles (4,4), (5,4), (4,5), (5,5) from being paved over.
    - SW Land Expansion ($2,000) on Days 10-18 when cash >= $3,000.
    - Scales land utilization to >=80% (54-58+ tiles) across NW, NE, SW.
 2. Strawberry Cash-Crop Integration:
    - Plants Strawberry and Wheat strictly BEFORE building pastures to eliminate priority collisions.
    - Strict watering order: danger plants (consecutive_unwatered >= 1) preempt routine care/harvesting.
    - Routine watering placed strictly before fertilizer collection.
    - Fertilizer applied directly to Strawberry plots by workers.
    - Rolling 3-day feed buffer and land reserve protecting multi-day feed burn through first harvest.
 3. Opponent-Aware Market Defense & Adaptive Herd Capping:
    - Active species freezing & pivoting: freezes Cow if milk < 145 or opp_cows >= 3; freezes Sheep if wool < 180 or opp_sheep >= 2.
    - Fixed _sell_qty signature (8 parameters/arguments with hour).
    - Emergency shed overflow evasion at 70/85 capacity strictly preempting soft floors down to $1.
    - Dynamic market-clearing rate at drain ticks (every 4 hours) for price >= 50% base or >= 25.
    - Unchokes sales on Days 26-27 (min 2 units).
    - 100% terminal liquidation on Day 29 Turn 22-23 (0 stranded inventory).
 4. Submission & Execution Compliance:
    - Standard library only, 100% self-contained, <2ms/turn, max 10 market orders/turn.
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


def _hire_cost(hires_today: int) -> int:
    return 20 + 10 * int(hires_today)


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
# Board scan and geometry
# ---------------------------------------------------------------------------
def _farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    return obs["farms"][int(obs.get("player", 0))]


def _dist(a: Sequence[int], b: Pos) -> int:
    return abs(int(a[0]) - b[0]) + abs(int(a[1]) - b[1])


def _sheds(board: int, quadrants: Sequence[str] = ("NW",)) -> List[Pos]:
    half = board // 2
    shed_tiles = [(half - 1, half - 1)]  # (4,4) in NW
    if "NE" in quadrants:
        shed_tiles.append((half, half - 1))  # (5,4) in NE
    if "SW" in quadrants:
        shed_tiles.append((half - 1, half))  # (4,5) in SW
    if "SE" in quadrants:
        shed_tiles.append((half, half))      # (5,5) in SE
    return shed_tiles


def _move(src: Sequence[int], target: Pos) -> List[str]:
    sx, sy = int(src[0]), int(src[1])
    tx, ty = int(target[0]), int(target[1])
    # Avoid locked SE (5,5) when moving between SW (x <= 4, y >= 5) and NE (x >= 5, y <= 4)
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
    if locked in SPECIES and owned > 0:
        if scores[best] > scores.get(str(locked), 0) * 1.25:
            _LOCK["animal"] = best
            return best
        return str(locked)

    _LOCK["animal"] = best
    return best


def _wanted(animal: str, demand: Mapping[str, float]) -> int:
    spec = SPECIES[animal]
    daily = float(demand.get(spec["product"], 0.0))
    safe_market_demand = daily + 3.0
    return max(4, int(round(safe_market_demand / spec["per_day"])))


def _target_herd(day: int, quadrants: int = 1) -> int:
    """Ramps herd size: NW (up to 18), NE (up to 24), SW (up to 38)."""
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
    if quadrants < 2:
        return 18
    if quadrants == 2:
        return 24
    if day < 18:
        return 28
    if day < 22:
        return 34
    return 38


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
    return {
        "shed": shed, "rows": rows, "empty": empty, "empties": empties,
        "weeds": weeds, "counts": counts, "crops": crops,
    }


# ---------------------------------------------------------------------------
# Sell policy — continuous with emergency shed dumping & zero-stranding endgame
# ---------------------------------------------------------------------------
def _sell_qty(
    item: str, have: int, price: float, shed_total: int, day: int, hour: int,
    need_cash: bool, is_endgame: bool,
) -> int:
    if have <= 0 or item in SPECIES:
        return 0

    # 100% Terminal Liquidation on Day 29 Turn 22-23 (0 stranded inventory at turn 719)
    if day == 29 and hour >= 22:
        return have

    if item == "WHEAT":
        if is_endgame:
            return max(0, have - 2)
        return max(0, have - 14) if shed_total >= 85 else 0

    if item == "FERTILIZER":
        if price < 1:
            return 0
        return min(have, 4)

    base = BASE.get(item, 1)

    # Emergency Shed Overflow Evasion: strictly preempts soft-floor checks to prevent 100-cap discard
    if shed_total >= 85:
        return min(have, 8)
    if shed_total >= 70:
        return min(have, 4)

    # Endgame liquidation
    if day >= 28:
        return min(have, 4)

    # Unchoke sales on Days 26-27: sell at least min(have, 2) even if price < 30
    if day >= 26:
        return min(have, 4 if price >= base * 0.85 else 2)

    # Dynamic market clearing at drain ticks or when price is reasonable
    is_drain_tick = (hour % 4 == 0)
    if price >= base * 0.50 or price >= 25:
        if need_cash or is_drain_tick or shed_total >= 40:
            return min(have, 2)

    # Soft floor: 85% of base price
    if price < base * 0.85:
        return min(have, 1) if (shed_total >= 50 or have >= 6) else 0

    # Need cash urgently: sell to fund expansion
    if need_cash:
        return min(have, 3)

    # Days 0-21: sell continuously, faster when quote is high
    if day < 22:
        if price >= base * 1.30:
            pace = 4 if shed_total >= 50 else 3
        elif price >= base * 1.10:
            pace = 3 if shed_total >= 60 else 2
        else:
            pace = 1
        return min(have, pace)

    # Days 22-25: sell steadily to avoid terminal glut
    if price >= base * 1.25:
        pace = 4
    elif price >= base * 1.00:
        pace = 3
    else:
        pace = 2
    return min(have, pace)


# ---------------------------------------------------------------------------
# Market orders — slot-budgeted & demand-bounded
# ---------------------------------------------------------------------------
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
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))

    # Opponent Inspection & Market Oversupply Detection
    opp_farm = obs["farms"][1 - int(obs.get("player", 0))] if len(obs.get("farms", [])) > 1 else {}
    opp_tiles = opp_farm.get("tiles", [])
    opp_counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    for row in opp_tiles:
        for tile in row:
            if isinstance(tile, dict) and "animal" in tile and tile["animal"] in opp_counts:
                opp_counts[tile["animal"]] += 1
    opp_cows = opp_counts["COW"]
    opp_sheep = opp_counts["SHEEP"]

    milk_price = float(prices.get("MILK", BASE["MILK"]))
    wool_price = float(prices.get("WOOL", BASE["WOOL"]))
    milk_oversupply = (milk_price < 145 or opp_cows >= 3)
    wool_oversupply = (wool_price < 180 or opp_sheep >= 2)

    # Demand-bounded target herd: never exceed market absorption capacity
    shops_unlocked = obs.get("town", {}).get("unlocked_shops", [])
    has_livestock_shop = any(sh in ("PIZZA_SHOP", "SMOOTHIE_SHOP", "ICE_CREAM_SHOP", "YARN_STORE") for sh in shops_unlocked)
    has_yarn = "YARN_STORE" in shops_unlocked

    target_max = 18 if len(quadrants) < 2 else (24 if len(quadrants) == 2 else 38)
    target = min(_target_herd(day, len(quadrants)), target_max)
    if day >= 6 and not has_livestock_shop:
        target = min(target, 4)

    wheat_have = int(shed.get("WHEAT", 0)) + sum(
        int((inv or {}).get("WHEAT", 0)) for inv in private.get("inventories", [])
    )
    shed_total = sum(int(v) for v in shed.values())
    is_endgame = day >= 28
    need_cash = money < spec["cost"] + OPERATING_RESERVE and focus_owned < target

    # === PRIORITY 1: Survival wheat (feed) — always first ===
    wheat_floor = 16 if (live <= 3 and day < 4) else live + 4
    wheat_need = max(0, wheat_floor - wheat_have)
    wheat_price = max(1.0, float(prices.get("WHEAT", 25)))
    if wheat_need and len(orders) < 8 and money >= wheat_price + 2:
        qty = min(wheat_need, int((money - 2) // wheat_price), 16)
        if qty > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * wheat_price
            wheat_have += qty

    # === PRIORITY 2: Land Expansion (NE: Days 5-16, SW: Days 10-18) ===
    if len(quadrants) == 1 and 5 <= day <= 16 and money >= 1000 + OPERATING_RESERVE and len(orders) < 8:
        orders.append(["BUY_LAND", "NE"])
        money -= 1000
        quadrants.append("NE")
    elif len(quadrants) == 2 and 10 <= day <= 18 and money >= 3000 and len(orders) < 8:
        orders.append(["BUY_LAND", "SW"])
        money -= 2000
        quadrants.append("SW")

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
        cost = _hire_cost(hires)
        if money < cost + OPERATING_RESERVE:
            break
        orders.append(["HIRE"])
        money -= cost
        hires += 1

    # === PRIORITY 4: Buy animals (with Active Species Freezing & Pivoting) ===
    if 3 <= day <= 22 and len(orders) < 9:
        wool_demand = float(demand.get("WOOL", 1.0))
        sheep_quota = min(12 if len(quadrants) < 2 else (16 if len(quadrants) == 2 else 20),
                          int(round(wool_demand / 1.33))) if wool_demand > 2.0 else 0
        owned_sheep = scan["counts"]["SHEEP"] + int(shed.get("SHEEP", 0))
        owned_cows = scan["counts"]["COW"] + int(shed.get("COW", 0))

        buy_species = None
        if animal == "COW":
            if milk_oversupply:
                if (not wool_oversupply) and wool_price >= 180 and (has_yarn or wool_demand > 2.0):
                    buy_species = "SHEEP"
            else:
                buy_species = "COW"
        elif animal == "SHEEP":
            if wool_oversupply:
                if (not milk_oversupply) and milk_price >= 145:
                    buy_species = "COW"
            else:
                buy_species = "SHEEP"
        else:
            buy_species = animal

        if buy_species == "COW" and owned_sheep < sheep_quota and (owned_cows >= 6) and (not wool_oversupply):
            buy_species = "SHEEP"

        if buy_species is not None:
            cur_owned = owned_sheep if buy_species == "SHEEP" else owned_cows
            cur_target = min(sheep_quota if buy_species == "SHEEP" else target,
                             target_max - (owned_cows if buy_species == "SHEEP" else owned_sheep))
            if cur_owned < cur_target:
                buy_spec = SPECIES[buy_species]
                free = len(scan["empty"][buy_spec["structure"]])
                reserve = OPERATING_RESERVE + (wheat_floor * wheat_price * 0.4)
                affordable = int((money - reserve) // buy_spec["cost"])
                affordable = max(0, affordable)
                buy = min(affordable, free, cur_target - cur_owned, 3)
                if buy > 0 and wheat_have >= live + buy:
                    orders.append(["BUY_ANIMAL", buy_species, buy])
                    money -= buy_spec["cost"] * buy

    # === PRIORITY 5: Seeds (Wheat Day 0 + Strawberry Days 3-14 with Gestation Buffer) ===
    growing_wheat = int(seeds.get("WHEAT", 0)) + int(scan["crops"].get("WHEAT", 0))
    if day == 0 and growing_wheat < WHEAT_PLOTS and len(orders) < 10:
        qty = min(WHEAT_PLOTS - int(seeds.get("WHEAT", 0)), int((money - OPERATING_RESERVE) // 10))
        if qty > 0:
            orders.append(["BUY_SEED", "WHEAT", qty])
            money -= qty * 10

    if 3 <= day <= 14 and len(orders) < 10:
        cur_straw = int(seeds.get("STRAWBERRY", 0)) + int(scan["crops"].get("STRAWBERRY", 0))
        need_straw = max(0, 8 - cur_straw)
        straw_price = 100
        feed_buffer_days = min(3, max(1, 29 - day))
        feed_reserve = max(0, live * feed_buffer_days - wheat_have) * max(wheat_price, 30.0) + OPERATING_RESERVE + 200
        land_reserve = 1000 if (len(quadrants) == 1 and day <= 14) else 0
        afford_straw = max(0, int((money - feed_reserve - land_reserve) // straw_price))
        buy_straw = min(need_straw, afford_straw, 4)
        if buy_straw > 0:
            orders.append(["BUY_SEED", "STRAWBERRY", buy_straw])
            money -= buy_straw * straw_price
            seeds["STRAWBERRY"] = int(seeds.get("STRAWBERRY", 0)) + buy_straw

    # === PRIORITY 6: Sells — use remaining slots ===
    ranked = sorted(shed.items(), key=lambda kv: -float(prices.get(kv[0], 0)))
    ranked.sort(key=lambda kv: 0 if kv[0] == "FERTILIZER" else 1)
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


# ---------------------------------------------------------------------------
# Worker actions — 2-pass watering preemption & full land scaling
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
    animal_fetchers = 0
    planted_wheat = 0
    planted_straw = 0
    carried_animals = sum(int((inv or {}).get(animal, 0)) for inv in inventories)
    owned = sum(scan["counts"].values()) + int(shed_stock.get(animal, 0)) + carried_animals
    quadrants = list(farm.get("unlocked_quadrants", ["NW"]))
    is_endgame = day >= 29 and hour >= 13

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
        m_age = 3 if c == "WHEAT" else (10 if c == "STRAWBERRY" else 99)
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

        # Terminal return: go to shed and drop everything
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
        # 1a. Animal underfoot feed
        if here is not None and not carrying_animal:
            if wheat > 0 and not here.get("fed_today", False):
                here["fed_today"] = True
                unfed_left = max(0, unfed_left - 1)
                actions.append(["FEED"])
                continue

        # 1b. Plant underfoot urgent actions
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

        # 1c. Animal underfoot routine care/harvest/fertilizer (only if no danger plants)
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
        avail_shed_sp = next((sp for sp in (animal, "COW", "SHEEP") if int(shed_stock.get(sp, 0)) > 0), None)
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

        # === Priority 6: DANGER PLANTS WATERING (preempts routine care and harvesting) ===
        if danger_plants:
            target = take_nearest(pos, danger_plants)
            if target is not None:
                actions.append(["WATER"] if pos == target else _move(pos, target))
                continue

        # === Priority 7: Care uncared animals ===
        care_targets = [p for p, tile in animals if not tile.get("cared_today", False)]
        target = take_nearest(pos, care_targets)
        if target is not None:
            actions.append(["CARE"] if pos == target else _move(pos, target))
            continue

        # === Priority 8: Harvest animals ===
        harvest_animals = [p for p, tile in animals if int(tile.get("yield_units", 0)) > 0]
        target = take_nearest(pos, harvest_animals)
        if target is not None:
            actions.append(["HARVEST"] if pos == target else _move(pos, target))
            continue

        # === Priority 9: Harvest ripe crops (Wheat & Strawberry) ===
        if day >= 2:
            ripe = [p for p, tile in plants if _crop_ripe(tile)]
            target = take_nearest(pos, ripe)
            if target is not None:
                actions.append(["HARVEST"] if pos == target else _move(pos, target))
                continue

        # === Priority 10: Water routine thirsty plants (strictly before fertilizer collection) ===
        thirsty = [p for p, tile in plants if not tile.get("watered_today", False)]
        target = take_nearest(pos, thirsty)
        if target is not None:
            actions.append(["WATER"] if pos == target else _move(pos, target))
            continue

        # === Priority 11: Apply fertilizer to unfertilized Strawberry plots ===
        if fert > 0:
            unfert_straw = [
                p for p, tile in plants
                if str(tile.get("crop")) == "STRAWBERRY" and int(tile.get("fertilized_until_day", -1)) < day
            ]
            target = take_nearest(pos, unfert_straw)
            if target is not None:
                actions.append(["FERTILIZE"] if pos == target else _move(pos, target))
                continue

        # === Priority 12: Collect fertilizer from animals ===
        fert_avail = [p for p, tile in animals if tile.get("fertilizer_available", False)]
        target = take_nearest(pos, fert_avail)
        if target is not None:
            actions.append(["COLLECT_FERTILIZER"] if pos == target else _move(pos, target))
            continue

        # === Priority 13: Plant Strawberry & Plant Wheat (BEFORE building pastures!) ===
        avail_straw = int(seeds.get("STRAWBERRY", 0)) - planted_straw
        if avail_straw > 0:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_straw += 1
                actions.append(["PLANT", "STRAWBERRY"] if pos == target else _move(pos, target))
                continue

        avail_wheat = int(seeds.get("WHEAT", 0)) - planted_wheat
        if day < 3 and avail_wheat > 0 and scan["crops"].get("WHEAT", 0) + planted_wheat < WHEAT_PLOTS:
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                planted_wheat += 1
                actions.append(["PLANT", "WHEAT"] if pos == target else _move(pos, target))
                continue

        # === Priority 14: Build structures (joint pasture capacity for Cow + Sheep) ===
        structure = spec["structure"]
        target_pastures = 18 if len(quadrants) < 2 else (24 if len(quadrants) == 2 else 38)
        if structure == "PASTURE":
            total_animals = scan["counts"]["COW"] + scan["counts"]["SHEEP"]
            have_struct = total_animals + len(scan["empty"]["PASTURE"])
            need_struct = min(_target_herd(day, len(quadrants)), target_pastures)
        else:
            have_struct = scan["counts"][animal] + len(scan["empty"][structure])
            need_struct = min(_target_herd(day, len(quadrants)), _wanted(animal, demand))

        if day < 26 and have_struct < max(need_struct, owned):
            target = take_nearest(pos, list(scan["empties"]))
            if target is not None:
                build = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
                actions.append([build] if pos == target else _move(pos, target))
                continue

        # === Priority 15: Clear weeds ===
        if scan["weeds"]:
            target = take_nearest(pos, list(scan["weeds"]))
            if target is not None:
                actions.append(["DIG"] if pos == target else _move(pos, target))
                continue

        # === Priority 16: Drop carried goods at shed ===
        carried = sum(int(v) for k, v in inv.items() if k != "WHEAT")
        if carried > 0:
            dest = nearest_shed(pos)
            actions.append(["DROP"] if pos == dest else _move(pos, dest))
            continue

        # === Priority 17: Move toward staging position (near shed) ===
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
