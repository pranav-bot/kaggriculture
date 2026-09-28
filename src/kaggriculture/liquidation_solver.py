"""Retrograde DP & MILP Liquidation Solver for Kaggriculture (Days 20–29).

Solves the multi-product terminal liquidation problem for the final 10 days of the game:
1. Ingests agent state on Day 20 (Hour 0): livestock, field crops, seeds, shed inventory.
2. Models town shop consumption ticks (6 ticks/day, draining 6-12+ units/tick across shop types).
3. Computes optimal daily SELL quotas using Mixed-Integer Linear Programming (MILP)
   via ``scipy.optimize.milp``, continuous non-linear optimization via ``scipy.optimize.minimize``,
   and exact discrete Retrograde Dynamic Programming (Bellman backward induction).
4. Strictly enforces that no new seeds are planted if their maturation/harvest cycle
   extends past Day 30 (turn 720).
5. Returns a daily ``Liquidation_Intent`` dictionary that overrides the standard Beam Search
   during the final 10 days (Days 20-29).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp, minimize

from kaggriculture.env.items import (
    ANIMALS_DATA,
    CROPS_DATA,
    MARKET_I0,
    MARKET_PARAMS,
    PRICE_FLOOR,
    PRODUCTS_LIST,
    SHED_CAPACITY,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    TOWN_CENTER_SELL_INTERVAL,
    TOWN_SHOP_SELL_INTERVAL,
    TURNS_PER_DAY,
    calculate_shop_turn_consumption,
    calculate_town_daily_consumption,
    market_price,
)

logger = logging.getLogger(__name__)

# Terminal timeline constants
START_DAY = 20
END_DAY = 29
PLANNING_HORIZON_DAYS = 10  # Days 20..29 inclusive
GAME_OVER_DAY = 30          # Turn 720 is Day 30 Hour 0
TICKS_PER_DAY = TURNS_PER_DAY // TOWN_SHOP_SELL_INTERVAL  # 6 ticks/day (hours 0, 4, 8, 12, 16, 20)
SHOP_TICK_HOURS = tuple(range(0, TURNS_PER_DAY, TOWN_SHOP_SELL_INTERVAL))  # (0, 4, 8, 12, 16, 20)

# Default unlocked shops by Day 20 (6 shops unlocked: Day 3, 6, 9, 12, 15, 18)
DEFAULT_UNLOCKED_SHOPS_DAY20 = [
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE"
]


# =============================================================================
# 1. Liquidation Intent Data Structures
# =============================================================================

@dataclass
class LiquidationIntent:
    """Daily strategic liquidation plan that overrides Beam Search on Days 20–29."""
    day: int
    macro_intent: str = "TERMINAL_LIQUIDATION"
    sell_quotas: Dict[str, int] = field(default_factory=dict)
    tick_sell_quotas: Dict[str, int] = field(default_factory=dict)
    plant_quotas: Dict[str, int] = field(default_factory=dict)
    prohibit_planting: List[str] = field(default_factory=list)
    prohibit_purchases: List[str] = field(default_factory=lambda: [
        "COW", "SHEEP", "GOOSE", "EXPAND_NE", "EXPAND_SW", "EXPAND_SE"
    ])
    harvest_targets: List[Tuple[int, int, str]] = field(default_factory=list)
    care_targets: List[str] = field(default_factory=lambda: ["COW", "SHEEP"])
    projected_revenue: float = 0.0
    shed_buffer_target: int = 0
    terminal_flush: bool = False
    override_beam_search: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to standard Python dictionary."""
        d = asdict(self)
        d["override_beam_search"] = self.override_beam_search
        return d


# =============================================================================
# 2. Crop Maturation & Harvestability Rules
# =============================================================================

def get_maturation_days(crop: str) -> int:
    """Returns the time to first harvestable yield for a given crop species."""
    crop_upper = str(crop).upper()
    if crop_upper in CROPS_DATA:
        return int(CROPS_DATA[crop_upper]["time_to_first_yield"])
    # Fallback heuristics
    defaults = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
    return defaults.get(crop_upper, 4)


def is_seed_viable_for_planting(crop: str, current_day: int, horizon_day: int = GAME_OVER_DAY) -> bool:
    """Determines whether planting a seed on `current_day` can yield before `horizon_day`.

    Game ends at turn 719 (Day 29, Hour 23). Any crop whose first yield day is
    >= horizon_day (Day 30) will NEVER produce a harvestable unit before the match terminates.
    """
    maturation = get_maturation_days(crop)
    # Earliest yield appears on current_day + maturation
    earliest_yield_day = current_day + maturation
    return earliest_yield_day < horizon_day


def get_forbidden_crops(current_day: int, horizon_day: int = GAME_OVER_DAY) -> List[str]:
    """List of all crop species whose maturation extends past the game horizon."""
    forbidden = []
    for crop in CROPS_DATA:
        if not is_seed_viable_for_planting(crop, current_day, horizon_day):
            forbidden.append(crop)
    return sorted(forbidden)


# =============================================================================
# 3. Town Shop Consumption Model (6 ticks/day, 6-12+ units/tick)
# =============================================================================

@dataclass
class TownShopModel:
    """Models town shop and center consumption ticks across Days 20–29."""
    unlocked_shops: List[str] = field(default_factory=lambda: list(DEFAULT_UNLOCKED_SHOPS_DAY20))

    def get_tick_drain_by_product(self, day: int) -> Dict[str, int]:
        """Units consumed per 4-hour tick across all active town shops."""
        active_shops = self.get_active_shops_for_day(day)
        drain = {p: 0 for p in PRODUCTS_LIST}
        for shop in active_shops:
            turn_c = calculate_shop_turn_consumption(shop)
            for p, qty in turn_c.items():
                drain[p] = drain.get(p, 0) + qty
        return drain

    def get_tick_total_drain(self, day: int) -> int:
        """Total units drained across all products on a single shop tick."""
        d = self.get_tick_drain_by_product(day)
        return sum(d.values())

    def get_daily_drain_by_product(self, day: int) -> Dict[str, int]:
        """Total units consumed over an entire 24-hour day (6 ticks + town center)."""
        active_shops = self.get_active_shops_for_day(day)
        daily = calculate_town_daily_consumption(active_shops)
        return daily

    def get_active_shops_for_day(self, day: int) -> List[str]:
        """Returns shops unlocked up to `day` (new shop unlocks every 3 days)."""
        all_shop_keys = list(SHOPS.keys())
        # Up to day 29, max 8 shops
        unlocked_count = min(len(all_shop_keys), (day // 3) + 1)
        # Use provided unlocked shops if known, otherwise infer from day
        if self.unlocked_shops and len(self.unlocked_shops) >= unlocked_count:
            return self.unlocked_shops[:unlocked_count]
        return all_shop_keys[:unlocked_count]


# =============================================================================
# 4. State Ingestion & Yield Forecasting on Day 20 (Hour 0)
# =============================================================================

@dataclass
class IngestedDay20State:
    """State ingested on Day 20, Hour 0, with 10-day projected production schedules."""
    initial_cash: float
    current_day: int
    current_hour: int
    shed_inventory: Dict[str, int]
    seed_inventory: Dict[str, int]
    hand_inventory_totals: Dict[str, int]
    animal_counts: Dict[str, int]
    field_crop_counts: Dict[str, int]
    daily_projected_yields: Dict[str, List[int]]  # {product: [yield_day20, ..., yield_day29]}
    market_inventory: Dict[str, int]
    market_prices: Dict[str, int]
    unlocked_shops: List[str]
    prohibited_seeds: List[str]


def ingest_state_on_day_20(obs: Mapping[str, Any]) -> IngestedDay20State:
    """Parses observation on Day 20 (Hour 0) and projects upcoming asset yields for Days 20–29."""
    current_day = int(obs.get("day", START_DAY))
    current_hour = int(obs.get("hour", 0))
    player = int(obs.get("player", 0))
    farms = obs.get("farms", [{}])
    farm = farms[player] if player < len(farms) else {}

    cash = float(farm.get("money", 0))
    tiles = farm.get("tiles", [])

    # Private shed and seeds
    private = obs.get("private", {})
    shed = dict(private.get("shed", farm.get("shed", {})))
    seeds = dict(private.get("seeds", farm.get("seeds", {})))

    # Hand personal inventories
    hand_totals: Dict[str, int] = {}
    for inv in private.get("inventories", farm.get("inventories", [])):
        if isinstance(inv, dict):
            for k, v in inv.items():
                hand_totals[k] = hand_totals.get(k, 0) + int(v)

    # Count animals and field crops
    animal_counts: Dict[str, int] = {}
    crop_counts: Dict[str, int] = {}
    daily_yields: Dict[str, List[int]] = {p: [0] * PLANNING_HORIZON_DAYS for p in PRODUCTS_LIST}

    # Inspect tiles
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")

            # 1. Livestock (PASTURE / COOP)
            if kind in ("PASTURE", "COOP"):
                anim = tile.get("animal")
                if anim:
                    anim_upper = str(anim).upper()
                    animal_counts[anim_upper] = animal_counts.get(anim_upper, 0) + 1
                    placed_day = int(tile.get("placed_day", 0))
                    yield_units = int(tile.get("yield_units", 0))

                    # Project production over Days 20..29
                    if anim_upper == "COW":
                        # First harvest = placed_day + 8, then 3 milk every 2 days
                        for d_idx, day in enumerate(range(START_DAY, END_DAY + 1)):
                            since = day - placed_day - 8
                            if since >= 0 and since % 2 == 0:
                                y = 6 if since == 0 else 3
                                daily_yields["MILK"][d_idx] += y
                            # Fertilizer: 1 unit per cared cow per day
                            daily_yields["FERTILIZER"][d_idx] += 1

                    elif anim_upper == "SHEEP":
                        # First harvest = placed_day + 6, then 4 wool every 3 days
                        for d_idx, day in enumerate(range(START_DAY, END_DAY + 1)):
                            since = day - placed_day - 6
                            if since >= 0 and since % 3 == 0:
                                y = 6 if since == 0 else 4
                                daily_yields["WOOL"][d_idx] += y
                            daily_yields["FERTILIZER"][d_idx] += 1

                    elif anim_upper == "GOOSE":
                        # First harvest = placed_day + 4, then 2 eggs every day
                        for d_idx, day in enumerate(range(START_DAY, END_DAY + 1)):
                            since = day - placed_day - 4
                            if since >= 0:
                                y = 4 if since == 0 else 2
                                daily_yields["EGG"][d_idx] += y

            # 2. Field crops (PLANT)
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop:
                    crop_upper = str(crop).upper()
                    crop_counts[crop_upper] = crop_counts.get(crop_upper, 0) + 1
                    planted_day = int(tile.get("planted_day", 0))
                    yield_units = int(tile.get("yield_units", 0))

                    if crop_upper == "STRAWBERRY":
                        # First harvest = planted_day + 10, then every 2 days x4
                        for h_step in range(4):
                            harv_day = planted_day + 10 + h_step * 2
                            if START_DAY <= harv_day <= END_DAY:
                                d_idx = harv_day - START_DAY
                                daily_yields["STRAWBERRY"][d_idx] += 4

                    elif crop_upper == "TOMATO":
                        # First harvest = planted_day + 8, then every 1 day x4
                        for h_step in range(4):
                            harv_day = planted_day + 8 + h_step * 1
                            if START_DAY <= harv_day <= END_DAY:
                                d_idx = harv_day - START_DAY
                                daily_yields["TOMATO"][d_idx] += 4

                    elif crop_upper == "MELON":
                        harv_day = planted_day + 10
                        if START_DAY <= harv_day <= END_DAY:
                            d_idx = harv_day - START_DAY
                            daily_yields["MELON"][d_idx] += 6

                    elif crop_upper == "WHEAT":
                        harv_day = planted_day + 2
                        if START_DAY <= harv_day <= END_DAY:
                            d_idx = harv_day - START_DAY
                            daily_yields["WHEAT"][d_idx] += 4

                    elif crop_upper == "CARROT":
                        harv_day = planted_day + 2
                        if START_DAY <= harv_day <= END_DAY:
                            d_idx = harv_day - START_DAY
                            daily_yields["CARROT"][d_idx] += 3

    # Market information
    market = obs.get("market", {})
    market_inv = dict(market.get("inventory", market.get("inventories", {})))
    market_prices = dict(market.get("prices", {}))
    unlocked_shops = list(market.get("unlocked_shops", DEFAULT_UNLOCKED_SHOPS_DAY20))

    # Prohibited seeds (cycle extends past Day 30)
    prohibited = get_forbidden_crops(current_day, GAME_OVER_DAY)

    return IngestedDay20State(
        initial_cash=cash,
        current_day=current_day,
        current_hour=current_hour,
        shed_inventory=shed,
        seed_inventory=seeds,
        hand_inventory_totals=hand_totals,
        animal_counts=animal_counts,
        field_crop_counts=crop_counts,
        daily_projected_yields=daily_yields,
        market_inventory=market_inv,
        market_prices=market_prices,
        unlocked_shops=unlocked_shops,
        prohibited_seeds=prohibited,
    )


# =============================================================================
# 5. Price Depreciation & Revenue Formulas
# =============================================================================

def calculate_batch_revenue(product: str, quantity: int, start_market_inv: int) -> float:
    """Exact cash generated from selling `quantity` units sequentially starting at `start_market_inv`."""
    if quantity <= 0:
        return 0.0
    total = 0.0
    inv = int(start_market_inv)
    p_name = str(product).upper()
    for _ in range(int(quantity)):
        price = market_price(p_name, inv)
        total += price
        inv += 1
    return total


def estimate_marginal_price_tiers(product: str) -> Tuple[float, float, float]:
    """Computes marginal price estimates for Tier 1 (within drain), Tier 2, and Tier 3."""
    param = MARKET_PARAMS.get(product, MARKET_PARAMS["WHEAT"])
    base = float(param["base"])
    func = param["above_func"]
    target = float(param["above_target"])
    t_val = float(param["T"])

    if func == "sq":
        beta = (target * base) / (t_val ** 2)
        tier1 = base
        tier2 = max(1.0, base - beta * (20 ** 2))
        tier3 = max(1.0, base - beta * (50 ** 2))
    elif func == "linear":
        gamma = (target * base) / t_val
        tier1 = base
        tier2 = max(1.0, base - gamma * 15)
        tier3 = max(1.0, base - gamma * 40)
    else:
        tier1 = base
        tier2 = max(1.0, base * 0.8)
        tier3 = max(1.0, base * 0.4)

    return tier1, tier2, tier3


# =============================================================================
# 6. Mixed-Integer Linear Programming (MILP) Liquidation Solver
# =============================================================================

class MILPLiquidationSolver:
    """Fast, exact Mixed-Integer Linear Programming solver using ``scipy.optimize.milp``.

    Models non-linear price depreciation using piecewise linear marginal revenue tiers:
    - Tier 1: [0, D_p] units (absorbed by town shops each day) -> sells at pristine base price.
    - Tier 2: (D_p, 2 * D_p] units -> sells at moderate price depreciation.
    - Tier 3: > 2 * D_p units -> severe price depreciation.

    Subject to:
    - Flow balance & cumulative availability constraints.
    - Shed capacity limit <= 100 units at all times.
    - Terminal liquidation on Day 29.
    """

    def __init__(self, shop_model: Optional[TownShopModel] = None) -> None:
        self.shop_model = shop_model or TownShopModel()

    def solve(
        self,
        initial_shed: Dict[str, int],
        daily_yields: Dict[str, List[int]],
        market_inv: Optional[Dict[str, int]] = None,
        shed_capacity: int = SHED_CAPACITY,
    ) -> Dict[str, List[int]]:
        """Solves for the optimal daily sell quotas for Days 20..29.

        Returns {product: [quota_day20, quota_day21, ..., quota_day29]}.
        """
        active_products = [
            p for p in PRODUCTS_LIST
            if initial_shed.get(p, 0) > 0 or any(y > 0 for y in daily_yields.get(p, []))
        ]
        if not active_products:
            return {p: [0] * PLANNING_HORIZON_DAYS for p in PRODUCTS_LIST}

        T = PLANNING_HORIZON_DAYS
        # Compute daily drains for active products
        daily_drains: Dict[str, List[int]] = {}
        for p in active_products:
            daily_drains[p] = [
                max(1, self.shop_model.get_daily_drain_by_product(day).get(p, 1))
                for day in range(START_DAY, END_DAY + 1)
            ]

        # 3 Tiers per product per day: 3 * T * |active_products| variables
        n_vars = len(active_products) * T * 3
        c = np.zeros(n_vars)
        integrality = np.ones(n_vars)  # strictly integer variables
        lb = np.zeros(n_vars)
        ub = np.zeros(n_vars)

        var_map: Dict[Tuple[str, int, int], int] = {}
        idx = 0

        for p in active_products:
            t1_price, t2_price, t3_price = estimate_marginal_price_tiers(p)
            for t in range(T):
                d_cap = daily_drains[p][t]

                # Tier 1
                var_map[(p, t, 1)] = idx
                c[idx] = -t1_price  # minimize negative revenue
                lb[idx] = 0
                ub[idx] = d_cap
                idx += 1

                # Tier 2
                var_map[(p, t, 2)] = idx
                c[idx] = -t2_price
                lb[idx] = 0
                ub[idx] = d_cap
                idx += 1

                # Tier 3
                var_map[(p, t, 3)] = idx
                c[idx] = -t3_price
                lb[idx] = 0
                ub[idx] = 200  # large upper bound
                idx += 1

        rows: List[np.ndarray] = []
        lhs: List[float] = []
        rhs: List[float] = []

        # Constraint 1: Terminal liquidation per product (sum of sales == total available)
        for p in active_products:
            tot = initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T))
            row = np.zeros(n_vars)
            for t in range(T):
                for tier in (1, 2, 3):
                    row[var_map[(p, t, tier)]] = 1.0
            rows.append(row)
            lhs.append(tot)
            rhs.append(tot)

        # Constraint 2: Cumulative availability per product per day
        for p in active_products:
            for t in range(T - 1):
                avail_t = initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T)[:t + 1])
                row = np.zeros(n_vars)
                for tau in range(t + 1):
                    for tier in (1, 2, 3):
                        row[var_map[(p, tau, tier)]] = 1.0
                rows.append(row)
                lhs.append(0.0)
                rhs.append(avail_t)

        # Constraint 3: Shed capacity constraint (items remaining in shed <= shed_capacity)
        for t in range(T):
            tot_produced = sum(
                initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T)[:t + 1])
                for p in active_products
            )
            min_sales_needed = max(0.0, float(tot_produced - shed_capacity))
            row = np.zeros(n_vars)
            for p in active_products:
                for tau in range(t + 1):
                    for tier in (1, 2, 3):
                        row[var_map[(p, tau, tier)]] = 1.0
            rows.append(row)
            lhs.append(min_sales_needed)
            rhs.append(1e9)

        A = np.array(rows)
        constraints = LinearConstraint(A, lhs, rhs)
        bounds = Bounds(lb, ub)

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

        quotas: Dict[str, List[int]] = {p: [0] * T for p in PRODUCTS_LIST}

        if res.success:
            for p in active_products:
                for t in range(T):
                    q = sum(res.x[var_map[(p, t, tier)]] for tier in (1, 2, 3))
                    quotas[p][t] = int(round(q))
        else:
            logger.warning("MILP solver did not converge (%s); falling back to heuristic rate.", res.message)
            # Fallback heuristic: uniform sales capped by daily drain
            for p in active_products:
                tot = initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T))
                base_q = tot // T
                rem = tot % T
                for t in range(T):
                    quotas[p][t] = base_q + (1 if t >= T - rem else 0)

        return quotas


# =============================================================================
# 7. Continuous Non-Linear Solver (scipy.optimize.minimize)
# =============================================================================

class ScipyMinimizeLiquidationSolver:
    """Continuous non-linear optimizer using ``scipy.optimize.minimize`` (SLSQP).

    Directly optimizes the analytical non-linear revenue function with quadratic and
    linear depreciation curves before projecting to discrete integers.
    """

    def __init__(self, shop_model: Optional[TownShopModel] = None) -> None:
        self.shop_model = shop_model or TownShopModel()

    def solve(
        self,
        initial_shed: Dict[str, int],
        daily_yields: Dict[str, List[int]],
        market_inv: Optional[Dict[str, int]] = None,
        shed_capacity: int = SHED_CAPACITY,
    ) -> Dict[str, List[int]]:
        """Solves non-linear liquidation problem using SLSQP."""
        active_products = [
            p for p in PRODUCTS_LIST
            if initial_shed.get(p, 0) > 0 or any(y > 0 for y in daily_yields.get(p, []))
        ]
        if not active_products:
            return {p: [0] * PLANNING_HORIZON_DAYS for p in PRODUCTS_LIST}

        T = PLANNING_HORIZON_DAYS
        n_prods = len(active_products)
        m_inv = market_inv or {p: MARKET_I0 for p in active_products}

        drains = {
            p: [self.shop_model.get_daily_drain_by_product(day).get(p, 10) for day in range(START_DAY, END_DAY + 1)]
            for p in active_products
        }

        def objective(x: np.ndarray) -> float:
            total_rev = 0.0
            for idx, p in enumerate(active_products):
                q = x[idx * T : (idx + 1) * T]
                param = MARKET_PARAMS.get(p, MARKET_PARAMS["WHEAT"])
                base = float(param["base"])
                t_param = float(param["T"])
                target = float(param["above_target"])
                func = param["above_func"]

                inv = float(m_inv.get(p, MARKET_I0))
                for t in range(T):
                    qt = max(0.0, float(q[t]))
                    excess = max(0.0, inv - MARKET_I0)
                    if func == "sq":
                        beta = (target * base) / (t_param ** 2)
                        r = qt * (base - beta * (excess ** 2)) - beta * excess * (qt ** 2) - (beta / 3.0) * (qt ** 3)
                    elif func == "linear":
                        gamma = (target * base) / t_param
                        r = qt * (base - gamma * excess) - (gamma / 2.0) * (qt ** 2)
                    else:
                        r = qt * base
                    total_rev += r
                    inv = max(float(MARKET_I0), inv + qt - float(drains[p][t]))
            return -total_rev

        cons = []
        bounds = [(0.0, 100.0) for _ in range(n_prods * T)]

        for idx, p in enumerate(active_products):
            tot = float(initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T)))
            cons.append({
                "type": "eq",
                "fun": lambda x, idx=idx, tot=tot: np.sum(x[idx * T : (idx + 1) * T]) - tot,
            })
            for t in range(T - 1):
                avail_t = float(initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T)[:t + 1]))
                cons.append({
                    "type": "ineq",
                    "fun": lambda x, idx=idx, t=t, avail=avail_t: avail - np.sum(x[idx * T : idx * T + t + 1]),
                })

        for t in range(T):
            cons.append({
                "type": "ineq",
                "fun": lambda x, t=t: float(shed_capacity) - sum(
                    initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T)[:t + 1]) - np.sum(x[idx * T : idx * T + t + 1])
                    for idx, p in enumerate(active_products)
                ),
            })

        x0 = []
        for p in active_products:
            tot = initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T))
            x0.extend([tot / T] * T)
        x0 = np.array(x0, dtype=float)

        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 150})

        quotas: Dict[str, List[int]] = {p: [0] * T for p in PRODUCTS_LIST}
        if res.success or res.fun < 0:
            for idx, p in enumerate(active_products):
                continuous = res.x[idx * T : (idx + 1) * T]
                # Round to integer preserving total sum
                rounded = [int(math.floor(v)) for v in continuous]
                diff = int(round(sum(continuous) - sum(rounded)))
                for k in range(diff):
                    rounded[-(k + 1)] += 1
                quotas[p] = rounded
        else:
            # Fallback
            for p in active_products:
                tot = initial_shed.get(p, 0) + sum(daily_yields.get(p, [0] * T))
                quotas[p] = [tot // T] * T
                quotas[p][-1] += tot % T

        return quotas


# =============================================================================
# 8. Discrete Retrograde Dynamic Programming (Bellman Backward Induction)
# =============================================================================

class RetrogradeDPSolver:
    """Discrete Retrograde Dynamic Programming solver using Bellman backward induction.

    Solves single-commodity optimal liquidation backward from Day 29 down to Day 20.
    At Day 29: V_29(s, di) = batch_revenue(s, I_0 + di), pi_29 = s.
    For d = 28..20:
        V_d(s, di) = max_{0 <= q <= s} [ batch_revenue(q, I_0 + di) + V_{d+1}(s - q + Y, di + q - D) ]
    """

    def __init__(self, shop_model: Optional[TownShopModel] = None) -> None:
        self.shop_model = shop_model or TownShopModel()

    def solve_commodity(
        self,
        product: str,
        initial_shed: int,
        daily_yields: List[int],
        start_excess_inv: int = 0,
        max_shed_state: int = 60,
        max_surplus_state: int = 50,
        step_granularity: int = 1,
    ) -> List[int]:
        """Runs Bellman backward induction for one commodity and returns optimal daily quotas."""
        T = PLANNING_HORIZON_DAYS
        drains = [
            self.shop_model.get_daily_drain_by_product(day).get(product, 12)
            for day in range(START_DAY, END_DAY + 1)
        ]

        V: Dict[Tuple[int, int, int], float] = {}
        pi: Dict[Tuple[int, int, int], int] = {}

        # Terminal state: Day 29 (d = 9)
        d_terminal = T - 1
        for s in range(0, max_shed_state + 1, step_granularity):
            for di in range(0, max_surplus_state + 1, step_granularity):
                V[(d_terminal, s, di)] = calculate_batch_revenue(product, s, MARKET_I0 + di)
                pi[(d_terminal, s, di)] = s

        # Backward induction: d = 8 down to 0 (Days 28 down to 20)
        for d in range(T - 2, -1, -1):
            drain_d = drains[d]
            yield_next = daily_yields[d + 1] if d + 1 < len(daily_yields) else 0

            for s in range(0, max_shed_state + 1, step_granularity):
                for di in range(0, max_surplus_state + 1, step_granularity):
                    best_val = -1e18
                    best_q = 0

                    # Candidates: 0, drain, s, and granular intervals
                    q_candidates = set(range(0, s + 1, step_granularity))
                    q_candidates.add(min(s, drain_d))
                    q_candidates.add(s)

                    for q in sorted(q_candidates):
                        if q > s:
                            continue
                        r = calculate_batch_revenue(product, q, MARKET_I0 + di)
                        next_s = min(max_shed_state, s - q + yield_next)
                        next_di = min(max_surplus_state, max(0, di + q - drain_d))

                        # Snap to nearest granular state
                        snap_s = round(next_s / step_granularity) * step_granularity
                        snap_di = round(next_di / step_granularity) * step_granularity

                        val = r + V.get((d + 1, snap_s, snap_di), 0.0)
                        if val > best_val:
                            best_val = val
                            best_q = q

                    V[(d, s, di)] = best_val
                    pi[(d, s, di)] = best_q

        # Forward rollout
        quotas: List[int] = []
        curr_s = min(max_shed_state, initial_shed)
        curr_di = min(max_surplus_state, start_excess_inv)

        for d in range(T):
            snap_s = round(curr_s / step_granularity) * step_granularity
            snap_di = round(curr_di / step_granularity) * step_granularity
            q = pi.get((d, snap_s, snap_di), curr_s if d == T - 1 else min(curr_s, drains[d]))
            quotas.append(int(q))

            if d < T - 1:
                y_next = daily_yields[d + 1] if d + 1 < len(daily_yields) else 0
                curr_s = min(max_shed_state, curr_s - q + y_next)
                curr_di = min(max_surplus_state, max(0, curr_di + q - drains[d]))

        return quotas


# =============================================================================
# 9. Master Terminal Liquidation Solver
# =============================================================================

class TerminalLiquidationSolver:
    """Master solver calculating the complete Days 20–29 liquidation schedule.

    Orchestrates:
    - State ingestion at Day 20 (or dynamic turn re-evaluation).
    - Joint MILP multi-product optimization under shed capacity <= 100.
    - Generation of daily ``LiquidationIntent`` dictionaries for Days 20 through 29.
    """

    def __init__(
        self,
        solver_backend: str = "milp",
        shop_model: Optional[TownShopModel] = None,
    ) -> None:
        self.backend = solver_backend
        self.shop_model = shop_model or TownShopModel()
        self.milp_solver = MILPLiquidationSolver(self.shop_model)
        self.scipy_solver = ScipyMinimizeLiquidationSolver(self.shop_model)
        self.dp_solver = RetrogradeDPSolver(self.shop_model)

        self._cached_schedule: Optional[Dict[str, List[int]]] = None
        self._cached_day20_state: Optional[IngestedDay20State] = None

    def plan_liquidation(self, obs: Mapping[str, Any]) -> Dict[int, LiquidationIntent]:
        """Ingests Day 20 state, solves for optimal quotas, and returns {day: LiquidationIntent}."""
        state = ingest_state_on_day_20(obs)
        self._cached_day20_state = state
        self.shop_model.unlocked_shops = state.unlocked_shops

        # Solve for 10-day quotas across all commodities
        if self.backend == "scipy":
            quotas = self.scipy_solver.solve(
                initial_shed=state.shed_inventory,
                daily_yields=state.daily_projected_yields,
                market_inv=state.market_inventory,
            )
        else:
            quotas = self.milp_solver.solve(
                initial_shed=state.shed_inventory,
                daily_yields=state.daily_projected_yields,
                market_inv=state.market_inventory,
            )

        self._cached_schedule = quotas

        # Build daily LiquidationIntent objects
        daily_intents: Dict[int, LiquidationIntent] = {}
        for d_idx, day in enumerate(range(START_DAY, END_DAY + 1)):
            day_quotas = {p: quotas[p][d_idx] for p in PRODUCTS_LIST if quotas[p][d_idx] > 0}

            # Per-tick quotas: divide daily quota evenly across the 6 consumption ticks
            tick_quotas: Dict[str, int] = {}
            for p, q in day_quotas.items():
                tick_quotas[p] = max(1, math.ceil(q / TICKS_PER_DAY))

            # Prohibited seeds for this specific day
            forbidden_seeds = get_forbidden_crops(day, GAME_OVER_DAY)
            plant_quotas = {crop: 0 for crop in forbidden_seeds}

            # Expected revenue today
            day_rev = sum(
                calculate_batch_revenue(p, q, state.market_inventory.get(p, MARKET_I0))
                for p, q in day_quotas.items()
            )

            intent = LiquidationIntent(
                day=day,
                macro_intent="TERMINAL_LIQUIDATION",
                sell_quotas=day_quotas,
                tick_sell_quotas=tick_quotas,
                plant_quotas=plant_quotas,
                prohibit_planting=forbidden_seeds,
                prohibit_purchases=["COW", "SHEEP", "GOOSE", "EXPAND_NE", "EXPAND_SW", "EXPAND_SE"],
                care_targets=["COW", "SHEEP"],
                projected_revenue=day_rev,
                terminal_flush=(day == END_DAY),
                override_beam_search=True,
            )
            daily_intents[day] = intent

        return daily_intents

    def get_intent_for_day(self, obs: Mapping[str, Any]) -> LiquidationIntent:
        """Returns the LiquidationIntent for the current day in `obs` (Days 20..29)."""
        current_day = int(obs.get("day", START_DAY))
        if self._cached_schedule is None or current_day == START_DAY:
            intents = self.plan_liquidation(obs)
            return intents.get(current_day, intents[START_DAY])

        # If already cached, retrieve or re-plan if day changed
        intents = self.plan_liquidation(obs) if self._cached_schedule is None else None
        if self._cached_schedule is not None and START_DAY <= current_day <= END_DAY:
            d_idx = current_day - START_DAY
            day_quotas = {p: self._cached_schedule[p][d_idx] for p in PRODUCTS_LIST if self._cached_schedule[p][d_idx] > 0}
            tick_quotas = {p: max(1, math.ceil(q / TICKS_PER_DAY)) for p, q in day_quotas.items()}
            forbidden = get_forbidden_crops(current_day, GAME_OVER_DAY)

            return LiquidationIntent(
                day=current_day,
                macro_intent="TERMINAL_LIQUIDATION",
                sell_quotas=day_quotas,
                tick_sell_quotas=tick_quotas,
                plant_quotas={c: 0 for c in forbidden},
                prohibit_planting=forbidden,
                prohibit_purchases=["COW", "SHEEP", "GOOSE", "EXPAND_NE", "EXPAND_SW", "EXPAND_SE"],
                terminal_flush=(current_day == END_DAY),
                override_beam_search=True,
            )

        # Fallback for out of range days
        return LiquidationIntent(day=current_day, override_beam_search=False)


# =============================================================================
# 10. Runtime Liquidation Controller (Overriding Beam Search Days 20–29)
# =============================================================================

class LiquidationController:
    """Runtime controller that executes the liquidation schedule on Days 20–29.

    Replaces step-level Beam Search during the final 10 days:
    1. Synchronizes market sell orders with the 6 town shop consumption ticks (hours 0, 4, 8, 12, 16, 20).
    2. Prohibits late seed planting and capital purchases (animals, land).
    3. Directs farmhands to care for cows/sheep and harvest ripe field crops.
    4. Executes complete terminal flush on Day 29, Hour 23.
    """

    def __init__(self, solver: Optional[TerminalLiquidationSolver] = None) -> None:
        self.solver = solver or TerminalLiquidationSolver()
        self._current_intent: Optional[LiquidationIntent] = None
        self._last_planned_day: int = -1
        self._sold_today: Dict[str, int] = {}

    def is_in_liquidation_phase(self, obs: Mapping[str, Any]) -> bool:
        """True if game is in the final 10 days (Days 20..29)."""
        day = int(obs.get("day", 0))
        return START_DAY <= day <= END_DAY

    def update_intent(self, obs: Mapping[str, Any]) -> LiquidationIntent:
        """Updates or returns the active LiquidationIntent for the current turn."""
        day = int(obs.get("day", 0))
        if day != self._last_planned_day:
            self._current_intent = self.solver.get_intent_for_day(obs)
            self._last_planned_day = day
            self._sold_today = {p: 0 for p in PRODUCTS_LIST}

        return self._current_intent or LiquidationIntent(day=day)

    def generate_market_orders(self, obs: Mapping[str, Any]) -> List[List[str]]:
        """Generates market SELL orders aligned with town shop consumption ticks."""
        intent = self.update_intent(obs)
        hour = int(obs.get("hour", 0))
        day = int(obs.get("day", 0))
        step = int(obs.get("step", 0))

        player = int(obs.get("player", 0))
        farm = obs.get("farms", [{}])[player] if player < len(obs.get("farms", [])) else {}
        shed = dict(obs.get("private", {}).get("shed", farm.get("shed", {})))

        orders: List[List[str]] = []

        # 1. Day 29 Terminal Flush (last 4 hours of Day 29: hours 20..23)
        if day == END_DAY and hour >= 20:
            for item, qty in shed.items():
                if qty > 0:
                    orders.append(["SELL", str(item).upper(), str(qty)])
            return orders

        # 2. Consumption Tick Execution (hours 0, 4, 8, 12, 16, 20)
        is_shop_tick = (hour in SHOP_TICK_HOURS)
        if not is_shop_tick:
            return []

        # Execute tick quota for items in shed
        for item, tick_qty in intent.tick_sell_quotas.items():
            avail = shed.get(item, 0)
            target_to_sell = min(avail, tick_qty)
            if target_to_sell > 0:
                orders.append(["SELL", item, str(target_to_sell)])
                self._sold_today[item] = self._sold_today.get(item, 0) + target_to_sell

        return orders

    def filter_mechanical_actions(
        self,
        operations: Dict[str, Any],
        obs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Filters operations to enforce liquidation constraints (zero late seeds, zero capital buys)."""
        intent = self.update_intent(obs)
        filtered = dict(operations)

        # 1. Inject or override market orders
        liquidation_market_orders = self.generate_market_orders(obs)
        if liquidation_market_orders:
            filtered["market"] = liquidation_market_orders

        # 2. Sanitize market orders against prohibited seeds and capital purchases
        prohibited = set(intent.prohibit_planting)
        sanitized_market = []
        for o in filtered.get("market", []):
            if isinstance(o, (list, tuple)) and len(o) >= 2:
                cmd = str(o[0]).upper()
                target = str(o[1]).upper()
                if cmd in ("BUY_SEED", "BUY") and target in prohibited:
                    continue
                if cmd in ("BUY", "BUY_ANIMAL") and target in ("COW", "SHEEP", "GOOSE"):
                    continue
                if cmd in ("EXPAND", "BUY_EXPANSION", "BUY_LAND"):
                    continue
            sanitized_market.append(o)
        filtered["market"] = sanitized_market

        # 3. Strip any forbidden seed plantings from farmer/hand action lists
        for actor_key in ["farmer"] + [f"hand_{i}" for i in range(10)]:
            action = filtered.get(actor_key)
            if isinstance(action, (list, tuple)) and len(action) >= 2:
                cmd = action[0].upper() if isinstance(action[0], str) else ""
                target = action[1].upper() if isinstance(action[1], str) else ""
                # Strip PLANT of forbidden crops
                if cmd == "PLANT" and target in prohibited:
                    filtered[actor_key] = ["PASS"]
                # Strip capital animal/land purchases
                if cmd in ("BUY", "BUY_ANIMAL") and target in ("COW", "SHEEP", "GOOSE"):
                    filtered[actor_key] = ["PASS"]
                if cmd in ("EXPAND", "BUY_EXPANSION", "BUY_LAND"):
                    filtered[actor_key] = ["PASS"]

        filtered["_liquidation_intent"] = intent.to_dict()
        filtered["_override_beam_search"] = True
        return filtered
