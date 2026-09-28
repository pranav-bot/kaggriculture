#!/usr/bin/env python3
"""Comprehensive scipy optimizer for every product's sell timing.

Extends quant_milk_opt and quant_catalog with:
  - Joint multi-product optimization (sell timing across all goods simultaneously)
  - Stage-aware sell policies (different strategies for days 0-10, 11-20, 21-29)
  - Herd ramp optimization (when to buy animals, how many)
  - Cash-flow reinvestment modeling (sell product -> buy more animals)
  - Opponent-aware pricing (how to respond to shared book pressure)

Usage:
  python scripts/quant_full_product.py             # full analysis
  python scripts/quant_full_product.py --quick      # fast grid only
  python scripts/quant_full_product.py --product MILK --drain 24
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggriculture.env.items import (
    MARKET_PARAMS, MARKET_I0, PRICE_FLOOR,
    shape_func, market_price,
    ANIMALS_DATA, CROPS_DATA, FERTILIZER_DATA,
    SHOPS, SHED_CAPACITY, TURNS_PER_DAY,
    TOWN_SHOP_SELL_INTERVAL,
)

DAYS = 30
SHED = SHED_CAPACITY  # 100
TICKS_PER_DAY = TURNS_PER_DAY // TOWN_SHOP_SELL_INTERVAL  # 6


# ---------------------------------------------------------------------------
# Price helper using the engine's exact formula
# ---------------------------------------------------------------------------
def price(item: str, inventory: int) -> int:
    """Exact engine market price."""
    return market_price(item, inventory)


def price_from_deficit(item: str, deficit: float) -> int:
    """Price given deficit (positive = below I0, good for seller)."""
    return price(item, int(MARKET_I0 - deficit))


# ---------------------------------------------------------------------------
# Production schedules
# ---------------------------------------------------------------------------
@dataclass
class ProductionModel:
    """Models production of a single product over 30 days."""
    item: str
    daily_production: list  # units produced each day (length 30)
    animal_cost: float = 0.0
    seed_cost: float = 0.0
    n_tiles: int = 0
    description: str = ""


def cow_production(n_cows: int, place_schedule: List[int]) -> List[int]:
    """Milk production: 6 on first harvest, 3 every 2 days after."""
    daily = [0] * DAYS
    for placed_day in place_schedule[:n_cows]:
        for day in range(DAYS):
            since = day - placed_day - 8  # first_yield_day = 8
            if since < 0 or since % 2 != 0:
                continue
            daily[day] += 6 if since == 0 else 3
    return daily


def sheep_production(n_sheep: int, place_schedule: List[int]) -> List[int]:
    """Wool production: 6 on first harvest, 4 every 3 days after."""
    daily = [0] * DAYS
    for placed_day in place_schedule[:n_sheep]:
        for day in range(DAYS):
            since = day - placed_day - 6  # first_yield_day = 6
            if since < 0 or since % 3 != 0:
                continue
            daily[day] += 6 if since == 0 else 4
    return daily


def goose_production(n_geese: int, place_schedule: List[int]) -> List[int]:
    """Egg production: 4 on first harvest, 2 every day after."""
    daily = [0] * DAYS
    for placed_day in place_schedule[:n_geese]:
        for day in range(DAYS):
            since = day - placed_day - 4  # first_yield_day = 4
            if since < 0 or since % 1 != 0:
                continue
            daily[day] += 4 if since == 0 else 2
    return daily


def fertilizer_production(n_animals: int, place_schedule: List[int]) -> List[int]:
    """Fertilizer: collected from animals. ~1 per animal every 2 days once producing."""
    daily = [0] * DAYS
    for placed_day in place_schedule[:n_animals]:
        animal_first = 8  # assume cows
        for day in range(DAYS):
            since = day - placed_day - animal_first
            if since >= 0 and since % 2 == 0:
                daily[day] += 1
    return daily


def slow_ramp(cap: int) -> List[int]:
    """Standard slow ramp: 4/8/14/cap placed on days 4/8/11/14."""
    schedule = []
    targets = [(4, 4), (8, 8), (11, 14), (14, cap)]
    owned = 0
    for day, target in targets:
        target = min(cap, target)
        while owned < target:
            schedule.append(day)
            owned += 1
    return schedule


def fast_ramp(cap: int) -> List[int]:
    """Aggressive ramp: 6/12/18/cap placed on days 3/6/9/12."""
    schedule = []
    targets = [(3, 6), (6, 12), (9, 18), (12, cap)]
    owned = 0
    for day, target in targets:
        target = min(cap, target)
        while owned < target:
            schedule.append(day)
            owned += 1
    return schedule


def cash_ramp(cap: int) -> List[int]:
    """Cash-flow driven ramp: buy as fast as money allows.
    Assumes $3000 start, $400/cow, income from fertilizer + early milk."""
    schedule = []
    targets = [(3, 4), (5, 6), (7, 10), (9, 14), (12, 18), (15, cap)]
    owned = 0
    for day, target in targets:
        target = min(cap, target)
        while owned < target:
            schedule.append(day)
            owned += 1
    return schedule


# ---------------------------------------------------------------------------
# Core simulator: sell a single product with configurable policy
# ---------------------------------------------------------------------------
def simulate_product(
    item: str,
    production: List[int],
    drain: float,
    policy: Dict[str, Any],
    opponent_sells: float = 0.0,
    opponent_start_day: int = 14,
) -> Dict[str, Any]:
    """Simulate selling a product over 30 days.

    Policy dict keys:
      - "start_day": int, first day to start selling
      - "daily_cap": int, max units to sell per day
      - "floor": int, minimum price to sell at (default: base price)
      - "stage_caps": optional list of (day_start, day_end, cap) overrides
      - "stage_floors": optional list of (day_start, day_end, floor) overrides
      - "endgame_dump_day": int, day to dump everything (default: 29)
      - "shed_pressure": int, shed level above which we force sell
    """
    base = MARKET_PARAMS[item]["base"]
    start_day = policy.get("start_day", 0)
    daily_cap = policy.get("daily_cap", 16)
    floor_price = policy.get("floor", base)
    stage_caps = policy.get("stage_caps", [])
    stage_floors = policy.get("stage_floors", [])
    endgame_dump = policy.get("endgame_dump_day", 29)
    shed_pressure = policy.get("shed_pressure", 90)

    deficit = 0.0
    shed = 0
    cash = 0.0
    sold = 0
    peak_price = 0
    min_price = 9999
    unsold_at_end = 0
    daily_cash = []

    for day in range(DAYS):
        shed += production[day]
        day_cash = 0.0

        # Opponent sells first
        if day >= opponent_start_day and opponent_sells > 0:
            for _ in range(int(opponent_sells)):
                p = price_from_deficit(item, deficit)
                if p < base:
                    break
                deficit -= 1.0

        # Determine effective cap and floor for this day
        eff_cap = daily_cap
        eff_floor = floor_price
        for ds, de, c in stage_caps:
            if ds <= day <= de:
                eff_cap = c
        for ds, de, f in stage_floors:
            if ds <= day <= de:
                eff_floor = f

        # Endgame dump
        if day >= endgame_dump:
            eff_cap = shed
            eff_floor = 1

        # Shed pressure override
        if shed >= shed_pressure:
            eff_cap = max(eff_cap, shed - shed_pressure + 4)

        # Sell with tick-level granularity
        per_tick = max(0, eff_cap // TICKS_PER_DAY)
        extra = eff_cap - per_tick * TICKS_PER_DAY

        for tick in range(TICKS_PER_DAY):
            budget = per_tick + (1 if tick < extra else 0)
            if day >= start_day:
                for _ in range(budget):
                    if shed <= 0:
                        break
                    p = price_from_deficit(item, deficit)
                    if p < eff_floor and day < endgame_dump:
                        break
                    shed -= 1
                    deficit -= 1.0
                    cash += p
                    day_cash += p
                    sold += 1
                    peak_price = max(peak_price, p)
                    min_price = min(min_price, p)

            deficit += drain / TICKS_PER_DAY

        # Shed overflow
        overflow = max(0, shed - SHED)
        shed -= overflow

        daily_cash.append(day_cash)

    unsold_at_end = shed

    return {
        "cash": cash,
        "sold": sold,
        "unsold": unsold_at_end,
        "peak_price": peak_price,
        "min_price": min_price if min_price < 9999 else 0,
        "avg_price": cash / sold if sold > 0 else 0,
        "daily_cash": daily_cash,
    }


# ---------------------------------------------------------------------------
# Scipy optimizers
# ---------------------------------------------------------------------------
def optimize_sell_schedule(
    item: str,
    production: List[int],
    drain: float,
    opponent: float = 0.0,
) -> Dict[str, Any]:
    """Grid search + Nelder-Mead for 5-stage sell policy.

    Optimizes: [start_day, daily_cap, floor_frac, mid_cap_frac, end_cap_frac]
    where floor_frac * base = floor price.
    """
    base = MARKET_PARAMS[item]["base"]

    # Stage definitions: early (0-9), mid-early (10-16), mid (17-22), late (23-27), endgame (28-29)
    def objective(x: np.ndarray) -> float:
        start = int(np.clip(x[0], 0, 27))
        cap = int(np.clip(x[1], 1, 40))
        floor_frac = float(np.clip(x[2], 0.0, 1.5))
        mid_cap_mult = float(np.clip(x[3], 0.5, 3.0))
        end_cap_mult = float(np.clip(x[4], 1.0, 10.0))

        policy = {
            "start_day": start,
            "daily_cap": cap,
            "floor": int(base * floor_frac),
            "stage_caps": [
                (17, 22, int(cap * mid_cap_mult)),
                (23, 27, int(cap * end_cap_mult)),
            ],
            "endgame_dump_day": 28,
        }
        result = simulate_product(item, production, drain, policy, opponent)
        return -result["cash"]

    # Grid search
    best_x = np.array([16, 8, 1.0, 1.0, 2.0])
    best_val = objective(best_x)

    starts = [0, 8, 12, 16, 20, 24]
    caps = [2, 4, 8, 16, 24]
    floors = [0.5, 0.8, 1.0, 1.2]

    for s in starts:
        for c in caps:
            for f in floors:
                x = np.array([s, c, f, 1.0, 2.0])
                val = objective(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()

    # Nelder-Mead refinement
    result = minimize(
        objective, best_x,
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 0.5, "fatol": 10.0},
    )
    if -result.fun > -best_val:
        best_x = result.x
        best_val = result.fun

    # Reconstruct best policy
    start = int(np.clip(best_x[0], 0, 27))
    cap = int(np.clip(best_x[1], 1, 40))
    floor_frac = float(np.clip(best_x[2], 0.0, 1.5))
    mid_mult = float(np.clip(best_x[3], 0.5, 3.0))
    end_mult = float(np.clip(best_x[4], 1.0, 10.0))

    policy = {
        "start_day": start,
        "daily_cap": cap,
        "floor": int(base * floor_frac),
        "stage_caps": [
            (17, 22, int(cap * mid_mult)),
            (23, 27, int(cap * end_mult)),
        ],
        "endgame_dump_day": 28,
    }

    sim = simulate_product(item, production, drain, policy, opponent)

    return {
        "item": item,
        "drain": drain,
        "opponent": opponent,
        "policy": policy,
        "result": sim,
        "params": {
            "start_day": start,
            "daily_cap": cap,
            "floor": int(base * floor_frac),
            "mid_cap": int(cap * mid_mult),
            "end_cap": int(cap * end_mult),
        },
    }


def optimize_herd_ramp(
    animal: str,
    drains: List[float],
    caps: List[int] = [8, 12, 14, 18, 24],
) -> Dict[str, Any]:
    """Find optimal herd size and ramp schedule for an animal type."""
    product = ANIMALS_DATA[animal]["product"]
    cost = ANIMALS_DATA[animal]["cost"]

    prod_fn = {
        "COW": cow_production,
        "SHEEP": sheep_production,
        "GOOSE": goose_production,
    }[animal]

    ramps = {
        "slow": slow_ramp,
        "fast": fast_ramp,
        "cash": cash_ramp,
    }

    results = []
    for drain in drains:
        for cap in caps:
            for ramp_name, ramp_fn in ramps.items():
                schedule = ramp_fn(cap)
                production = prod_fn(cap, schedule)
                opt = optimize_sell_schedule(product, production, drain)
                net = opt["result"]["cash"] - cost * cap
                results.append({
                    "animal": animal,
                    "drain": drain,
                    "cap": cap,
                    "ramp": ramp_name,
                    "gross": opt["result"]["cash"],
                    "net": net,
                    "sold": opt["result"]["sold"],
                    "unsold": opt["result"]["unsold"],
                    "policy": opt["params"],
                })

    results.sort(key=lambda r: -r["net"])
    return {
        "animal": animal,
        "best": results[0] if results else None,
        "all": results,
    }


def optimize_joint_production(opponent: float = 0.0) -> Dict[str, Any]:
    """Joint optimization: milk + fertilizer sell timing.

    This models the key interaction: selling fertilizer funds more cows,
    which produce more milk. The cow count affects both production streams.
    """
    def objective(x: np.ndarray) -> float:
        n_cows = int(np.clip(x[0], 4, 24))
        milk_start = int(np.clip(x[1], 0, 27))
        milk_cap = int(np.clip(x[2], 1, 40))
        milk_floor_frac = float(np.clip(x[3], 0.5, 1.5))
        fert_floor = int(np.clip(x[4], 1, 100))
        fert_cap = int(np.clip(x[5], 1, 24))

        schedule = cash_ramp(n_cows)
        milk_prod = cow_production(n_cows, schedule)
        fert_prod = fertilizer_production(n_cows, schedule)

        base_milk = MARKET_PARAMS["MILK"]["base"]
        milk_policy = {
            "start_day": milk_start,
            "daily_cap": milk_cap,
            "floor": int(base_milk * milk_floor_frac),
            "endgame_dump_day": 28,
        }
        fert_policy = {
            "start_day": 0,
            "daily_cap": fert_cap,
            "floor": fert_floor,
            "endgame_dump_day": 28,
        }

        milk_sim = simulate_product("MILK", milk_prod, 24.0, milk_policy, opponent)
        fert_sim = simulate_product("FERTILIZER", fert_prod, 0.0, fert_policy)

        # Costs
        cow_cost = n_cows * 400
        wheat_cost = n_cows * 20 * 25  # rough feed cost
        hire_cost = 1500  # rough estimate

        total = milk_sim["cash"] + fert_sim["cash"] - cow_cost - wheat_cost - hire_cost
        return -total

    result = differential_evolution(
        objective,
        bounds=[
            (4, 24),    # n_cows
            (0, 27),    # milk_start
            (1, 40),    # milk_cap
            (0.5, 1.5), # milk_floor_frac
            (1, 100),   # fert_floor
            (1, 24),    # fert_cap
        ],
        seed=42,
        popsize=15,
        maxiter=40,
        tol=0.005,
        workers=1,
        updating="immediate",
    )

    x = result.x
    return {
        "n_cows": int(np.clip(x[0], 4, 24)),
        "milk_start": int(np.clip(x[1], 0, 27)),
        "milk_cap": int(np.clip(x[2], 1, 40)),
        "milk_floor": int(MARKET_PARAMS["MILK"]["base"] * np.clip(x[3], 0.5, 1.5)),
        "fert_floor": int(np.clip(x[4], 1, 100)),
        "fert_cap": int(np.clip(x[5], 1, 24)),
        "total_cash": -result.fun,
        "opponent": opponent,
    }


# ---------------------------------------------------------------------------
# Analysis of all products
# ---------------------------------------------------------------------------
def analyze_all_products(quick: bool = False) -> Dict[str, Any]:
    """Run optimization for every product and print results."""
    results = {}

    # --- Animal products ---
    print("=" * 80)
    print("ANIMAL PRODUCT OPTIMIZATION")
    print("=" * 80)

    for animal in ["COW", "SHEEP", "GOOSE"]:
        product = ANIMALS_DATA[animal]["product"]
        cost = ANIMALS_DATA[animal]["cost"]
        drains = [6, 12, 24] if not quick else [12]
        caps_list = [8, 14, 18] if not quick else [18]

        print(f"\n{'─' * 60}")
        print(f"  {animal} -> {product} (cost ${cost}/animal)")
        print(f"{'─' * 60}")

        opt = optimize_herd_ramp(animal, drains, caps_list)
        best = opt["best"]
        if best:
            print(f"  BEST: {best['cap']} animals, {best['ramp']} ramp, drain={best['drain']}")
            print(f"    gross=${best['gross']:,.0f}  net=${best['net']:,.0f}")
            print(f"    sold={best['sold']}  unsold={best['unsold']}")
            print(f"    policy: start={best['policy']['start_day']}  cap={best['policy']['daily_cap']}  floor=${best['policy']['floor']}")

        # Show top 5 configs
        for i, r in enumerate(opt["all"][:5]):
            print(f"    #{i+1}: cap={r['cap']} ramp={r['ramp']} drain={r['drain']} -> net=${r['net']:,.0f}")

        results[animal] = opt

    # --- Fertilizer ---
    print(f"\n{'=' * 80}")
    print("FERTILIZER OPTIMIZATION (no town drain)")
    print(f"{'=' * 80}")

    for n_cows in ([18] if quick else [8, 14, 18]):
        schedule = slow_ramp(n_cows)
        fert_prod = fertilizer_production(n_cows, schedule)
        total_prod = sum(fert_prod)

        for floor in [1, 20, 40, 60, 80]:
            policy = {"start_day": 0, "daily_cap": 16, "floor": floor, "endgame_dump_day": 28}
            sim = simulate_product("FERTILIZER", fert_prod, 0.0, policy)
            print(f"  {n_cows} cows, floor=${floor:3d}: cash=${sim['cash']:8,.0f}  sold={sim['sold']:3d}/{total_prod}  unsold={sim['unsold']}")

        # Scipy optimize
        opt = optimize_sell_schedule("FERTILIZER", fert_prod, 0.0)
        print(f"  scipy best: floor=${opt['params']['floor']}  cap={opt['params']['daily_cap']}  cash=${opt['result']['cash']:,.0f}")

    results["FERTILIZER"] = opt

    # --- Crop products ---
    print(f"\n{'=' * 80}")
    print("CROP PRODUCT ANALYSIS (8-tile farms)")
    print(f"{'=' * 80}")

    crop_setups = {
        "WHEAT": {"prod": [0]*4 + [8*4] + [0]*25, "cost": 10*8},
        "CARROT": {"prod": [0]*3 + [8*3] + [0]*26, "cost": 20*8},
        "TOMATO": {"prod": [0]*8 + [8, 8, 8, 8] + [0]*18, "cost": 50*8},
        "STRAWBERRY": {"prod": [0]*10 + [8, 0, 8, 0, 8, 0, 8] + [0]*13, "cost": 100*8},
        "MELON": {"prod": [0]*10 + [8*6] + [0]*19, "cost": 80*8},
    }

    for crop, setup in crop_setups.items():
        drains_test = [1, 6, 12] if not quick else [6]
        print(f"\n  {crop} (seed cost: ${setup['cost']})")
        for drain in drains_test:
            opt_crop = optimize_sell_schedule(crop, setup["prod"], drain)
            net = opt_crop["result"]["cash"] - setup["cost"]
            print(f"    drain={drain:2d}: cash=${opt_crop['result']['cash']:8,.0f}  net=${net:8,.0f}  sold={opt_crop['result']['sold']}")
        results[crop] = opt_crop

    # --- Joint milk+fert optimization ---
    print(f"\n{'=' * 80}")
    print("JOINT MILK + FERTILIZER OPTIMIZATION")
    print(f"{'=' * 80}")

    for opp in ([0.0, 12.0] if not quick else [0.0]):
        joint = optimize_joint_production(opp)
        print(f"\n  opponent sells {opp:.0f}/day:")
        print(f"    optimal cows: {joint['n_cows']}")
        print(f"    milk: start day {joint['milk_start']}, cap {joint['milk_cap']}, floor ${joint['milk_floor']}")
        print(f"    fert: floor ${joint['fert_floor']}, cap {joint['fert_cap']}")
        print(f"    total cash: ${joint['total_cash']:,.0f}")
        results["joint_" + str(int(opp))] = joint

    return results


# ---------------------------------------------------------------------------
# Opponent modeling: Bayesian Online Change Point Detection (BOCD)
# ---------------------------------------------------------------------------
# Economic rationale (engine-grounded, see MARKET_PARAMS in env/items.py):
#   * WOOL above_func="sq" (above_target 3.20): price falls QUADRATICALLY in
#     shared-book surplus. A simultaneous opponent dump craters Wool to $1.
#   * MILK above_func="linear" (above_target 1.60): price falls LINEARLY.
#   * Scarcity upside is weak (WOOL log/0.20, MILK sqrt/0.60), so dumping hurts
#     far more than hoarding helps -- the shared book is negatively asymmetric.
#   * FERTILIZER is the ideal tripwire: no town shop or town-center consumes
#     it (drain == 0 always), so window-to-window changes in its market
#     inventory equal executed player sells (up to partial-fill noise).
# Causal story: opponent CEASES selling fertilizer -> they are stockpiling it
# for herd expansion (or net-buying it) -> a Milk/Wool production boom follows
# -> simultaneous dump crashes the shared book. Fertilizer-sale cessation is a
# LEADING indicator of an incoming Milk/Wool flood, so we preemptively
# liquidate our own Milk/Wool BEFORE the opponent acts.

OPPONENT_HOARDING_DETECTED = "OPPONENT_HOARDING_DETECTED"
HOARDING_PROB_THRESHOLD = 0.85
LIQUIDATION_ITEMS = ("MILK", "WOOL")
LIQUIDATION_PENALTY_MULT = 2.0  # per-unit holding penalty = mult x base price


def _gauss_logpdf(x: float, mean: float, var: float) -> float:
    """Log of the Normal pdf (math-only, no scipy.stats dependency)."""
    var = max(1e-9, float(var))
    return -0.5 * (math.log(2.0 * math.pi * var) + (float(x) - mean) ** 2 / var)


def parse_opponent_state(
    obs: Dict[str, Any],
    *,
    seat: int = 0,
    prev_market_inventory: Optional[Dict[str, float]] = None,
    our_executed_sells: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Parse the opponent's PUBLIC state from a live observation.

    Returns a dict with:
      - "opponent_cash_reserves": opponent liquid cash (public farm field).
      - "opponent_fertilizer_sold": opponent's executed FERTILIZER sells since
        the previous observation, inferred from public market accounting
        (None when no previous inventory snapshot is available).
      - "opponent_net_flow": per-item inferred opponent net outflow
        (positive = net selling into the book, negative = net buying).
      - "turn": current step number.

    Accounting identity per item (FERTILIZER drain is exactly 0):
        opp_net[t] = (inv[t] - inv[t-1]) - our_executed_sells[t]
    Callers should pass EXECUTED (post-partial-fill) own quantities; requested
    SELL amounts overstate when the shed is short. Residual noise (partial
    fills, opponent buys) is absorbed by the BOCD observation variance.
    """
    opp_seat = 1 - int(seat)
    farms = obs.get("farms") or []
    opp_farm = farms[opp_seat] if 0 <= opp_seat < len(farms) else {}
    opp_cash = float(opp_farm.get("money", 0.0) or 0.0)

    market = obs.get("market") or {}
    cur_inv = market.get("inventory") or {}

    opp_net: Dict[str, float] = {}
    opp_fert: Optional[float] = None
    if prev_market_inventory is not None:
        ours = our_executed_sells or {}
        keys = set(cur_inv.keys()) | set(prev_market_inventory.keys())
        for item in keys:
            try:
                delta = float(cur_inv.get(item, 0.0) or 0.0) - float(
                    prev_market_inventory.get(item, 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue
            opp_net[item] = delta - float(ours.get(item, 0.0) or 0.0)
        opp_fert = opp_net.get("FERTILIZER")

    return {
        "opponent_cash_reserves": opp_cash,
        "opponent_fertilizer_sold": opp_fert,
        "opponent_net_flow": opp_net,
        "turn": int(obs.get("step", 0) or 0),
    }


class OpponentTracker:
    """Rolling time-series of opponent public-state observables."""

    def __init__(self, maxlen: int = 720) -> None:
        self.opponent_fertilizer_sold: deque = deque(maxlen=maxlen)
        self.opponent_cash_reserves: deque = deque(maxlen=maxlen)
        self.turns: deque = deque(maxlen=maxlen)
        self._prev_market_inventory: Optional[Dict[str, float]] = None

    def update(
        self,
        obs: Dict[str, Any],
        *,
        seat: int = 0,
        our_executed_sells: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Parse one observation and extend the rolling series."""
        parsed = parse_opponent_state(
            obs,
            seat=seat,
            prev_market_inventory=self._prev_market_inventory,
            our_executed_sells=our_executed_sells,
        )
        market = obs.get("market") or {}
        try:
            self._prev_market_inventory = {
                k: float(v or 0.0) for k, v in (market.get("inventory") or {}).items()
            }
        except (TypeError, ValueError):
            pass
        if parsed["opponent_fertilizer_sold"] is not None:
            self.opponent_fertilizer_sold.append(float(parsed["opponent_fertilizer_sold"]))
        self.opponent_cash_reserves.append(float(parsed["opponent_cash_reserves"]))
        self.turns.append(parsed["turn"])
        return parsed

    def window_sum(self, series: deque, window: int) -> Optional[float]:
        """Sum of the last `window` entries (None when history is short)."""
        if len(series) < window:
            return None
        return float(sum(list(series)[-window:]))

    def __len__(self) -> int:
        return len(self.turns)


class BayesianChangePointDetector:
    """Gaussian Bayesian Online Change Point Detection (Adams & MacKay 2007).

    Tracks the posterior over run length r_t (windows since the last regime
    shift) under a conjugate Normal unknown-mean / known-variance model.
    `update(x)` returns P(regime shift | x_1:t), defined as the posterior
    mass on short run lengths r_t <= grace -- i.e. the probability that the
    current regime began within the last `grace` windows. (Note: with a
    constant hazard, the filtered P(r_t = 0) is identically the hazard rate,
    so the decision statistic must aggregate short runs rather than read
    off r = 0.) No policy or action is ever queried -- inference is over the
    observed public series only.
    """

    def __init__(
        self,
        *,
        hazard: float = 0.05,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        sigma2: float = 400.0,
        max_run: int = 40,
        grace: int = 2,
    ) -> None:
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be in (0, 1)")
        self.hazard = float(hazard)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.sigma2 = float(sigma2)
        self.max_run = int(max_run)
        self.grace = int(grace)
        self.run_posterior = np.array([1.0], dtype=np.float64)  # P(r_0 = 0) = 1
        self.run_sums = np.array([0.0], dtype=np.float64)  # sum of last r obs
        self.history: List[float] = []
        self.last_cp_prob: float = 0.0

    def _predictive_logpdf(self, x: float, n: float, s: float) -> float:
        kappa = self.kappa0 + n
        mean = (self.kappa0 * self.mu0 + s) / kappa
        var = self.sigma2 * (1.0 + 1.0 / kappa)
        return _gauss_logpdf(x, mean, var)

    def update(self, x: float) -> float:
        """Ingest one observation; return P(changepoint at this step)."""
        x = float(x)
        r_prev = self.run_posterior
        s_prev = self.run_sums
        n_prev = np.arange(len(r_prev), dtype=np.float64)  # run r holds r obs

        log_pred = np.array(
            [self._predictive_logpdf(x, n, s) for n, s in zip(n_prev, s_prev)]
        )
        # Absolute predictive densities (stable: shift by max, then scale back).
        # NOTE: do NOT renormalize across run lengths here -- the absolute
        # scale is what lets changepoint mass compete with growth mass.
        log_max = float(log_pred.max())
        pred = np.exp(log_pred - log_max) * math.exp(log_max)

        growth = r_prev * pred * (1.0 - self.hazard)
        cp_mass = float(np.sum(r_prev * pred * self.hazard))

        new_len = min(len(r_prev) + 1, self.max_run + 1)
        new_post = np.zeros(new_len, dtype=np.float64)
        new_post[0] = cp_mass
        take = min(len(growth), new_len - 1)
        new_post[1 : 1 + take] = growth[:take]
        if len(growth) > take:  # fold truncated tail mass onto longest run
            new_post[-1] += float(np.sum(growth[take:]))
        total = new_post.sum()
        if total <= 0.0 or not np.isfinite(total):
            new_post = np.zeros(new_len, dtype=np.float64)
            new_post[0] = 1.0
            total = 1.0
        self.run_posterior = new_post / total

        new_sums = np.zeros(new_len, dtype=np.float64)
        take_s = min(len(s_prev), new_len - 1)
        new_sums[1 : 1 + take_s] = s_prev[:take_s] + x
        self.run_sums = new_sums

        self.history.append(x)
        # Posterior probability of a (recent) regime shift: mass on short runs.
        keep = min(len(self.run_posterior), self.grace + 1)
        self.last_cp_prob = float(self.run_posterior[:keep].sum())
        return self.last_cp_prob

    @property
    def map_run_length(self) -> int:
        return int(np.argmax(self.run_posterior))

    def regime_means(self, pre_window: int = 6) -> Tuple[float, float]:
        """(pre_mean, post_mean) around the MAP changepoint.

        post_mean averages the current MAP run; pre_mean averages the
        `pre_window` points before it. A hoarding signature is
        post_mean << pre_mean (opponent stopped selling fertilizer).
        """
        r = self.map_run_length
        hist = self.history
        post = hist[len(hist) - r :] if r > 0 and hist else hist[-1:]
        start = max(0, len(hist) - r - pre_window)
        pre = hist[start : len(hist) - r] if len(hist) - r > start else hist[:1]
        post_mean = float(np.mean(post)) if post else 0.0
        pre_mean = float(np.mean(pre)) if pre else 0.0
        return pre_mean, post_mean


class HoardingMonitor:
    """Turn-level monitor: tracker + windowed BOCD + latched event flag.

    Feed live observations via `update()`; when the fertilizer-sale series
    shows a regime shift with posterior P > 0.85 AND the post-change mean is
    below the pre-change mean (cessation/hoarding signature), the monitor
    latches the OPPONENT_HOARDING_DETECTED event. The flag persists until
    `clear()` so the strategic override cannot flicker mid-liquidation.
    """

    EVENT = OPPONENT_HOARDING_DETECTED

    def __init__(
        self,
        *,
        window_turns: int = 24,
        min_windows: int = 4,
        threshold: float = HOARDING_PROB_THRESHOLD,
        min_drop: float = 10.0,
        detector_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tracker = OpponentTracker()
        self.detector = BayesianChangePointDetector(**(detector_kwargs or {}))
        self.window_turns = int(window_turns)
        self.min_windows = int(min_windows)
        self.threshold = float(threshold)
        self.min_drop = float(min_drop)
        self._pending_turns = 0
        self.n_windows = 0
        self.last_cp_prob = 0.0
        self.last_pre_mean = 0.0
        self.last_post_mean = 0.0
        self.event_latched = False
        self.trigger_info: Optional[Dict[str, Any]] = None

    def update(
        self,
        obs: Dict[str, Any],
        *,
        seat: int = 0,
        our_executed_sells: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        parsed = self.tracker.update(obs, seat=seat, our_executed_sells=our_executed_sells)
        fert = parsed["opponent_fertilizer_sold"]
        if fert is not None:
            self._pending_turns += 1
            if self._pending_turns >= self.window_turns:
                wsum = self.tracker.window_sum(
                    self.tracker.opponent_fertilizer_sold, self.window_turns
                )
                self._pending_turns = 0
                if wsum is not None:
                    self.n_windows += 1
                    self.last_cp_prob = self.detector.update(wsum)
                    pre, post = self.detector.regime_means()
                    self.last_pre_mean, self.last_post_mean = pre, post
                    if (
                        not self.event_latched
                        and self.n_windows >= self.min_windows
                        and self.last_cp_prob > self.threshold
                        and (pre - post) >= self.min_drop
                    ):
                        self.event_latched = True
                        self.trigger_info = {
                            "event": self.EVENT,
                            "p_changepoint": self.last_cp_prob,
                            "pre_mean": pre,
                            "post_mean": post,
                            "turn": parsed["turn"],
                            "n_windows": self.n_windows,
                        }
        return self.status()

    def status(self) -> Dict[str, Any]:
        return {
            "event": self.EVENT if self.event_latched else None,
            "event_latched": self.event_latched,
            "p_changepoint": self.last_cp_prob,
            "pre_mean": self.last_pre_mean,
            "post_mean": self.last_post_mean,
            "n_windows": self.n_windows,
            "trigger_info": self.trigger_info,
        }

    def clear(self) -> None:
        self.event_latched = False
        self.trigger_info = None


# ---------------------------------------------------------------------------
# Strategic override: preemptive Milk/Wool liquidation
# ---------------------------------------------------------------------------
def holding_penalty_for_inventory(
    inventory: Dict[str, float],
    *,
    items: Tuple[str, ...] = LIQUIDATION_ITEMS,
    mult: float = LIQUIDATION_PENALTY_MULT,
) -> float:
    """Penalty (in $) for holding liquidation-target inventory.

    Per-unit penalty = mult x base price (MILK $320/u, WOOL $400/u at the
    default mult=2.0). Any beam candidate that keeps holding these goods
    scores strictly worse than liquidating them now at realistic prices, so
    the search is forced into immediate preemptive liquidation.
    """
    penalty = 0.0
    for item in items:
        qty = float((inventory or {}).get(item, 0.0) or 0.0)
        base = float(MARKET_PARAMS.get(item, {}).get("base", 0.0) or 0.0)
        penalty += max(0.0, qty) * mult * base
    return float(penalty)


def adjust_beam_score(
    score: float,
    inventory: Dict[str, float],
    *,
    hoarding_active: bool,
) -> float:
    """Apply the strategic override to one beam-search candidate score."""
    if not hoarding_active:
        return float(score)
    return float(score) - holding_penalty_for_inventory(inventory)


def apply_hoarding_override_to_policy(
    policy: Dict[str, Any],
    *,
    items: Tuple[str, ...] = LIQUIDATION_ITEMS,
) -> Dict[str, Any]:
    """Rewrite a sell policy into an immediate-liquidation policy.

    start_day=0 (sell NOW), unbounded daily cap, floor=$1 (accept any price),
    dump from today: forces simulate/beam-search policies to clear Milk/Wool
    before the opponent's anticipated dump lands. Non-target keys pass
    through untouched.
    """
    _ = items  # target set is documented for beam-search penalty parity
    overridden = dict(policy)
    overridden.update(
        {
            "start_day": 0,
            "daily_cap": 10**6,
            "floor": 1,
            "stage_caps": [],
            "stage_floors": [],
            "endgame_dump_day": 0,
            "shed_pressure": 0,
            "hoarding_override": True,
            "liquidation_items": list(items),
        }
    )
    return overridden


def dump_impact_table() -> List[Dict[str, Any]]:
    """Engine-exact price impact of a simultaneousdump (shared-book surplus)."""
    rows = []
    for item in LIQUIDATION_ITEMS:
        base = MARKET_PARAMS[item]["base"]
        row = {"item": item, "base": base, "above_func": MARKET_PARAMS[item]["above_func"]}
        for surplus in [0, 25, 50, 100]:
            row[f"price_+{surplus}"] = price(item, MARKET_I0 + surplus)
        rows.append(row)
    return rows


def demo_hoarding_detection(seed: int = 42) -> Dict[str, Any]:
    """Synthetic selling -> hoarding scenario through the full live path."""
    rng = np.random.default_rng(seed)
    # Regime 1 (days 0-9): opponent steadily sells ~60 fertilizer/window.
    # Regime 2 (days 10+): opponent ceases selling (hoarding) -> ~0/window.
    truth = [float(rng.normal(60.0, 8.0)) for _ in range(10)] + [0.0] * 6
    monitor = HoardingMonitor(window_turns=6, min_windows=3)
    fert_inv = float(MARKET_I0)
    seat = 0
    fired_at = None
    # Prime with an initial snapshot so turn 0 has a previous inventory
    # (live usage: prime the monitor on the first observation of the episode).
    prime = {
        "step": 0,
        "day": 0,
        "farms": [{"money": 3000.0}, {"money": 3000.0}],
        "market": {"inventory": {"FERTILIZER": fert_inv}, "prices": {}},
        "town": {"unlocked_shops": []},
    }
    monitor.update(prime, seat=seat, our_executed_sells={"FERTILIZER": 0.0})
    for day, opp_sold in enumerate(truth):
        for _ in range(6):  # 6 turns per window; drip the window total evenly
            fert_inv += opp_sold / 6.0
            obs = {
                "step": day * 6,
                "day": day,
                "farms": [
                    {"money": 3000.0},
                    {"money": 3000.0 + day * 50.0},
                ],
                "market": {"inventory": {"FERTILIZER": fert_inv}, "prices": {}},
                "town": {"unlocked_shops": []},
            }
            status = monitor.update(obs, seat=seat, our_executed_sells={"FERTILIZER": 0.0})
            if status["event_latched"] and fired_at is None:
                fired_at = {"day": day, **(status["trigger_info"] or {})}
    return {
        "dump_impact": dump_impact_table(),
        "trigger": fired_at,
        "final_status": monitor.status(),
        "n_windows": monitor.n_windows,
    }


def run_hoarding_replay(
    episode_path: str, *, seat: int = 0, window_turns: int = 24
) -> Dict[str, Any]:
    """Run the monitor over a logged episode (validates accounting + detector)."""
    with gzip.open(episode_path, "rt") as f:
        ep = json.load(f)
    steps = ep.get("steps") or []
    monitor = HoardingMonitor(window_turns=window_turns, min_windows=3)

    def requested_sells(action: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for m in action.get("market") or []:
            if isinstance(m, list) and len(m) >= 3 and m[0] == "SELL":
                out[m[1]] = out.get(m[1], 0.0) + float(m[2])
        return out

    for t in range(len(steps)):
        entry = steps[t][seat]
        monitor.update(
            entry.get("observation") or {},
            seat=seat,
            our_executed_sells=requested_sells(entry.get("action") or {}),
        )
    fert = list(monitor.tracker.opponent_fertilizer_sold)
    return {
        "episode": episode_path,
        "turns": len(monitor.tracker),
        "windows": monitor.n_windows,
        "status": monitor.status(),
        "inferred_opp_fert_mean": float(np.mean(fert)) if fert else 0.0,
        "inferred_opp_fert_max": float(np.max(fert)) if fert else 0.0,
    }


# ---------------------------------------------------------------------------
# Price sensitivity analysis
# ---------------------------------------------------------------------------
def price_sensitivity() -> None:
    """Show how each product's price responds to inventory changes."""
    print(f"\n{'=' * 80}")
    print("PRICE SENSITIVITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"{'Product':>12} {'Base':>5} {'Scarcity':>8} {'T':>4} "
          f"{'@+10':>6} {'@+50':>6} {'@+100':>7} {'@+200':>7} "
          f"{'@-10':>6} {'@-50':>6} {'@-100':>7} {'@-200':>7}")
    print("-" * 100)

    for item in MARKET_PARAMS:
        base = MARKET_PARAMS[item]["base"]
        p0 = price(item, MARKET_I0)
        vals = []
        for delta in [10, 50, 100, 200, -10, -50, -100, -200]:
            inv = MARKET_I0 - delta  # deficit = delta
            p = price(item, inv)
            vals.append(p - p0)

        print(f"{item:>12} {base:5d} {MARKET_PARAMS[item]['below_func']:>8} "
              f"{MARKET_PARAMS[item]['T']:4d} "
              + " ".join(f"{v:+6d}" for v in vals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Full product optimizer.")
    parser.add_argument("--quick", action="store_true", help="Quick grid only")
    parser.add_argument("--product", type=str, help="Single product to optimize")
    parser.add_argument("--drain", type=float, default=12.0, help="Drain rate")
    parser.add_argument("--sensitivity", action="store_true", help="Show price sensitivity")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--hoarding-demo", action="store_true",
                        help="Run synthetic opponent-hoarding BOCD demo")
    parser.add_argument("--hoarding-replay", type=str, metavar="EPISODE_JSON_GZ",
                        help="Run the hoarding monitor over a logged episode")
    parser.add_argument("--replay-seat", type=int, default=0, help="Seat to monitor from")
    args = parser.parse_args()

    if args.hoarding_demo:
        demo = demo_hoarding_detection()
        print("=" * 80)
        print("SIMULTANEOUS-DUMP IMPACT (engine-exact shared-book prices)")
        print("=" * 80)
        for row in demo["dump_impact"]:
            cells = "  ".join(f"+{s}: ${row[f'price_+{s}']}" for s in [0, 25, 50, 100])
            print(f"  {row['item']:>5} (base ${row['base']}, {row['above_func']}): {cells}")
        print(f"\n{'=' * 80}")
        print("BOCD HOARDING DETECTION (selling ~60/window -> 0/window)")
        print(f"{'=' * 80}")
        trig = demo["trigger"]
        if trig:
            print(f"  {OPPONENT_HOARDING_DETECTED} fired at window-day {trig['day']}")
            print(f"    P(changepoint)={trig['p_changepoint']:.3f} (> 0.85)")
            print(f"    pre_mean={trig['pre_mean']:.1f} -> post_mean={trig['post_mean']:.1f}")
        else:
            print("  no trigger (unexpected on synthetic shift)")
        print(f"  windows processed: {demo['n_windows']}")
        return

    if args.hoarding_replay:
        rep = run_hoarding_replay(args.hoarding_replay, seat=args.replay_seat)
        print(f"Replay: {rep['episode']} (seat {args.replay_seat})")
        print(f"  turns={rep['turns']} windows={rep['windows']}")
        print(f"  inferred opp fertilizer/turn: mean={rep['inferred_opp_fert_mean']:.2f} "
              f"max={rep['inferred_opp_fert_max']:.2f}")
        st = rep["status"]
        print(f"  event={st['event']} P={st['p_changepoint']:.3f} "
              f"pre={st['pre_mean']:.1f} post={st['post_mean']:.1f}")
        return

    if args.sensitivity:
        price_sensitivity()
        return

    results = analyze_all_products(quick=args.quick)

    price_sensitivity()

    if args.save:
        # Convert to serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj

        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=make_serializable)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
