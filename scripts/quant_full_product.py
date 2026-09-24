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
import json
import math
import sys
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
    args = parser.parse_args()

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
