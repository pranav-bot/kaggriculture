#!/usr/bin/env python3
"""CLI runner and evaluator for the Retrograde DP & MILP Liquidation Solver.

Ingests Day 20 (Hour 0) farm state, runs the liquidation solver, prints the daily
liquidation schedule table with per-tick quotas, and compares total terminal cash
against a naïve Day 29 bulk dump.

Usage:
    python scripts/solve_liquidation.py --demo
    python scripts/solve_liquidation.py --backend milp
    python scripts/solve_liquidation.py --backend scipy
    python scripts/solve_liquidation.py --backend dp
    python scripts/solve_liquidation.py --replay replays/other_agents/rank1/112521191.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kaggriculture.env.items import (
    MARKET_I0,
    MARKET_PARAMS,
    PRODUCTS_LIST,
    market_price,
)
from kaggriculture.liquidation_solver import (
    GAME_OVER_DAY,
    PLANNING_HORIZON_DAYS,
    START_DAY,
    END_DAY,
    TICKS_PER_DAY,
    LiquidationIntent,
    RetrogradeDPSolver,
    TerminalLiquidationSolver,
    TownShopModel,
    calculate_batch_revenue,
    get_forbidden_crops,
    ingest_state_on_day_20,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("liquidation_solver")

# ANSI Colors
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def build_default_day20_state() -> Dict[str, Any]:
    """Generates a representative competitive Day 20 farm state for benchmarking."""
    tiles = [[None] * 10 for _ in range(10)]
    # Cows (Pastures)
    for r in range(4):
        for c in range(3):
            tiles[r][c] = {"kind": "PASTURE", "animal": "COW", "placed_day": 4, "yield_units": 0}
    # Sheep (Pastures)
    for r in range(4):
        for c in range(3, 5):
            tiles[r][c] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 6, "yield_units": 0}
    # Strawberries (Plants)
    for r in range(5, 8):
        for c in range(4):
            tiles[r][c] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 10, "yield_units": 0}

    return {
        "step": 480,
        "day": 20,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 28500.0,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [[4, 5], [5, 4], [5, 5]],
                "unlocked_quadrants": ["NW", "NE", "SW"],
            },
            {
                "money": 24000.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "shed": {
                "MILK": 24,
                "WOOL": 18,
                "STRAWBERRY": 22,
                "FERTILIZER": 40,
                "WHEAT": 16,
            },
            "seeds": {
                "STRAWBERRY": 12,
                "MELON": 8,
                "TOMATO": 6,
                "WHEAT": 10,
                "CARROT": 4,
            },
            "inventories": [{}, {}, {}],
        },
        "market": {
            "inventory": {
                "MILK": 10000,
                "WOOL": 10000,
                "STRAWBERRY": 10000,
                "FERTILIZER": 10000,
                "WHEAT": 10000,
                "CARROT": 10000,
                "TOMATO": 10000,
                "MELON": 10000,
                "EGG": 10000,
            },
            "prices": {
                "MILK": 160,
                "WOOL": 200,
                "STRAWBERRY": 120,
                "FERTILIZER": 100,
                "WHEAT": 25,
                "CARROT": 35,
                "TOMATO": 60,
                "MELON": 250,
                "EGG": 50,
            },
            "unlocked_shops": [
                "BAKERY",
                "PIZZA_SHOP",
                "BRUNCH_SPOT",
                "YARN_STORE",
                "ICE_CREAM_SHOP",
                "PET_CAFE",
                "SMOOTHIE_SHOP",
            ],
        },
    }


def compute_naive_day29_dump_revenue(
    initial_shed: Dict[str, int],
    daily_yields: Dict[str, List[int]],
    market_inv: Optional[Dict[str, int]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Calculates terminal revenue if the agent dumps 100% of accumulated assets on Day 29."""
    m_inv = market_inv or {p: MARKET_I0 for p in PRODUCTS_LIST}
    total_rev = 0.0
    per_product: Dict[str, float] = {}

    for p in PRODUCTS_LIST:
        tot_qty = initial_shed.get(p, 0) + sum(daily_yields.get(p, []))
        if tot_qty > 0:
            rev = calculate_batch_revenue(p, tot_qty, m_inv.get(p, MARKET_I0))
            per_product[p] = rev
            total_rev += rev

    return total_rev, per_product


def print_liquidation_schedule(
    schedule: Dict[int, LiquidationIntent],
    state: Any,
    naive_rev: float,
    opt_rev: float,
) -> None:
    """Prints a beautiful console report detailing the optimal 10-day liquidation plan."""
    print("\n" + "=" * 100)
    print(f"{BOLD}{CYAN}KAGGRICULTURE TERMINAL LIQUIDATION SCHEDULE (DAYS 20–29){RESET}")
    print("=" * 100)
    print(f"Initial Liquid Cash: ${state.initial_cash:,.0f} | Active Animals: {state.animal_counts} | Crops: {state.field_crop_counts}")
    print(f"Town Shops Unlocked: {', '.join(state.unlocked_shops)}")
    print(f"Seed Planting Prohibitions on Day 20: {', '.join(state.prohibited_seeds)}")
    print("-" * 100)

    # Header
    print(f"{BOLD}{'Day':<5} {'Sell Quotas (Daily Total)':<45} {'Per-Tick Quotas (6 ticks/day)':<32} {'Day Rev':<12}{RESET}")
    print("-" * 100)

    for day in sorted(schedule.keys()):
        intent = schedule[day]
        quotas_str = ", ".join(f"{p}:{q}" for p, q in intent.sell_quotas.items()) if intent.sell_quotas else "None"
        tick_str = ", ".join(f"{p}:{q}" for p, q in intent.tick_sell_quotas.items()) if intent.tick_sell_quotas else "None"
        day_rev_str = f"${intent.projected_revenue:,.0f}"

        highlight = BOLD if day in (20, 29) else ""
        flush_tag = f" {YELLOW}[TERMINAL FLUSH]{RESET}" if intent.terminal_flush else ""
        print(f"{highlight}{day:<5} {quotas_str:<45} {tick_str:<32} {day_rev_str:<12}{RESET}{flush_tag}")

    print("=" * 100)
    print(f"{BOLD}COMPARATIVE FINANCIAL IMPACT:{RESET}")
    print(f"  • Naïve Day 29 Dump Revenue:       {RED}${naive_rev:,.2f}{RESET} (Catastrophic price crash to $1 floor)")
    print(f"  • Optimal Solver Schedule Revenue: {GREEN}${opt_rev:,.2f}{RESET} (Absorbed smoothly by town shop ticks)")
    delta = opt_rev - naive_rev
    pct = (delta / naive_rev * 100.0) if naive_rev > 0 else 0.0
    print(f"  • Total Net Gain from Solver:      {BOLD}{GREEN}+${delta:,.2f} (+{pct:.1f}%){RESET}")
    print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrograde DP & MILP Liquidation Solver for Kaggriculture.")
    parser.add_argument("--backend", choices=["milp", "scipy", "retrograde", "dp"], default="milp", help="Optimization solver backend")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demonstration on synthetic Day 20 state")
    parser.add_argument("--replay", type=str, default=None, help="Path to replay json/json.gz to extract Day 20 state from")
    args = parser.parse_args()

    # Ingest observation
    if args.replay and Path(args.replay).is_file():
        logger.info("Loading replay state from %s...", args.replay)
        open_fn = gzip.open if args.replay.endswith(".gz") else open
        with open_fn(args.replay, "rt", encoding="utf-8") as f:
            data = json.load(f)
        # Find step 480 (Day 20 Hour 0)
        steps = data.get("steps") or data
        obs = None
        for s in steps:
            s_obs = s[0].get("observation") if isinstance(s, list) and s else s.get("observation")
            if s_obs and int(s_obs.get("day", 0)) == 20 and int(s_obs.get("hour", 0)) == 0:
                obs = s_obs
                break
        if obs is None:
            logger.warning("Could not find Day 20 Hour 0 in replay; falling back to demo state.")
            obs = build_default_day20_state()
    else:
        obs = build_default_day20_state()

    logger.info("Initializing TerminalLiquidationSolver with backend '%s'...", args.backend)
    solver = TerminalLiquidationSolver(solver_backend=args.backend)
    schedule = solver.plan_liquidation(obs)
    state = solver._cached_day20_state

    # Calculate optimal revenue vs naive dump revenue
    opt_rev = sum(intent.projected_revenue for intent in schedule.values())
    naive_rev, _ = compute_naive_day29_dump_revenue(
        initial_shed=state.shed_inventory,
        daily_yields=state.daily_projected_yields,
        market_inv=state.market_inventory,
    )

    print_liquidation_schedule(schedule, state, naive_rev, opt_rev)


if __name__ == "__main__":
    main()
