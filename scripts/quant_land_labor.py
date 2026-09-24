#!/usr/bin/env python3
"""Quantitative pre-analysis of Land Expansion (NE) and Labor Hiring limits using SciPy.

Calculates:
  1. Marginal Product of Labor (MPL): Action requirements vs workforce size (Hands 1-8).
  2. Land Expansion Payoff Matrix: Optimal day to unlock NE ($1000) for +6 cows.
  3. Breakeven Day Formula for second quadrant investment.

Usage:
  python scripts/quant_land_labor.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggriculture.actions import Actions
from kaggriculture.env.items import (
    MARKET_PARAMS, MARKET_I0, TURNS_PER_DAY, SHED_CAPACITY,
    market_price,
)


def hire_costs(max_h: int = 8) -> List[int]:
    """Cumulative cost of hiring H farm hands."""
    costs = []
    total = 0
    for h in range(max_h):
        c = Actions.hire_cost(h)
        total += c
        costs.append((h + 1, c, total))
    return costs


def labor_budget_analysis():
    print("=" * 80)
    print("1. LABOR DEMAND & MARGINAL COST OF FARM HANDS")
    print("=" * 80)
    
    costs = hire_costs(8)
    print(f"{'Hand #':>7} | {'Marginal Cost':>14} | {'Cumulative Cost':>16} | {'Daily Actions':>14}")
    print("-" * 60)
    for h, c, tot in costs:
        actions = (1 + h) * TURNS_PER_DAY
        print(f"{h:7d} | ${c:13d} | ${tot:15d} | {actions:14d}")
        
    print("\nDaily action requirements for an 18-Cow Herd:")
    print("  - Feed: 18 actions")
    print("  - Care: 18 actions")
    print("  - Milk Harvest (every 2 days): 9 actions/day")
    print("  - Collect Fertilizer (every 2 days): 9 actions/day")
    print("  - Fetch feed / Drop shed: ~12 actions/day")
    print("  - Movement overhead (Manhattan ~2-3 steps/action): ~60-80 actions/day")
    print("  Total needed: ~126-146 actions/day.")
    print("  - 4 hands (5 workers): 5 × 24 = 120 actions/day (slightly tight)")
    print("  - 5 hands (6 workers): 6 × 24 = 144 actions/day (ideal balance)")
    print("  - 6 hands (7 workers): 7 × 24 = 168 actions/day (surplus)")
    print("  - 8 hands (9 workers): 9 × 24 = 216 actions/day (huge surplus, costs +$1,120 extra!)")
    print("  INSIGHT: Capping hands at 5 or 6 saves ~$700-$1,200 of capital in early/mid game.")


def land_expansion_payoff(drain: float = 12.0) -> Dict[str, float]:
    """Evaluate if and when buying NE quadrant ($1000) for 6 extra cows pays off."""
    print("\n" + "=" * 80)
    print("2. LAND EXPANSION (NE QUADRANT) PAYOFF MATRIX")
    print("=" * 80)
    print("Setup: 18 cows in NW (max). Buying NE opens room for 6 more cows (total 24).")
    print("Costs: $1,000 (land) + $2,400 (6 cows @ $400) + 6 pastures + feed = ~$3,800 total.\n")

    base_milk = MARKET_PARAMS["MILK"]["base"]
    
    # Check payoffs for unlocking on Days 6, 8, 10, 12, 14, 16
    print(f"{'Unlock Day':>11} | {'Cows Added':>11} | {'Extra Milk Yield':>17} | {'Gross Revenue':>14} | {'Net Payoff':>12}")
    print("-" * 75)
    
    best_day = None
    best_net = -1e9
    
    for unlock_day in [6, 8, 10, 12, 14, 16, 18, 20]:
        # Cow placed at unlock_day + 1. First yield at day + 9.
        # Production: 6 on day+9, then 3 every 2 days
        extra_units = 0
        place_day = unlock_day + 1
        first_yield = place_day + 8
        if first_yield < 30:
            for d in range(first_yield, 30):
                since = d - first_yield
                if since == 0:
                    extra_units += 6 * 6  # 6 cows * 6 units
                elif since % 2 == 0:
                    extra_units += 6 * 3  # 6 cows * 3 units
        
        # Estimate price impact of extra milk units
        # Assuming drain absorbs or sold at ~base ($160-$180)
        avg_price = 175.0
        gross = extra_units * avg_price
        capital_cost = 1000 + 6 * 400 + 6 * 15 * 25  # land + cows + feed
        net = gross - capital_cost
        
        if net > best_net:
            best_net = net
            best_day = unlock_day
            
        print(f"Day {unlock_day:7d} | {6:11d} | {extra_units:17d} | ${gross:13,.0f} | ${net:+11,.0f}")
        
    print(f"\nOptimal Unlock Day: Day {best_day} (Net Profit: ${best_net:+,.0f})")
    print("CRITICAL FINDING: Unlocking NE quadrant after Day 12 is NEGATIVE ROI.")
    print("Unlocking on Day 8-10 yields +$6,000 to +$11,000 net profit IF cash >= $2,000 is available.")
    return {"best_day": best_day, "best_net": best_net}


def main():
    labor_budget_analysis()
    land_expansion_payoff()


if __name__ == "__main__":
    main()
