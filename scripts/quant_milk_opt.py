#!/usr/bin/env python3
"""Fit a four-stage milk sell schedule with scipy.

The simulator is the official milk curve, a cared-cow yield, and a shed
cap. It is not the full farm. It ranks sell paths before an agent is built.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

BASE = 160.0
AMP_BELOW = 0.60 * BASE / math.sqrt(122.0)
AMP_ABOVE = 1.60 * BASE / 122.0
SHED = 100


def milk_price(deficit: float) -> float:
    if deficit >= 0:
        return BASE + AMP_BELOW * math.sqrt(deficit)
    return max(1.0, BASE - AMP_ABOVE * (-deficit))


def cow_yield(day: int, placed: int) -> int:
    """Yield that appears at dawn of `day` for one cow placed on `placed`."""
    since = day - placed - 8
    if since < 0 or since % 2 != 0:
        return 0
    # First cycle stacks every cared day of the wait, capped at 6.
    # Later cycles: one off-day of care plus the reset, so 3.
    if since == 0:
        return 6
    return 3


def place_days(cap: int) -> list[int]:
    """Slow ramp: 4 / 8 / 14 / cap, clipped to the cap."""
    schedule = []
    targets = [(4, 4), (8, 8), (11, 14), (14, cap)]
    owned = 0
    for day, target in targets:
        target = min(cap, target)
        while owned < target:
            schedule.append(day)
            owned += 1
    return schedule


def simulate(fractions: np.ndarray, drain: float, cap: int) -> float:
    """Cash from milk only. Fractions are per stage, in [0, 1]."""
    stages = [(0, 10), (11, 16), (17, 22), (23, 29)]
    placed = place_days(cap)
    deficit = 0.0
    shed = 0
    cash = 0.0
    # Animals and wheat are a fixed bill so paths differ only by milk sales.
    cash -= 400 * cap + 25 * cap * 20
    for day in range(30):
        for p in placed:
            shed += cow_yield(day, p)
        frac = 0.0
        for (lo, hi), f in zip(stages, fractions):
            if lo <= day <= hi:
                frac = float(f)
                break
        # Sell along the curve, one unit at a time, and stop under base.
        want = int(round(frac * shed))
        for _ in range(want):
            if shed <= 0:
                break
            price = milk_price(deficit)
            if price < BASE:
                break
            shed -= 1
            deficit -= 1.0
            cash += price
        overflow = max(0, shed - SHED)
        shed -= overflow
        deficit += drain
    return cash


GRID = (0.0, 0.1, 0.25, 0.5, 1.0)


def optimise(drain: float, cap: int) -> tuple[np.ndarray, float]:
    """Grid first, then Nelder-Mead from the best cell.

    A single start at (0, 0.2, 0.5, 1) stayed there. The grid is what
    decides the basin.
    """
    best_x = np.zeros(4)
    best_cash = -1e18
    for a in GRID:
        for b in GRID:
            for c in GRID:
                for d in GRID:
                    x = np.array([a, b, c, d])
                    cash = simulate(x, drain, cap)
                    if cash > best_cash:
                        best_cash = cash
                        best_x = x

    def objective(x: np.ndarray) -> float:
        return -simulate(np.clip(x, 0.0, 1.0), drain, cap)

    result = minimize(
        objective,
        best_x,
        method="Nelder-Mead",
        options={"maxiter": 60, "xatol": 0.02, "fatol": 20.0},
    )
    refined = np.clip(result.x, 0.0, 1.0)
    refined_cash = simulate(refined, drain, cap)
    if refined_cash > best_cash:
        return refined, refined_cash
    return best_x, best_cash


def baselines(drain: float, cap: int) -> dict[str, float]:
    paths = {
        "hold": np.array([0.0, 0.0, 0.0, 1.0]),
        "half": np.array([0.0, 0.5, 0.5, 1.0]),
        "match": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    return {name: simulate(frac, drain, cap) for name, frac in paths.items()}


def simulate_band(
    hi: float, lo: float, hi_units: float, lo_units: float,
    drain: float, cap: int, opponent: float,
) -> float:
    """Sell a daily unit cap chosen by the quoted price.

    An opponent, if any, sells `opponent` units a day from day 14, ahead of
    us, and stops under the base price. Their cash is not ours.
    """
    placed = place_days(cap)
    deficit = 0.0
    shed = 0
    cash = 3000.0 - 400 * cap
    for day in range(30):
        for p in placed:
            shed += cow_yield(day, p)
        if day >= 14 and opponent > 0:
            for _ in range(int(opponent)):
                if milk_price(deficit) < BASE or deficit <= 0:
                    break
                deficit -= 1.0
        price = milk_price(deficit)
        if day >= 27 and price >= BASE:
            daily = shed
        elif price >= hi:
            daily = hi_units
        elif price >= lo:
            daily = lo_units
        else:
            daily = 0
        for _ in range(int(daily)):
            if shed <= 0:
                break
            price = milk_price(deficit)
            if price < BASE:
                break
            shed -= 1
            deficit -= 1.0
            cash += price
        overflow = max(0, shed - SHED)
        shed -= overflow
        deficit += drain
    return cash


def optimise_band(drain: float, cap: int, opponent: float) -> tuple[np.ndarray, float]:
    """Differential evolution on (hi price, lo price, hi units, lo units)."""
    from scipy.optimize import differential_evolution

    def objective(x: np.ndarray) -> float:
        hi, lo, hi_u, lo_u = x
        if lo > hi:
            lo = hi
        return -simulate_band(hi, lo, hi_u, lo_u, drain, cap, opponent)

    result = differential_evolution(
        objective,
        bounds=[(170, 280), (160, 250), (0, 36), (0, 24)],
        seed=0,
        popsize=6,
        mutation=0.5,
        recombination=0.7,
        atol=20,
        tol=0.01,
        workers=1,
        updating="immediate",
        maxiter=12,
    )
    hi, lo, hi_u, lo_u = result.x
    if lo > hi:
        lo = hi
    best = np.array([hi, lo, hi_u, lo_u])
    return best, simulate_band(hi, lo, hi_u, lo_u, drain, cap, opponent)


def main() -> None:
    print(f"amp_below={AMP_BELOW:.3f} amp_above={AMP_ABOVE:.3f}")
    print(f"{'drain':>6} {'cap':>4} {'hold':>10} {'half':>10} {'sell_all':>10} {'scipy':>10} fractions")
    for drain in (8.0, 18.0, 30.0):
        for cap in (12, 18):
            base = baselines(drain, cap)
            frac, cash = optimise(drain, cap)
            rounded = " ".join(f"{v:.2f}" for v in frac)
            print(
                f"{drain:6.0f} {cap:4d} {base['hold']:10.0f} {base['half']:10.0f} "
                f"{base['match']:10.0f} {cash:10.0f}  {rounded}"
            )
    print("\nprice-band policy (hi, lo, units when rich, units when mid)")
    print(f"{'drain':>6} {'cap':>4} {'opp':>4} {'cash':>10}  hi lo rich mid")
    for drain in (12.0, 24.0):
        for cap in (14,):
            for opp in (0.0, 12.0, 24.0):
                params, cash = optimise_band(drain, cap, opp)
                text = " ".join(f"{v:.1f}" for v in params)
                print(f"{drain:6.0f} {cap:4d} {opp:4.0f} {cash:10.0f}  {text}")


if __name__ == "__main__":
    main()
