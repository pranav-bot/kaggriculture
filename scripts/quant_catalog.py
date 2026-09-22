#!/usr/bin/env python3
"""Price impact and a sell-timing grid for every product.

The price function is the engine's. Production schedules are the cared
yields. This ranks enterprises before an agent is written.
"""

from __future__ import annotations

import math

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    MARKET_PARAMS,
    market_price,
)

I0 = 10000
DAYS = 30
SHED = 100


def price(item: str, deficit: float) -> int:
    return int(market_price(item, I0 - deficit))


def amp_note(item: str) -> str:
    p = MARKET_PARAMS[item]
    base = p["base"]
    below = price(item, 0)
    up = price(item, 50) - below
    down = price(item, -50) - below
    return (
        f"base {base:3d}  scarcity {p['below_func']:6} T={p['T']:3d}  "
        f"Δ at +50={up:+4d}  Δ at -50={down:+5d}"
    )


def animal_production(
    n: int, first_day: int, interval: int, opening: int, later: int, place_day: int,
) -> list[int]:
    daily = [0] * DAYS
    for i in range(n):
        placed = place_day + i // 2
        for day in range(DAYS):
            since = day - placed - first_day
            if since < 0 or since % interval != 0:
                continue
            daily[day] += opening if since == 0 else later
    return daily


def lump(day: int, units: int) -> list[int]:
    daily = [0] * DAYS
    if 0 <= day < DAYS:
        daily[day] = units
    return daily


def simulate(
    item: str, production: list[int], drain: float, start: int, daily_cap: int,
    floor: int | None = None,
) -> tuple[float, int]:
    """Cash and units sold. Stops under `floor` (default: the base price).

    Drain and sales are split across six ticks, matching shop consumption
    every four hours. A one-shot daily dump overstates how fast the quote falls.
    """
    base = MARKET_PARAMS[item]["base"]
    stop = base if floor is None else floor
    deficit = 0.0
    shed = 0
    cash = 0.0
    sold = 0
    for day in range(DAYS):
        shed += production[day]
        per_tick = max(0, daily_cap // 6)
        extra = daily_cap - per_tick * 6
        for tick in range(6):
            budget = per_tick + (1 if tick < extra else 0)
            if day >= start:
                for _ in range(budget):
                    if shed <= 0:
                        break
                    quoted = price(item, deficit)
                    if quoted < stop:
                        break
                    shed -= 1
                    deficit -= 1.0
                    cash += quoted
                    sold += 1
            deficit += drain / 6.0
        overflow = max(0, shed - SHED)
        shed -= overflow
    return cash, sold


def best_path(item: str, production: list[int], drain: float, floor: int | None = None) -> tuple[int, int, float, int]:
    starts = (0, 8, 12, 16, 20, 24, 27)
    caps = (0, 1, 4, 8, 16, 40, 200)
    best = (0, 0, -1.0, 0)
    for start in starts:
        for cap in caps:
            cash, sold = simulate(item, production, drain, start, cap, floor)
            if cash > best[2]:
                best = (start, cap, cash, sold)
    return best


def show_curves() -> None:
    print("price impact (engine quotes)")
    for item in MARKET_PARAMS:
        print(f"  {item:12} {amp_note(item)}")


def show_enterprises() -> None:
    print("\nbest sell path: start_day, daily cap, cash, units sold")
    herds = {
        "MILK": animal_production(8, 8, 2, 6, 3, 4),
        "WOOL": animal_production(8, 6, 3, 6, 4, 4),
        "EGG": animal_production(8, 4, 1, 4, 2, 4),
    }
    costs = {"MILK": 400 * 8, "WOOL": 500 * 8, "EGG": 300 * 8}
    for item, prod in herds.items():
        for drain in (1, 6, 12, 24):
            start, cap, cash, sold = best_path(item, prod, drain)
            net = cash - costs[item]
            print(
                f"  {item:6} drain {drain:2d}  start {start:2d} cap {cap:3d}  "
                f"cash {cash:8.0f} sold {sold:4d}  net {net:8.0f}"
            )
    crops = {
        "MELON": (lump(12, 8 * 6), 80 * 8, (1,)),
        "STRAWBERRY": (
            [sum(x) for x in zip(lump(10, 8), lump(12, 8), lump(14, 8), lump(16, 8))],
            100 * 8,
            (1, 6, 18),
        ),
        "TOMATO": (
            [sum(x) for x in zip(lump(8, 8), lump(9, 8), lump(10, 8), lump(11, 8))],
            50 * 8,
            (1, 6),
        ),
        "CARROT": (lump(3, 8 * 3), 20 * 8, (1, 12, 24)),
        "WHEAT": (lump(4, 8 * 4), 10 * 8, (1, 12)),
    }
    for item, (prod, cost, drains) in crops.items():
        for drain in drains:
            start, cap, cash, sold = best_path(item, prod, drain)
            print(
                f"  {item:12} drain {drain:2d}  start {start:2d} cap {cap:3d}  "
                f"cash {cash:8.0f} sold {sold:4d}  net {cash - cost:8.0f}"
            )
    fert = [0] * DAYS
    for day in range(8, DAYS):
        fert[day] = 12
    print("\nfertilizer, 12/day from day 8, drain 0")
    for floor in (1, 40, 60, 80, 100):
        start, cap, cash, sold = best_path("FERTILIZER", fert, 0, floor=floor)
        print(
            f"  floor {floor:3d}  start {start:2d} cap {cap:3d}  "
            f"cash {cash:8.0f} sold {sold:4d}"
        )


def scipy_fertilizer() -> None:
    """Differential evolution on fertilizer floor and daily cap. Drain is 0."""
    import numpy as np
    from scipy.optimize import differential_evolution

    fert = [0] * DAYS
    for day in range(8, DAYS):
        fert[day] = 12

    def objective(x: np.ndarray) -> float:
        floor, cap = x
        cash, _sold = simulate("FERTILIZER", fert, 0.0, 0, int(round(cap)), floor=int(round(floor)))
        return -cash

    result = differential_evolution(
        objective,
        bounds=[(1, 100), (1, 24)],
        seed=0,
        popsize=8,
        maxiter=20,
        atol=10,
        workers=1,
        updating="immediate",
    )
    floor, cap = result.x
    cash, sold = simulate(
        "FERTILIZER", fert, 0.0, 0, int(round(cap)), floor=int(round(floor)),
    )
    print(
        f"\nscipy fertilizer  floor≈{floor:.0f} cap≈{cap:.0f}  "
        f"cash {cash:.0f} sold {sold}"
    )


def main() -> None:
    show_curves()
    show_enterprises()
    scipy_fertilizer()


if __name__ == "__main__":
    main()
