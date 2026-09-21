"""Seven-turn terminal deposit search skeleton (research/03; days 27–29 only)."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any, Callable

from kaggriculture.env.items import PRODUCTS_LIST

TERMINAL_START_STEP = 712
TERMINAL_FINAL_STEP = 718
TERMINAL_HORIZON = TERMINAL_FINAL_STEP - TERMINAL_START_STEP + 1
TERMINAL_DAY_MIN = 27
TERMINAL_DAY_MAX = 29
MAX_SEARCH_MS = 200.0

PRODUCTS = tuple(PRODUCTS_LIST)


class Unsupported(ValueError):
    pass


def terminal_search_window(obs: dict[str, Any]) -> bool:
    day = int(obs.get("day", 0) or 0)
    return TERMINAL_DAY_MIN <= day <= TERMINAL_DAY_MAX


def _shed_access_tiles(board_size: int = 10) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _walk(start: tuple[int, int], end: tuple[int, int]) -> list[list[str]]:
    x, y = start
    tx, ty = end
    return (
        [["EAST"]] * max(0, tx - x)
        + [["WEST"]] * max(0, x - tx)
        + [["SOUTH"]] * max(0, ty - y)
        + [["NORTH"]] * max(0, y - ty)
    )


def return_to_shed_drop(pos: tuple[int, int], board_size: int = 10) -> list[list[Any]]:
    targets = _shed_access_tiles(board_size)
    target = min(targets, key=lambda xy: (abs(pos[0] - xy[0]) + abs(pos[1] - xy[1]), targets.index(xy)))
    return _walk(pos, target) + [["DROP"]]


def dominates(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if candidate.get("overflow_units"):
        return False
    for new, old in zip(candidate.get("rows", []), baseline.get("rows", [])):
        if any(new.get("pre_market_shed", {}).get(item, 0) < old.get("pre_market_shed", {}).get(item, 0) for item in PRODUCTS):
            return False
        if any(new.get("sold", {}).get(item, 0) < old.get("sold", {}).get(item, 0) for item in PRODUCTS):
            return False
        for deposited_new, deposited_old in zip(new.get("deposited_by_actor", []), old.get("deposited_by_actor", [])):
            if any(deposited_new.get(item, 0) < deposited_old.get(item, 0) for item in PRODUCTS):
                return False
    return True


def plan_value(run: dict[str, Any], prices: dict[str, float]) -> float:
    shed = run.get("private", {}).get("shed", {})
    sold = run.get("sold", {})
    return sum((int(sold.get(item, 0)) + int(shed.get(item, 0))) * prices.get(item, 1.0) for item in PRODUCTS)


def plan_terminal(
    obs: dict[str, Any],
    config: dict[str, Any],
    baseline_remaining: list[dict[str, Any]],
    simulate: Callable[..., dict[str, Any]],
    *,
    max_simulations: int = 64,
    proposals_per_actor: int = 4,
) -> dict[str, Any]:
    """Search for a dominating 7-turn schedule; abort if planning exceeds MAX_SEARCH_MS."""
    begun = perf_counter()
    fallback: dict[str, Any] = {
        "accepted": False,
        "reason": "",
        "actions": None,
        "simulations": 0,
        "planning_ms": 0.0,
    }

    def over_budget() -> bool:
        return (perf_counter() - begun) * 1000.0 >= MAX_SEARCH_MS

    try:
        if not terminal_search_window(obs):
            raise Unsupported("terminal search only runs on days 27–29")
        step = int(obs.get("step", -1))
        if step != TERMINAL_START_STEP:
            raise Unsupported(f"planning requires step {TERMINAL_START_STEP}")
        if len(baseline_remaining) != TERMINAL_HORIZON:
            raise Unsupported(f"baseline must contain {TERMINAL_HORIZON} turns")

        if over_budget():
            raise Unsupported("latency budget exceeded before baseline simulation")

        baseline = simulate(obs, config, baseline_remaining, detailed=True)
        prices = {item: max(1.0, float(obs.get("market", {}).get("prices", {}).get(item, 1))) for item in PRODUCTS}
        baseline_value = plan_value(baseline, prices)
        best_value = baseline_value
        current = deepcopy(baseline_remaining)
        simulations = 1

        # Skeleton: proposal generation hooks live here (harvest→DROP suffixes per actor).
        proposals: list[list[dict[str, Any]]] = []
        _ = (max_simulations, proposals_per_actor, proposals)

        if over_budget():
            raise Unsupported("latency budget exceeded during proposal search")

        if best_value <= baseline_value:
            return {
                **fallback,
                "reason": "no positive physical delivery gain",
                "simulations": simulations,
                "planning_ms": (perf_counter() - begun) * 1000,
            }

        physical = simulate(obs, config, current)
        simulations += 1
        if not dominates(physical, baseline):
            raise Unsupported("no zero-overflow dominating continuation")

        return {
            "accepted": True,
            "reason": "joint physical dominance",
            "baseline": deepcopy(baseline_remaining),
            "actions": current,
            "simulations": simulations,
            "planning_ms": (perf_counter() - begun) * 1000,
        }
    except (Unsupported, KeyError, TypeError, ValueError, IndexError) as exc:
        return {
            **fallback,
            "reason": str(exc),
            "simulations": fallback.get("simulations", 0),
            "planning_ms": (perf_counter() - begun) * 1000,
        }
