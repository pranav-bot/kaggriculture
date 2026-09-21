"""Seven-turn terminal deposit search (research/03 + research/05 B1 prototype)."""

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


def _merge_turn(base: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    if proposal.get("farmer"):
        merged["farmer"] = deepcopy(proposal["farmer"])
    if proposal.get("hands"):
        merged["hands"] = deepcopy(proposal["hands"])
    return merged


def plan_terminal(
    obs: dict[str, Any],
    config: dict[str, Any],
    baseline_remaining: list[dict[str, Any]],
    simulate: Callable[..., dict[str, Any]],
    *,
    max_simulations: int = 64,
    proposals_per_actor: int = 4,
) -> dict[str, Any]:
    """Search harvest→DROP proposals over a 7-turn horizon; 200 ms budget per call."""
    begun = perf_counter()
    fallback: dict[str, Any] = {
        "accepted": False,
        "reason": "",
        "actions": None,
        "simulations": 0,
        "planning_ms": 0.0,
        "changes": [],
    }
    simulations = 0

    def over_budget() -> bool:
        return (perf_counter() - begun) * 1000.0 >= MAX_SEARCH_MS

    def finish(**extra: Any) -> dict[str, Any]:
        return {
            **fallback,
            **extra,
            "simulations": simulations,
            "planning_ms": (perf_counter() - begun) * 1000.0,
        }

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

        baseline_run = simulate(obs, config, baseline_remaining, detailed=True)
        simulations += 1
        prices = {item: max(1.0, float(obs.get("market", {}).get("prices", {}).get(item, 1))) for item in PRODUCTS}
        baseline_value = plan_value(baseline_run, prices)
        current = deepcopy(baseline_remaining)
        best_value = baseline_value
        changes: list[dict[str, Any]] = []

        from .proposals import propose_harvest_drop_routes

        proposals = propose_harvest_drop_routes(
            obs,
            proposals_per_actor=proposals_per_actor,
        )

        for turn_idx in range(TERMINAL_HORIZON):
            if over_budget() or simulations >= max_simulations:
                break
            local_best = current[turn_idx]
            local_value = best_value
            for proposal in proposals:
                if over_budget() or simulations >= max_simulations:
                    break
                trial_actions = deepcopy(current)
                trial_actions[turn_idx] = _merge_turn(trial_actions[turn_idx], proposal)
                trial_run = simulate(obs, config, trial_actions, detailed=True)
                simulations += 1
                trial_value = plan_value(trial_run, prices)
                if trial_value > local_value and dominates(trial_run, baseline_run):
                    local_best = trial_actions[turn_idx]
                    local_value = trial_value
            if local_value > best_value:
                current[turn_idx] = local_best
                best_value = local_value
                changes.append({"turn": turn_idx, "value": local_value})

        if over_budget():
            raise Unsupported("latency budget exceeded during proposal search")

        if best_value <= baseline_value:
            return finish(reason="no positive physical delivery gain")

        physical = simulate(obs, config, current, detailed=True)
        simulations += 1
        if not dominates(physical, baseline_run):
            raise Unsupported("no zero-overflow dominating continuation")

        return finish(
            accepted=True,
            reason="joint physical dominance",
            baseline=deepcopy(baseline_remaining),
            actions=current,
            changes=changes,
        )
    except (Unsupported, KeyError, TypeError, ValueError, IndexError) as exc:
        return finish(reason=str(exc))
