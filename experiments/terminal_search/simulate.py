"""Stub simulator for terminal-search experiments (wire unit_model later)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .terminal_search import PRODUCTS


def simulate_terminal_schedule(
    obs: dict[str, Any],
    config: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    """Return a minimal detailed run record compatible with dominates()."""
    private = deepcopy(obs.get("private", {}))
    sold = {item: 0 for item in PRODUCTS}
    rows = []
    for action in actions:
        market = action.get("market") or []
        for order in market:
            if order and order[0] == "SELL" and len(order) >= 3:
                sold[str(order[1])] = sold.get(str(order[1]), 0) + int(order[2])
        rows.append(
            {
                "pre_market_shed": dict(private.get("shed", {})),
                "sold": dict(sold),
                "deposited_by_actor": [{item: 0 for item in PRODUCTS} for _ in private.get("inventories", [{}])],
            }
        )
    return {
        "rows": rows,
        "sold": sold,
        "overflow_units": 0,
        "private": private,
        "farm": deepcopy(obs.get("farms", [{}])[int(obs.get("player", 0))]),
        "actions": deepcopy(actions),
        "events": [],
        "states": [],
    }
