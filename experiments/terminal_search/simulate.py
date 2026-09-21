"""Minimal terminal-week schedule simulation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .state import TerminalState
from .terminal_search import PRODUCTS
from .unit_step import apply_turn


def _empty_product_map() -> dict[str, int]:
    return {item: 0 for item in PRODUCTS}


def simulate_terminal_schedule(
    obs: dict[str, Any],
    config: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    """Replay *actions* and return a run record compatible with dominates()."""
    board_size = int(config.get("boardSize", 10) or 10)
    state = TerminalState.from_obs(obs)
    sold = _empty_product_map()
    rows: list[dict[str, Any]] = []

    for action in actions:
        pre_market_shed = dict(state.private.get("shed") or {})
        deposited = apply_turn(state, action, board_size=board_size)
        for order in action.get("market") or []:
            if isinstance(order, list) and order and str(order[0]).upper() == "SELL" and len(order) >= 3:
                item = str(order[1]).upper()
                qty = int(order[2])
                sold[item] = sold.get(item, 0) + qty
                shed = dict(state.private.get("shed") or {})
                shed[item] = max(0, int(shed.get(item, 0) or 0) - qty)
                state.private["shed"] = shed

        dep_maps = []
        for dep in deposited:
            row_map = _empty_product_map()
            for item, qty in dep.items():
                if item in row_map:
                    row_map[item] = int(qty)
            dep_maps.append(row_map)

        rows.append(
            {
                "pre_market_shed": pre_market_shed,
                "sold": dict(sold),
                "deposited_by_actor": dep_maps or [_empty_product_map()],
            }
        )

    return {
        "rows": rows,
        "sold": sold,
        "overflow_units": state.overflow_units,
        "private": state.private,
        "farm": state.farm,
        "actions": deepcopy(actions),
        "events": [],
        "states": [],
    }
