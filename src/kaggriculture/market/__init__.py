"""Midgame market guardrail: O(1) safe-volume math for the Beam Search layer.

The shared wholesale book starts at a 10,000-unit baseline. Above it, Wool
falls QUADRATICALLY (P = 200 - 0.058*dx^2) and Milk LINEARLY. Dumping blindly
(e.g. 79 Wool at once -> $1 floor) destroys our own margins, so every
proposed SELL_ALL is clamped algebraically to the volume whose LAST unit
still clears a minimum margin. No lookahead, no simulation loops: closed-form
pricing inverses only (~1us per call).

Past turn 600 (Day 25) the manager disables itself and yields all market
decisions to the separate terminal liquidation solver.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from kaggriculture.env.items import MARKET_I0, MARKET_PARAMS, market_price

TERMINAL_HANDOFF_TURN = 600  # Day 25: terminal solver takes over
BASELINE_INVENTORY = MARKET_I0  # 10,000-unit wholesale baseline

# Default per-unit margin floors (tunable per call).
DEFAULT_MIN_MARGIN = {"WOOL": 150.0, "MILK": 80.0}

# Beam macro-intent each item's uncapped sale maps to (24-class space).
SELL_INTENT = {"MILK": "DUMP_MILK", "WOOL": "DUMP_WOOL"}


def _above_amplitude(item: str) -> tuple:
    """(base, amp, func) for the surplus side of the book."""
    p = MARKET_PARAMS[str(item).upper()]
    base, T, func = float(p["base"]), float(p["T"]), p["above_func"]
    from kaggriculture.env.items import shape_func
    amp = float(p["above_target"]) * base / shape_func(func, T, T)
    return base, amp, func


def _max_surplus(base: float, amp: float, func: str, margin: float) -> float:
    """Largest surplus x with price(base - amp*f(x)) >= margin (algebraic)."""
    if margin >= base:
        return 0.0
    room = (base - margin) / amp
    if func == "linear":
        return room
    if func == "sq":
        return math.sqrt(max(0.0, room))
    if func == "sqrt":
        return max(0.0, room) ** 2
    if func == "log":
        return math.exp(max(0.0, room)) - 1.0
    raise ValueError(f"no closed-form inverse for pricing func {func!r}")


def calculate_safe_sale_volume(
    current_price: int,
    baseline_inventory: int,
    item_type: str,
    min_margin_threshold: float,
) -> int:
    """Maximum integer units sellable before the LAST unit's revenue < margin.

    The k-th unit of a sale executes at the pre-sale book level, so the
    marginal (last) unit of an n-unit sale sees surplus
    x = (baseline_inventory - MARKET_I0) + n - 1. Requiring
    price(x) >= min_margin_threshold gives n <= X_max - surplus + 1 with
    X_max from the inverse pricing curve. `current_price` is accepted for
    call-site convenience (it equals market_price at baseline_inventory).
    """
    item = str(item_type).upper()
    base, amp, func = _above_amplitude(item)
    inv = int(baseline_inventory)
    surplus = inv - MARKET_I0
    x_max = _max_surplus(base, amp, func, float(min_margin_threshold))
    # Tiny epsilon: exact-boundary float dust must not steal a unit, and
    # must not grant one either -- verified against engine rounding in tests.
    n = int(math.floor(x_max - surplus + 1.0 + 1e-9))
    return max(0, n)


def micro_batch_order(item: str, qty: int) -> list:
    """Mechanical market order for a capped sale."""
    return ["SELL", str(item).upper(), max(0, int(qty))]


def is_midgame(turn_number: int) -> bool:
    """True on Days 1-24 (turns 0-599). At turn >= 600 yield to terminal."""
    try:
        return int(turn_number) < TERMINAL_HANDOFF_TURN
    except (TypeError, ValueError):
        return False


class MidgameMarketManager:
    """Strict guardrail between step-level Beam Search and the shared book."""

    def __init__(
        self,
        min_margins: Optional[Mapping[str, float]] = None,
        handoff_turn: int = TERMINAL_HANDOFF_TURN,
    ) -> None:
        self.min_margins = {**DEFAULT_MIN_MARGIN, **dict(min_margins or {})}
        self.handoff_turn = int(handoff_turn)
        self.intercepts = 0
        self.yields = 0

    def safe_volume(self, item: str, market_inventory: int) -> int:
        """O(1) safe sale volume at the current book level."""
        item = str(item).upper()
        margin = float(self.min_margins.get(item, 0.0))
        price = market_price(item, int(market_inventory))
        return calculate_safe_sale_volume(price, int(market_inventory), item, margin)

    def guard_sell_intent(
        self,
        item: str,
        requested_qty: int,
        market_inventory: int,
        turn_number: int,
    ) -> Dict[str, Any]:
        """Intercept a proposed SELL_ALL; clamp to MICRO_BATCH_SELL if unsafe.

        Returns a decision dict the beam layer executes directly. Past the
        handoff turn the manager is inert and yields to the terminal solver.
        """
        item = str(item).upper()
        requested = max(0, int(requested_qty))
        if not is_midgame(turn_number) or turn_number >= self.handoff_turn:
            self.yields += 1
            return {"action": "YIELD_TO_TERMINAL", "item": item,
                    "requested_qty": requested, "reason": "terminal-phase handoff"}
        safe = self.safe_volume(item, market_inventory)
        base_intent = SELL_INTENT.get(item, "SELL_MARKET")
        if requested <= safe:
            return {"action": base_intent, "item": item, "qty": requested,
                    "safe_qty": safe, "clamped": False,
                    "order": micro_batch_order(item, requested)}
        self.intercepts += 1
        return {"action": "MICRO_BATCH_SELL", "item": item,
                "requested_qty": requested, "safe_qty": safe,
                "deferred_qty": requested - safe, "clamped": True,
                "base_intent": base_intent,
                "order": micro_batch_order(item, safe)}
