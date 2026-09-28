"""State and order serialization between Python observation structures and Rust engine JSON."""

from __future__ import annotations

import copy
from typing import Any, Mapping


def extract_engine_state(obs: Mapping[str, Any], seed: int = 42) -> dict[str, Any]:
    """Serialize an observation or state mapping into a Rust engine LOADSTATE-compatible dictionary.

    Args:
        obs: Observation or state mapping from Kaggle environment or dataset frame.
        seed: Random seed for the engine state.

    Returns:
        Dictionary formatted for the Rust simulator's state_from_json loader.
    """
    state = copy.deepcopy(dict(obs))
    state["seed"] = int(seed)

    if "step" not in state:
        state["step"] = int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
    if "day" not in state:
        state["day"] = int(state["step"]) // 24
    if "hour" not in state:
        state["hour"] = int(state["step"]) % 24
    if "done" not in state:
        state["done"] = bool(state["step"] >= 719)

    farms = state.get("farms") or []
    if not isinstance(farms, list) or len(farms) < 2:
        # Default 2-player farms structure
        state["farms"] = [
            {"money": 3000.0, "farmer": [4, 4], "hands": [], "tiles": [[None]*10 for _ in range(10)], "unlocked_quadrants": ["NW"], "hires_today": 0},
            {"money": 3000.0, "farmer": [4, 4], "hands": [], "tiles": [[None]*10 for _ in range(10)], "unlocked_quadrants": ["NW"], "hires_today": 0},
        ]

    # Ensure private block is a 2-element list [seat0_private, seat1_private]
    private = state.get("private")
    if isinstance(private, Mapping):
        state["private"] = [copy.deepcopy(private), copy.deepcopy(private)]
    elif not isinstance(private, list) or len(private) < 2:
        state["private"] = [
            {"carried": {}, "seeds": {}, "shed": {}},
            {"carried": {}, "seeds": {}, "shed": {}},
        ]

    if "market" not in state:
        state["market"] = {"inventory": {}, "prices": {}}
    if "town" not in state:
        state["town"] = {"unlocked_shops": []}

    return state


def serialize_market_order(
    order_type: str = "BUY",
    item: str = "WHEAT",
    quantity: int = 100,
    price: float | int | None = None,
    **kwargs: Any,
) -> str:
    """Serialize market transaction parameters into a tape order string.

    Args:
        order_type: Transaction verb ('BUY', 'SELL', 'BUY_SEED', 'BUY_ANIMAL').
        item: Commodity or animal identifier ('WHEAT', 'COW', etc.).
        quantity: Integer units.
        price: Optional limit price or wholesale quote.

    Returns:
        Order string suitable for simulation tape injection.
    """
    cmd = f"{order_type.upper()} {item.upper()} {int(quantity)}"
    if price is not None:
        cmd += f" {int(price)}"
    return cmd
