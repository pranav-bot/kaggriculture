"""Backward mapping of raw JSON mechanical moves to discrete macro-intents."""

from __future__ import annotations

from typing import Any, Mapping

MACRO_INTENT_CLASSES = [
    "PASS",                 # 0
    "EXPAND_NE",            # 1
    "EXPAND_SW",            # 2
    "EXPAND_SE",            # 3
    "HIRE_WORKER",          # 4
    "PLANT_CROPS",          # 5
    "WATER_CROPS",          # 6
    "HARVEST_CROPS",        # 7
    "MAINTAIN_CROPS",       # 8
    "FERTILIZE_CROPS",      # 9
    "BUILD_PASTURE",        # 10
    "BUY_COW",              # 11
    "BUY_SHEEP",            # 12
    "CARE_ANIMAL",          # 13
    "FEED_ANIMAL",          # 14
    "COLLECT_FERTILIZER",   # 15
    "CLEAR_WEED",           # 16
    "BUY_SEEDS",            # 17
    "BUY_MARKET",           # 18
    "SELL_CROPS",           # 19
    "DUMP_MILK",            # 20
    "DUMP_WOOL",            # 21
    "SELL_MARKET",          # 22
    "MOVE_WORKERS",         # 23
]

_CLASS_TO_IDX = {name: idx for idx, name in enumerate(MACRO_INTENT_CLASSES)}


def _normalize_tokens(action_item: Any) -> list[str]:
    """Convert string or list action item into uppercase string tokens."""
    if isinstance(action_item, str):
        return action_item.strip().upper().split()
    if isinstance(action_item, (list, tuple)):
        tokens = []
        for x in action_item:
            if isinstance(x, str):
                tokens.extend(x.strip().upper().split())
            else:
                tokens.append(str(x).upper())
        return tokens
    return []


def classify_macro_intent(move_dict: Any) -> int:
    """Classify raw JSON mechanical action into one of 24 discrete macro classes.

    Args:
        move_dict: Mapping containing 'farmer', 'hands', and 'market' action lists.

    Returns:
        Categorical integer index in [0, 23].
    """
    if not isinstance(move_dict, Mapping):
        return 0  # PASS

    market_orders = move_dict.get("market") or []
    farmer_moves = move_dict.get("farmer") or []
    hands_moves = move_dict.get("hands") or []

    # Priority 1: Market Orders
    for order in market_orders:
        tokens = _normalize_tokens(order)
        if not tokens:
            continue
        order_type = tokens[0]

        # Land Expansion
        if order_type in {"BUY_LAND", "EXPAND"}:
            if len(tokens) > 1:
                target = tokens[1]
                if "NE" in target:
                    return _CLASS_TO_IDX["EXPAND_NE"]
                if "SW" in target:
                    return _CLASS_TO_IDX["EXPAND_SW"]
                if "SE" in target:
                    return _CLASS_TO_IDX["EXPAND_SE"]
            return _CLASS_TO_IDX["EXPAND_NE"]

        # Labor Hiring
        if order_type in {"HIRE", "HIRE_HAND"}:
            return _CLASS_TO_IDX["HIRE_WORKER"]

        # Animal Purchasing
        if order_type in {"BUY_ANIMAL", "BUY"}:
            if "COW" in tokens:
                return _CLASS_TO_IDX["BUY_COW"]
            if "SHEEP" in tokens:
                return _CLASS_TO_IDX["BUY_SHEEP"]

        # Seed Purchasing
        if order_type in {"BUY_SEED", "BUY_SEEDS"}:
            return _CLASS_TO_IDX["BUY_SEEDS"]

        # Liquidating / Dumping Commodities
        if order_type in {"SELL", "SELL_PRODUCT", "DUMP"}:
            if "MILK" in tokens:
                return _CLASS_TO_IDX["DUMP_MILK"]
            if "WOOL" in tokens:
                return _CLASS_TO_IDX["DUMP_WOOL"]
            if any(crop in tokens for crop in ["WHEAT", "CORN", "SOY", "MELON", "STRAWBERRY"]):
                return _CLASS_TO_IDX["SELL_CROPS"]
            return _CLASS_TO_IDX["SELL_MARKET"]

        # General Market Buying
        if order_type in {"BUY", "BUY_PRODUCT"}:
            return _CLASS_TO_IDX["BUY_MARKET"]

    # Priority 2: Farmer & Hands Spatial/Physical Actions
    all_unit_moves = []
    if farmer_moves:
        all_unit_moves.append(farmer_moves)
    all_unit_moves.extend(hands_moves)

    for unit_move in all_unit_moves:
        tokens = _normalize_tokens(unit_move)
        if not tokens:
            continue
        cmd = tokens[0]

        if "WATER" in cmd:
            return _CLASS_TO_IDX["MAINTAIN_CROPS"]
        if "PLANT" in cmd:
            return _CLASS_TO_IDX["PLANT_CROPS"]
        if "HARVEST" in cmd:
            return _CLASS_TO_IDX["HARVEST_CROPS"]
        if "FERTILIZE" in cmd:
            return _CLASS_TO_IDX["FERTILIZE_CROPS"]
        if "PASTURE" in cmd or "BUILD" in cmd:
            return _CLASS_TO_IDX["BUILD_PASTURE"]
        if "WEED" in cmd or "CLEAR" in cmd:
            return _CLASS_TO_IDX["CLEAR_WEED"]
        if "CARE" in cmd:
            return _CLASS_TO_IDX["CARE_ANIMAL"]
        if "FEED" in cmd:
            return _CLASS_TO_IDX["FEED_ANIMAL"]
        if "COLLECT" in cmd:
            return _CLASS_TO_IDX["COLLECT_FERTILIZER"]

    # Priority 3: Unit movement
    for unit_move in all_unit_moves:
        tokens = _normalize_tokens(unit_move)
        if tokens and tokens[0] in {"NORTH", "SOUTH", "EAST", "WEST", "MOVE", "N", "S", "E", "W"}:
            return _CLASS_TO_IDX["MOVE_WORKERS"]

    return _CLASS_TO_IDX["PASS"]
