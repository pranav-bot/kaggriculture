from __future__ import annotations
from typing import Any

from mechanics import MAX_MARKET_ORDERS, read
from state import GameState

VALID_UNIT_OPS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "PICKUP", "PLACE",
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
    "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
}
VALID_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}


def validate_joint_action(state: GameState, unit_actions: list[list[Any]], market_orders: list[list[Any]]) -> dict[str, Any]:
    expected = len(read(state.farm, "hands", []) or []) + 1
    actions = [a if isinstance(a, list) and a and a[0] in VALID_UNIT_OPS else ["PASS"] for a in unit_actions[:expected]]
    actions.extend([["PASS"] for _ in range(expected - len(actions))])

    seeds_left = {str(crop): int(amount) for crop, amount in state.seeds.items()}
    capped: list[list[Any]] = []
    for action in actions:
        if len(action) >= 2 and action[0] == "PLANT":
            crop = str(action[1])
            if seeds_left.get(crop, 0) <= 0:
                capped.append(["PASS"])
            else:
                capped.append(action)
                seeds_left[crop] = seeds_left.get(crop, 0) - 1
        else:
            capped.append(action)
    actions = capped

    market = [order for order in market_orders if isinstance(order, list) and order and order[0] in VALID_MARKET_OPS]
    return {"farmer": actions[0], "hands": actions[1:], "market": market[:MAX_MARKET_ORDERS]}


def safe_fallback(state: GameState) -> dict[str, Any]:
    hands = len(read(state.farm, "hands", []) or [])
    sell = [["SELL", item, int(amount)] for item, amount in state.shed.items() if int(amount) > 0][:MAX_MARKET_ORDERS]
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(hands)], "market": sell}
