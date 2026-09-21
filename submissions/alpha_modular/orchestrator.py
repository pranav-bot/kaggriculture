from __future__ import annotations
import time
from typing import Any

from .config import BASELINE_CONFIG, CHAMPION_CONFIG
from .market_orders import make_market_orders
from .routing import assign_actions
from .state import GameState
from .tasks import materialize_tasks
from .validator import safe_fallback, validate_joint_action


def decide(observation: Any, config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    state = GameState.from_observation(observation)
    fallback = safe_fallback(state)
    try:
        tasks = materialize_tasks(state, config)
        if time.monotonic() - started > 0.20:
            return fallback
        unit_actions, planned_drop = assign_actions(state, tasks, config)
        market_orders = make_market_orders(state, config, planned_drop)
        if time.monotonic() - started > 0.40:
            return fallback
        return validate_joint_action(state, unit_actions, market_orders)
    except Exception:
        return fallback


def agent(observation: Any) -> dict[str, Any]:
    return decide(observation, CHAMPION_CONFIG)


def baseline_agent(observation: Any) -> dict[str, Any]:
    return decide(observation, BASELINE_CONFIG)
