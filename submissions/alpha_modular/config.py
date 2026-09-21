from __future__ import annotations
from copy import deepcopy
from typing import Any

BASELINE_CONFIG: dict[str, Any] = {
    "name": "baseline-v1",
    "max_active": 18,
    "early_mix": {"CARROT": 14, "WHEAT": 4},
    "late_mix": {"CARROT": 14, "WHEAT": 4},
    "mix_switch_day": 30,
    "crop_order": ["CARROT", "WHEAT"],
    "target_unlocked": 1,
    "hands_by_unlocked": {1: 5},
    "sell_mode": "immediate",
    "premium_batch": 100,
    "terminal_day": 29,
    "terminal_return_hour": 15,
    "operating_reserve": 100,
    "adaptive_mix": False,
}

CHAMPION_CONFIG: dict[str, Any] = {
    "name": "champion-v1",
    "max_active": 40,
    "early_mix": {"MELON": 14, "CARROT": 14, "WHEAT": 12},
    "late_mix": {"CARROT": 20, "WHEAT": 20},
    "mix_switch_day": 13,
    "crop_order": ["MELON", "CARROT", "WHEAT"],
    "target_unlocked": 2,
    "hands_by_unlocked": {1: 6, 2: 9},
    "sell_mode": "batched",
    "premium_batch": 8,
    "terminal_day": 29,
    "terminal_return_hour": 13,
    "operating_reserve": 100,
    "adaptive_mix": True,
}


def cloned(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)
