"""Unit tests for submissions/alpha_modular (Architecture Alpha)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "submissions"))

from alpha_modular.config import CHAMPION_CONFIG, cloned
from alpha_modular.market_orders import _sale_quantity, make_market_orders
from alpha_modular.mechanics import hire_cost
from alpha_modular.orchestrator import decide
from alpha_modular.state import GameState
from alpha_modular.validator import validate_joint_action


def _base_obs(**overrides):
    obs = {
        "player": 0,
        "step": 1,
        "day": 0,
        "hour": 1,
        "farms": [
            {
                "money": 3000.0,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "farmer": [0, 0],
                "hands": [[1, 1], [2, 2]],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": 3000.0,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {},
            "seeds": {"MELON": 1},
            "inventories": [{}, {}, {}],
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }
    obs.update(overrides)
    return obs


def test_overplant_passes_excess_units():
    """Validator converts overplant PLANT actions to PASS instead of invalidating."""
    obs = _base_obs()
    state = GameState.from_observation(obs)
    unit_actions = [
        ["PLANT", "MELON"],
        ["PLANT", "MELON"],
        ["PLANT", "MELON"],
    ]
    result = validate_joint_action(state, unit_actions, [])
    plant_ops = [a for a in [result["farmer"], *result["hands"]] if a and a[0] == "PLANT"]
    assert len(plant_ops) == 1
    pass_count = sum(1 for a in [result["farmer"], *result["hands"]] if a == ["PASS"])
    assert pass_count >= 2


def test_batched_premium_sell_quantity():
    state = GameState.from_observation(
        {
            "player": 0,
            "day": 5,
            "hour": 10,
            "farms": [{"money": 1000, "tiles": [[None] * 10] * 10, "unlocked_quadrants": ["NW"]}],
            "private": {"shed": {"MELON": 50, "WHEAT": 50}, "seeds": {}},
            "market": {"prices": {"MELON": 250, "WHEAT": 25}, "inventory": {"MELON": 10000, "WHEAT": 10000}},
            "town": {},
        }
    )
    config = cloned(CHAMPION_CONFIG)
    assert _sale_quantity("MELON", 50, state, config) == 8
    assert _sale_quantity("WHEAT", 50, state, config) == 50

    orders = make_market_orders(state, config)
    melon_sells = [o for o in orders if o[0] == "SELL" and o[1] == "MELON"]
    assert melon_sells and melon_sells[0][2] == 8


def test_operating_reserve_blocks_broke_hires():
    """HIRE is skipped when cash cannot cover hire cost plus operating reserve."""
    reserve = 100
    money = hire_cost(0) + reserve - 1
    state = GameState.from_observation(
        {
            "player": 0,
            "day": 0,
            "hour": 0,
            "farms": [
                {
                    "money": float(money),
                    "tiles": [[None] * 10] * 10,
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                }
            ],
            "private": {"shed": {}, "seeds": {}},
            "market": {"prices": {}, "inventory": {}},
            "town": {},
        }
    )
    config = cloned(CHAMPION_CONFIG)
    config["hands_by_unlocked"] = {1: 3}
    config["operating_reserve"] = reserve
    orders = make_market_orders(state, config)
    assert not any(o[0] == "HIRE" for o in orders)

    state_ok = GameState.from_observation(
        {
            "player": 0,
            "day": 0,
            "hour": 0,
            "farms": [
                {
                    "money": float(hire_cost(0) + reserve),
                    "tiles": [[None] * 10] * 10,
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                }
            ],
            "private": {"shed": {}, "seeds": {}},
            "market": {"prices": {}, "inventory": {}},
            "town": {},
        }
    )
    orders_ok = make_market_orders(state_ok, config)
    assert any(o[0] == "HIRE" for o in orders_ok)


def test_decide_returns_valid_action_shape():
    action = decide(_base_obs(), CHAMPION_CONFIG)
    assert "farmer" in action and "hands" in action and "market" in action


@pytest.mark.skipif(
    not (_REPO / "src" / "kaggriculture").exists(),
    reason="kaggriculture package not available",
)
def test_full_episode_terminal_cash():
    from kaggriculture.env import Environment

    submissions_dir = str(_REPO / "submissions")
    if submissions_dir not in sys.path:
        sys.path.insert(0, submissions_dir)
    from alpha_modular.main import agent

    env = Environment(configuration={"episodeSteps": 720})
    final = env.run_env(agent, "random")
    cash = env.get_farm(final, 0)["money"]
    assert final is not None
    print(f"terminal_cash={cash}")
