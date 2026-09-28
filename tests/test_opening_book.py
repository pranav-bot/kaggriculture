"""Unit tests for Opening Book Generator and OpeningBookController.

Validates:
1. Rank 1 replay parsing and consensus extraction (scripts/opening_book_generator.py)
2. OpeningBookController active on Days 0-15 overriding Beam Search
3. Fertilizer buffering on Days 1-4
4. NE and SW quadrant expansion triggers on Days 6 and 9
5. Clean handoff on Day 16 releasing control to MacroOptionManager
6. Integrated GrandStrategyController execution across opening and midgame phases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from kaggriculture.opening_book import OPENING_BOOK_SCHEDULE, OpeningBookController
from scripts.opening_book_generator import parse_rank1_replays


@pytest.fixture
def mock_obs() -> Dict[str, Any]:
    return {
        "step": 0,
        "day": 0,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "farmer": [4, 4],
                "hands": [],
                "tiles": [[None] * 10 for _ in range(10)],
                "unlocked_quadrants": ["NW"],
                "money": 3000.0,
                "hires_today": 0,
            },
            {
                "farmer": [4, 4],
                "hands": [],
                "tiles": [[None] * 10 for _ in range(10)],
                "unlocked_quadrants": ["NW"],
                "money": 3000.0,
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {"FERTILIZER": 10, "MILK": 0, "WOOL": 0, "WHEAT": 20},
            "seeds": {},
            "inventories": [{}],
        },
        "market": {
            "prices": {"FERTILIZER": 100, "MILK": 160, "WOOL": 200, "WHEAT": 25},
            "inventory": {"FERTILIZER": 10000, "MILK": 10000, "WOOL": 10000},
        },
        "town": {"unlocked_shops": []},
    }


def test_opening_book_generator_rank1():
    """Verify rank 1 replay parsing tracks expansion and cow purchases."""
    replay_dir = Path("replays/other_agents/rank1")
    if not replay_dir.exists():
        pytest.skip("replays/other_agents/rank1 not found")

    result = parse_rank1_replays(replay_dir)
    assert result["total_replays_analyzed"] >= 1
    assert "consensus_milestones" in result
    ms = result["consensus_milestones"]
    assert ms["expand_ne"]["day"] in (5, 6)
    assert ms["expand_sw"]["day"] in (8, 9)
    assert ms["handoff_day"] == 16
    assert len(result["daily_build_order"]) == 16


def test_opening_book_controller_active_days_0_to_15(mock_obs: Dict[str, Any]):
    """Verify OpeningBookController is active on Days 0-15 and overrides Beam Search."""
    ctrl = OpeningBookController()

    for d in range(16):
        mock_obs["day"] = d
        mock_obs["step"] = d * 24
        assert ctrl.is_active(mock_obs) is True

        ops = ctrl.act(mock_obs)
        assert ops["_opening_book_active"] is True
        assert ops["_opening_day"] == d
        assert "_opening_macro_intent" in ops
        assert ops["_opening_macro_intent"] == OPENING_BOOK_SCHEDULE[d]["macro_intent"]


def test_fertilizer_buffering_days_1_to_4(mock_obs: Dict[str, Any]):
    """Verify that fertilizer sales are buffered (suppressed) on Days 1-4."""
    ctrl = OpeningBookController()

    # Day 3 should buffer fertilizer
    mock_obs["day"] = 3
    mock_obs["step"] = 3 * 24 + 6
    mock_obs["private"]["shed"]["FERTILIZER"] = 15

    ops = ctrl.act(mock_obs)
    market = ops.get("market", [])
    # Verify no SELL FERTILIZER order was emitted
    fert_sells = [o for o in market if len(o) >= 2 and o[0] == "SELL" and o[1] == "FERTILIZER"]
    assert len(fert_sells) == 0, f"Expected fertilizer to be buffered on Day 3, but found: {fert_sells}"


def test_quadrant_expansion_triggers(mock_obs: Dict[str, Any]):
    """Verify NE and SW quadrant expansions trigger at the designated hours."""
    ctrl = OpeningBookController()

    # Day 6, Hour 2 with $1500 money -> should trigger BUY_LAND NE
    mock_obs["day"] = 6
    mock_obs["hour"] = 2
    mock_obs["farms"][0]["money"] = 1500.0
    mock_obs["farms"][0]["unlocked_quadrants"] = ["NW"]

    ops = ctrl.act(mock_obs)
    market = ops.get("market", [])
    expand_orders = [o for o in market if o[0] in ("BUY_LAND", "EXPAND") and len(o) >= 2 and o[1] == "NE"]
    assert len(expand_orders) >= 1, f"Expected BUY_LAND NE order on Day 6 Hour 2, got {market}"

    # Day 9, Hour 2 with $2500 money -> should trigger BUY_LAND SW
    mock_obs["day"] = 9
    mock_obs["hour"] = 2
    mock_obs["farms"][0]["money"] = 2500.0
    mock_obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]

    ops2 = ctrl.act(mock_obs)
    market2 = ops2.get("market", [])
    expand_sw = [o for o in market2 if o[0] in ("BUY_LAND", "EXPAND") and len(o) >= 2 and o[1] == "SW"]
    assert len(expand_sw) >= 1, f"Expected BUY_LAND SW order on Day 9 Hour 2, got {market2}"


def test_clean_handoff_day_16(mock_obs: Dict[str, Any]):
    """Verify clean release of control to MacroOptionManager on Day 16."""
    ctrl = OpeningBookController()

    mock_obs["day"] = 16
    mock_obs["step"] = 16 * 24
    mock_obs["hour"] = 0

    assert ctrl.is_active(mock_obs) is False

    ops = ctrl.act(mock_obs)
    assert ops["_opening_book_active"] is False
    assert ops["_handed_off_to_option_critic"] is True
    assert ctrl.active is False
    # MacroOptionManager should have locked in an intent at Hour 0
    assert ctrl.option_manager.current_intent is not None


def test_grand_strategy_controller_integration(mock_obs: Dict[str, Any]):
    """Verify GrandStrategyController unites OpeningBook, MacroOptionManager, and PoisonedWellTrap."""
    from scratch_grandmaster import GrandStrategyController

    gsc = GrandStrategyController()

    # Day 5: opening book active
    mock_obs["day"] = 5
    mock_obs["step"] = 5 * 24
    ops_day5 = gsc.act(mock_obs)
    assert ops_day5["_opening_book_active"] is True
    assert "_trap_stats" in ops_day5

    # Day 16: cleanly handed off to MacroOptionManager
    mock_obs["day"] = 16
    mock_obs["step"] = 16 * 24
    ops_day16 = gsc.act(mock_obs)
    assert ops_day16["_opening_book_active"] is False
    assert ops_day16["_handed_off_to_option_critic"] is True
