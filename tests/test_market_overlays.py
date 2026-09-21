"""Unit tests for helpers/market_overlays (research/08 thresholds)."""

import pytest

from kaggriculture.helpers.market_overlays import (
    clone_like,
    collision_guard,
    premium_phase_shift,
    reset_market_overlay_state,
)


@pytest.fixture(autouse=True)
def _clean_overlay_state():
    reset_market_overlay_state()
    yield
    reset_market_overlay_state()


def _obs(
    *,
    day: int = 10,
    money: float = 3000.0,
    shed: dict | None = None,
    market: dict | None = None,
    farms: list | None = None,
    player: int = 0,
):
    return {
        "player": player,
        "day": day,
        "step": day * 24,
        "farms": farms
        or [
            {"money": money, "hands": [], "unlocked_quadrants": ["NW"], "tiles": [[None] * 10 for _ in range(10)]},
            {"money": 3000, "hands": [], "unlocked_quadrants": ["NW"], "tiles": [[None] * 10 for _ in range(10)]},
        ],
        "private": {"shed": shed or {"MILK": 10}},
        "market": market or {"inventory": {"MILK": 10000}, "prices": {"MILK": 160}},
    }


def test_premium_phase_shift_delays_when_safe():
    obs = _obs(day=10, money=2000, shed={"MILK": 10})
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]}
    result = premium_phase_shift(obs, base)
    assert result["market"] == []


def test_premium_phase_shift_no_delay_when_cash_below_1600():
    obs = _obs(day=10, money=1599, shed={"MILK": 10})
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]}
    result = premium_phase_shift(obs, base)
    assert result["market"] == [["SELL", "MILK", 5]]


def test_premium_phase_shift_no_delay_when_shed_at_82():
    shed = {"MILK": 10, "WHEAT": 72}
    obs = _obs(day=10, money=2000, shed=shed)
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]}
    result = premium_phase_shift(obs, base)
    assert result["market"] == [["SELL", "MILK", 5]]


def test_premium_phase_shift_no_delay_on_day_27():
    obs = _obs(day=27, money=2000, shed={"MILK": 10})
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 5]]}
    result = premium_phase_shift(obs, base)
    assert result["market"] == [["SELL", "MILK", 5]]


def test_collision_guard_holds_on_inventory_delta():
    obs1 = _obs(market={"inventory": {"MILK": 10000}, "prices": {"MILK": 160}})
    collision_guard(obs1, {"farmer": ["PASS"], "hands": [], "market": []})
    obs2 = _obs(
        market={"inventory": {"MILK": 10003}, "prices": {"MILK": 160}},
        money=2000,
        shed={"MILK": 8},
    )
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 4]]}
    result = collision_guard(obs2, base)
    assert result["market"] == []


def test_collision_guard_holds_on_price_delta():
    obs1 = _obs(market={"inventory": {"WOOL": 10000}, "prices": {"WOOL": 200}})
    collision_guard(obs1, {"farmer": ["PASS"], "hands": [], "market": []})
    obs2 = _obs(
        market={"inventory": {"WOOL": 10000}, "prices": {"WOOL": 196}},
        money=2000,
        shed={"WOOL": 6},
    )
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WOOL", 3]]}
    result = collision_guard(obs2, base)
    assert result["market"] == []


def test_collision_guard_no_hold_when_cash_below_1500():
    obs1 = _obs(market={"inventory": {"MILK": 10000}, "prices": {"MILK": 160}})
    collision_guard(obs1, {"farmer": ["PASS"], "hands": [], "market": []})
    obs2 = _obs(
        market={"inventory": {"MILK": 10005}, "prices": {"MILK": 160}},
        money=1499,
        shed={"MILK": 8},
    )
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 4]]}
    result = collision_guard(obs2, base)
    assert result["market"] == [["SELL", "MILK", 4]]


def test_collision_guard_no_hold_when_shed_at_84():
    obs1 = _obs(market={"inventory": {"MILK": 10000}, "prices": {"MILK": 160}})
    collision_guard(obs1, {"farmer": ["PASS"], "hands": [], "market": []})
    shed = {"MILK": 4, "WHEAT": 80}
    obs2 = _obs(
        market={"inventory": {"MILK": 10005}, "prices": {"MILK": 160}},
        money=2000,
        shed=shed,
    )
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 4]]}
    result = collision_guard(obs2, base)
    assert result["market"] == [["SELL", "MILK", 4]]


def test_collision_guard_no_hold_on_day_27():
    obs1 = _obs(day=26, market={"inventory": {"MILK": 10000}, "prices": {"MILK": 160}})
    collision_guard(obs1, {"farmer": ["PASS"], "hands": [], "market": []})
    obs2 = _obs(
        day=27,
        market={"inventory": {"MILK": 10005}, "prices": {"MILK": 160}},
        money=2000,
        shed={"MILK": 8},
    )
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 4]]}
    result = collision_guard(obs2, base)
    assert result["market"] == [["SELL", "MILK", 4]]


def test_clone_like_true_for_similar_public_farms():
    tiles = [[{"kind": "PLANT", "crop": "WHEAT"}] + [None] * 9 for _ in range(10)]
    farms = [
        {"hands": [[4, 4]], "unlocked_quadrants": ["NW"], "tiles": tiles},
        {"hands": [[4, 4]], "unlocked_quadrants": ["NW"], "tiles": tiles},
    ]
    assert clone_like(_obs(farms=farms)) is True


def test_clone_like_false_when_land_differs():
    tiles = [[None] * 10 for _ in range(10)]
    farms = [
        {"hands": [], "unlocked_quadrants": ["NW"], "tiles": tiles},
        {"hands": [], "unlocked_quadrants": ["NW", "NE"], "tiles": tiles},
    ]
    assert clone_like(_obs(farms=farms)) is False
