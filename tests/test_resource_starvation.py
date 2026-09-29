"""Tests for the Resource Starvation subroutine (resource_starvation)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kaggriculture.env.items import MARKET_I0
from kaggriculture.helpers.market_prediction import (
    simulate_buy_slippage,
    simulate_sell_slippage,
)
from kaggriculture.helpers.resource_starvation import (
    buy_cost,
    denial_purchase_volume,
    estimate_melon_tiles,
    fertilizer_price,
    holding_loss,
    is_rush_intent,
    opponent_fertilizer_cost,
    opponent_melon_loss,
    plan_denial,
    plan_denial_from_obs,
    read_fertilizer_market,
    required_fertilizer_volume,
    select_rush_intent,
    sell_revenue,
    should_execute_denial,
)


def test_execution_math_matches_engine_slippage():
    for q, inv in [(25, 10000), (10, 9975), (1, 10000), (40, 9950)]:
        assert buy_cost(q, inv) == simulate_buy_slippage("FERTILIZER", q, inv).total_revenue
    for q, inv in [(25, 9975), (10, 10000)]:
        assert sell_revenue(q, inv) == simulate_sell_slippage("FERTILIZER", q, inv).total_revenue


def test_fertilizer_price_follows_engine_curve():
    assert fertilizer_price(MARKET_I0) == 100
    assert fertilizer_price(9900) == 120  # 0.2 $/unit scarcity slope
    assert fertilizer_price(9500) == 200


def test_intent_ingestion_gates_on_rush():
    assert is_rush_intent("PREPARING_MELON_RUSH") is True
    assert is_rush_intent("melon_rush") is True
    assert is_rush_intent("HOARDING_FERTILIZER") is True
    assert is_rush_intent("PREPARING_EXPANSION") is False
    assert is_rush_intent("LIVESTOCK_RUSH") is False
    # Forecaster distribution: confident rush passes, weak signal abstains.
    strong = {"PREPARING_EXPANSION": 0.2, "HOARDING_FERTILIZER": 0.7,
              "LIVESTOCK_RUSH": 0.05, "CROP_ROTATION": 0.05}
    assert select_rush_intent(strong, 0.5) == ("HOARDING_FERTILIZER", 0.7)
    weak = {"PREPARING_EXPANSION": 0.7, "HOARDING_FERTILIZER": 0.2,
            "LIVESTOCK_RUSH": 0.05, "CROP_ROTATION": 0.05}
    assert select_rush_intent(weak, 0.5) is None


def test_requirement_tiles_times_coverage_minus_supply():
    # 10 melon tiles x ceil(7/3)=3 units/tile = 30.
    assert required_fertilizer_volume("PREPARING_MELON_RUSH", 10) == 30
    assert required_fertilizer_volume("PREPARING_MELON_RUSH", 10,
                                      opponent_self_supply=6) == 24
    assert required_fertilizer_volume("PREPARING_EXPANSION", 10) == 0
    assert required_fertilizer_volume("PREPARING_MELON_RUSH", 0) == 0


def test_tile_estimate_prefers_override_then_vacant():
    assert estimate_melon_tiles(None, melon_tiles=7) == 7
    farm = {"tiles": [[None] * 10 for _ in range(10)]}
    assert estimate_melon_tiles(farm) == 100


def test_market_read_handles_obs_and_bare_dicts():
    assert read_fertilizer_market(None) == (MARKET_I0, 100)
    obs = {"market": {"inventory": {"FERTILIZER": 9900}, "prices": {}}}
    assert read_fertilizer_market(obs) == (9900, 120)
    assert read_fertilizer_market({"inventory": {"FERTILIZER": 9500}}) == (9500, 200)


def test_denial_volume_is_least_locking_quantity():
    # V=10, opp cash $1050 at pristine market: q*=20 is the exact boundary.
    q = denial_purchase_volume(10, MARKET_I0, 1050, our_cash=5000, shed_room=100)
    assert q == 20
    assert opponent_fertilizer_cost(10, MARKET_I0 - (q - 1)) <= 1050
    assert opponent_fertilizer_cost(10, MARKET_I0 - q) > 1050  # strict lockout


def test_denial_already_locked_returns_zero():
    # V=30 at base costs ~$3090 > $1000 cash: locked with no spend.
    assert denial_purchase_volume(30, MARKET_I0, 1000, our_cash=5000, shed_room=100) == 0


def test_denial_infeasible_when_opponent_too_rich_or_we_are_broke():
    assert denial_purchase_volume(10, MARKET_I0, 10**9, our_cash=5000, shed_room=100) is None
    assert denial_purchase_volume(10, MARKET_I0, 1050, our_cash=10, shed_room=100) is None
    assert denial_purchase_volume(10, MARKET_I0, 1050, our_cash=5000, shed_room=0) is None
    assert denial_purchase_volume(0, MARKET_I0, 1050, our_cash=5000, shed_room=100) is None


def test_melon_loss_viability_gate_and_margin():
    # 5 tiles x (6 x 250 - 80) = $7100 while plantable; $0 once too late.
    assert opponent_melon_loss(5, current_day=5) == 7100
    assert opponent_melon_loss(5, current_day=25) == 0
    assert opponent_melon_loss(0, current_day=5) == 0


def test_holding_loss_round_trip_is_nearly_lossless():
    # Linear symmetric curve: buy-then-resell near I0 loses only rounding.
    assert holding_loss(25, MARKET_I0) <= 5
    assert holding_loss(0, MARKET_I0) == 0
    assert holding_loss(10, MARKET_I0, own_fertilizer_need=10) == 0


def test_end_to_end_execute_with_strict_gates():
    plan = plan_denial(
        "PREPARING_MELON_RUSH", opponent_cash=1050, market_inventory=MARKET_I0,
        our_cash=5000, shed_room=100, melon_tiles=5, fertilizer_per_tile=2,
        current_day=5,
    )
    assert plan.execute is True
    assert plan.denial_volume == 20
    assert plan.locked_out is True
    assert plan.holding_loss < plan.opponent_loss  # strict risk gate
    assert plan.to_market_orders() == [["BUY_PRODUCT", "FERTILIZER", 20]]


def test_end_to_end_abstains():
    # Non-rush intent.
    assert should_execute_denial(
        "PREPARING_EXPANSION", opponent_cash=1050, market_inventory=MARKET_I0,
        our_cash=5000, shed_room=100, melon_tiles=5, current_day=5) is False
    # Infeasible lockout (opponent too rich).
    poor = plan_denial(
        "PREPARING_MELON_RUSH", opponent_cash=10**9, market_inventory=MARKET_I0,
        our_cash=5000, shed_room=100, melon_tiles=5, fertilizer_per_tile=2,
        current_day=5)
    assert poor.execute is False and poor.denial_volume is None
    assert poor.to_market_orders() == []
    # Missed window has no value: risk gate fails even when lockout is feasible.
    late = plan_denial(
        "PREPARING_MELON_RUSH", opponent_cash=1050, market_inventory=MARKET_I0,
        our_cash=5000, shed_room=100, melon_tiles=5, fertilizer_per_tile=2,
        current_day=25)
    assert late.execute is False
    # Boundary strictness: equal losses must NOT execute (monkeypatched scale).
    import kaggriculture.helpers.resource_starvation as rs
    orig = rs.opponent_melon_loss
    try:
        rs.opponent_melon_loss = lambda *a, **k: 0  # zero opp loss, feasible lockout
        tie = plan_denial(
            "PREPARING_MELON_RUSH", opponent_cash=1050, market_inventory=MARKET_I0,
            our_cash=5000, shed_room=100, melon_tiles=5, fertilizer_per_tile=2,
            current_day=5, own_fertilizer_need=0)
        assert tie.execute is False
    finally:
        rs.opponent_melon_loss = orig


def test_obs_wrapper_reads_cash_shed_market_day():
    def empty_farm(money):
        return {"money": money, "tiles": [[None] * 10 for _ in range(10)],
                "unlocked_quadrants": ["NW"], "farmer": [4, 4], "hands": []}

    obs = {
        "player": 0,
        "day": 5,
        "farms": [empty_farm(5000), empty_farm(1050)],
        "private": {"shed": {}},
        "market": {"inventory": {"FERTILIZER": MARKET_I0, "MELON": MARKET_I0},
                   "prices": {"MELON": 250}},
    }
    plan = plan_denial_from_obs(
        obs, "PREPARING_MELON_RUSH", melon_tiles=5, fertilizer_per_tile=2)
    assert plan.execute is True
    assert plan.denial_volume == 20
    assert plan.opponent_cash == 1050
