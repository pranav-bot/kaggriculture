"""Tests for demand-calendar sell MPC."""

from kaggriculture.env.items import MARKET_I0
from kaggriculture.helpers.sell_mpc import (
    plan_sell_horizon,
    sell_now_vs_wait_revenue,
    should_delay_for_demand,
    turns_until_product_consumption,
)


def _base_obs(
    *,
    step: int = 1,
    day: int = 5,
    shops: list[str] | None = None,
    shed: dict | None = None,
    wheat_inv: int = MARKET_I0,
    wheat_price: int = 25,
):
    return {
        "step": step,
        "day": day,
        "hour": step % 24,
        "player": 0,
        "farms": [{"money": 4000.0, "tiles": [], "hands": [], "unlocked_quadrants": ["NW"]}],
        "private": {"shed": shed or {"WHEAT": 20}, "seeds": {}, "inventories": [{}]},
        "market": {
            "inventory": {"WHEAT": wheat_inv},
            "prices": {"WHEAT": wheat_price},
        },
        "town": {"unlocked_shops": shops or ["BAKERY"]},
    }


def test_turns_until_bakery_wheat():
    turns, drained = turns_until_product_consumption(["BAKERY"], current_step=1, product="WHEAT")
    assert turns == 3
    assert drained == 1


def test_should_delay_when_wait_beats_now():
    rev_now, rev_wait = sell_now_vs_wait_revenue("WHEAT", 10, MARKET_I0, drained_before_sell=50)
    assert should_delay_for_demand(
        shed_total=40,
        turns_until=2,
        drained=50,
        revenue_now=rev_now,
        revenue_wait=rev_wait,
    ) == (rev_wait > rev_now)


def test_plan_sell_horizon_waits_before_shop_tick():
    obs = _base_obs(step=23, shed={"WHEAT": 8})
    rev_now, rev_wait = sell_now_vs_wait_revenue("WHEAT", 8, MARKET_I0, 2)
    orders = plan_sell_horizon(obs, {"WHEAT": 8}, max_orders=5, horizon=12)
    if rev_wait > rev_now:
        assert not any(o[0] == "SELL" for o in orders)


def test_plan_sell_horizon_sells_under_pressure():
    obs = _base_obs(step=1, shed={"WHEAT": 30})
    orders = plan_sell_horizon(
        obs,
        obs["private"]["shed"],
        max_orders=5,
        horizon=12,
        shed_cap=10,
    )
    assert any(o[0] == "SELL" for o in orders)
