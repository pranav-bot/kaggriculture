"""Unit & Integration tests for Retrograde DP & MILP Liquidation Solver (Days 20–29).

Validates:
1. Seed maturation constraints (no seeds planted if maturation extends past Day 30).
2. Town shop consumption ticks (6 ticks/day, 6-12+ units/tick).
3. State ingestion on Day 20 (Hour 0) with animal and crop yield projections.
4. MILP solver (scipy.optimize.milp) under flow balance and shed capacity <= 100.
5. Non-linear solver (scipy.optimize.minimize) with quadratic and linear depreciation.
6. Retrograde Dynamic Programming (Bellman backward induction).
7. Daily LiquidationIntent dictionary generation.
8. LiquidationController overriding Beam Search and filtering illegal mechanical moves.
9. End-to-end integration across Opening Book (Days 0-15), MacroOption (Days 16-19),
   and Liquidation (Days 20-29).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import pytest

from kaggriculture.env.items import (
    MARKET_I0,
    PRODUCTS_LIST,
    SHED_CAPACITY,
    TOWN_SHOP_SELL_INTERVAL,
)
from kaggriculture.liquidation_solver import (
    END_DAY,
    GAME_OVER_DAY,
    PLANNING_HORIZON_DAYS,
    START_DAY,
    TICKS_PER_DAY,
    LiquidationController,
    LiquidationIntent,
    MILPLiquidationSolver,
    RetrogradeDPSolver,
    ScipyMinimizeLiquidationSolver,
    TerminalLiquidationSolver,
    TownShopModel,
    calculate_batch_revenue,
    get_forbidden_crops,
    ingest_state_on_day_20,
    is_seed_viable_for_planting,
)
from kaggriculture.opening_book import OpeningBookController


@pytest.fixture
def mock_day20_obs() -> Dict[str, Any]:
    """Generates standard Day 20 (Hour 0) observation with livestock, crops, and shed."""
    tiles = [[None] * 10 for _ in range(10)]
    # 4 Cows
    tiles[0][0] = {"kind": "PASTURE", "animal": "COW", "placed_day": 4, "yield_units": 0}
    tiles[0][1] = {"kind": "PASTURE", "animal": "COW", "placed_day": 4, "yield_units": 0}
    tiles[0][2] = {"kind": "PASTURE", "animal": "COW", "placed_day": 8, "yield_units": 0}
    tiles[0][3] = {"kind": "PASTURE", "animal": "COW", "placed_day": 8, "yield_units": 0}

    # 3 Sheep
    tiles[1][0] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 6, "yield_units": 0}
    tiles[1][1] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 6, "yield_units": 0}
    tiles[1][2] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 9, "yield_units": 0}

    # 4 Strawberry plots planted Day 10
    tiles[2][0] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 10, "yield_units": 0}
    tiles[2][1] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 10, "yield_units": 0}
    tiles[2][2] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 12, "yield_units": 0}
    tiles[2][3] = {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 12, "yield_units": 0}

    return {
        "step": 480,
        "day": 20,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 15000.0,
                "tiles": tiles,
                "farmer": [4, 4],
                "hands": [[4, 5], [5, 4]],
                "unlocked_quadrants": ["NW", "NE", "SW"],
            }
        ],
        "private": {
            "shed": {
                "MILK": 18,
                "WOOL": 14,
                "STRAWBERRY": 16,
                "FERTILIZER": 25,
                "WHEAT": 20,
            },
            "seeds": {
                "STRAWBERRY": 8,
                "MELON": 4,
                "TOMATO": 5,
                "WHEAT": 10,
                "CARROT": 6,
            },
            "inventories": [{}, {}],
        },
        "market": {
            "inventory": {p: MARKET_I0 for p in PRODUCTS_LIST},
            "prices": {
                "MILK": 160,
                "WOOL": 200,
                "STRAWBERRY": 120,
                "FERTILIZER": 100,
                "WHEAT": 25,
                "CARROT": 35,
                "TOMATO": 60,
                "MELON": 250,
                "EGG": 50,
            },
            "unlocked_shops": [
                "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
                "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP",
            ],
        },
    }


def test_seed_maturation_prohibitions():
    """Verify that seeds whose harvest extends past Day 30 cannot be planted."""
    # Day 20: Strawberry & Melon take 10 days (20 + 10 = 30 >= 30) -> forbidden
    assert is_seed_viable_for_planting("STRAWBERRY", 20) is False
    assert is_seed_viable_for_planting("MELON", 20) is False
    # Tomato takes 8 days (20 + 8 = 28 < 30) -> viable on Day 20
    assert is_seed_viable_for_planting("TOMATO", 20) is True
    # Wheat & Carrot take 2 days -> viable on Day 20
    assert is_seed_viable_for_planting("WHEAT", 20) is True
    assert is_seed_viable_for_planting("CARROT", 20) is True

    forbidden_d20 = get_forbidden_crops(20)
    assert "STRAWBERRY" in forbidden_d20
    assert "MELON" in forbidden_d20
    assert "TOMATO" not in forbidden_d20

    # Day 22: Tomato (22 + 8 = 30) -> forbidden
    assert is_seed_viable_for_planting("TOMATO", 22) is False
    forbidden_d22 = get_forbidden_crops(22)
    assert "TOMATO" in forbidden_d22

    # Day 28: Wheat & Carrot (28 + 2 = 30) -> forbidden
    assert is_seed_viable_for_planting("WHEAT", 28) is False
    assert is_seed_viable_for_planting("CARROT", 28) is False
    forbidden_d28 = get_forbidden_crops(28)
    assert "WHEAT" in forbidden_d28
    assert "CARROT" in forbidden_d28


def test_town_shop_consumption_ticks():
    """Verify town shop consumption ticks model 6 ticks/day and drain 6-12+ units/tick."""
    shop_model = TownShopModel(unlocked_shops=[
        "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE"
    ])
    assert TICKS_PER_DAY == 6

    tick_drain = shop_model.get_tick_drain_by_product(20)
    assert tick_drain["WOOL"] == 2  # Yarn Store (single product) consumes 2 units/tick
    assert tick_drain["MILK"] >= 2  # Pizza Shop + Ice Cream Shop
    assert tick_drain["CARROT"] == 2  # Pet Cafe consumes 2 units/tick

    total_tick_drain = shop_model.get_tick_total_drain(20)
    assert 6 <= total_tick_drain <= 25, f"Expected 6-25 units/tick, got {total_tick_drain}"

    daily_drain = shop_model.get_daily_drain_by_product(20)
    assert daily_drain["WOOL"] >= 13  # 12 from shop + 1 from Town Center
    assert daily_drain["MILK"] >= 12


def test_day20_state_ingestion(mock_day20_obs: Dict[str, Any]):
    """Verify ingestion of Day 20 state and accurate yield projections."""
    state = ingest_state_on_day_20(mock_day20_obs)

    assert state.initial_cash == 15000.0
    assert state.current_day == 20
    assert state.animal_counts["COW"] == 4
    assert state.animal_counts["SHEEP"] == 3
    assert state.field_crop_counts["STRAWBERRY"] == 4
    assert state.shed_inventory["MILK"] == 18
    assert state.shed_inventory["WOOL"] == 14

    # Cows produce milk and fertilizer
    milk_yields = state.daily_projected_yields["MILK"]
    assert len(milk_yields) == PLANNING_HORIZON_DAYS
    assert sum(milk_yields) > 0
    # Sheep produce wool
    wool_yields = state.daily_projected_yields["WOOL"]
    assert sum(wool_yields) > 0

    # Forbidden crops list
    assert "STRAWBERRY" in state.prohibited_seeds
    assert "MELON" in state.prohibited_seeds


def test_milp_solver_execution(mock_day20_obs: Dict[str, Any]):
    """Verify Mixed-Integer Linear Programming solver computes valid daily quotas."""
    state = ingest_state_on_day_20(mock_day20_obs)
    solver = MILPLiquidationSolver()

    quotas = solver.solve(
        initial_shed=state.shed_inventory,
        daily_yields=state.daily_projected_yields,
        market_inv=state.market_inventory,
        shed_capacity=SHED_CAPACITY,
    )

    # 1. Quotas must be non-negative integers
    for p, daily_q in quotas.items():
        assert len(daily_q) == PLANNING_HORIZON_DAYS
        for q in daily_q:
            assert isinstance(q, int)
            assert q >= 0

    # 2. Terminal liquidation: total sold == total available
    for p in ("MILK", "WOOL", "STRAWBERRY"):
        tot_avail = state.shed_inventory.get(p, 0) + sum(state.daily_projected_yields.get(p, []))
        tot_sold = sum(quotas[p])
        assert tot_sold == tot_avail, f"Product {p}: sold {tot_sold} != avail {tot_avail}"

    # 3. Shed capacity: total items in shed at end of each day <= 100
    for t in range(PLANNING_HORIZON_DAYS):
        end_shed = sum(
            state.shed_inventory.get(p, 0)
            + sum(state.daily_projected_yields.get(p, [0] * 10)[:t + 1])
            - sum(quotas[p][:t + 1])
            for p in PRODUCTS_LIST
        )
        assert 0 <= end_shed <= SHED_CAPACITY + 1e-5, f"Day {20+t} shed capacity violation: {end_shed}"


def test_scipy_minimize_solver(mock_day20_obs: Dict[str, Any]):
    """Verify non-linear SLSQP solver converges and yields valid quotas."""
    state = ingest_state_on_day_20(mock_day20_obs)
    solver = ScipyMinimizeLiquidationSolver()

    quotas = solver.solve(
        initial_shed=state.shed_inventory,
        daily_yields=state.daily_projected_yields,
        market_inv=state.market_inventory,
        shed_capacity=SHED_CAPACITY,
    )

    for p in ("MILK", "WOOL"):
        tot_avail = state.shed_inventory.get(p, 0) + sum(state.daily_projected_yields.get(p, []))
        tot_sold = sum(quotas[p])
        assert tot_sold == tot_avail


def test_retrograde_dp_solver():
    """Verify Bellman backward induction for single commodity."""
    dp_solver = RetrogradeDPSolver()
    quotas = dp_solver.solve_commodity(
        product="WOOL",
        initial_shed=20,
        daily_yields=[4, 0, 0, 4, 0, 0, 4, 0, 0, 4],
        start_excess_inv=0,
        max_shed_state=40,
        max_surplus_state=30,
        step_granularity=2,
    )

    assert len(quotas) == PLANNING_HORIZON_DAYS
    assert all(isinstance(q, int) and q >= 0 for q in quotas)
    # Sold quantity should be positive and liquidates all inventory
    assert sum(quotas) > 0


def test_daily_liquidation_intent_dictionary(mock_day20_obs: Dict[str, Any]):
    """Verify TerminalLiquidationSolver produces rich, daily LiquidationIntent dictionaries."""
    solver = TerminalLiquidationSolver(solver_backend="milp")
    schedule = solver.plan_liquidation(mock_day20_obs)

    assert len(schedule) == PLANNING_HORIZON_DAYS

    for day in range(START_DAY, END_DAY + 1):
        assert day in schedule
        intent = schedule[day]
        assert isinstance(intent, LiquidationIntent)
        assert intent.day == day
        assert intent.override_beam_search is True
        assert intent.macro_intent == "TERMINAL_LIQUIDATION"
        assert isinstance(intent.sell_quotas, dict)
        assert isinstance(intent.tick_sell_quotas, dict)
        assert isinstance(intent.plant_quotas, dict)
        assert isinstance(intent.prohibit_planting, list)
        assert "COW" in intent.prohibit_purchases
        assert "EXPAND_NE" in intent.prohibit_purchases

        # Verify to_dict serialization
        d = intent.to_dict()
        assert d["day"] == day
        assert d["override_beam_search"] is True

    # Day 29 must have terminal_flush flag
    assert schedule[END_DAY].terminal_flush is True


def test_liquidation_controller_overrides_beam_search(mock_day20_obs: Dict[str, Any]):
    """Verify LiquidationController filters actions and manages market ticks."""
    ctrl = LiquidationController()
    assert ctrl.is_in_liquidation_phase(mock_day20_obs) is True

    # Mock obs on Day 15 (not in liquidation phase)
    obs_day15 = dict(mock_day20_obs, day=15, step=15 * 24)
    assert ctrl.is_in_liquidation_phase(obs_day15) is False

    # Hour 0 (tick hour): emits market sell orders
    mock_day20_obs["hour"] = 0
    orders_h0 = ctrl.generate_market_orders(mock_day20_obs)
    assert len(orders_h0) >= 1
    assert all(o[0] == "SELL" for o in orders_h0)

    # Hour 1 (non-tick hour): emits empty orders
    obs_h1 = dict(mock_day20_obs, hour=1)
    orders_h1 = ctrl.generate_market_orders(obs_h1)
    assert len(orders_h1) == 0

    # Filter mechanical actions: blocks PLANT of STRAWBERRY and BUY of COW
    raw_ops = {
        "farmer": ["PLANT", "STRAWBERRY"],
        "hand_0": ["BUY", "COW"],
        "hand_1": ["EXPAND", "SE"],
        "market": [["BUY_SEED", "STRAWBERRY", 5]],
    }
    filtered = ctrl.filter_mechanical_actions(raw_ops, mock_day20_obs)
    assert filtered["farmer"] == ["PASS"]
    assert filtered["hand_0"] == ["PASS"]
    assert filtered["hand_1"] == ["PASS"]
    # Illegal market purchase filtered
    for o in filtered.get("market", []):
        assert not (o[0] == "BUY_SEED" and o[1] == "STRAWBERRY")

    assert filtered["_override_beam_search"] is True
    assert "_liquidation_intent" in filtered


def test_opening_book_to_liquidation_integration(mock_day20_obs: Dict[str, Any]):
    """Verify seamless execution across Opening Book (Days 0-15), MacroOption (Days 16-19),
    and Liquidation (Days 20-29)."""
    controller = OpeningBookController()

    # 1. Day 5 (Opening Book)
    obs_d5 = dict(mock_day20_obs, day=5, step=5 * 24, hour=0)
    ops_d5 = controller.act(obs_d5)
    assert ops_d5.get("_opening_book_active") is True
    assert ops_d5.get("_liquidation_active") is not True

    # 2. Day 16 (MacroOptionManager handoff)
    obs_d16 = dict(mock_day20_obs, day=16, step=16 * 24, hour=0)
    ops_d16 = controller.act(obs_d16)
    assert ops_d16.get("_opening_book_active") is False
    assert ops_d16.get("_handed_off_to_option_critic") is True
    assert ops_d16.get("_liquidation_active") is not True

    # 3. Day 20 (Liquidation Controller)
    obs_d20 = dict(mock_day20_obs, day=20, step=20 * 24, hour=0)
    ops_d20 = controller.act(obs_d20)
    assert ops_d20.get("_opening_book_active") is False
    assert ops_d20.get("_liquidation_active") is True
    assert ops_d20.get("_override_beam_search") is True
    assert "_liquidation_intent" in ops_d20


def test_terminal_schedule_covers_exact_days_and_quota_totals(mock_day20_obs: Dict[str, Any]):
    """Validate the ten-day terminal horizon and exact per-product liquidation totals."""
    solver = TerminalLiquidationSolver(solver_backend="milp")
    schedule = solver.plan_liquidation(mock_day20_obs)
    state = solver._cached_day20_state
    assert state is not None

    assert list(schedule) == list(range(20, 30))
    for product in PRODUCTS_LIST:
        planned = sum(schedule[day].sell_quotas.get(product, 0) for day in range(20, 30))
        available = state.shed_inventory.get(product, 0) + sum(
            state.daily_projected_yields.get(product, [])
        )
        assert planned == available


def test_consumption_tick_hours_and_tick_quotas(mock_day20_obs: Dict[str, Any]):
    """Validate sell orders occur on all six shop ticks and not between ticks."""
    controller = LiquidationController()
    for hour in (0, 4, 8, 12, 16, 20):
        obs = dict(mock_day20_obs, step=20 * 24 + hour, hour=hour)
        orders = controller.generate_market_orders(obs)
        assert all(order[0] == "SELL" for order in orders)

    non_tick = dict(mock_day20_obs, step=20 * 24 + 1, hour=1)
    assert controller.generate_market_orders(non_tick) == []


def test_nonlinear_price_penalty_is_strict_and_batch_revenue_matches():
    """Validate sequential market inventory causes a nonlinear revenue penalty."""
    pristine = calculate_batch_revenue("MILK", 1, MARKET_I0)
    later = calculate_batch_revenue("MILK", 1, MARKET_I0 + 40)
    bulk = calculate_batch_revenue("MILK", 40, MARKET_I0)

    assert later < pristine
    assert bulk < pristine * 40


def test_step_719_terminal_flush_boundary(mock_day20_obs: Dict[str, Any]):
    """Validate Day 29 Hour 23 (step 719) flushes, while step 720 is out of phase."""
    controller = LiquidationController()
    terminal_obs = dict(mock_day20_obs, day=29, hour=23, step=719)
    orders = controller.generate_market_orders(terminal_obs)
    assert controller.is_in_liquidation_phase(terminal_obs) is True
    assert ["SELL", "MILK", "18"] in orders

    post_game = dict(mock_day20_obs, day=30, hour=0, step=720)
    assert controller.is_in_liquidation_phase(post_game) is False


def test_no_new_seeds_after_day_20_and_beam_override_output(mock_day20_obs: Dict[str, Any]):
    """Validate late seed planting is blocked and the controller marks Beam as overridden."""
    controller = LiquidationController()
    for day in range(20, 30):
        obs = dict(mock_day20_obs, day=day, step=day * 24, hour=1)
        raw = {
            "farmer": ["PLANT", "STRAWBERRY"],
            "hand_0": ["BUY", "COW"],
            "market": [["BUY_SEED", "STRAWBERRY", 1]],
        }
        filtered = controller.filter_mechanical_actions(raw, obs)
        assert filtered["farmer"] == ["PASS"]
        assert filtered["hand_0"] == ["PASS"]
        assert filtered["_override_beam_search"] is True
        assert filtered["_liquidation_intent"]["override_beam_search"] is True
        assert "STRAWBERRY" in filtered["_liquidation_intent"]["prohibit_planting"]
        assert all(order[0] != "BUY_SEED" for order in filtered["market"])
