"""Tests for the continuous Marginal Labor Value calculator (labor_roi)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kaggriculture.features.macro_intents import _CLASS_TO_IDX
from kaggriculture.helpers.labor_roi import (
    HIRE_WORKER_INDEX,
    STRAWBERRY_CYCLE_DAYS,
    STRAWBERRY_GROSS_MULTIPLE,
    compounded_opportunity_cost,
    decaying_harvest_value,
    dig_tile_npv,
    explain_hire_decision,
    extra_worker_capacity,
    fib,
    filter_beam_trajectory,
    filter_hire_market_orders,
    hire_cost,
    marginal_cost_curve,
    marginal_cost_of_nth_worker,
    marginal_value_of_extra_worker,
    max_roi_positive_hires,
    normalize_pending_queue,
    scheduler_demand_from_batches,
    scheduler_unfulfilled_tasks,
    should_hire_worker,
    strawberry_growth_factor,
    veto_hire_worker_macro,
)


def test_fib_matches_engine():
    assert [fib(i) for i in range(8)] == [1, 1, 2, 3, 5, 8, 13, 21]
    assert hire_cost(0) == 1
    assert hire_cost(4) == 5


def test_hire_macro_index_matches_beam():
    assert HIRE_WORKER_INDEX == 4
    assert _CLASS_TO_IDX["HIRE_WORKER"] == 4


def test_growth_factor_geometric_and_terminal():
    g0 = strawberry_growth_factor(0)
    assert g0 == STRAWBERRY_GROSS_MULTIPLE ** (30.0 / STRAWBERRY_CYCLE_DAYS)
    assert g0 > 10.0  # Day-0 dollar compounds >10x by turn 720
    assert strawberry_growth_factor(720) == 1.0
    assert strawberry_growth_factor(719) < strawberry_growth_factor(700)
    # Compounded cost is direct x G(t).
    assert compounded_opportunity_cost(100, 720) == 100.0
    assert compounded_opportunity_cost(1, 0) > 10.0


def test_marginal_cost_rises_on_fib_ladder_and_falls_into_horizon():
    early = marginal_cost_curve(0, max_n=6)
    assert [c.direct_cost for c in early] == [1, 1, 2, 3, 5, 8]
    # Non-decreasing compounded cost (first two Fibonacci rungs tie at $1),
    # strictly increasing once the ladder moves past fib(1).
    comp = [c.compounded_cost for c in early]
    assert all(b >= a for a, b in zip(comp, comp[1:]))
    assert all(b > a for a, b in zip(comp[1:], comp[2:]))
    # Same N gets cheaper later in the season (less compounding time).
    assert marginal_cost_of_nth_worker(6, 0).compounded_cost > \
        marginal_cost_of_nth_worker(6, 600).compounded_cost


def test_extra_worker_capacity_one_action_per_hour():
    assert extra_worker_capacity(0) == 24  # Day 0 Hour 0: full day
    assert extra_worker_capacity(5) == 19
    assert extra_worker_capacity(23) == 1
    assert extra_worker_capacity(24) == 24  # next dawn: full day again


def test_marginal_value_only_counts_decaying_ops():
    v = marginal_value_of_extra_worker(
        {"HARVEST": 3, "DIG": 2, "WATER": 100, "PLANT": 50}, 5,
    )
    # 5 decaying tasks, WATER/PLANT ignored.
    assert v.n_decaying_tasks == 5
    assert v.n_clearable == 5
    assert v.rescuable_value > 0
    v_none = marginal_value_of_extra_worker({"WATER": 10, "PLANT": 5}, 5)
    assert v_none.rescuable_value == 0.0
    assert should_hire_worker(5000, {"WATER": 10}, 5) is False


def test_marginal_value_caps_at_remaining_hours():
    # 30 harvests at $120 but only 19 slots left at Hour 5 -> $2280.
    v = marginal_value_of_extra_worker([{"op": "HARVEST", "value": 120.0}] * 30, 5)
    assert v.extra_actions == 19
    assert v.n_clearable == 19
    assert v.rescuable_value == 19 * 120.0
    assert v.value_per_action == 120.0
    assert v.top_task_value == 120.0


def test_dig_npv_viability_cutoff():
    assert dig_tile_npv(5) == 380.0  # replant matures well before Day 30
    assert dig_tile_npv(19) == 380.0  # Day 19 + 10 = 29 < 30: still viable
    assert dig_tile_npv(20) == 0.0  # Day 20 + 10 = 30: never matures
    assert dig_tile_npv(29) == 0.0


def test_harvest_decay_loses_one_unit_per_two_steps():
    assert decaying_harvest_value(480, 0) == 480.0
    assert decaying_harvest_value(480, 1) == 360.0  # -1 unit @ $120
    assert decaying_harvest_value(120, 10) == 0.0  # fully decayed, floored


def test_normalize_accepts_all_queue_shapes():
    assert normalize_pending_queue(None, 0) == []
    assert normalize_pending_queue([], 0) == []
    by_count = normalize_pending_queue({"HARVEST": 2, "DIG": 1}, current_day=5)
    assert len(by_count) == 3
    by_str = normalize_pending_queue(["HARVEST", "dig"], current_day=5)
    assert [t.op for t in by_str] == ["HARVEST", "DIG"]
    by_dict = normalize_pending_queue(
        [{"op": "HARVEST", "yield_units": 4, "price": 100.0}], current_day=5)
    assert by_dict[0].value == 400.0


def test_scheduler_query_overflow_and_batches():
    # 1 worker (20 slots/day) cannot cover 14 HARVEST + 5 DIG + 10 WATER.
    unf = scheduler_unfulfilled_tasks(
        {"HARVEST": 14, "DIG": 5, "WATER": 10},
        n_workers_available=1, current_day=12,
    )
    assert len(unf) == 9  # 29 demand - 20 slots
    assert all(t.op in ("HARVEST", "DIG") for t in unf)
    # 9 workers cover everything.
    assert scheduler_unfulfilled_tasks(
        {"HARVEST": 14, "DIG": 5, "WATER": 10},
        n_workers_available=9, current_day=12,
    ) == []
    # Batch wrapper queries the real CropScheduler lifecycle.
    dem = scheduler_demand_from_batches([(2, 5)], day=12)
    assert dem.get("HARVEST", 0) == 5  # batch age 10 -> harvest day


def test_threshold_strict_inequality_and_affordability():
    # Empty queue never hires, however rich.
    assert should_hire_worker(10000, [], 100) is False
    # Unaffordable hire (reserve breach) never hires.
    assert should_hire_worker(5, [{"op": "HARVEST", "value": 5000.0}], 100,
                               operating_reserve=60.0) is False
    # Boundary: rescue exactly equal to compounded cost must NOT hire.
    cost = marginal_cost_of_nth_worker(1, 600).compounded_cost
    assert should_hire_worker(5000, [{"op": "HARVEST", "value": cost}], 600,
                               unlocked_quadrants=["NW", "NE", "SW"]) is False
    assert should_hire_worker(5000, [{"op": "HARVEST", "value": cost + 0.01}], 600,
                               unlocked_quadrants=["NW", "NE", "SW"]) is True


def test_liquidity_race_veto_beats_big_rescue():
    big = [{"op": "HARVEST", "value": 120.0}] * 10  # $1200 rescue
    t_day5 = 5 * 24
    # Cash exactly $1000: $1 hire knocks NE out of reach -> veto despite ROI.
    assert should_hire_worker(1000, big, t_day5, hires_today=0,
                               unlocked_quadrants=["NW"]) is False
    d = explain_hire_decision(1000, big, t_day5, hires_today=0,
                               unlocked_quadrants=["NW"])
    assert d.hire is False and d.land_guard_pass is False
    # Safely above NE: same queue hires.
    assert should_hire_worker(1200, big, t_day5, hires_today=0,
                               unlocked_quadrants=["NW"]) is True
    # After the race (Day 16+) the land guard no longer applies.
    assert should_hire_worker(1000, big, 16 * 24, hires_today=0,
                               unlocked_quadrants=["NW"]) is True


def test_veto_and_trajectory_override():
    assert veto_hire_worker_macro(500, [], 10) is True
    assert veto_hire_worker_macro(
        5000, [{"op": "HARVEST", "value": 500.0}] * 24, 400,
        hires_today=0, unlocked_quadrants=["NW", "NE", "SW"]) is False
    # Hire slots (4) become PASS (0); all other macros untouched.
    assert filter_beam_trajectory([1, 4, 5, 4], 500, [], 10) == [1, 0, 5, 0]
    rich_q = [{"op": "HARVEST", "value": 500.0}] * 24
    assert filter_beam_trajectory([1, 4, 5], 5000, rich_q, 400,
                                   hires_today=0,
                                   unlocked_quadrants=["NW", "NE", "SW"]) == [1, 4, 5]


def test_market_order_filter_marginal_batch():
    orders = [["SELL", "WHEAT", 5], ["HIRE"], ["HIRE"], ["BUY_SEED", "WHEAT", 2]]
    # No decaying work: both HIRE orders stripped, rest preserved in order.
    assert filter_hire_market_orders(orders, 500, [], 10) == [
        ["SELL", "WHEAT", 5], ["BUY_SEED", "WHEAT", 2]]
    # Rich queue late-game: both HIRE orders survive (cheap compounded cost).
    rich_q = [{"op": "HARVEST", "value": 500.0}] * 40
    kept = filter_hire_market_orders(
        [["HIRE"], ["HIRE"]], 5000, rich_q, 600, hires_today=0,
        unlocked_quadrants=["NW", "NE", "SW"])
    assert kept == [["HIRE"], ["HIRE"]]


def test_max_hires_replaces_fixed_target_hands():
    # 50 harvests, Hour 5 (19 slots/hire): hires 1-2 clear 19 each, hire 3
    # clears the last 12 — all ROI-positive at $120 vs ~$16-32 compounded.
    assert max_roi_positive_hires(5000, [{"op": "HARVEST", "value": 120.0}] * 50, 5) == 3
    assert max_roi_positive_hires(5000, [], 5) == 0
