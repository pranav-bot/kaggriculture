"""Tests for the online CFR policy selector (meta/cfr.py)."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kaggriculture.meta.cfr import (
    CFRPolicySelector,
    POLICY_A,
    POLICY_B,
    POLICY_C,
    DayUpdate,
    RegretTable,
    call_value_net,
    counterfactual_values,
    evaluate_leaf_value,
    fallback_leaf_value,
    is_end_of_day,
    normalize_policy,
    obs_day,
    obs_hour,
    project_counterfactual_leaves,
    regret_matching_strategy,
)


def _values(a, b, c):
    return {POLICY_A: float(a), POLICY_B: float(b), POLICY_C: float(c)}


# --- regret matching ---------------------------------------------------------

def test_initial_strategy_is_uniform():
    sel = CFRPolicySelector()
    assert sel.strategy_for_tomorrow() == {POLICY_A: 1 / 3, POLICY_B: 1 / 3, POLICY_C: 1 / 3}
    assert sel.average_strategy() == {POLICY_A: 1 / 3, POLICY_B: 1 / 3, POLICY_C: 1 / 3}


def test_pairwise_counterfactual_regret_update():
    sel = CFRPolicySelector()
    upd = sel.end_of_day_update(0, POLICY_A, _values(100, 150, 80))
    assert isinstance(upd, DayUpdate)
    # r_p = u_p - u_played: B gains 50, C loses 20, A is zero by construction.
    assert upd.regrets == {POLICY_A: 0.0, POLICY_B: 50.0, POLICY_C: -20.0}
    assert upd.cumulative_regret == upd.regrets
    assert sel.cumulative_regret == {POLICY_A: 0.0, POLICY_B: 50.0, POLICY_C: -20.0}
    # Only positive regret gets mass: tomorrow is pure B.
    assert sel.strategy_for_tomorrow() == {POLICY_A: 0.0, POLICY_B: 1.0, POLICY_C: 0.0}


def test_strategy_proportional_to_positive_regret():
    strat = regret_matching_strategy({POLICY_A: 10.0, POLICY_B: 30.0, POLICY_C: -5.0})
    assert strat == {POLICY_A: 0.25, POLICY_B: 0.75, POLICY_C: 0.0}
    assert sum(strat.values()) == 1.0


def test_all_nonpositive_regret_falls_back_to_uniform():
    assert regret_matching_strategy({POLICY_A: 0.0, POLICY_B: -3.0, POLICY_C: -1.0}) == {
        POLICY_A: 1 / 3, POLICY_B: 1 / 3, POLICY_C: 1 / 3}


def test_regret_accumulates_across_days():
    table = RegretTable()
    table.add(0, POLICY_A, _values(100, 150, 80))
    table.add(1, POLICY_B, _values(200, 120, 130))
    # Day 2: r = u - u_B = {80, 0, 10}; totals {80, 50, -10}.
    assert table.cumulative == {POLICY_A: 80.0, POLICY_B: 50.0, POLICY_C: -10.0}
    assert len(table.history) == 2
    assert table.history[0].day == 0 and table.history[1].played_policy == POLICY_B


def test_average_strategy_converges_toward_best_fixed_policy():
    sel = CFRPolicySelector()
    for day in range(30):
        sel.end_of_day_update(day, POLICY_A, _values(100, 150, 120))
    # Regrets accumulate {A:0, B:+50/d, C:+20/d} -> mix stays propto {0, 5, 2}.
    strat = sel.strategy_for_tomorrow()
    assert strat[POLICY_B] == 1500.0 / 2100.0
    assert strat[POLICY_C] == 600.0 / 2100.0
    assert strat[POLICY_A] == 0.0
    avg = sel.average_strategy()
    assert avg[POLICY_A] == 0.0
    assert abs(avg[POLICY_B] - strat[POLICY_B]) < 1e-9  # stationary mix
    assert abs(avg[POLICY_C] - strat[POLICY_C]) < 1e-9
    assert abs(sum(avg.values()) - 1.0) < 1e-9
    assert sel.max_average_regret() == 50.0  # constant per-day edge, T-normalized


def test_no_regret_property_average_regret_vanishes():
    sel = CFRPolicySelector()
    seen = []
    for day in range(20):
        # Adversary alternates which unplayed policy looks best.
        vals = _values(100, 150, 80) if day % 2 == 0 else _values(100, 80, 150)
        sel.end_of_day_update(day, POLICY_A, vals)
        seen.append(sel.max_average_regret())
    assert seen[-1] < seen[0]
    assert seen[-1] <= 50.0 * 20 / 20  # bounded; strictly decaying here
    assert sel.max_average_regret() == max(sel.cumulative_regret.values()) / 20


def test_mixed_strategy_is_unpredictable_under_split_regret():
    sel = CFRPolicySelector()
    sel.end_of_day_update(0, POLICY_C, _values(110, 110, 100))
    strat = sel.strategy_for_tomorrow()
    assert strat == {POLICY_A: 0.5, POLICY_B: 0.5, POLICY_C: 0.0}
    # A Rank-1 sniffer cannot deduce a deterministic pick: both draw.
    draws = {sel.select_for_tomorrow(random.Random(s)) for s in range(50)}
    assert draws == {POLICY_A, POLICY_B}
    # Pure-B evidence still samples B every time (degenerate mix is exact).
    sel2 = CFRPolicySelector()
    sel2.end_of_day_update(0, POLICY_A, _values(100, 150, 80))
    assert {sel2.select_for_tomorrow(random.Random(s)) for s in range(10)} == {POLICY_B}


def test_to_meta_distribution_replaces_pure_best_response():
    sel = CFRPolicySelector()
    pure_before = sel.to_meta_distribution()
    assert sum(pure_before.values()) == 1.0
    sel.end_of_day_update(0, POLICY_A, _values(100, 150, 80))
    assert sel.to_meta_distribution() == {POLICY_A: 0.0, POLICY_B: 1.0, POLICY_C: 0.0}


# --- IQL leaf evaluation ------------------------------------------------------

class _TensorLike:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


def test_value_net_protocol_predict_item_and_list():
    assert call_value_net(type("N", (), {"predict": lambda s, x: _TensorLike(7.5)})(), None) == 7.5
    assert call_value_net(lambda x: [3.0, 99.0], None) == 3.0
    assert evaluate_leaf_value({"money": 5.0}, None) == 5.0


def test_fallback_material_formula_matches_beam_layer():
    leaf = {"money": 1000.0, "shed": {"FERTILIZER": 4}, "herd": {"COW": 2}}
    assert fallback_leaf_value(leaf) == 1000.0 + 4 * 100.0 + 2 * 400.0


def test_broken_net_falls_back_per_leaf_without_poisoning():
    def bad_net(x):
        raise RuntimeError("no weights on this box")

    vals = counterfactual_values(
        {POLICY_A: {"money": 10.0}, POLICY_B: {"money": 20.0}, POLICY_C: {"money": 5.0}},
        bad_net,
    )
    assert vals == {POLICY_A: 10.0, POLICY_B: 20.0, POLICY_C: 5.0}


def test_iql_net_values_drive_regret():
    sel = CFRPolicySelector(value_net=lambda leaf: leaf["money"] * 2.0)
    upd = sel.end_of_day_update(
        3, POLICY_A,
        {POLICY_A: {"money": 50.0}, POLICY_B: {"money": 90.0}, POLICY_C: {"money": 40.0}},
    )
    assert upd.values == {POLICY_A: 100.0, POLICY_B: 180.0, POLICY_C: 80.0}
    assert upd.regrets[POLICY_B] == 80.0


def test_projected_leaves_apply_policy_deltas():
    leaves = project_counterfactual_leaves(
        {"money": 1000.0, "MILK": 0.0},
        {POLICY_A: {"MILK": 20.0}, POLICY_B: {"money": 50.0}},
    )
    assert leaves[POLICY_A] == {"money": 1000.0, "MILK": 20.0}
    assert leaves[POLICY_B] == {"money": 1050.0, "MILK": 0.0}
    assert leaves[POLICY_C] == {"money": 1000.0, "MILK": 0.0}


# --- live-game plumbing ---------------------------------------------------------

def test_hour23_gate_and_obs_helpers():
    assert is_end_of_day({"hour": 23, "day": 4}) is True
    assert is_end_of_day({"hour": 12, "day": 4}) is False
    assert is_end_of_day({"step": 119}) is True  # 119 % 24 == 23
    assert obs_day({"step": 119}) == 4 and obs_hour({"step": 119}) == 23
    assert obs_day({"day": 7, "hour": 0}) == 7


def test_maybe_update_only_fires_at_hour23():
    sel = CFRPolicySelector()
    assert sel.maybe_update_from_obs({"day": 2, "hour": 12}, POLICY_A, _values(1, 2, 3)) is None
    assert sel._iterations == 0
    upd = sel.maybe_update_from_obs({"day": 2, "hour": 23}, POLICY_A, _values(1, 2, 3))
    assert upd.day == 2 and sel._iterations == 1


def test_task_name_aliases_resolve():
    assert normalize_policy("Milk Flooder") == POLICY_A
    assert normalize_policy("strawberry_contingency") == POLICY_B
    assert normalize_policy("Poisoned Well") == POLICY_C
    sel = CFRPolicySelector()
    sel.end_of_day_update(0, "Milk Flooder", _values(100, 150, 80))
    assert sel.strategy_for_tomorrow()[POLICY_B] == 1.0


def test_reset_clears_live_state():
    sel = CFRPolicySelector()
    sel.end_of_day_update(0, POLICY_A, _values(100, 150, 80))
    sel.reset()
    assert sel.cumulative_regret == {POLICY_A: 0.0, POLICY_B: 0.0, POLICY_C: 0.0}
    assert sel.strategy_for_tomorrow()[POLICY_A] == 1 / 3
    assert sel.average_strategy()[POLICY_C] == 1 / 3
