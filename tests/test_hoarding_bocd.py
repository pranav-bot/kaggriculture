"""Tests for Subtask 5: BOCD opponent-hoarding detection + liquidation override.

Covers scripts/quant_full_product.py (detector, tracker, monitor, override)
and the holding-penalty hook in scratch_grandmaster.step_level_beam_search.
"""

import gzip
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from kaggriculture.env.items import MARKET_I0, MARKET_PARAMS, market_price
from scripts.quant_full_product import (
    HOARDING_PROB_THRESHOLD,
    LIQUIDATION_ITEMS,
    OPPONENT_HOARDING_DETECTED,
    BayesianChangePointDetector,
    HoardingMonitor,
    OpponentTracker,
    adjust_beam_score,
    apply_hoarding_override_to_policy,
    demo_hoarding_detection,
    holding_penalty_for_inventory,
    parse_opponent_state,
    run_hoarding_replay,
    simulate_product,
)


def _obs(step, fert_inv, cash0=3000.0, cash1=3000.0):
    return {
        "step": step,
        "farms": [{"money": cash0}, {"money": cash1}],
        "market": {"inventory": {"FERTILIZER": fert_inv}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


# --- Engine price asymmetry: the reason preemption matters -----------------


def test_wool_drops_quadratically():
    d1 = market_price("WOOL", MARKET_I0) - market_price("WOOL", MARKET_I0 + 25)
    d2 = market_price("WOOL", MARKET_I0 + 25) - market_price("WOOL", MARKET_I0 + 50)
    assert d2 > d1 > 0  # accelerating (convex) collapse


def test_milk_drops_linearly():
    amp = 1.6 * 160 / 122  # above_target * base / T, engine-exact
    d1 = market_price("MILK", MARKET_I0) - market_price("MILK", MARKET_I0 + 10)
    d2 = market_price("MILK", MARKET_I0 + 10) - market_price("MILK", MARKET_I0 + 20)
    assert d1 == d2 == pytest.approx(round(amp * 10), abs=1.0)


def test_simultaneous_dump_is_catastrophic():
    assert market_price("WOOL", MARKET_I0 + 100) == 1
    assert market_price("MILK", MARKET_I0 + 100) == 1
    assert MARKET_PARAMS["WOOL"]["above_func"] == "sq"
    assert MARKET_PARAMS["MILK"]["above_func"] == "linear"


# --- BOCD detector ----------------------------------------------------------


def test_detector_fires_on_selling_to_hoarding_shift():
    rng = np.random.default_rng(42)
    series = [float(rng.normal(60.0, 8.0)) for _ in range(10)] + [0.0] * 6
    det = BayesianChangePointDetector()
    probs = [det.update(x) for x in series]
    assert max(probs[10:]) > HOARDING_PROB_THRESHOLD
    assert HOARDING_PROB_THRESHOLD == pytest.approx(0.85)


def test_detector_quiet_on_steady_series():
    worst = 0.0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        det = BayesianChangePointDetector()
        probs = [det.update(float(rng.normal(60.0, 8.0))) for _ in range(16)]
        worst = max(worst, max(probs[3:]))  # skip burn-in
    assert worst < HOARDING_PROB_THRESHOLD


def test_detector_invalid_hazard_rejected():
    with pytest.raises(ValueError):
        BayesianChangePointDetector(hazard=1.5)


# --- Public-state parsing ---------------------------------------------------


def test_parse_opponent_state_accounting_is_exact():
    prev = {"FERTILIZER": 10000.0}
    obs = _obs(7, 10007.0, cash0=500.0, cash1=750.0)
    parsed = parse_opponent_state(
        obs, seat=0, prev_market_inventory=prev, our_executed_sells={"FERTILIZER": 2.0}
    )
    assert parsed["opponent_fertilizer_sold"] == pytest.approx(5.0)  # 7 - 2
    assert parsed["opponent_cash_reserves"] == pytest.approx(750.0)
    assert parsed["turn"] == 7


def test_parse_without_prev_returns_none_fert():
    parsed = parse_opponent_state(_obs(0, 10000.0), seat=0)
    assert parsed["opponent_fertilizer_sold"] is None
    assert parsed["opponent_cash_reserves"] == pytest.approx(3000.0)


def test_tracker_rolling_series_and_windows():
    tr = OpponentTracker()
    tr.update(_obs(0, 10000.0), seat=0)
    for t in range(1, 7):
        tr.update(
            _obs(t, 10000.0 + 3 * t),
            seat=0,
            our_executed_sells={"FERTILIZER": 1.0},
        )
    assert len(tr.opponent_fertilizer_sold) == 6
    assert list(tr.opponent_fertilizer_sold) == pytest.approx([2.0] * 6)
    assert tr.window_sum(tr.opponent_fertilizer_sold, 6) == pytest.approx(12.0)
    assert tr.window_sum(tr.opponent_fertilizer_sold, 7) is None


# --- Monitor: flag semantics -------------------------------------------------


def test_monitor_latches_hoarding_event():
    rng = np.random.default_rng(42)
    truth = [float(rng.normal(60.0, 8.0)) for _ in range(10)] + [0.0] * 6
    monitor = HoardingMonitor(window_turns=6, min_windows=3)
    fert = float(MARKET_I0)
    monitor.update(_obs(0, fert), seat=0, our_executed_sells={"FERTILIZER": 0.0})
    for day, sold in enumerate(truth, start=1):
        for _ in range(6):
            fert += sold / 6.0
            status = monitor.update(
                _obs(day, fert), seat=0, our_executed_sells={"FERTILIZER": 0.0}
            )
    assert status["event"] == OPPONENT_HOARDING_DETECTED
    assert status["event_latched"] is True
    info = status["trigger_info"]
    assert info["p_changepoint"] > 0.85
    assert info["pre_mean"] > info["post_mean"]  # cessation signature
    # Latch persists on further hoarding observations.
    later = monitor.update(_obs(99, fert), seat=0, our_executed_sells={"FERTILIZER": 0.0})
    assert later["event_latched"] is True
    monitor.clear()
    assert monitor.status()["event"] is None


def test_monitor_ignores_upward_shift():
    monitor = HoardingMonitor(window_turns=1, min_windows=3)
    fert = float(MARKET_I0)
    monitor.update(_obs(0, fert), seat=0, our_executed_sells={"FERTILIZER": 0.0})
    for i in range(1, 8):
        fert += 60.0
        monitor.update(_obs(i, fert), seat=0, our_executed_sells={"FERTILIZER": 0.0})
    for i in range(8, 14):
        fert += 120.0
        status = monitor.update(
            _obs(i, fert), seat=0, our_executed_sells={"FERTILIZER": 0.0}
        )
    assert status["event_latched"] is False  # surge != hoarding


# --- Strategic override ------------------------------------------------------


def test_holding_penalty_targets_milk_wool_only():
    inv = {"MILK": 10.0, "WOOL": 5.0, "WHEAT": 100.0, "FERTILIZER": 50.0}
    assert holding_penalty_for_inventory(inv) == pytest.approx(10 * 320.0 + 5 * 400.0)
    assert holding_penalty_for_inventory({}) == pytest.approx(0.0)
    assert set(LIQUIDATION_ITEMS) == {"MILK", "WOOL"}


def test_adjust_beam_score_only_when_active():
    assert adjust_beam_score(1000.0, {"MILK": 10.0}, hoarding_active=False) == 1000.0
    assert adjust_beam_score(1000.0, {"MILK": 10.0}, hoarding_active=True) == pytest.approx(
        1000.0 - 3200.0
    )


def test_override_policy_forces_immediate_liquidation():
    base_policy = {"start_day": 20, "daily_cap": 4, "floor": 150,
                   "stage_caps": [(17, 22, 8)], "endgame_dump_day": 28}
    overridden = apply_hoarding_override_to_policy(base_policy)
    assert overridden["start_day"] == 0
    assert overridden["floor"] == 1
    assert overridden["endgame_dump_day"] == 0
    assert overridden["daily_cap"] >= 10**5
    assert base_policy["start_day"] == 20  # input not mutated

    production = [0] * 10 + [6] * 10 + [0] * 10  # 60 milk total
    sim = simulate_product("MILK", production, 24.0, overridden)
    assert sim["unsold"] == 0
    assert sim["sold"] == sum(production)


def test_demo_end_to_end():
    demo = demo_hoarding_detection()
    assert demo["trigger"] is not None
    assert demo["trigger"]["p_changepoint"] > 0.85
    assert len(demo["dump_impact"]) == 2


def test_replay_mode_runs_on_real_episode():
    import glob as _glob

    files = _glob.glob(os.path.join(ROOT, "datasets", "il", "episodes", "*", "*.json.gz"))
    assert files, "no episode files available"
    rep = run_hoarding_replay(files[0], seat=0)
    assert rep["turns"] > 700
    assert rep["windows"] >= 20
    assert np.isfinite(rep["inferred_opp_fert_mean"])


# --- Beam-search hook --------------------------------------------------------


def test_beam_search_holding_penalty_hook():
    sys.path.insert(0, ROOT)
    import importlib

    sg = importlib.import_module("scratch_grandmaster")

    def sim(action):
        return {"money": 100.0, "hold": int(action)}

    # No hook: legacy behavior, every leaf scores 100.
    winner = sg.step_level_beam_search(
        {"money": 100.0, "hold": 0}, sim, action_space=[0, 1],
        action_to_simulator=lambda a, s: a,
    )
    assert winner.score == pytest.approx(100.0)

    # Hook penalizing action 1 holdings: beam must converge on all zeros.
    winner = sg.step_level_beam_search(
        {"money": 100.0, "hold": 0}, sim, action_space=[0, 1],
        action_to_simulator=lambda a, s: a,
        holding_penalty_fn=lambda state: 50.0 * state.get("hold", 0),
    )
    assert tuple(winner.actions) == (0,) * 24
    assert winner.score == pytest.approx(100.0)
