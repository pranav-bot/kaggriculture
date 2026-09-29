"""tests/test_challenger_verification.py — Challenger 1 Empirical Test Suite.

Adversarial Stress Verification Suite covering:
1. Multi-Seed Head-to-Head Matches against Top 10 Ghost Opponents.
2. Latency profile assertions (mean < 2.0ms, max < 50.0ms).
3. Watchdog overage drawdown, threshold tripping (< 5.0s), and fallback routing.
4. Live simulation delay and error injection robustness.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from sim_engine import FastSimulation, is_kagg_available
from kaggsim.serve import load_agent, call_agent, obs_for
from kaggsim.constants import FINAL_STEP

CANDIDATE_PATH = ROOT / "submissions" / "hybrid_grandmaster_v2" / "main.py"

TOP_5_GHOSTS = [
    ("Boey", ROOT / "submissions" / "ladder_ghost_114845092" / "main.py"),
    ("DSM", ROOT / "submissions" / "ladder_ghost_114846266" / "main.py"),
    ("DECEM", ROOT / "submissions" / "ladder_ghost_114847113" / "main.py"),
    ("Vadim", ROOT / "submissions" / "ladder_ghost_114838145" / "main.py"),
    ("yuto083", ROOT / "submissions" / "ladder_ghost_114847353" / "main.py"),
]

SEEDS = [42, 100, 2026]


def setup_module():
    assert CANDIDATE_PATH.is_file(), f"Candidate submission not found at {CANDIDATE_PATH}"
    for name, path in TOP_5_GHOSTS:
        assert path.is_file(), f"Ghost opponent {name} not found at {path}"


def test_challenger_head_to_head_five_ghosts():
    """Verify Candidate vs 5 Top 10 Ghost Opponents across diverse seeds.
    
    Requirements:
    - 5 unique ghost opponents (Boey, DSM, DECEM, Vadim, yuto083).
    - Seeds: 42, 100, 2026.
    - All matches run to Step 719 with zero simulator faults or forfeitures.
    - Candidate turn latency: mean < 2.0ms, max < 50.0ms.
    - Candidate achieves positive cash margin.
    """
    if not is_kagg_available():
        pytest.skip("Rust simulation engine not available")

    with FastSimulation() as sim:
        for opp_name, ghost_path in TOP_5_GHOSTS:
            for seed in SEEDS:
                cand_fn = load_agent(str(CANDIDATE_PATH))
                ghost_fn = load_agent(str(ghost_path))

                p0_money, p1_money, final_st = sim.run_match(cand_fn, ghost_fn, seed=seed)

                # 1. Zero fault / clean completion assertion
                assert int(final_st.get("step", 0)) == 719, (
                    f"Match vs {opp_name} (seed {seed}) ended prematurely at step {final_st.get('step')}"
                )
                assert final_st.get("forfeit") is None, (
                    f"Match vs {opp_name} (seed {seed}) was forfeited: {final_st.get('forfeit')}"
                )

                # 2. Score and margin assertion
                assert p0_money > 0, f"Candidate money non-positive: {p0_money}"
                assert p0_money > p1_money, (
                    f"Candidate (${p0_money:,.2f}) failed to beat {opp_name} (${p1_money:,.2f}) on seed {seed}"
                )

                # 3. Latency assertions
                stats = cand_fn.stats() if hasattr(cand_fn, "stats") else {}
                turn_durations = cand_fn.turn_history if hasattr(cand_fn, "turn_history") else []

                assert len(turn_durations) == 719, f"Expected 719 turns, recorded {len(turn_durations)}"

                dur_ms = [d * 1000.0 for d in turn_durations]
                mean_lat = statistics.mean(dur_ms)
                max_lat = max(dur_ms)

                assert mean_lat < 2.0, (
                    f"Mean turn latency {mean_lat:.3f}ms exceeded 2.0ms threshold vs {opp_name} (seed {seed})"
                )
                assert max_lat < 50.0, (
                    f"Max turn latency {max_lat:.3f}ms exceeded 50.0ms limit vs {opp_name} (seed {seed})"
                )

                # 4. Watchdog integrity
                assert stats.get("circuit_broken") is False
                assert stats.get("fallback_trigger_count") == 0
                assert stats.get("remaining_overage_s") == 60.0


def test_challenger_both_seats_symmetry():
    """Verify Candidate performs cleanly in Seat 1 (Player 1) as well as Seat 0."""
    if not is_kagg_available():
        pytest.skip("Rust simulation engine not available")

    with FastSimulation() as sim:
        cand_fn = load_agent(str(CANDIDATE_PATH))
        ghost_fn = load_agent(str(TOP_5_GHOSTS[0][1]))  # Boey

        # Seat 1: Ghost is P0, Candidate is P1
        p0_money, p1_money, final_st = sim.run_match(ghost_fn, cand_fn, seed=42)

        assert int(final_st.get("step", 0)) == 719
        assert final_st.get("forfeit") is None
        assert p1_money > p0_money, f"Candidate in Seat 1 (${p1_money}) lost to Ghost (${p0_money})"

        stats = cand_fn.stats()
        assert stats["circuit_broken"] is False
        assert stats["fallback_trigger_count"] == 0
        assert stats["avg_turn_duration_ms"] < 2.0


def test_challenger_watchdog_overage_drawdown_unit():
    """Verify exact mathematical overage bank drawdown on synthetic delays."""
    cand_fn = load_agent(str(CANDIDATE_PATH))
    cand_fn.reset()

    assert cand_fn.remaining_overage == 60.0
    assert cand_fn.cumulative_overage_used == 0.0

    mock_obs = {
        "step": 1, "day": 0, "hour": 1, "player": 0,
        "farms": [{"money": 5000.0, "tiles": [[None]*10]*10, "farmer": [4, 4], "hands": []}, {}],
        "private": {"shed": {}, "seeds": {}},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }

    # Turn taking 1.4s -> consumes 0.4s overage
    with patch("time.perf_counter", side_effect=[10.0, 11.4]):
        act = cand_fn(mock_obs)
        assert isinstance(act, dict)

    assert abs(cand_fn.cumulative_overage_used - 0.4) < 1e-4
    assert abs(cand_fn.remaining_overage - 59.6) < 1e-4
    assert cand_fn.circuit_broken is False

    # Next fast turn (0.01s) -> no additional overage consumed
    with patch("time.perf_counter", side_effect=[20.0, 20.01]):
        cand_fn(dict(mock_obs, step=2, hour=2))

    assert abs(cand_fn.cumulative_overage_used - 0.4) < 1e-4
    assert abs(cand_fn.remaining_overage - 59.6) < 1e-4


def test_challenger_watchdog_circuit_breaker_and_fallback():
    """Verify circuit breaker trips below 5.0s and routes turns to SafeFallbackController."""
    cand_fn = load_agent(str(CANDIDATE_PATH))
    cand_fn.reset()

    mock_obs = {
        "step": 5, "day": 0, "hour": 5, "player": 0,
        "farms": [{"money": 5000.0, "tiles": [[None]*10]*10, "farmer": [4, 4], "hands": []}, {}],
        "private": {"shed": {}, "seeds": {}},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }

    # Catastrophic turn: 56.5s duration -> consumes 55.5s overage -> remaining = 4.5s (< 5.0s threshold)
    with patch("time.perf_counter", side_effect=[100.0, 156.5]):
        cand_fn(mock_obs)

    assert cand_fn.circuit_broken is True
    assert cand_fn.remaining_overage < 5.0

    # Next turn: primary agent is completely bypassed, fallback handles act
    called_primary = False
    original_fn = cand_fn.agent_fn

    def canary_fn(*args, **kwargs):
        nonlocal called_primary
        called_primary = True
        return original_fn(*args, **kwargs)

    cand_fn.agent_fn = canary_fn

    act = cand_fn(dict(mock_obs, step=6, hour=6))
    assert called_primary is False, "Primary agent called while circuit breaker was tripped!"
    assert isinstance(act, dict)
    assert "farmer" in act
    assert act.get("_fallback_active") is True


def test_challenger_watchdog_exception_shield():
    """Verify exception shield catches unhandled exceptions and returns valid actions."""
    cand_fn = load_agent(str(CANDIDATE_PATH))
    cand_fn.reset()

    mock_obs = {
        "step": 10, "day": 0, "hour": 10, "player": 0,
        "farms": [{"money": 5000.0, "tiles": [[None]*10]*10, "farmer": [4, 4], "hands": []}, {}],
        "private": {"shed": {}, "seeds": {}},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }

    original_fn = cand_fn.agent_fn

    def exploding_agent(*args, **kwargs):
        raise ValueError("Injected adversarial failure")

    cand_fn.agent_fn = exploding_agent

    act = cand_fn(mock_obs)
    assert isinstance(act, dict)
    assert act.get("_exception_shield_triggered") is True
    assert "ValueError: Injected adversarial failure" in act.get("_last_error", "")
    assert cand_fn.fallback_trigger_count == 1
    assert "farmer" in act


def test_challenger_watchdog_reset_on_step0():
    """Verify watchdog resets state automatically when step 0 / day 0 / hour 0 is received."""
    cand_fn = load_agent(str(CANDIDATE_PATH))
    cand_fn.cumulative_overage_used = 45.0
    cand_fn.remaining_overage = 15.0
    cand_fn.fallback_trigger_count = 3
    cand_fn.circuit_broken = True

    obs_step0 = {
        "step": 0, "day": 0, "hour": 0, "player": 0,
        "farms": [{"money": 5000.0, "tiles": [[None]*10]*10, "farmer": [4, 4], "hands": []}, {}],
        "private": {"shed": {}, "seeds": {}},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }

    cand_fn(obs_step0)
    assert cand_fn.remaining_overage == 60.0
    assert cand_fn.cumulative_overage_used == 0.0
    assert cand_fn.circuit_broken is False
    assert cand_fn.fallback_trigger_count == 0


def test_challenger_live_match_delay_injection():
    """Verify live match with artificial turn delay completes to Step 719 without crashing."""
    if not is_kagg_available():
        pytest.skip("Rust simulation engine not available")

    cand_fn = load_agent(str(CANDIDATE_PATH))
    ghost_fn = load_agent(str(TOP_5_GHOSTS[0][1]))
    cand_fn.reset()

    real_agent = cand_fn.agent_fn

    def delayed_agent(obs: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        step = int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0))
        if step == 30:
            time.sleep(1.05)  # Soft limit breach (1.05s)
        return real_agent(obs, *args, **kwargs)

    cand_fn.agent_fn = delayed_agent

    with FastSimulation() as sim:
        js = sim.srv.reset(42)
        while js["step"] < FINAL_STEP:
            a0 = call_agent(cand_fn, obs_for(js, 0))
            a1 = call_agent(ghost_fn, obs_for(js, 1))
            js = sim.srv.step2(a0, a1)

    assert js["step"] == 719, f"Simulation terminated prematurely: {js['step']}"
    stats = cand_fn.stats()
    assert stats["cumulative_overage_used_s"] > 0.0, "Overage was not recorded"
    assert stats["remaining_overage_s"] < 60.0
