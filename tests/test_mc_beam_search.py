"""Tests for Monte Carlo opponent-uncertainty beam search."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scratch_grandmaster as sg


class TwoPlayerSim:
    """Dummy Rust-like rollout: our action helps, opponent action hurts."""

    def __init__(self, state=0):
        self.state = state

    def clone(self, state):
        return TwoPlayerSim(state)

    def step2(self, our_action, opp_action):
        return int(self.state) + int(our_action) * 10 - int(opp_action)


class SingleSeatSim:
    def __init__(self, state=0):
        self.state = state

    def clone(self, state):
        return SingleSeatSim(state)

    def step(self, action):
        return int(self.state) + int(action) * 10


def _value_net(state):
    return float(state)


def _intent_fn(state):
    probs = [0.0] * 24
    probs[1], probs[2], probs[3] = 0.5, 0.3, 0.2
    return probs


def test_expected_value_beats_optimistic():
    stats = {}
    winner = sg.step_level_beam_search(
        0, TwoPlayerSim(), IQL_Value_Net=_value_net,
        beam_width=3, horizon=24, action_space=[0, 1, 2],
        opponent_intent_fn=_intent_fn, level_budget_ms=5000.0,
        stats=stats)
    # E[v|a] = 10a - (0.5*1 + 0.3*2 + 0.2*3) = 10a - 1.7 -> always pick 2.
    # States accumulate via the most-likely branch (+19/level); the final
    # leaf scores the expectation: 23*19 + 18.3.
    assert tuple(winner.actions) == (2,) * 24
    assert winner.score == pytest.approx(23 * (20.0 - 1.0) + (20.0 - 1.7))
    assert stats["opp_queries"] == 24
    assert stats["degraded_levels"] == 0


def test_top_opponent_macros_grounding_and_renorm():
    top = sg.top_opponent_macros({"LIVESTOCK_RUSH": 0.5, "CROP_ROTATION": 0.3,
                                  "PREPARING_EXPANSION": 0.1,
                                  "HOARDING_FERTILIZER": 0.1}, k=3)
    assert [m for m, _ in top] == [11, 5, 1]
    assert sum(p for _, p in top) == pytest.approx(1.0)
    seq = [0.0] * 24
    seq[20], seq[21] = 0.6, 0.4
    assert sg.top_opponent_macros(seq, k=2) == [(20, 0.6), (21, 0.4)]
    assert sg.top_opponent_macros({}, k=3) == [(0, 1.0)]
    assert sg.top_opponent_macros({"DUMP_MILK": 1.0}, k=3) == [(20, 1.0)]


def test_level_budget_met_on_fast_path():
    stats = {}
    sg.step_level_beam_search(
        0, TwoPlayerSim(), IQL_Value_Net=_value_net,
        beam_width=3, horizon=24, action_space=[0, 1, 2],
        opponent_intent_fn=_intent_fn, level_budget_ms=5.0, stats=stats)
    assert stats["avg_level_ms"] < 5.0
    assert stats["degraded_levels"] == 0


def test_slow_sim_degrades_gracefully_not_fails():
    class SlowSim(TwoPlayerSim):
        def clone(self, state):
            return SlowSim(state)

        def step2(self, our_action, opp_action):
            time.sleep(0.003)
            return super().step2(our_action, opp_action)

    stats = {}
    winner = sg.step_level_beam_search(
        0, SlowSim(), IQL_Value_Net=_value_net,
        beam_width=3, horizon=24, action_space=[0, 1],
        opponent_intent_fn=_intent_fn, level_budget_ms=5.0, stats=stats)
    assert stats["degraded_levels"] > 0
    assert len(winner.actions) == 24


def test_single_seat_simulator_falls_back():
    winner = sg.step_level_beam_search(
        0, SingleSeatSim(), IQL_Value_Net=_value_net,
        beam_width=3, horizon=24, action_space=[0, 1, 2],
        opponent_intent_fn=_intent_fn, level_budget_ms=5000.0)
    # Opponent ignored: E[v|a] = 10a -> still picks 2, accumulated x24.
    assert tuple(winner.actions) == (2,) * 24
    assert winner.score == pytest.approx(24 * 20.0)


def test_broken_intent_fn_falls_back_to_legacy():
    winner = sg.step_level_beam_search(
        0, TwoPlayerSim(), IQL_Value_Net=_value_net,
        beam_width=3, horizon=24, action_space=[0, 1],
        opponent_intent_fn=lambda s: (_ for _ in ()).throw(RuntimeError("no net")),
        level_budget_ms=5000.0)
    assert len(winner.actions) == 24


def test_sequential_mode_matches_parallel():
    kw = dict(IQL_Value_Net=_value_net, beam_width=3, horizon=24,
              action_space=[0, 1, 2], opponent_intent_fn=_intent_fn,
              level_budget_ms=5000.0)
    a = sg.step_level_beam_search(0, TwoPlayerSim(), parallel_rollouts=True, **kw)
    b = sg.step_level_beam_search(0, TwoPlayerSim(), parallel_rollouts=False, **kw)
    assert tuple(a.actions) == tuple(b.actions)
    assert a.score == pytest.approx(b.score)
