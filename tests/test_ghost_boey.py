"""Ghost-opponent exploitability test (Boey, replay 112542379)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from kaggriculture.meta import (
    EXPANSION_RUSH,
    OPPONENT_EXPANSION_DETECTED,
    POLICY_A,
    POLICY_B,
    MetaController,
)
from scripts.ghost_boey_test import (
    GhostFailure,
    assert_ghost_result,
    ground_truth_expansions,
    run_ghost,
)

REPLAY = os.path.join(ROOT, "replays", "other_agents", "rank1", "112542379.json")


def _tiles(fill="LOCKED"):
    return [[fill for _ in range(10)] for _ in range(10)]


def _obs(quads, step=0):
    return {
        "step": step,
        "farms": [
            {"money": 3000.0, "tiles": _tiles(None), "farmer": [4, 4],
             "hands": [], "unlocked_quadrants": ["NW"]},
            {"money": 3000.0, "tiles": _tiles(None), "farmer": [4, 4],
             "hands": [], "unlocked_quadrants": list(quads)},
        ],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def test_expansion_edge_fires_exact_turn():
    mc = MetaController()
    assert mc.infer_latent_strategy(_obs(["NW"], step=10), 0) == "UNKNOWN"
    assert mc.infer_latent_strategy(_obs(["NW", "NE"], step=11), 0) == EXPANSION_RUSH
    assert mc.expansion_event["turn"] == 11
    assert mc.expansion_event["new_quadrants"] == ["NE"]
    d = mc.update(_obs(["NW", "NE", "SW"], step=12), 0, {})
    assert d["event"] == OPPONENT_EXPANSION_DETECTED
    assert d["active_policy"] == POLICY_B


def test_dwell_holds_counter_policy_across_flap():
    mc = MetaController(expansion_dwell_turns=72)
    mc.update(_obs(["NW"], step=0), 0, {})
    d = mc.update(_obs(["NW", "NE"], step=1), 0, {})
    assert d["active_policy"] == POLICY_B
    # Heuristic inference flaps (no expansion edge anymore) -> still held.
    for t in range(2, 10):
        d = mc.update(_obs(["NW", "NE"], step=t), 0, {})
        assert d["active_policy"] == POLICY_B, f"flapped at turn {t}"
        assert d["dwell_remaining"] == 1 + 72 - t
    # After dwell expiry, normal gating resumes.
    d = mc.update(_obs(["NW", "NE"], step=100), 0, {})
    assert d["dwell_remaining"] == 0


def test_no_dwell_flaps_and_diagnostic_fires():
    mc = MetaController(expansion_dwell_turns=0)
    mc.update(_obs(["NW"], step=0), 0, {})
    mc.update(_obs(["NW", "NE"], step=1), 0, {})
    d = mc.update(_obs(["NW", "NE"], step=2), 0, {})
    assert d["active_policy"] == POLICY_A  # flap-back without dwell


def test_ghost_boey_full_replay():
    res = run_ghost(REPLAY, 0, "Boey")
    assert res["opponent"] == "Boey"
    ne = res["expansions"][0]
    assert ne["new"] == ["NE"]  # ground truth derived from the log
    verdict = assert_ghost_result(res)
    assert verdict["flag_turn"] == ne["turn"] == 146
    assert verdict["lag_turns"] == 0
    assert verdict["policy_before"] == POLICY_A


def test_ghost_failure_names_exact_turn():
    ep = {"info": {"TeamNames": ["us", "them"]}, "steps": [
        [{"observation": _obs(["NW"], step=t),
          "action": {"market": []}}] for t in range(30)
    ]}
    # No expansion in this stream: harness must fail loudly, not pass vacuously.
    import scripts.ghost_boey_test as gb
    real_load = gb.load_replay
    gb.load_replay = lambda *a, **k: ep
    try:
        res = gb.run_ghost("fake", 0, "them")
        with pytest.raises(GhostFailure, match="no ground-truth expansion"):
            gb.assert_ghost_result(res)
    finally:
        gb.load_replay = real_load
