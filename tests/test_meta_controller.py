"""Tests for the PSRO MetaController (src/kaggriculture/meta)."""

import glob
import gzip
import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kaggriculture.features.macro_intents import MACRO_INTENT_CLASSES
from kaggriculture.meta import (
    CONSERVATIVE_SAVER,
    MELON_RUSH,
    MILK_FLOODER,
    POLICY_A,
    POLICY_B,
    POLICY_C,
    STRAWBERRY_CONTINGENCY,
    UNKNOWN,
    MetaController,
    build_default_profiles,
    opponent_spatial_footprint,
    own_reclaim_targets,
)


def _tiles(fill="LOCKED"):
    return [[fill for _ in range(10)] for _ in range(10)]


def _obs(own_tiles=None, opp_tiles=None, own_money=3000.0, opp_money=3000.0,
         opp_quads=None, step=100):
    return {
        "step": step,
        "player": 0,
        "farms": [
            {"money": own_money, "tiles": own_tiles or _tiles(None),
             "farmer": [4, 4], "hands": [], "unlocked_quadrants": ["NW"]},
            {"money": opp_money, "tiles": opp_tiles or _tiles(None),
             "farmer": [4, 4], "hands": [],
             "unlocked_quadrants": opp_quads or ["NW"]},
        ],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def _melon_rush_obs(cash=5000.0, n_empty=6):
    tiles = _tiles("LOCKED")
    for r in range(5, 10):
        for c in range(0, 5):
            tiles[r][c] = None
    # keep only n_empty empties, lock the rest of SW
    empties = [(r, c) for r in range(5, 10) for c in range(0, 5)]
    for r, c in empties[n_empty:]:
        tiles[r][c] = "LOCKED"
    return _obs(opp_tiles=tiles, opp_money=cash,
                opp_quads=["NW", "NE", "SW"])


# --- profiles ---------------------------------------------------------------


def test_three_distinct_profiles_cover_24_intents():
    profiles = build_default_profiles()
    assert set(profiles) == {POLICY_A, POLICY_B, POLICY_C}
    argmaxes = set()
    for pid, prof in profiles.items():
        prior = prof["intent_prior"]
        assert len(prior) == len(MACRO_INTENT_CLASSES) == 24
        assert sum(prior) == pytest.approx(1.0)
        assert all(p > 0 for p in prior)
        argmaxes.add(int(np.argmax(prior)))
    assert len(argmaxes) == 3  # distinct peaks


def test_profile_peaks_match_doctrine():
    profiles = build_default_profiles()
    top = lambda pid: MACRO_INTENT_CLASSES[int(np.argmax(profiles[pid]["intent_prior"]))]
    assert top(POLICY_A) in ("BUY_COW", "DUMP_MILK")  # milk-doctrine peak
    assert top(POLICY_B) == "PLANT_CROPS"
    assert top(POLICY_C) == "SELL_MARKET"


# --- perception ---------------------------------------------------------------


def test_footprint_counts_sw_empties():
    fp = opponent_spatial_footprint(_melon_rush_obs(n_empty=6), seat=0)
    assert fp["sw_empty"] == 6
    assert fp["cash"] == pytest.approx(5000.0)


def test_melon_rush_exact_rule():
    mc = MetaController()
    assert mc.infer_latent_strategy(_melon_rush_obs(cash=5000.0, n_empty=6), 0) == MELON_RUSH
    # cash at/under the hoard threshold -> not a rush
    assert mc.infer_latent_strategy(_melon_rush_obs(cash=2000.0, n_empty=6), 0) != MELON_RUSH
    # 5 empties is not > 5
    assert mc.infer_latent_strategy(_melon_rush_obs(cash=5000.0, n_empty=5), 0) != MELON_RUSH


def test_other_latent_clusters():
    mc = MetaController()
    hoard = {"event_latched": True, "event": "OPPONENT_HOARDING_DETECTED"}
    assert mc.infer_latent_strategy(_obs(), 0, hoard) == MILK_FLOODER

    tiles = _tiles("LOCKED")
    n = 0
    for r in range(10):
        for c in range(10):
            if n < 15:
                tiles[r][c] = {"kind": "PLANT", "crop": "STRAWBERRY"}
                n += 1
    assert mc.infer_latent_strategy(_obs(opp_tiles=tiles), 0, {}) == STRAWBERRY_CONTINGENCY

    saver = _obs(opp_tiles=_tiles("LOCKED"), opp_money=9000.0, opp_quads=["NW"])
    assert mc.infer_latent_strategy(saver, 0, {}) == CONSERVATIVE_SAVER
    assert mc.infer_latent_strategy(_obs(), 0, {}) == UNKNOWN


# --- gating -------------------------------------------------------------------


def test_gating_maps_clusters_to_best_responses():
    mc = MetaController()
    assert mc.gate(MILK_FLOODER) == POLICY_B
    assert mc.gate(MELON_RUSH) == POLICY_C
    assert mc.gate(STRAWBERRY_CONTINGENCY) == POLICY_A
    assert mc.gate(CONSERVATIVE_SAVER) == POLICY_A
    assert mc.gate(UNKNOWN) == POLICY_A
    assert mc.gate("NONSENSE") == POLICY_A  # unknown falls back safely


def test_update_switches_instantly_and_reports():
    mc = MetaController()
    d = mc.update(_melon_rush_obs(), 0, {})
    assert d["latent_strategy"] == MELON_RUSH
    assert d["active_policy"] == POLICY_C
    assert mc.active_policy == POLICY_C
    assert mc.meta_distribution() == {POLICY_A: 0.0, POLICY_B: 0.0, POLICY_C: 1.0}


# --- smooth transition ----------------------------------------------------------


def _pasture_farm():
    tiles = _tiles(None)
    tiles[1][1] = {"kind": "PASTURE"}  # empty -> reclaim
    tiles[2][2] = {"kind": "PASTURE", "animal": "COW"}  # occupied -> defer
    tiles[3][3] = {"kind": "WEED"}  # weed -> clear
    return tiles


def test_policy_b_entry_queues_digs_for_unoccupied_only():
    mc = MetaController()
    obs = _obs(own_tiles=_pasture_farm())
    mc.update(obs, 0, {"event_latched": True})  # hoarding -> MILK_FLOODER -> Policy_B
    assert mc.active_policy == POLICY_B
    queued = {(tuple(o["target"]), o["kind"]) for o in mc.transition_queue}
    assert ((1, 1), "PASTURE") in queued
    assert ((3, 3), "WEED") in queued
    assert not any(o["target"] == (2, 2) for o in mc.transition_queue)  # cow safe
    assert all(o["command"] == ["DIG"] for o in mc.transition_queue)


def test_transition_commands_render_and_drain():
    mc = MetaController()
    mc.plan_transition(_obs(own_tiles=_pasture_farm()), 0)
    assert len(mc.transition_queue) == 2
    # Farmer standing on the pasture target emits DIG immediately.
    cmds = mc.pop_transition_commands({"farmer": [1, 1], "hands": []})
    assert cmds["farmer"] == ["DIG"]
    assert len(mc.transition_queue) == 1
    # Distant unit steps toward its target; order stays queued.
    cmds = mc.pop_transition_commands({"farmer": [0, 0], "hands": []})
    assert cmds["farmer"] in (["SOUTH"], ["EAST"])
    assert len(mc.transition_queue) == 1


def test_own_reclaim_targets_marks_occupancy():
    targets = own_reclaim_targets(_obs(own_tiles=_pasture_farm()), 0)
    by_tile = {t["tile"]: t for t in targets}
    assert by_tile[(1, 1)]["occupied"] is False
    assert by_tile[(2, 2)]["occupied"] is True


# --- beam interface ---------------------------------------------------------------


def test_profile_for_beam_topk_and_penalty_hook():
    mc = MetaController(top_k_intents=6)
    mc.update(_melon_rush_obs(), 0, {})
    pkg = mc.profile_for_beam({"MILK": 4.0})
    assert pkg["policy_id"] == POLICY_C
    assert len(pkg["action_space"]) == 6
    assert len(pkg["intent_prior"]) == 24
    assert pkg["holding_penalty_fn"] is None  # no hoarding latched

    mc.update(_obs(own_tiles=_pasture_farm()), 0, {"event_latched": True})
    pkg = mc.profile_for_beam({"MILK": 4.0, "WOOL": 2.0})
    assert pkg["hoarding_active"] is True
    assert pkg["holding_penalty_fn"]({"MILK": 4.0, "WOOL": 2.0}) == pytest.approx(
        4 * 320.0 + 2 * 400.0
    )


def test_replay_smoke_over_real_episode():
    files = sorted(glob.glob(os.path.join(
        ROOT, "datasets", "il", "episodes", "00", "*.json.gz")))
    assert files
    ep = json.load(gzip.open(files[0]))
    mc = MetaController()
    seen = set()
    for t in range(0, len(ep["steps"]), 30):
        obs = ep["steps"][t][0]["observation"]
        d = mc.update(obs, 0, {})
        assert d["active_policy"] in (POLICY_A, POLICY_B, POLICY_C)
        seen.add(d["latent_strategy"])
    assert seen  # ran without error across game phases
