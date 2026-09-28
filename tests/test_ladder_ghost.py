"""Engine-free tests for scripts/ladder_ghost.py."""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ladder_ghost import (
    EXAMPLE_LOSS,
    generate_ghost,
    parse_build_order,
    render_tunable,
    run_hpo,
    sample_beam_weights,
)


def _obs(day, step, money=5000.0, shed=None, seeds=None, weed_at=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    if weed_at:
        tiles[weed_at[1]][weed_at[0]] = {"kind": "WEED"}
    return {
        "step": step, "day": day, "player": 0,
        "farms": [{"money": money, "tiles": tiles, "farmer": [4, 4],
                   "hands": [[4, 5]], "unlocked_quadrants": ["NW"]}],
        "private": {"shed": shed or {}, "seeds": seeds or {}},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def _load_ghost(tmp_path):
    ep, order = parse_build_order(dict(EXAMPLE_LOSS))
    main = generate_ghost(ep, "Boey", order, tmp_path / "ladder_ghost_1")
    spec = importlib.util.spec_from_file_location("ghost1", main)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ghost1"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_sorts_and_validates():
    ep, order = parse_build_order(dict(EXAMPLE_LOSS))
    assert ep == "112542379"
    assert [d["day"] for d in order] == sorted(d["day"] for d in order)
    with pytest.raises(ValueError, match="unknown action"):
        parse_build_order({"episode_id": 1,
                           "build_order": [{"day": 0, "action": "FLY"}]})
    with pytest.raises(ValueError, match="empty"):
        parse_build_order({"episode_id": 1, "build_order": []})


def test_ghost_fires_oneshot_orders_by_day(tmp_path):
    mod = _load_ghost(tmp_path)
    d0 = mod.agent(_obs(0, 0))
    kinds = [m[0] for m in d0["market"]]
    assert "BUY_ANIMAL" in kinds  # day-0 directive
    assert "BUY_LAND" not in kinds
    mod.agent(_obs(3, 72))  # advance: nothing new
    d5 = mod.agent(_obs(5, 120))
    assert ["BUY_LAND"] in d5["market"]  # day-5 EXPAND_NE
    d5b = mod.agent(_obs(5, 121))
    assert ["BUY_LAND"] not in d5b["market"]  # blind ledger: fires once


def test_ghost_provisions_seeds_and_dumps(tmp_path):
    mod = _load_ghost(tmp_path)
    mod.agent(_obs(0, 0))
    mod.agent(_obs(6, 144))  # HIRE directive fires here, freeing later cap
    d10 = mod.agent(_obs(10, 240))
    assert any(m[0] == "BUY_SEED" and m[1] == "STRAWBERRY" for m in d10["market"])
    d25 = mod.agent(_obs(25, 600, shed={"MILK": 20}))
    assert ["SELL", "MILK", 12] in d25["market"]


def test_ghost_km_routing_moves_to_weed(tmp_path):
    mod = _load_ghost(tmp_path)
    mod.agent(_obs(0, 0))
    out = mod.agent(_obs(3, 72, weed_at=(4, 4)))
    assert out["farmer"] == ["DIG"]  # standing on it
    out2 = mod.agent(_obs(3, 73, weed_at=(4, 6)))
    # Kuhn-Munkres sends the NEAREST unit (hand at [4,5]); farmer holds.
    assert out2["hands"][0] == ["SOUTH"]
    assert out2["farmer"] == ["PASS"]
    # greedy fallback path matches scipy path shape
    pairs = mod.kuhn_munkres_route([(0, 0)], [(2, 3)])
    assert pairs == [(0, 0)]


def test_ghost_never_crashes_and_resets_per_episode(tmp_path):
    mod = _load_ghost(tmp_path)
    mod.agent(_obs(0, 0))
    mod.agent(_obs(5, 120))
    ep2 = mod.agent(_obs(0, 0))  # new episode resets ledger
    assert ["BUY_LAND"] not in ep2["market"]
    assert mod.agent({})["farmer"] == ["PASS"]


def test_beam_weight_search_space_bounds():
    import optuna
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    w = sample_beam_weights(trial)
    assert 0.2 <= w["milk_prior_scale"] <= 5.0
    assert 4 <= w["top_k_intents"] <= 24
    assert set(w) == {"milk_prior_scale", "berry_prior_scale",
                      "poison_prior_scale", "holding_penalty_mult",
                      "milk_sell_margin", "berry_sell_margin", "top_k_intents"}


def test_hpo_picks_best_stub_and_ships_counter(tmp_path):
    def evaluator(weights, trial_no):
        return float(weights["berry_prior_scale"])  # monotonic: best = max scale

    res = run_hpo({"name": "ghost", "type": "tape", "path": "x"},
                  tmp_path / "hpo", n_trials=6, trial_seeds=[0],
                  workers=1, kagg=None, optuna_seed=0, evaluator=evaluator)
    summ = json.loads((tmp_path / "hpo" / "hpo_summary.json").read_text())
    peak = max(t["value"] for t in summ["trials"])
    assert res["best_value"] == pytest.approx(peak)
    assert res["best_weights"]["berry_prior_scale"] == pytest.approx(peak)
    assert (tmp_path / "hpo" / "counter_best" / "main.py").exists()
    assert (tmp_path / "hpo" / "counter_best" / "weights.json").exists()
    assert summ["n_trials"] == 6
