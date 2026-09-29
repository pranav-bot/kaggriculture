"""Tests for the ShadowInjector offline telemetry tool."""

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tools.shadow_telemetry import (
    ShadowInjector,
    load_agent,
    load_replay,
    main,
    obs_for_turn,
    sanitize,
)


def _obs(day, hour, step):
    tiles = [[None] * 10 for _ in range(10)]
    farm = {"money": 3000.0, "farmer": [4, 4], "hands": [],
            "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": tiles}
    return {"player": 0, "day": day, "hour": hour, "step": step,
            "farms": [farm, dict(farm)],
            "private": {"shed": {}, "seeds": {}, "inventories": []},
            "market": {"prices": {"MELON": 250}, "inventory": {}},
            "town": {"unlocked_shops": []}}


def _replay(path, hours=((0, 22), (0, 23), (1, 0), (1, 1))):
    steps = []
    for i, (day, hour) in enumerate(hours):
        steps.append([{"observation": _obs(day, hour, i), "action": None,
                       "reward": 0, "status": "ACTIVE", "info": {}}] * 2)
    payload = {"id": "loss_123", "steps": steps}
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return path


def _stub_agent_module():
    """Agent exposing every intercepted subsystem (no source edits needed).

    Built via exec into the stub namespace so the agent's internal calls
    resolve (and are therefore interceptable) in that namespace.
    """
    mod = types.ModuleType("stub_shadow_agent")
    sys.modules["stub_shadow_agent"] = mod
    exec(
        "from kaggriculture.meta.cfr import CFRPolicySelector\n"
        "DISCRETE_MACRO_ACTIONS = ['PASS', 'EXPAND_NE', 'EXPAND_SW',\n"
        "    'EXPAND_SE', 'HIRE_WORKER', 'PLANT_CROPS', 'WATER_CROPS',\n"
        "    'HARVEST_CROPS']\n"
        "cfr_selector = CFRPolicySelector(seed=0)\n"
        "def evaluate_leaf(state, net=None):\n"
        "    return float(state.get('v', 0.0)) + 10.0\n"
        "class _Winner:\n"
        "    actions = (5, 7, 4, 1)\n"
        "    score = 123.5\n"
        "def step_level_beam_search(*args, **kwargs):\n"
        "    evaluate_leaf({'v': 40.0})\n"
        "    evaluate_leaf({'v': 30.0})\n"
        "    evaluate_leaf({'v': 20.0})\n"
        "    return _Winner()\n"
        "def kuhn_munkres_route(units, targets):\n"
        "    return [(0, 0)]\n"
        "class OpponentIntentModel:\n"
        "    def predict_proba(self, buf):\n"
        "        return {'MELON_RUSH': 0.82, 'MILK_FLOODER': 0.10,\n"
        "                'STRAWBERRY_CONTINGENCY': 0.05, 'UNKNOWN': 0.03}\n"
        "class _Endgame:\n"
        "    def solve(self, inv):\n"
        "        return {'MELON': 12, 'MILK': 8}\n"
        "endgame_solver = _Endgame()\n"
        "def agent(obs):\n"
        "    evaluate_leaf({'v': 1.0})\n"
        "    step_level_beam_search(None, None)\n"
        "    OpponentIntentModel().predict_proba([])\n"
        "    kuhn_munkres_route([[4, 4]], [[4, 5]])\n"
        "    endgame_solver.solve({})\n"
        "    if obs.get('hour') == 23:\n"
        "        cfr_selector.end_of_day_update(\n"
        "            obs.get('day'), 'Policy_A',\n"
        "            {'Policy_A': 100.0, 'Policy_B': 150.0, 'Policy_C': 80.0})\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
        mod.__dict__,
    )
    return mod


def test_full_interception_captures_all_five_blocks(tmp_path):
    _stub_agent_module()
    replay = _replay(str(tmp_path / "loss_123.json"))
    out = main(["--replay", replay, "--agent", "stub_shadow_agent",
                "--out", str(tmp_path)])
    assert out.endswith("telemetry_loss_123.jsonl")
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 4
    assert [r["hour"] for r in rows] == [22, 23, 0, 1]

    # IQL leaf values intercepted every turn (1 direct + 3 inside beam).
    assert rows[0]["iql"]["present"] is True
    assert rows[0]["iql"]["n_evals"] == 4
    assert sorted(rows[0]["iql"]["values_sample"]) == [11.0, 30.0, 40.0, 50.0]

    # Beam winner trajectory + top-3 macros + IQL expected values.
    beam = rows[0]["beam"]
    assert beam["present"] is True
    assert beam["winner_trajectory"] == ["PLANT_CROPS", "HARVEST_CROPS",
                                         "HIRE_WORKER", "EXPAND_NE"]
    assert beam["top_macros"] == ["PLANT_CROPS", "HARVEST_CROPS", "HIRE_WORKER"]
    assert beam["iql_expected_values"] == [50.0, 40.0, 30.0]
    assert beam["winner_score"] == 123.5

    # Bayesian opponent intent probabilities.
    intent = rows[0]["opponent_intent"]
    assert intent["present"] is True
    assert intent["probabilities"]["MELON_RUSH"] == 0.82

    # Endgame safe sale volumes.
    endgame = rows[0]["endgame"]
    assert endgame["present"] is True
    assert endgame["safe_sale_volumes"] == {"raw": {"MELON": 12, "MILK": 8}}

    # KM routing assignment.
    km = rows[0]["km"]
    assert km["present"] is True
    assert km["assignment"] == [[0, 0]]
    assert km["cost_matrix_shape"] == [1, 1]
    assert km["total_manhattan_cost"] == 1

    # CFR table only gains evidence at Hour 23 (r = u - u_A).
    assert rows[0]["cfr"]["present"] is False
    hour23 = rows[1]["cfr"]
    assert hour23["present"] is True
    assert hour23["cumulative_regret"] == {"Policy_A": 0.0, "Policy_B": 50.0,
                                           "Policy_C": -20.0}
    assert hour23["policy_probabilities"] == {"Policy_A": 0.0, "Policy_B": 1.0,
                                               "Policy_C": 0.0}
    assert hour23["n_day_updates"] == 1

    # Agent action recorded untouched.
    assert rows[0]["agent_action"] == {"farmer": ["PASS"], "hands": [],
                                       "market": []}
    assert rows[0]["error"] is None


def test_graceful_degradation_on_production_agent(tmp_path):
    replay = _replay(str(tmp_path / "loss_9.json"),
                     hours=((2, 5), (2, 6)))
    out = main(["--replay", replay, "--agent", "agent_final",
                "--out", str(tmp_path), "--max-turns", "2"])
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 2
    for block in ("cfr", "beam", "opponent_intent", "endgame", "km"):
        assert rows[0][block]["present"] is False
    assert rows[0]["iql"]["present"] is False
    action = rows[0]["agent_action"]
    assert set(action) == {"farmer", "hands", "market"}
    assert rows[0]["error"] is None


def test_obs_rebuild_matches_replay_turn(tmp_path):
    replay = _replay(str(tmp_path / "r.json"))
    steps, episode_id = load_replay(replay)
    assert episode_id == "loss_123"
    obs = obs_for_turn(steps, 2, 0)
    assert (obs["day"], obs["hour"], obs["step"]) == (1, 0, 2)
    obs["farms"][0]["money"] = -1  # mutation must not leak into replay
    assert obs_for_turn(steps, 2, 0)["farms"][0]["money"] == 3000.0


def test_sanitize_never_raises_and_bounds():
    assert sanitize(float("nan")) == "nan"
    assert sanitize({"a": list(range(200))})["a"][-1] == {"<truncated>": "136 more"}
    assert isinstance(sanitize(object()), str)


def test_cli_requires_replay_and_agent(tmp_path):
    try:
        main(["--replay", str(tmp_path / "nope.json"), "--agent", "agent_final"])
    except (SystemExit, FileNotFoundError, OSError):
        pass
    else:
        raise AssertionError("missing replay should fail")


# ---------------------------------------------------------------------------
# Regressions: each test below pins a bug found by adversarial probing.
# ---------------------------------------------------------------------------

def _stub(name, source, obs):
    """Build a stub agent module and return a (replay, main-args) runner."""
    mod = types.ModuleType(name)
    exec(source, mod.__dict__)
    sys.modules[name] = mod
    return mod


def test_installs_every_hook_not_just_the_first(tmp_path):
    """Regression: any(genexpr) short-circuited, so only the FIRST-named entry
    point was patched and the entry point the agent actually calls was missed
    (silent ``present: false``)."""
    _stub("multi_beam_agent", """
DISCRETE_MACRO_ACTIONS = ['PASS', 'EXPAND_NE']
def step_level_beam_search(*a, **k):
    raise AssertionError('unused path')
def beam_search_operations(*a, **k):
    class _W:
        actions = (1,)
        score = 77.0
    return _W()
def agent(obs):
    return beam_search_operations(None, None)
""", None)
    replay = _replay(str(tmp_path / "r.json"), hours=((0, 0),))
    out = main(["--replay", replay, "--agent", "multi_beam_agent",
                "--out", str(tmp_path)])
    row = json.loads(open(out).readline())
    assert row["beam"]["present"] is True
    assert row["beam"]["winner_trajectory"] == ["EXPAND_NE"]
    assert row["beam"]["winner_score"] == 77.0


def test_episode_id_prefers_info_over_uuid(tmp_path):
    """Regression: Kaggle replay ``id`` is a UUID, so output was
    telemetry_<uuid>.jsonl instead of telemetry_<EPISODE_ID>.jsonl."""
    path = str(tmp_path / "episode-114833391-replay.json")
    steps = [[{"observation": _obs(0, 0, 0), "action": None, "reward": 0,
               "status": "ACTIVE", "info": {}}] * 2]
    with open(path, "w") as fh:
        json.dump({"id": "6f6e8660-bb6e-11f1-bda6-0242ac130204",
                   "info": {"EpisodeId": 114833391}, "steps": steps}, fh)
    _steps, episode_id = load_replay(path)
    assert episode_id == "114833391"
    out = main(["--replay", path, "--agent", "agent_final",
                "--out", str(tmp_path), "--max-turns", "1"])
    assert out.endswith("telemetry_114833391.jsonl")


def test_agent_spec_with_src_prefix_resolves_root_module():
    """Regression: the documented ``src.kaggriculture.agent_final`` form failed
    because the agent module actually lives at the repo root."""
    fn, mod = load_agent("src.kaggriculture.agent_final")
    assert callable(fn)
    assert mod.__name__ == "agent_final"


def test_beam_values_are_not_misattributed_to_macros(tmp_path):
    """Regression: leaf values were sorted then zipped onto the trajectory,
    which paired the winner's macro with another candidate's value (inverted)."""
    _stub("beam_attr_agent", """
DISCRETE_MACRO_ACTIONS = ['PASS', 'EXPAND_NE']
def evaluate_leaf(state, IQL_Value_Net=None):
    return float(state['v'])
class _W:
    actions = (0, 1)      # PASS then EXPAND_NE
    score = 99.0
def step_level_beam_search(*a, **k):
    evaluate_leaf({'v': 10.0})   # PASS
    evaluate_leaf({'v': 99.0})   # EXPAND_NE
    return _W()
def agent(obs):
    return step_level_beam_search(None, None)
""", None)
    replay = _replay(str(tmp_path / "r.json"), hours=((0, 0),))
    out = main(["--replay", replay, "--agent", "beam_attr_agent",
                "--out", str(tmp_path)])
    beam = json.loads(open(out).readline())["beam"]
    assert beam["top_macros"] == ["PASS", "EXPAND_NE"]
    # Attribution is not recoverable from an untagged leaf evaluator.
    assert beam["macro_iql_pairs"] is None
    assert beam["iql_values_attributed"] is False
    # The verified mapping is the beam's own score for the returned candidate.
    assert beam["winner_macro_value"] == 99.0
    assert beam["iql_expected_values"] == [99.0, 10.0]


def test_intent_captured_from_sequence_of_pairs(tmp_path):
    """Regression: top_opponent_macros returns [(macro, prob), ...]; the harvester
    only accepted Mappings and reported present: false."""
    _stub("intent_seq_agent", """
def top_opponent_macros(intent_dist, k=3):
    return [(0, 0.55), (1, 0.45)]
def agent(obs):
    return top_opponent_macros({'MILK_FLOODER': 0.55})
""", None)
    replay = _replay(str(tmp_path / "r.json"), hours=((0, 0),))
    out = main(["--replay", replay, "--agent", "intent_seq_agent",
                "--out", str(tmp_path)])
    intent = json.loads(open(out).readline())["opponent_intent"]
    assert intent["present"] is True
    assert intent["probabilities"] == {"0": 0.55, "1": 0.45}


def test_cfr_reads_live_table_under_any_attribute_name(tmp_path):
    """Regression: the live table was only read from four hardcoded attribute
    names, so a selector bound to anything else reported nulls; and the strategy
    was re-derived locally instead of asked of the controller."""
    _stub("cfr_alias_agent", """
from kaggriculture.meta.cfr import CFRPolicySelector
CFRPolicySelector = CFRPolicySelector
sel = CFRPolicySelector(seed=0)
def agent(obs):
    if obs.get('hour') == 23:
        sel.end_of_day_update(obs.get('day'), 'Policy_A',
                              {'Policy_A': 100.0, 'Policy_B': 150.0,
                               'Policy_C': 80.0})
    return {'farmer': ['PASS'], 'hands': [], 'market': []}
""", None)
    replay = _replay(str(tmp_path / "r.json"), hours=((0, 22), (0, 23)))
    out = main(["--replay", replay, "--agent", "cfr_alias_agent",
                "--out", str(tmp_path)])
    rows = [json.loads(x) for x in open(out)]
    cfr = rows[1]["cfr"]
    assert cfr["present"] is True
    assert cfr["cumulative_regret"] == {"Policy_A": 0.0, "Policy_B": 50.0,
                                        "Policy_C": -20.0}
    # played_policy must be read by NAME: patching a class method intercepts
    # unbound calls where arg 0 is self (arg 1 was the day, not the policy).
    assert cfr["played_policy"] == "Policy_A"
    # Must equal the controller's own distribution, not a local re-derivation.
    assert cfr["policy_probabilities"] == {"Policy_A": 0.0, "Policy_B": 1.0,
                                           "Policy_C": 0.0}


def test_seat_out_of_range_raises_clear_error(tmp_path):
    replay = _replay(str(tmp_path / "r.json"), hours=((0, 0),))
    steps, _ = load_replay(replay)
    try:
        obs_for_turn(steps, 0, 5)
    except IndexError as exc:
        assert "seat 5" in str(exc)
    else:
        raise AssertionError("out-of-range seat must raise")
