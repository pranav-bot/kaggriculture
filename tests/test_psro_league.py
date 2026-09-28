"""Tests for the PSRO league (scripts/psro_league.py). Engine-free via stubs."""

import importlib.util
import json
import os
import random
import sys

import pytest

np = pytest.importorskip("numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.psro_league import (
    BR_GRID_DEFAULT,
    League,
    Member,
    fictitious_play,
    matrix_from_summary,
    run_league,
    train_best_response,
    write_ensemble,
)

RPS = np.array([[0.5, 0.7, 0.3],
                [0.3, 0.5, 0.7],
                [0.7, 0.3, 0.5]])


class StubRunner:
    """Scripted tournament results: RPS matrix + per-candidate BR scores."""

    def __init__(self, br_scores=None):
        self.br_scores = br_scores or {}

    def run_matrix(self, members, seeds, tag):
        assert len(seeds) > 0
        n = len(members)
        M = np.full((n, n), 0.5)
        M[:3, :3] = RPS  # baseline trio is non-transitive by construction
        return M

    def gauntlet_scores(self, candidate, panel, seeds, tag):
        base = self.br_scores.get(candidate.name, 0.5)
        return {m.name: base for m in panel}


def _args(tmp, **kw):
    d = dict(league_dir=str(tmp / "league"),
             baselines=["melon_rusher", "care_mill", "agent_final"],
             seeds=4, workers=1, iters=1, fp_iters=5000, eps=0.02,
             eval_seeds=2, ensemble_dir=str(tmp / "league" / "ensemble"),
             kagg=None, build=False, final_only=False)
    d.update(kw)
    return type("A", (), d)()


# --- meta-solver ---------------------------------------------------------------


def test_fictitious_play_finds_uniform_rps_nash():
    nash, value, expl = fictitious_play(RPS, iters=20000, seed=0)
    assert nash == pytest.approx([1 / 3] * 3, abs=0.05)
    assert value == pytest.approx(0.0, abs=0.02)
    assert expl < 0.05


def test_fictitious_play_finds_pure_nash_when_dominant():
    M = np.array([[0.5, 0.9, 0.9],
                  [0.1, 0.5, 0.6],
                  [0.1, 0.4, 0.5]])
    nash, _, expl = fictitious_play(M, iters=20000, seed=0)
    assert nash[0] > 0.9
    assert expl < 0.05


def test_matrix_from_summary_orders_and_defaults():
    summary = {"matrix": {"b": {"a": 0.75}, "a": {"b": 0.25}}}
    M = matrix_from_summary(summary, ["a", "b", "c"])
    assert M[0, 1] == pytest.approx(0.25)
    assert M[1, 0] == pytest.approx(0.75)
    assert M[0, 2] == pytest.approx(0.5)  # missing -> draw
    assert M[0, 0] == pytest.approx(0.5)


# --- best response ---------------------------------------------------------------


def test_train_best_response_picks_max_mixture_score(tmp_path):
    league = League(tmp_path / "league")
    for n in ("m0", "m1"):
        league.add_member_dir(n, f"/fake/{n}/main.py")
    nash = np.array([0.7, 0.3])
    grid = [dict(BR_GRID_DEFAULT[0]), dict(BR_GRID_DEFAULT[2])]
    runner = StubRunner(br_scores={"br0c0": 0.4, "br0c1": 0.9})
    member, info = train_best_response(league, nash, runner, iteration=0,
                                       grid=grid, eval_seeds=[0, 1])
    assert member.name == "br_iter0"
    assert info["config"]["target_crop"] == "STRAWBERRY"
    assert info["mixture_score"] == pytest.approx(0.9)
    main = tmp_path / "league" / "members" / "br_iter0" / "main.py"
    assert main.exists() and "def agent(obs)" in main.read_text()


# --- end-to-end (stubbed engine) ---------------------------------------------------


def test_league_loop_writes_matrix_nash_and_ensemble(tmp_path):
    final = run_league(_args(tmp_path, iters=2, eps=-1.0), StubRunner())
    assert set(final["members"][:3]) == {"melon_rusher", "care_mill", "agent_final"}
    league = League(tmp_path / "league")
    assert any(m.name == "br_iter0" for m in league.members)
    nash = np.array(final["nash"])
    assert nash.sum() == pytest.approx(1.0)
    ens = tmp_path / "league" / "ensemble"
    assert (ens / "main.py").exists()
    saved = json.loads((ens / "nash.json").read_text())
    assert saved["weights"] == pytest.approx(list(nash), abs=1e-6)


def test_ensemble_samples_nash_mixture(tmp_path):
    dest = tmp_path / "ens"
    dest.mkdir()
    for i, tag in enumerate(("A", "B")):
        (dest / f"src{i}.py").write_text(f"def agent(obs):\n    return '{tag}'\n")
    league = League(tmp_path / "league")
    league.add_member_dir("aa", str(dest / "src0.py"))
    league.add_member_dir("bb", str(dest / "src1.py"))
    from scripts.psro_league import write_ensemble as _we
    _we(league, np.array([0.75, 0.25]), tmp_path / "out")

    spec = importlib.util.spec_from_file_location("ens_main", tmp_path / "out" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ens_main"] = mod
    spec.loader.exec_module(mod)
    mod._rng = random.Random(0)
    counts = {"A": 0, "B": 0}
    for _ in range(2000):
        counts[mod.agent({"step": 0})] += 1
    assert counts["A"] / 2000 == pytest.approx(0.75, abs=0.05)
    # mid-episode turns stick with the committed member
    first = mod.agent({"step": 0})
    assert mod.agent({"step": 5}) == first
