#!/usr/bin/env python3
"""PSRO League: Fictitious-Play master loop for non-transitive Kaggriculture.

Rock-paper-scissors economics (Milk Flooder > Saver > ... > Milk Flooder)
means no single policy is robust. This loop maintains an empirical game,
solves for its Nash mixture (Fictitious Play), trains a Best Response
against that mixture, and repeats. Output: a Nash-weighted ensemble
submission.

Pipeline per iteration:
  1. PAYOFFS: round-robin `kagg tournament` over league members x SEEDS
     (default 100) x both seats -> win-rate matrix M (summary["matrix"]).
  2. META-SOLVE: Fictitious Play on the zero-sum game M - 0.5 -> Nash mix.
  3. BEST RESPONSE: tune ActionController hyperparameters (the Beam Search /
     offline-policy surface available without retraining nets) to maximize
     expected score vs the Nash mixture; materialize the winner as a new
     league member.
  4. Repeat; stop when exploitability < --eps.

Usage:
  .venv/bin/python scripts/psro_league.py --iters 3
  .venv/bin/python scripts/psro_league.py --iters 1 --seeds 8 --eval-seeds 4  # smoke
  .venv/bin/python scripts/psro_league.py --final-only  # re-emit ensemble
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggriculture-simulation" / "src-python"))

BASELINE_MEMBERS = ["melon_rusher", "care_mill", "agent_final"]
DEFAULT_SEEDS = 100

# Best-response search space over the controller surface (no net retraining).
BR_GRID_DEFAULT = [
    {"target_crop": "MELON", "min_sell_margin": 0.6, "max_hires_per_day": 1},
    {"target_crop": "MELON", "min_sell_margin": 0.4, "max_hires_per_day": 2},
    {"target_crop": "STRAWBERRY", "min_sell_margin": 0.6, "max_hires_per_day": 1},
    {"target_crop": "STRAWBERRY", "min_sell_margin": 0.8, "max_hires_per_day": 1},
    {"target_crop": "WHEAT", "min_sell_margin": 0.6, "max_hires_per_day": 2},
    {"target_crop": "WHEAT", "min_sell_margin": 0.4, "max_hires_per_day": 1},
]

BR_MAIN_TEMPLATE = '''"""PSRO best-response agent (iter {iteration}, vs Nash {nash_str}).

Tuned controller config: {config_str}
Mixture score during selection: {score:.3f}
"""
import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants

controller = ActionController(
    target_crop=Plants.{target_crop},
    auto_water=True,
    auto_harvest=True,
    auto_fertilize=True,
    auto_sell=True,
    auto_expand_land=True,
    auto_hire_hands=True,
    max_hires_per_day={max_hires_per_day},
    auto_dig_weeds=True,
    min_sell_margin={min_sell_margin},
)


def agent(obs):
    """Kaggle entrypoint."""
    return controller.act(obs)
'''


# ---------------------------------------------------------------------------
# Tournament runners
# ---------------------------------------------------------------------------
class Member:
    def __init__(self, name: str, main_path: str) -> None:
        self.name = name
        self.main_path = main_path

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "python", "path": self.main_path}


class KaggRunner:
    """Round-robin payoff evaluation via the Rust `kagg tournament` engine."""

    def __init__(self, kagg: Optional[str] = None, workers: int = 4,
                 out_root: str = "tournaments") -> None:
        from kaggsim.tournament import run_tournament  # deferred: needs kagg
        self._run = run_tournament
        self.workers = workers
        self.out_root = out_root
        if kagg:
            os.environ["KAGG_BIN"] = kagg

    def _cfg(self, name: str, candidate: Member, panel: List[Member],
             seeds: Sequence[int]) -> Dict[str, Any]:
        return {
            "name": name,
            "candidate": candidate.spec(),
            "panel": [m.spec() for m in panel],
            "schedule": "round_robin",
            "seats": "both",
            "worlds": {"strategy": "list", "seeds": [int(s) for s in seeds]},
            "workers": self.workers,
            "python": {"stderr": "null"},
            "output": {"dir": self.out_root, "resume": True},
        }

    def run_matrix(self, members: List[Member], seeds: Sequence[int],
                   tag: str) -> np.ndarray:
        """Mean-score matrix M[i][j] over all pairs (win=1, draw=0.5)."""
        if len(members) < 2:
            raise ValueError("need >= 2 members for a payoff matrix")
        summary = self._run(
            self._cfg(tag, members[0], members[1:], seeds))
        return matrix_from_summary(summary, [m.name for m in members])

    def gauntlet_scores(self, candidate: Member, panel: List[Member],
                        seeds: Sequence[int], tag: str) -> Dict[str, float]:
        """Mean score of `candidate` vs each panel member."""
        summary = self._run(self._cfg(tag, candidate, panel, seeds))
        mat = summary.get("matrix", {})
        return {m.name: float(mat.get(candidate.name, {}).get(m.name, 0.5))
                for m in panel}


def matrix_from_summary(summary: Dict[str, Any],
                        order: List[str]) -> np.ndarray:
    """Ordered payoff matrix from a `summary.json` dict (missing -> 0.5)."""
    mat = summary.get("matrix", {})
    n = len(order)
    M = np.full((n, n), 0.5, dtype=np.float64)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                continue
            try:
                M[i, j] = float(mat.get(a, {}).get(b, 0.5))
            except (TypeError, ValueError):
                M[i, j] = 0.5
    return M


# ---------------------------------------------------------------------------
# Meta-solver: Fictitious Play for zero-sum Nash
# ---------------------------------------------------------------------------
def fictitious_play(M: np.ndarray, iters: int = 20000,
                    seed: int = 0) -> Tuple[np.ndarray, float, float]:
    """Nash mix of the symmetric zero-sum game U = (M - M^T)/2 via FP.

    Returns (nash, value, exploitability). Exploitability = best-response
    gain of the row player + best-response gain of the column player vs the
    average profile (0 at equilibrium).
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    U = (M - M.T) / 2.0  # antisymmetrize: draws/noise-safe zero-sum
    rng = np.random.default_rng(seed)
    row_count = np.zeros(n)
    col_count = np.zeros(n)
    row_count[rng.integers(n)] += 1
    col_count[rng.integers(n)] += 1
    for _ in range(iters):
        row_avg = row_count / row_count.sum()
        col_avg = col_count / col_count.sum()
        row_count[int(np.argmax(U @ col_avg))] += 1
        col_count[int(np.argmin(row_avg @ U))] += 1
    nash_row = row_count / row_count.sum()
    nash_col = col_count / col_count.sum()
    nash = (nash_row + nash_col) / 2.0  # symmetric game: pool both averages
    nash = nash / nash.sum()
    value = float(nash @ U @ nash)
    expl = float(np.max(U @ nash) - np.min(nash @ U))
    return nash, value, expl


# ---------------------------------------------------------------------------
# League state
# ---------------------------------------------------------------------------
class League:
    def __init__(self, league_dir: str | Path) -> None:
        self.dir = Path(league_dir)
        self.members_dir = self.dir / "members"
        self.members_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "league.json"
        self.members: List[Member] = []
        self.history: List[Dict[str, Any]] = []
        if self.state_path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.state_path.read_text())
        self.members = [Member(m["name"], m["main"])
                        for m in data.get("members", [])]
        self.history = data.get("history", [])

    def save(self) -> None:
        self.state_path.write_text(json.dumps({
            "members": [{"name": m.name, "main": m.main_path}
                        for m in self.members],
            "history": self.history,
        }, indent=1))

    def add_baseline(self, name: str) -> Member:
        """Copy a submissions/<name>/ template into the league directory."""
        src = ROOT / "submissions" / name
        if not (src / "main.py").exists():
            raise FileNotFoundError(f"baseline template missing: {src}")
        if any(m.name == name for m in self.members):
            return next(m for m in self.members if m.name == name)
        dest = self.members_dir / name
        dest.mkdir(exist_ok=True)
        for py in sorted(src.glob("*.py")):
            shutil.copy2(py, dest / py.name)
        member = Member(name, str(dest / "main.py"))
        self.members.append(member)
        self.save()
        return member

    def add_member_dir(self, name: str, main_path: str) -> Member:
        member = Member(name, main_path)
        self.members.append(member)
        self.save()
        return member

    @property
    def names(self) -> List[str]:
        return [m.name for m in self.members]


# ---------------------------------------------------------------------------
# Best-response training (controller hyperparameter tuning vs Nash mixture)
# ---------------------------------------------------------------------------
def render_br_main(config: Dict[str, Any], iteration: int,
                   nash: np.ndarray, names: List[str], score: float,
                   dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    nash_str = ",".join(f"{n}={p:.2f}" for n, p in zip(names, nash))
    main = dest / "main.py"
    main.write_text(BR_MAIN_TEMPLATE.format(
        iteration=iteration, nash_str=nash_str,
        config_str=json.dumps(config), score=score,
        target_crop=config["target_crop"],
        max_hires_per_day=config["max_hires_per_day"],
        min_sell_margin=config["min_sell_margin"]))
    (dest / "meta.json").write_text(json.dumps({
        "config": config, "iteration": iteration,
        "nash": {n: float(p) for n, p in zip(names, nash)},
        "mixture_score": score,
    }, indent=1))
    return main


def train_best_response(
    league: League,
    nash: np.ndarray,
    runner: Any,
    *,
    iteration: int,
    grid: Optional[List[Dict[str, Any]]] = None,
    eval_seeds: Optional[Sequence[int]] = None,
    scratch: Optional[Path] = None,
) -> Tuple[Member, Dict[str, Any]]:
    """Tune controller configs to maximize expected score vs Nash mixture."""
    grid = grid if grid is not None else BR_GRID_DEFAULT
    eval_seeds = list(eval_seeds) if eval_seeds is not None else list(range(8))
    scratch = scratch or (league.dir / "br_scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    results = []
    for gi, config in enumerate(grid):
        cand_dir = scratch / f"iter{iteration}_cand{gi}"
        main = render_br_main(config, iteration, nash, league.names, 0.0, cand_dir)
        cand = Member(f"br{iteration}c{gi}", str(main))
        scores = runner.gauntlet_scores(cand, league.members, eval_seeds,
                                        tag=f"br{iteration}c{gi}")
        mix_score = float(sum(nash[j] * scores.get(league.members[j].name, 0.5)
                              for j in range(len(league.members))))
        results.append({"config": config, "score": mix_score, "dir": str(cand_dir)})
        print(f"  BR cand {gi} {config} -> mixture score {mix_score:.3f}", flush=True)

    results.sort(key=lambda r: -r["score"])
    best = results[0]
    name = f"br_iter{iteration}"
    dest = league.members_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(Path(best["dir"]), dest)
    main = render_br_main(best["config"], iteration, nash, league.names,
                          best["score"], dest)
    member = league.add_member_dir(name, str(main))
    info = {"name": name, "config": best["config"], "mixture_score": best["score"],
            "n_candidates": len(grid)}
    league.history.append({"event": "best_response", **info})
    league.save()
    return member, info


# ---------------------------------------------------------------------------
# Nash-weighted ensemble submission
# ---------------------------------------------------------------------------
ENSEMBLE_MAIN = '''"""Nash-weighted PSRO ensemble (generated by scripts/psro_league.py).

Each episode commits to one member sampled from the league Nash mixture.
Members: {members_str}
Nash: {nash_str}
"""
import os
import random
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

{imports}

_MODULES = [{modules}]
_WEIGHTS = [{weights}]

_rng = random.Random()
_current = None


def _sample():
    r = _rng.random()
    cum = 0.0
    for mod, w in zip(_MODULES, _WEIGHTS):
        cum += w
        if r < cum:
            return mod
    return _MODULES[-1]


def agent(obs):
    """Kaggle entrypoint: per-episode Nash-mixture commit."""
    global _current
    step = 0
    try:
        step = int(obs.get("step", 0) or 0)
    except (TypeError, ValueError):
        pass
    if step == 0 or _current is None:
        _current = _sample()
    return _current.agent(obs)
'''


def write_ensemble(league: League, nash: np.ndarray, dest: Path) -> Path:
    """Materialize the Nash-weighted ensemble as a submission directory."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    imports, mods, weights = [], [], []
    for i, member in enumerate(league.members):
        mod = f"_m{i}_{member.name}_main"
        shutil.copy2(member.main_path, dest / f"{mod}.py")
        imports.append(f"import {mod}")
        mods.append(mod)
        weights.append(f"{float(nash[i]):.6f}")
    members_str = ", ".join(league.names)
    nash_str = ", ".join(f"{n}={p:.3f}" for n, p in zip(league.names, nash))
    (dest / "main.py").write_text(ENSEMBLE_MAIN.format(
        members_str=members_str, nash_str=nash_str,
        imports="\n".join(imports), modules=", ".join(mods),
        weights=", ".join(weights)))
    (dest / "nash.json").write_text(json.dumps({
        "members": league.names,
        "weights": [float(p) for p in nash],
    }, indent=1))
    return dest / "main.py"


# ---------------------------------------------------------------------------
# Master loop
# ---------------------------------------------------------------------------
def run_league(args: argparse.Namespace, runner: Any) -> Dict[str, Any]:
    league = League(args.league_dir)
    for name in args.baselines:
        league.add_baseline(name)
    print(f"League members: {league.names}", flush=True)

    seeds = list(range(args.seeds))
    final: Dict[str, Any] = {}
    for it in range(args.iters):
        print(f"=== PSRO iter {it}: payoff matrix "
              f"({len(league.members)} members x {len(seeds)} seeds) ===", flush=True)
        M = runner.run_matrix(league.members, seeds, tag=f"psro_iter{it}")
        nash, value, expl = fictitious_play(M, iters=args.fp_iters, seed=it)
        print(f"matrix:\n{np.round(M, 3)}", flush=True)
        print(f"nash={np.round(nash, 3)} value={value:+.3f} "
              f"exploitability={expl:.4f}", flush=True)
        record = {"iter": it, "members": league.names,
                  "matrix": M.tolist(), "nash": nash.tolist(),
                  "value": value, "exploitability": expl}
        league.history.append({"event": "matrix", **record})
        league.save()
        final = record
        if expl < args.eps:
            print(f"Converged (exploitability {expl:.4f} < {args.eps}).", flush=True)
            break
        if it == args.iters - 1:
            break
        member, info = train_best_response(
            league, nash, runner, iteration=it, eval_seeds=list(range(args.eval_seeds)))
        print(f"Added best response: {info}", flush=True)

    nash = np.array(final["nash"])
    ensemble_main = write_ensemble(league, nash, Path(args.ensemble_dir))
    print(f"Ensemble written: {ensemble_main}", flush=True)
    if args.build:
        cmd = [sys.executable, str(ROOT / "scripts" / "build_submission.py"),
               "--agent", str(ensemble_main), "--format", "tar"]
        print(f"$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    return final


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PSRO Fictitious-Play league.")
    ap.add_argument("--league-dir", default=str(ROOT / "experiments" / "psro_league"))
    ap.add_argument("--baselines", nargs="+", default=list(BASELINE_MEMBERS))
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--fp-iters", type=int, default=20000)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--ensemble-dir", default=None,
                    help="defaults to <league-dir>/ensemble")
    ap.add_argument("--kagg", default=None)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--final-only", action="store_true",
                    help="re-emit the ensemble from the last solved Nash")
    args = ap.parse_args(argv)

    runner = KaggRunner(kagg=args.kagg, workers=args.workers,
                        out_root=str(Path(args.league_dir) / "tournaments"))
    if args.ensemble_dir is None:
        args.ensemble_dir = str(Path(args.league_dir) / "ensemble")
    if args.final_only:
        league = League(args.league_dir)
        solved = [h for h in league.history if h.get("event") == "matrix"]
        if not solved:
            raise SystemExit("no solved matrix in league history")
        last = solved[-1]
        # Re-align in case members changed since: use stored record directly.
        write_ensemble(league, np.array(last["nash"]), Path(args.ensemble_dir))
        print(f"Ensemble re-emitted from iter {last['iter']}.")
        return 0
    run_league(args, runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
