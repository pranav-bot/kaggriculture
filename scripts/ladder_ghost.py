#!/usr/bin/env python3
"""Ladder-ghost pipeline: replay a parsed ladder loss locally, evolve a counter.

Reads a parsed ladder loss (loss_analysis.json build order), generates a
hardcoded deterministic ghost agent (submissions/ladder_ghost_<ID>/) that
blindly executes the opponent's macro-sequence with our Kuhn-Munkres routing
layer, evaluates our Hybrid Grandmaster against it in a local kagg
tournament, solves the resulting game with the PSRO meta-solver
(Fictitious Play), and -- if the grandmaster loses -- triggers Optuna HPO
over the Beam Search weights to find a counter configuration.

loss_analysis.json schema:
  {"episode_id": 112542379, "opponent_name": "Boey",
   "build_order": [{"day": 5, "action": "EXPAND_NE"},
                   {"day": 10, "action": "PLANT", "crop": "STRAWBERRY", "count": 14},
                   {"day": 25, "action": "DUMP", "product": "MILK"}]}
  Supported actions: EXPAND_NE/SW/SE, HIRE{count}, BUY_SEED{crop,count},
  BUY_ANIMAL{animal,count}, PLANT{crop,count}, SELL/DUMP{product}.

Usage:
  .venv/bin/python scripts/ladder_ghost.py --example-out loss_analysis.json
  .venv/bin/python scripts/ladder_ghost.py --loss loss_analysis.json --seeds 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from find_best_agent import aggregate, run_mass_tournament  # noqa: E402
from psro_league import fictitious_play  # noqa: E402

GRANDMASTER_DEFAULT = "agent_final"  # apex two-team (hybrid) grandmaster
LEAGUE_CONTEXT = ["melon_rusher", "care_mill"]
MAX_MARKET_ORDERS = 10

VALID_ACTIONS = {"EXPAND_NE", "EXPAND_SW", "EXPAND_SE", "HIRE", "BUY_SEED",
                 "BUY_ANIMAL", "PLANT", "SELL", "DUMP"}

EXAMPLE_LOSS = {
    "episode_id": 112542379,
    "opponent_name": "Boey",
    "our_score": 69225,
    "opp_score": 170961,
    "build_order": [
        {"day": 0, "action": "BUY_ANIMAL", "animal": "COW", "count": 3},
        {"day": 0, "action": "PLANT", "crop": "WHEAT", "count": 8},
        {"day": 5, "action": "EXPAND_NE"},
        {"day": 6, "action": "HIRE", "count": 8},
        {"day": 10, "action": "PLANT", "crop": "STRAWBERRY", "count": 14},
        {"day": 25, "action": "DUMP", "product": "MILK"},
    ],
}


# ---------------------------------------------------------------------------
# 1. Build-order parsing
# ---------------------------------------------------------------------------
def parse_build_order(doc: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Validate + day-sort the extracted macro-sequence. Returns (ep_id, order)."""
    ep = str(doc.get("episode_id", "unknown"))
    raw = doc.get("build_order") or []
    if not raw:
        raise ValueError("loss_analysis.json has empty build_order")
    order = []
    for i, step in enumerate(raw):
        act = str(step.get("action", "")).upper()
        if act not in VALID_ACTIONS:
            raise ValueError(f"build_order[{i}]: unknown action {act!r}")
        day = int(step.get("day", 0))
        norm: Dict[str, Any] = {"day": day, "action": act}
        for key in ("crop", "animal", "product"):
            if step.get(key) is not None:
                norm[key] = str(step[key]).upper()
        if step.get("count") is not None:
            norm["count"] = int(step["count"])
        order.append(norm)
    order.sort(key=lambda s: s["day"])
    return ep, order


# ---------------------------------------------------------------------------
# 2. Ghost agent generation (Kuhn-Munkres routing vendored from our layer)
# ---------------------------------------------------------------------------
GHOST_TEMPLATE = '''"""Ghost of ladder episode {ep_id} ({opp_name}) -- DO NOT EDIT BY HAND.

Deterministic replay of the extracted macro-sequence below. One-shot market
orders fire once via a blind ledger; continuous directives (PLANT/DUMP) run
while active. Field work uses our Kuhn-Munkres routing layer (scipy if
present, greedy fallback -- same as scratch_grandmaster).
Schedule: {schedule_str}
"""
import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

SCHEDULE = {schedule_repr}
MAX_ORDERS = 10

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:
    _lsa = None

_fired = set()  # blind one-shot ledger: (day, index)


def kuhn_munkres_route(unit_positions, targets):
    """Our routing layer: min-Manhattan assignment (vendored)."""
    if not unit_positions or not targets:
        return []
    costs = [[abs(int(u[0]) - int(t[0])) + abs(int(u[1]) - int(t[1]))
              for t in targets] for u in unit_positions]
    if _lsa is None:  # greedy fallback
        pairs, used = [], set()
        for row in range(len(costs)):
            best, bj = None, -1
            for col in range(len(targets)):
                if col not in used and (best is None or costs[row][col] < best):
                    best, bj = costs[row][col], col
            if bj >= 0:
                used.add(bj)
                pairs.append((row, bj))
        return pairs
    rows, cols = _lsa(costs)
    return list(zip(rows.tolist(), cols.tolist()))


def _step(src, dst):
    dx, dy = int(dst[0]) - int(src[0]), int(dst[1]) - int(src[1])
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _tile(farm, r, c):
    try:
        return farm["tiles"][r][c]
    except (IndexError, TypeError, KeyError):
        return "LOCKED"


def agent(obs):
    """Kaggle entrypoint: blind macro-sequence execution."""
    try:
        return _act(obs)
    except Exception:
        n_hands = len(((obs.get("farms") or [{{}}])[int(obs.get("player", 0))]).get("hands", []))
        return {{"farmer": ["PASS"], "hands": [["PASS"]] * n_hands, "market": []}}


def _act(obs):
    if int(obs.get("step", 0) or 0) == 0:
        _fired.clear()  # new episode: blind ledger resets
    seat = int(obs.get("player", 0))
    farm = (obs.get("farms") or [{{}}])[seat] if seat < len(obs.get("farms") or []) else {{}}
    day = int(obs.get("day", 0) or 0)
    money = float(farm.get("money", 0.0) or 0.0)
    private = obs.get("private") or {{}}
    shed = dict(private.get("shed") or {{}})
    seeds = dict((str(k).upper(), int(v or 0)) for k, v in (private.get("seeds") or {{}}).items())
    farmer = list(farm.get("farmer", [4, 4]))
    hands = [list(p) for p in (farm.get("hands") or [])]
    market, used = [], [0]

    def order(o):
        if used[0] < MAX_ORDERS:
            market.append(o)
            used[0] += 1

    # --- one-shot directives (blind ledger) ---
    for i, d in enumerate(SCHEDULE):
        if d["day"] > day or (d["day"], i) in _fired:
            continue
        a = d["action"]
        if a in ("EXPAND_NE", "EXPAND_SW", "EXPAND_SE"):
            order(["BUY_LAND"])
            _fired.add((d["day"], i))
        elif a == "HIRE":
            for _ in range(int(d.get("count", 1))):
                order(["HIRE"])
            _fired.add((d["day"], i))
        elif a == "BUY_SEED":
            order(["BUY_SEED", d["crop"], int(d.get("count", 4))])
            _fired.add((d["day"], i))
        elif a == "BUY_ANIMAL":
            order(["BUY_ANIMAL", d["animal"], int(d.get("count", 1))])
            _fired.add((d["day"], i))

    # --- seed provisioning for active PLANT directives ---
    for d in SCHEDULE:
        if d["day"] <= day and d["action"] == "PLANT":
            have = seeds.get(d["crop"], 0)
            need = max(0, int(d.get("count", 8)) - have)
            if need > 0 and money > 500:
                order(["BUY_SEED", d["crop"], min(need, 8)])

    # --- continuous DUMP directives: liquidate shed stock ---
    for d in SCHEDULE:
        if d["day"] <= day and d["action"] in ("SELL", "DUMP"):
            qty = int(shed.get(d["product"], 0) or 0)
            if qty > 0:
                order(["SELL", d["product"], min(qty, 12)])

    # --- field work via Kuhn-Munkres routing ---
    tiles = farm.get("tiles") or []
    empties, weeds, thirsty, ripe, animals = [], [], [], [], []
    for r in range(10):
        for c in range(10):
            cell = _tile(farm, r, c)
            if cell is None:
                empties.append((c, r))
            elif isinstance(cell, dict) and cell.get("kind") == "WEED":
                weeds.append((c, r))
            elif isinstance(cell, dict) and cell.get("kind") == "PLANT":
                if not cell.get("watered_today"):
                    thirsty.append((c, r))
                if int(cell.get("yield_units", 0) or 0) > 0:
                    ripe.append((c, r))
            elif isinstance(cell, dict) and cell.get("kind") == "PASTURE":
                if cell.get("animal"):
                    animals.append((c, r))
    plant_crops = [d["crop"] for d in SCHEDULE
                   if d["day"] <= day and d["action"] == "PLANT"
                   and seeds.get(d["crop"], 0) > 0 and empties]
    units = [("farmer", tuple(farmer))] + [(f"h{{i}}", tuple(p)) for i, p in enumerate(hands)]
    jobs = ([(t, ["DIG"]) for t in weeds]
            + [(t, ["WATER"]) for t in thirsty]
            + [(t, ["HARVEST"]) for t in ripe]
            + [(t, ["FEED"]) for t in animals]
            + [(t, ["CARE"]) for t in animals])
    if plant_crops:
        jobs += [(t, ["PLANT", plant_crops[0]]) for t in empties]
    frontier = jobs[:len(units) * 2]
    acts = {{"farmer": ["PASS"], "hands": [["PASS"]] * len(hands)}}
    if frontier and units:
        pairs = kuhn_munkres_route([u[1] for u in units], [j[0] for j in frontier])
        for ui, ji in pairs:
            name, pos = units[ui]
            tgt, cmd = frontier[ji]
            go = cmd if tuple(pos) == tuple(tgt) else _step(pos, tgt)
            if name == "farmer":
                acts["farmer"] = go
            else:
                acts["hands"][int(name[1:])] = go
    acts["market"] = market
    return acts
'''


def generate_ghost(ep_id: str, opp_name: str,
                   order: List[Dict[str, Any]], dest: Path) -> Path:
    """Write submissions/ladder_ghost_<ID>/main.py. Returns main path."""
    dest.mkdir(parents=True, exist_ok=True)
    parts = []
    for d in order:
        what = d.get("crop") or d.get("animal") or d.get("product") or ""
        cnt = f"x{d['count']}" if "count" in d else ""
        parts.append(f"D{d['day']}:{d['action']}" + (f" {what}{cnt}" if what or cnt else ""))
    sched_str = "; ".join(parts)
    (dest / "main.py").write_text(GHOST_TEMPLATE.format(
        ep_id=ep_id, opp_name=opp_name, schedule_str=sched_str,
        schedule_repr=repr(order)))
    (dest / "manifest.json").write_text(json.dumps(
        {"episode_id": ep_id, "opponent": opp_name, "build_order": order,
         "routing": "kuhn-munkres (vendored, scipy/greedy)"}, indent=1))
    return dest / "main.py"


# ---------------------------------------------------------------------------
# 3-4. Inject + evaluate via the PSRO meta-solver path
# ---------------------------------------------------------------------------
def spec_for(name: str, submissions_dir: Path) -> Dict[str, Any]:
    main = submissions_dir / name / "main.py"
    if not main.is_file():
        raise FileNotFoundError(f"agent missing: {main}")
    return {"name": name, "type": "python", "path": str(main)}


def evaluate_league(specs: List[Dict[str, Any]], seeds: Sequence[int],
                    kagg: Optional[str], workers: int,
                    out_dir: str) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """Round-robin matrix + FP Nash over the league (grandmaster vs ghost...)."""
    from kaggsim.tournament import run_tournament  # local engine front end
    import os as _os
    if kagg:
        _os.environ["KAGG_BIN"] = kagg
    panel = [dict(s) for s in specs]
    first, rest = panel[0], panel[1:]
    cfg = {"name": "ghost-league", "candidate": first, "panel": rest,
           "schedule": "round_robin", "seats": "both",
           "worlds": {"strategy": "list", "seeds": [int(s) for s in seeds]},
           "workers": workers, "python": {"stderr": "null"},
           "output": {"dir": out_dir, "resume": True}}
    summary = run_tournament(cfg)
    names = [s["name"] for s in specs]
    mat = summary.get("matrix", {})
    n = len(names)
    M = np.full((n, n), 0.5)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                try:
                    M[i, j] = float(mat.get(a, {}).get(b, 0.5))
                except (TypeError, ValueError):
                    M[i, j] = 0.5
    nash, value, expl = fictitious_play(M, iters=20000, seed=0)
    return M, {"nash": nash, "value": value, "exploitability": expl,
               "summary": summary}, names


# ---------------------------------------------------------------------------
# 5. Optuna HPO over Beam Search weights (triggered only on a local loss)
# ---------------------------------------------------------------------------
BEAM_SEARCH_SPACE = {
    "milk_prior_scale": ("loguniform", 0.2, 5.0),
    "berry_prior_scale": ("loguniform", 0.2, 5.0),
    "poison_prior_scale": ("loguniform", 0.2, 5.0),
    "holding_penalty_mult": ("uniform", 0.5, 4.0),
    "milk_sell_margin": ("uniform", 0.3, 0.9),
    "berry_sell_margin": ("uniform", 0.3, 0.9),
    "top_k_intents": ("int", 4, 24),
}

TUNABLE_TEMPLATE = '''"""Optuna-tuned counter (trial {trial_no}, score {score:.3f}). Beam weights: {wstr}"""
import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants
from kaggriculture.meta import MetaController, POLICY_A, POLICY_B, POLICY_C

WEIGHTS = {weights_repr}
_profiles = None

def _controller():
    from kaggriculture.meta import build_default_profiles
    global _profiles
    if _profiles is None:
        _profiles = build_default_profiles()
        _profiles["Policy_A"]["intent_prior"] = [
            min(1.0, p * WEIGHTS["milk_prior_scale"]) for p in _profiles["Policy_A"]["intent_prior"]]
        _profiles["Policy_B"]["intent_prior"] = [
            min(1.0, p * WEIGHTS["berry_prior_scale"]) for p in _profiles["Policy_B"]["intent_prior"]]
        _profiles["Policy_C"]["intent_prior"] = [
            min(1.0, p * WEIGHTS["poison_prior_scale"]) for p in _profiles["Policy_C"]["intent_prior"]]
    return _profiles

_mc = MetaController(profiles=_controller(),
                     holding_penalty_fn=lambda inv: sum(
                         max(0.0, float((inv or {{}}).get(k, 0.0) or 0.0))
                         * WEIGHTS["holding_penalty_mult"]
                         * {{ "MILK": 160.0, "WOOL": 200.0 }}[k] for k in ("MILK", "WOOL")),
                     top_k_intents=WEIGHTS["top_k_intents"])
_ctrls = {{
    POLICY_A: ActionController(target_crop=Plants.MELON, target_animal="COW",
        auto_hire_hands=True, max_hires_per_day=2,
        min_sell_margin=WEIGHTS["milk_sell_margin"]),
    POLICY_B: ActionController(target_crop=Plants.STRAWBERRY,
        auto_hire_hands=True, max_hires_per_day=1,
        min_sell_margin=WEIGHTS["berry_sell_margin"]),
    POLICY_C: ActionController(target_crop=Plants.WHEAT, min_sell_margin=0.4),
}}

def agent(obs):
    """Kaggle entrypoint: meta-gated hybrid response."""
    try:
        seat = int(obs.get("player", 0))
        pkg = _mc.profile_for_beam()
        return _ctrls.get(pkg["policy_id"], _ctrls[POLICY_A]).act(obs)
    except Exception:
        return _ctrls[POLICY_A].act(obs)
'''


def sample_beam_weights(trial: Any) -> Dict[str, Any]:
    return {
        "milk_prior_scale": trial.suggest_float("milk_prior_scale", 0.2, 5.0, log=True),
        "berry_prior_scale": trial.suggest_float("berry_prior_scale", 0.2, 5.0, log=True),
        "poison_prior_scale": trial.suggest_float("poison_prior_scale", 0.2, 5.0, log=True),
        "holding_penalty_mult": trial.suggest_float("holding_penalty_mult", 0.5, 4.0),
        "milk_sell_margin": trial.suggest_float("milk_sell_margin", 0.3, 0.9),
        "berry_sell_margin": trial.suggest_float("berry_sell_margin", 0.3, 0.9),
        "top_k_intents": trial.suggest_int("top_k_intents", 4, 24),
    }


def render_tunable(weights: Dict[str, Any], trial_no: int, score: float,
                   dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "main.py").write_text(TUNABLE_TEMPLATE.format(
        trial_no=trial_no, score=score,
        wstr=",".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in weights.items()),
        weights_repr=repr(weights)))
    (dest / "weights.json").write_text(json.dumps(weights, indent=1))
    return dest / "main.py"


def run_hpo(ghost_spec: Dict[str, Any], out_dir: Path, *,
            n_trials: int, trial_seeds: Sequence[int], workers: int,
            kagg: Optional[str], optuna_seed: int,
            evaluator: Any = None) -> Dict[str, Any]:
    """Optuna maximize of tunable-hybrid win rate vs the ghost."""
    import optuna
    from kaggsim.tournament import run_tournament
    import os as _os
    if kagg:
        _os.environ["KAGG_BIN"] = kagg
    out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(weights: Dict[str, Any], trial_no: int) -> float:
        if evaluator is not None:
            return float(evaluator(weights, trial_no))
        trial_dir = out_dir / f"trial_{trial_no:03d}"
        main = render_tunable(weights, trial_no, 0.0, trial_dir)
        cfg = {"name": f"hpo-t{trial_no}", "candidate": {
                   "name": "tunable", "type": "python", "path": str(main)},
               "panel": [ghost_spec], "schedule": "round_robin",
               "seats": "both",
               "worlds": {"strategy": "list", "seeds": [int(s) for s in trial_seeds]},
               "workers": workers, "python": {"stderr": "null"},
               "output": {"dir": str(out_dir / "tournaments"), "resume": False}}
        summary = run_tournament(cfg)
        return float(summary.get("matrix", {}).get("tunable", {}).get(
            ghost_spec["name"], 0.5))

    def objective(trial: Any) -> float:
        w = sample_beam_weights(trial)
        score = evaluate(w, trial.number)
        trial.set_user_attr("weights", w)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=optuna_seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    best_w = dict(best.user_attrs["weights"])
    counter_dir = out_dir / "counter_best"
    if counter_dir.exists():
        shutil.rmtree(counter_dir)
    render_tunable(best_w, best.number, best.value, counter_dir)
    (out_dir / "hpo_summary.json").write_text(json.dumps(
        {"best_value": best.value, "best_weights": best_w,
         "n_trials": n_trials,
         "trials": [{"no": t.number, "value": t.value,
                     "weights": t.user_attrs.get("weights", {})}
                    for t in study.trials]}, indent=1))
    return {"best_value": best.value, "best_weights": best_w,
            "counter_dir": str(counter_dir)}


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ladder-ghost injection + counter HPO.")
    ap.add_argument("--loss", default="loss_analysis.json")
    ap.add_argument("--example-out", default=None,
                    help="Write an example loss_analysis.json and exit.")
    ap.add_argument("--submissions-dir", default=str(ROOT / "submissions"))
    ap.add_argument("--grandmaster", default=GRANDMASTER_DEFAULT)
    ap.add_argument("--league", nargs="*", default=list(LEAGUE_CONTEXT))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kagg", default=None)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--trial-seeds", type=int, default=4)
    ap.add_argument("--optuna-seed", type=int, default=0)
    ap.add_argument("--no-hpo", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if args.example_out:
        Path(args.example_out).write_text(json.dumps(EXAMPLE_LOSS, indent=1))
        print(f"wrote example {args.example_out}")
        return 0

    doc = json.loads(Path(args.loss).read_text())
    ep_id, order = parse_build_order(doc)
    opp_name = str(doc.get("opponent_name", "ladder"))
    print(f"Parsed loss {ep_id} ({opp_name}): {len(order)} directives", flush=True)

    # 1-2. Generate + inject the ghost.
    subs = Path(args.submissions_dir)
    ghost_name = f"ladder_ghost_{ep_id}"
    ghost_main = generate_ghost(ep_id, opp_name, order, subs / ghost_name)
    print(f"Injected ghost: {ghost_main}", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "experiments" / ghost_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. League tournament through the PSRO meta-solver path.
    names = [args.grandmaster, ghost_name] + [l for l in args.league if l != args.grandmaster]
    specs = [spec_for(n, subs) for n in names]
    M, solved, _ = evaluate_league(
        specs, list(range(args.seeds)), args.kagg, args.workers,
        str(out_dir / "tournaments"))
    gi = names.index(args.grandmaster)
    hi = names.index(ghost_name)
    gm_wr = float(M[gi, hi])
    print(f"payoff matrix:\n{np.round(M, 3)}", flush=True)
    print(f"nash={np.round(solved['nash'], 3)} "
          f"exploitability={solved['exploitability']:.4f}", flush=True)
    print(f"{args.grandmaster} vs {ghost_name}: win rate {gm_wr:.3f}", flush=True)
    (out_dir / "league_result.json").write_text(json.dumps(
        {"members": names, "matrix": M.tolist(), "nash": solved["nash"].tolist(),
         "grandmaster_win_rate": gm_wr}, indent=1))

    # 4. Conditional Optuna HPO on Beam Search weights.
    if gm_wr < 0.5 and not args.no_hpo:
        print(f"LOSS locally ({gm_wr:.3f} < 0.5): triggering Optuna HPO "
              f"({args.trials} trials)...", flush=True)
        ghost_spec = specs[hi]
        hpo = run_hpo(ghost_spec, out_dir / "hpo", n_trials=args.trials,
                      trial_seeds=list(range(args.trial_seeds)),
                      workers=args.workers, kagg=args.kagg,
                      optuna_seed=args.optuna_seed)
        print(f"HPO best win rate vs ghost: {hpo['best_value']:.3f}", flush=True)
        print(f"counter agent: {hpo['counter_dir']}/main.py", flush=True)
        ship = subs / f"ladder_counter_{ep_id}"
        if ship.exists():
            shutil.rmtree(ship)
        shutil.copytree(hpo["counter_dir"], ship)
        print(f"Shipped counter-strategy: {ship}/main.py", flush=True)
    elif gm_wr < 0.5:
        print("LOSS locally but --no-hpo: skipping counter optimization.")
    else:
        print(f"Grandmaster holds ({gm_wr:.3f} >= 0.5): no HPO needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
