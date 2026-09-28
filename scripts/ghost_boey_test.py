#!/usr/bin/env python3
"""Ghost Opponent exploitability test: Boey (Rank 1, replay 112542379).

Replays Boey's HISTORICAL public states turn-by-turn through our Bayesian
Change Point Detector (fertilizer hoarding) and PSRO MetaController WITHOUT
running self-play, and asserts:

  1. On the exact turn Boey's NE expansion becomes observable, the
     MetaController registers OPPONENT_EXPANSION_DETECTED.
  2. Our active MacroIntent shifts from the linear-growth default (Policy_A)
     to the Strawberry Contingency (Policy_B) within 24 in-game hours
     (24 turns) of the flag, and holds it.

Ground truth is DERIVED from the replay log (first turn the opponent's public
unlocked-quadrant set grows), never hardcoded. Log truth: NE executes at step
146 (Day 6, hour 2) once Boey's cash first covers the $1,000 price; SW follows
at step 218 (Day 9, hour 2). (Forensic docs previously claimed "Day 5" for NE;
corrected to Day 6 after this harness surfaced the mismatch. The $170,961
figure seen in gauntlet tables is Boey-tape's realized cash against our
candidate agent, not this file's logged $130,115.)

Any failure raises GhostFailure pointing at the exact turn the opponent
model failed, with quad/latent/policy histories attached.

Usage:
  .venv/bin/python scripts/ghost_boey_test.py
  .venv/bin/python scripts/ghost_boey_test.py --replay replays/other_agents/rank1/112542379.json --seat 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.quant_full_product import HoardingMonitor
from kaggriculture.meta import (
    EXPANSION_RUSH,
    OPPONENT_EXPANSION_DETECTED,
    POLICY_A,
    POLICY_B,
    MetaController,
)

DEFAULT_REPLAY = ROOT / "replays" / "other_agents" / "rank1" / "112542379.json"
SWITCH_DEADLINE_TURNS = 24  # 24 in-game hours == 24 turns


class GhostFailure(AssertionError):
    """Diagnostic exploitability failure with the exact failing turn."""


def load_replay(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def find_seats(ep: Dict[str, Any], opponent_name: str = "Boey") -> Tuple[int, int]:
    """(our_seat, opponent_seat) from replay team names; default 0 / 1."""
    teams = ((ep.get("info") or {}).get("TeamNames") or []) if isinstance(ep.get("info"), dict) else []
    if opponent_name in teams:
        opp = int(teams.index(opponent_name))
        return 1 - opp, opp
    return 0, 1


def ground_truth_expansions(ep: Dict[str, Any], seat: int, opp: int) -> List[Dict[str, Any]]:
    """Quad-set growth events from OUR seat view (mirrors live perception)."""
    events = []
    prev: Optional[Tuple[str, ...]] = None
    for t, entry in enumerate(ep.get("steps") or []):
        try:
            farms = entry[seat]["observation"]["farms"]
            quads = tuple(farms[opp].get("unlocked_quadrants") or [])
        except (IndexError, KeyError, TypeError):
            continue
        if prev is not None and set(quads) - set(prev):
            obs = entry[seat]["observation"]
            events.append({
                "turn": t,
                "day": obs.get("day"),
                "hour": obs.get("hour"),
                "prev": list(prev),
                "new": sorted(set(quads) - set(prev)),
            })
        prev = quads
    return events


def requested_sells(action: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in action.get("market") or []:
        if isinstance(m, list) and len(m) >= 3 and m[0] == "SELL":
            out[m[1]] = out.get(m[1], 0.0) + float(m[2])
    return out


def run_ghost(replay: str | Path, our_seat: int = 0,
              opponent_name: str = "Boey") -> Dict[str, Any]:
    """Feed historical public states through BOCD + MetaController."""
    ep = load_replay(replay)
    seat, opp = our_seat, 1 - our_seat
    teams = ((ep.get("info") or {}).get("TeamNames") or []) if isinstance(ep.get("info"), dict) else []
    if teams and opponent_name in teams:
        opp = int(teams.index(opponent_name))
        seat = 1 - opp
    truth = ground_truth_expansions(ep, seat, opp)

    monitor = HoardingMonitor(window_turns=24, min_windows=3)
    controller = MetaController()
    trace: List[Dict[str, Any]] = []
    for t, entry in enumerate(ep.get("steps") or []):
        try:
            node = entry[seat]
            obs = node["observation"]
        except (IndexError, KeyError, TypeError):
            continue
        bocd = monitor.update(obs, seat=seat,
                              our_executed_sells=requested_sells(node.get("action") or {}))
        decision = controller.update(obs, seat, bocd)
        trace.append({"turn": t, **{k: decision[k] for k in
                      ("event", "latent_strategy", "gated_latent", "active_policy",
                       "dwell_remaining", "queued_digs")},
                      "quads": decision["footprint"]["quadrants"],
                      "hoarding": decision["hoarding"]})
    final_money = None
    try:
        farms = ep["steps"][-1][seat]["observation"]["farms"]
        final_money = float(farms[opp].get("money", 0.0) or 0.0)
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    return {"replay": str(replay), "seat": seat, "opp": opp,
            "opponent": teams[opp] if opp < len(teams) else f"seat{opp}",
            "final_opp_money": final_money,
            "expansions": truth, "trace": trace}


def _slice(trace: List[Dict[str, Any]], lo: int, hi: int) -> str:
    lines = []
    for d in trace:
        if lo <= d["turn"] <= hi:
            lines.append(
                f"t={d['turn']} ev={d['event']} lat={d['latent_strategy']} "
                f"pol={d['active_policy']} quads={d['quads']}")
    return "\n".join(lines)


def assert_ghost_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """Assertion checks 1 + 2. Raises GhostFailure naming the exact turn."""
    trace = res["trace"]
    by_turn = {d["turn"]: d for d in trace}
    if not res["expansions"]:
        raise GhostFailure("no ground-truth expansion found in replay; "
                           "opponent model has nothing to detect")
    first = res["expansions"][0]
    g = first["turn"]

    # --- Check 1: flag on the exact observable turn -------------------------
    d = by_turn.get(g)
    if d is None or d["event"] != OPPONENT_EXPANSION_DETECTED \
            or d["latent_strategy"] != EXPANSION_RUSH:
        raise GhostFailure(
            f"opponent model MISSED the expansion: ground-truth NE unlock at "
            f"turn {g} (day {first['day']}h{first['hour']}, "
            f"{first['prev']} -> +{first['new']}), but controller reported "
            f"event={d['event'] if d else 'NO-TRACE'} "
            f"latent={d['latent_strategy'] if d else 'NO-TRACE'}.\n"
            f"Trace t={max(0, g-3)}..{g+3}:\n{_slice(trace, max(0, g-3), g+3)}")

    # --- Check 2: counter-policy within 24h, then held ------------------------
    switch = next((t for t in range(g, g + SWITCH_DEADLINE_TURNS + 1)
                   if by_turn.get(t, {}).get("active_policy") == POLICY_B), None)
    if switch is None or switch - g > SWITCH_DEADLINE_TURNS:
        raise GhostFailure(
            f"policy switch TOO LATE: flag at turn {g}, first {POLICY_B} at "
            f"{switch} (> {SWITCH_DEADLINE_TURNS}h deadline) -- shared book "
            f"exposed to the expander's dump.\n"
            f"Trace t={g}..{g + SWITCH_DEADLINE_TURNS + 2}:\n"
            f"{_slice(trace, g, g + SWITCH_DEADLINE_TURNS + 2)}")
    late = [t for t in range(g, g + SWITCH_DEADLINE_TURNS + 1)
            if by_turn.get(t, {}).get("active_policy") != POLICY_B]
    if late:
        raise GhostFailure(
            f"counter-policy NOT HELD: {POLICY_B} engaged at turn {switch} "
            f"but flapped away at turns {late} (deadline window "
            f"{g}..{g + SWITCH_DEADLINE_TURNS}). Flap slice:\n"
            f"{_slice(trace, max(0, min(late) - 2), min(max(late) + 2, g + SWITCH_DEADLINE_TURNS))}")

    before = by_turn.get(max(0, g - 1), {}).get("active_policy")
    return {"flag_turn": g, "flag_day": first["day"], "flag_hour": first["hour"],
            "new_quadrants": first["new"], "switch_turn": switch,
            "lag_turns": switch - g, "policy_before": before,
            "final_opp_money": res["final_opp_money"],
            "n_expansions": len(res["expansions"])}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ghost-opponent exploitability test.")
    ap.add_argument("--replay", default=str(DEFAULT_REPLAY))
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--opponent", default="Boey")
    args = ap.parse_args(argv)

    res = run_ghost(args.replay, args.seat, args.opponent)
    print(f"Ghost: {res['opponent']} ({args.replay}), "
          f"opp terminal cash ${res['final_opp_money']:,.0f}")
    for e in res["expansions"]:
        print(f"  ground-truth expansion: turn {e['turn']} "
              f"(day {e['day']}h{e['hour']}) +{e['new']}")
    try:
        verdict = assert_ghost_result(res)
    except GhostFailure as exc:
        print(f"\nGHOST TEST FAILED\n{exc}")
        return 1
    print(f"\nGHOST TEST PASSED: {OPPONENT_EXPANSION_DETECTED} at turn "
          f"{verdict['flag_turn']} (day {verdict['flag_day']}h{verdict['flag_hour']}), "
          f"{verdict['policy_before']} -> {POLICY_B} at turn {verdict['switch_turn']} "
          f"(lag {verdict['lag_turns']}h <= {SWITCH_DEADLINE_TURNS}h, held).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
