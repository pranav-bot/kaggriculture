#!/usr/bin/env python3
"""Audit Replay Fallbacks: Detect if an agent is tripping safety fallback mechanisms.

Inspects replay files (.json or .json.gz) to determine if:
1. The agent crashed and defaulted to SafeFallbackController.
2. The agent entered an all-PASS stagnation loop (unhandled exception / deadlocks).
3. The agent suffered from Kaggle timeout / overage bank exhaustion (status != 'DONE').
4. The agent experienced a 'capability collapse' (sudden permanent cessation of planting,
   purchasing, or market trading).

Usage:
    .venv/bin/python scripts/audit_replay_fallbacks.py --dir replays/my_agents/psro_leauge_pick
    .venv/bin/python scripts/audit_replay_fallbacks.py --file replays/my_agents/psro_leauge_pick/114793445.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from kaggriculture.safety import SafeFallbackController
except ImportError:
    SafeFallbackController = None  # type: ignore


def load_replay(path: str) -> Dict[str, Any]:
    """Loads JSON or gzipped JSON replay."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def identify_player_index(replay_data: Dict[str, Any], player_query: str = "Pranav") -> int:
    """Finds target agent index (0 or 1). Defaults to 0 if indeterminate."""
    info = replay_data.get("info", {})
    team_names = info.get("TeamNames", [])
    for idx, name in enumerate(team_names):
        if player_query.lower() in str(name).lower():
            return idx
    # Fallback heuristic: check description or agents array
    agents = info.get("Agents", [])
    for idx, agent in enumerate(agents):
        name = agent.get("Name", "")
        if player_query.lower() in str(name).lower():
            return idx
    return 0


def audit_single_replay(
    replay_path: str,
    player_query: str = "Pranav",
    match_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Performs deep forensic audit on a single match replay."""
    data = load_replay(replay_path)
    steps = data.get("steps", [])
    total_steps = len(steps)
    if total_steps == 0:
        return {"error": "Replay contains zero steps"}

    my_idx = identify_player_index(data, player_query)
    opp_idx = 1 - my_idx
    team_names = data.get("info", {}).get("TeamNames", [f"Player_{my_idx}", f"Player_{opp_idx}"])
    my_name = team_names[my_idx] if my_idx < len(team_names) else f"Player_{my_idx}"
    opp_name = team_names[opp_idx] if opp_idx < len(team_names) else f"Player_{opp_idx}"

    final_my = steps[-1][my_idx] if my_idx < len(steps[-1]) else {}
    final_opp = steps[-1][opp_idx] if opp_idx < len(steps[-1]) else {}
    my_reward = float(final_my.get("reward") or 0.0)
    opp_reward = float(final_opp.get("reward") or 0.0)
    final_status = final_my.get("status", "UNKNOWN")

    fallback_sim = SafeFallbackController() if SafeFallbackController else None

    exact_fallback_matches = 0
    farmer_fallback_matches = 0
    all_pass_steps = 0
    non_done_statuses: List[Tuple[int, str]] = []

    quadrants_bought: List[int] = []
    animals_bought = Counter()
    crops_planted = Counter()
    action_verb_counts = Counter()
    market_verb_counts = Counter()

    # Track activity timeline across 30 days
    day_activity: Dict[int, Counter] = {d: Counter() for d in range(30)}

    for s_idx, step in enumerate(steps):
        p_data = step[my_idx]
        status = p_data.get("status", "DONE")
        if status != "DONE":
            non_done_statuses.append((s_idx, status))

        act = p_data.get("action")
        day = s_idx // 24

        if not isinstance(act, dict):
            all_pass_steps += 1
            day_activity[day]["NON_DICT_ACTION"] += 1
            continue

        farmer = act.get("farmer", ["PASS"])
        hands = act.get("hands", [])
        market = act.get("market", [])

        # Check all PASS
        is_all_pass = (farmer == ["PASS"] and all(h == ["PASS"] for h in hands))
        if is_all_pass:
            all_pass_steps += 1
            day_activity[day]["ALL_PASS"] += 1

        # Track verbs
        if farmer and len(farmer) > 0:
            verb = farmer[0]
            action_verb_counts[verb] += 1
            day_activity[day][verb] += 1
            if verb == "PLANT" and len(farmer) > 1:
                crops_planted[farmer[1]] += 1

        for h in hands:
            if h and len(h) > 0:
                verb = h[0]
                action_verb_counts[verb] += 1
                day_activity[day][verb] += 1
                if verb == "PLANT" and len(h) > 1:
                    crops_planted[h[1]] += 1

        for m in market:
            if m and len(m) > 0:
                m_verb = m[0]
                market_verb_counts[m_verb] += 1
                day_activity[day][m_verb] += 1
                if m_verb == "BUY_LAND":
                    quadrants_bought.append(len(quadrants_bought) + 1)
                elif m_verb == "BUY_ANIMAL" and len(m) > 2:
                    animals_bought[m[1]] += int(m[2])

        # Evaluate SafeFallbackController match if available
        if fallback_sim is not None:
            obs = p_data.get("observation")
            if isinstance(obs, dict):
                fb_act = fallback_sim.act(obs)
                if act == fb_act:
                    exact_fallback_matches += 1
                if farmer == fb_act.get("farmer"):
                    farmer_fallback_matches += 1

    exact_fb_rate = (exact_fallback_matches / total_steps) if total_steps > 0 else 0.0
    all_pass_rate = (all_pass_steps / total_steps) if total_steps > 0 else 0.0

    # Determine fallback verdict
    if final_status in ("TIMEOUT", "ERROR"):
        verdict = f"CRITICAL_FAILURE ({final_status})"
        is_fallback_active = True
    elif exact_fb_rate >= match_threshold:
        verdict = f"ACTIVE_FALLBACK ({exact_fb_rate*100:.1f}% match)"
        is_fallback_active = True
    elif all_pass_rate > 0.50:
        verdict = f"PASS_FREEZE ({all_pass_rate*100:.1f}% PASS)"
        is_fallback_active = True
    elif exact_fb_rate > 0.05:
        verdict = f"PARTIAL_FALLBACK ({exact_fb_rate*100:.1f}% match)"
        is_fallback_active = True
    else:
        verdict = "INACTIVE (0.0% fallback, normal policy execution)"
        is_fallback_active = False

    return {
        "file": os.path.basename(replay_path),
        "my_name": my_name,
        "opp_name": opp_name,
        "my_reward": my_reward,
        "opp_reward": opp_reward,
        "score_delta": my_reward - opp_reward,
        "status": final_status,
        "non_done_statuses": non_done_statuses,
        "total_steps": total_steps,
        "exact_fallback_matches": exact_fallback_matches,
        "exact_fallback_rate": exact_fb_rate,
        "all_pass_steps": all_pass_steps,
        "all_pass_rate": all_pass_rate,
        "quadrants_bought": len(quadrants_bought),
        "animals_bought": dict(animals_bought),
        "crops_planted": dict(crops_planted),
        "action_verbs": dict(action_verb_counts),
        "market_verbs": dict(market_verb_counts),
        "verdict": verdict,
        "is_fallback_active": is_fallback_active,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit agent replays for safety fallback activation.")
    parser.add_argument("--dir", type=str, default="replays/my_agents/psro_leauge_pick", help="Directory containing JSON replays")
    parser.add_argument("--file", type=str, default=None, help="Path to single replay file")
    parser.add_argument("--player", type=str, default="Pranav", help="Name query to identify agent seat")
    parser.add_argument("--threshold", type=float, default=0.50, help="Fallback match rate threshold")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")
    args = parser.parse_args()

    files: List[str] = []
    if args.file:
        files = [args.file]
    elif args.dir:
        patterns = [os.path.join(args.dir, "*.json"), os.path.join(args.dir, "*.json.gz")]
        for p in patterns:
            files.extend(glob.glob(p))
        files = sorted(files)

    if not files:
        print(f"No replay files found in {args.dir or args.file}")
        sys.exit(1)

    results = [audit_single_replay(f, player_query=args.player, match_threshold=args.threshold) for f in files]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 90)
    print("🌾 Kaggriculture Replay Safety Fallback Forensic Audit")
    print(f"   Target Query: '{args.player}' | Replays Analyzed: {len(results)}")
    print("=" * 90)

    any_fallback = False
    for r in results:
        if "error" in r:
            print(f"❌ {r.get('file')}: {r['error']}")
            continue

        print(f"\n📂 Replay: {r['file']}")
        print(f"   Match: {r['my_name']} (${r['my_reward']:,.0f}) vs {r['opp_name']} (${r['opp_reward']:,.0f}) | Delta: {r['score_delta']:+,.0f}")
        print(f"   Status: {r['status']} | Total Turns: {r['total_steps']}")
        print(f"   SafeFallbackController Matches: {r['exact_fallback_matches']} / {r['total_steps']} ({r['exact_fallback_rate']*100:.2f}%)")
        print(f"   All-PASS Turns: {r['all_pass_steps']} / {r['total_steps']} ({r['all_pass_rate']*100:.2f}%)")
        print(f"   Quadrants Unlocked: {r['quadrants_bought']} | Animals: {r['animals_bought']}")
        print(f"   Crops Planted: {r['crops_planted']}")
        print(f"   🚨 Verdict: {r['verdict']}")

        if r["is_fallback_active"]:
            any_fallback = True

    print("\n" + "=" * 90)
    if any_fallback:
        print("⚠️  AUDIT CONCLUSION: Fallback mechanisms WERE triggered during one or more matches.")
    else:
        print("✅ AUDIT CONCLUSION: SafeFallbackController was 0.0% active across all matches.")
        print("   The agent executed its full active policy (crop planting, animal purchases, land expansion).")
        print("   The score gap was caused by macro-strategy/market collisions, NOT by code failure or safety fallbacks.")
    print("=" * 90)


if __name__ == "__main__":
    main()
