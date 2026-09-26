#!/usr/bin/env python3
"""Run an agent against real replay opponents using recorded actions.

Powered by `kaggriculture-simulation` (Rust engine) with automatic
fallback to `kaggle_environments`.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sim_engine import FastSimulation, is_kagg_available, run_official_match
from agent_utils import load_agent


def evaluate_replay(
    agent_path: str,
    replay_path: str,
    our_player: int = 0,
    sim: FastSimulation | None = None,
):
    with open(replay_path) as f:
        d = json.load(f)

    info = d.get("info", {})
    seed = info.get("seed")
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]
    steps = d.get("steps", [])
    opp_player = 1 - our_player

    if sim is not None:
        our_money, opp_money, win, _ = sim.run_replay(
            agent_path, d, our_player=our_player
        )
    else:
        # Fallback to official python runner
        opp_acts = [s[opp_player].get("action") for s in steps]

        def bot_opp(obs):
            step = obs.step
            if step + 1 < len(opp_acts) and opp_acts[step + 1] is not None:
                return opp_acts[step + 1]
            return {"farmer": ["PASS"], "hands": [], "market": []}

        from kaggle_environments import make
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        agent = load_agent(agent_path, module_suffix=f"eval_{Path(replay_path).stem}_p{our_player}")
        players = [agent, bot_opp] if our_player == 0 else [bot_opp, agent]
        res = env.run(players)
        our_money = float(res[-1][our_player].observation.farms[our_player]["money"])
        opp_money = float(res[-1][opp_player].observation.farms[opp_player]["money"])
        win = our_money > opp_money

    diff = our_money - opp_money
    win_str = "WIN 🏆" if diff > 0 else "LOSS ❌"
    opp_name = agents[opp_player]
    print(f"Replay {Path(replay_path).name:15} | Seed {seed:10} | Opponent (P{opp_player}): {opp_name:18}")
    print(f"   Our Agent: ${our_money:9,.0f} | Opponent: ${opp_money:9,.0f} | Diff: ${diff:9,.0f} | {win_str}")
    return our_money, opp_money, win


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent against replay opponents.")
    parser.add_argument("agent", nargs="?", default="two_team_grandmaster", help="Agent template or file to test (e.g. two_team_grandmaster, sovereign_apex, apex_engine)")
    parser.add_argument("replays", nargs="*", help="Specific replay files or globs")
    parser.add_argument("--official", action="store_true", help="Force official kaggle_environments engine")
    parser.add_argument("--quick", action="store_true", help="Test top 4 landmark replays only")
    args = parser.parse_args()

    agent_path = args.agent

    if args.replays:
        replays = []
        for a in args.replays:
            if "*" in a:
                replays.extend(glob.glob(a, recursive=True))
            else:
                replays.append(a)
    elif args.quick:
        replays = [
            "replays/other_agents/rank1/112542379.json",  # Boey
            "replays/other_agents/rank1/112544834.json",  # M & M & P & Q
            "replays/other_agents/rank2/112555622.json",  # DECEM
            "replays/my_agents/agent_final: Sovereign Apex k+/112619304.json",  # BenPalmer59
        ]
    else:
        replays = sorted(glob.glob("replays/**/*.json", recursive=True))

    use_fast = is_kagg_available() and not args.official
    engine_name = "kaggriculture-simulation (Rust)" if use_fast else "kaggle_environments (Python)"
    print(f"Engine: {engine_name} | Target Agent: {agent_path}")
    print(f"Loaded {len(replays)} replay matches to evaluate.\n")

    t_start = time.perf_counter()
    wins = 0
    total = 0
    our_totals = []
    opp_totals = []

    def run_eval(sim=None):
        nonlocal wins, total
        for r in replays:
            if Path(r).exists():
                try:
                    with open(r) as f:
                        d = json.load(f)
                    ag_names = [a.get("Name", "") for a in d.get("info", {}).get("Agents", [])]
                    our_p = 1 if (len(ag_names) > 1 and "Pranav" in ag_names[1]) else 0

                    our_m, opp_m, w = evaluate_replay(agent_path, r, our_player=our_p, sim=sim)
                    if w:
                        wins += 1
                    total += 1
                    our_totals.append(our_m)
                    opp_totals.append(opp_m)
                except Exception as e:
                    print(f"Error evaluating {r}: {e}")

    if use_fast:
        with FastSimulation() as sim:
            run_eval(sim)
    else:
        run_eval(None)

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    avg_our = sum(our_totals) / len(our_totals) if our_totals else 0
    avg_opp = sum(opp_totals) / len(opp_totals) if opp_totals else 0
    win_pct = wins / total * 100 if total else 0
    print(f"Gauntlet Results: {wins}/{total} Replays Won ({win_pct:.1f}%) | Time: {elapsed:.2f}s ({elapsed/max(1,total):.2f}s/replay)")
    print(f"Average Cash: Our Agent = ${avg_our:,.0f} | Opponents = ${avg_opp:,.0f} | Margin = ${avg_our - avg_opp:,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
