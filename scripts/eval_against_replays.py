#!/usr/bin/env python3
"""Run an agent against real replay opponents using recorded actions."""

import glob
import json
import sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agent_utils import load_agent


def evaluate_replay(agent_path: str, replay_path: str, our_player: int = 0):
    with open(replay_path) as f:
        d = json.load(f)

    info = d.get("info", {})
    seed = info.get("seed")
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]
    steps = d.get("steps", [])

    opp_player = 1 - our_player
    opp_acts = [s[opp_player].get("action") for s in steps]

    def bot_opp(obs):
        step = obs.step
        if step + 1 < len(opp_acts) and opp_acts[step + 1] is not None:
            return opp_acts[step + 1]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    agent = load_agent(agent_path, module_suffix=f"eval_{Path(replay_path).stem}_p{our_player}")
    
    players = [agent, bot_opp] if our_player == 0 else [bot_opp, agent]
    res = env.run(players)

    our_money = res[-1][our_player].observation.farms[our_player]["money"]
    opp_money = res[-1][opp_player].observation.farms[opp_player]["money"]
    diff = our_money - opp_money
    win = "WIN 🏆" if diff > 0 else "LOSS ❌"
    opp_name = agents[opp_player]
    print(f"Replay {Path(replay_path).name:15} | Seed {seed:10} | Opponent (P{opp_player}): {opp_name:18}")
    print(f"   Our Agent: ${our_money:9,.0f} | Opponent: ${opp_money:9,.0f} | Diff: ${diff:9,.0f} | {win}")
    return our_money, opp_money, diff > 0


def main():
    agent_path = sys.argv[1] if len(sys.argv) > 1 else "scratch_straw_empire.py"
    target_args = sys.argv[2:]

    if target_args:
        replays = []
        for a in target_args:
            if "*" in a:
                replays.extend(glob.glob(a, recursive=True))
            else:
                replays.append(a)
    else:
        # Default gauntlet across categories
        replays = sorted(glob.glob("replays/**/*.json", recursive=True))

    wins = 0
    total = 0
    our_totals = []
    opp_totals = []
    for r in replays:
        if Path(r).exists():
            try:
                # In replays/my_agents/, our agent was P0 originally, so test our_player=0
                # In general, if replay has Pranav Advani as P1, test our_player=1
                with open(r) as f:
                    d = json.load(f)
                ag_names = [a.get("Name", "") for a in d.get("info", {}).get("Agents", [])]
                our_p = 1 if (len(ag_names) > 1 and "Pranav" in ag_names[1]) else 0
                
                our_m, opp_m, w = evaluate_replay(agent_path, r, our_player=our_p)
                if w:
                    wins += 1
                total += 1
                our_totals.append(our_m)
                opp_totals.append(opp_m)
            except Exception as e:
                print(f"Error evaluating {r}: {e}")

    print("=" * 70)
    avg_our = sum(our_totals) / len(our_totals) if our_totals else 0
    avg_opp = sum(opp_totals) / len(opp_totals) if opp_totals else 0
    win_pct = wins / total * 100 if total else 0
    print(f"Gauntlet Results: {wins}/{total} Replays Won ({win_pct:.1f}%)")
    print(f"Average Cash: Our Agent = ${avg_our:,.0f} | Opponents = ${avg_opp:,.0f} | Margin = ${avg_our - avg_opp:,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
