#!/usr/bin/env python3
"""Run an agent against real replay opponents using recorded actions."""

import json
import sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agent_utils import load_agent


def evaluate_replay(agent_path: str, replay_path: str):
    with open(replay_path) as f:
        d = json.load(f)

    info = d.get("info", {})
    seed = info.get("seed")
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]
    steps = d.get("steps", [])

    # The replay recorded: Player 0 vs Player 1
    # We test our agent in both positions:
    # 1. Our agent as Player 0 vs recorded Player 1
    p1_acts = [s[1].get("action") for s in steps]
    def bot_p1(obs):
        step = obs.step
        if step + 1 < len(p1_acts) and p1_acts[step + 1] is not None:
            return p1_acts[step + 1]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    agent = load_agent(agent_path, module_suffix=f"eval_{Path(replay_path).stem}")
    res = env.run([agent, bot_p1])

    p0_money = res[-1][0].observation.farms[0]["money"]
    p1_money = res[-1][1].observation.farms[1]["money"]
    diff = p0_money - p1_money
    win = "WIN 🏆" if diff > 0 else "LOSS ❌"
    print(f"Replay {Path(replay_path).name:15} | Seed {seed} | Opponent: {agents[1]:15}")
    print(f"   Our Agent: ${p0_money:9,.0f} | Opponent: ${p1_money:9,.0f} | Diff: ${diff:9,.0f} | {win}")
    return p0_money, p1_money, diff > 0


def main():
    agent_path = sys.argv[1] if len(sys.argv) > 1 else "agent_final.py"
    target_replays = sys.argv[2:] if len(sys.argv) > 2 else []

    if not target_replays:
        replays = [
            "replays/my_agents/agent_final: Sovereign Apex k+/112619304.json",
            "replays/my_agents/agent_final: Sovereign Apex k+/112618133.json",
        ]
    else:
        replays = target_replays

    wins = 0
    total = 0
    for r in replays:
        if Path(r).exists():
            _, _, w = evaluate_replay(agent_path, r)
            if w: wins += 1
            total += 1
    print("=" * 60)
    print(f"Summary: {wins}/{total} Replays Won ({wins/total*100:.1f}%)" if total else "No replays run")


if __name__ == "__main__":
    main()
