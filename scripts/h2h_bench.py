#!/usr/bin/env python3
"""Head-to-head benchmark: run agent_a vs agent_b on multiple seeds.

Unlike compare_agents.py which runs each agent solo vs a passive opponent,
this runs them AGAINST each other on the same episode — the real test.

Usage:
  python scripts/h2h_bench.py surge_mill rl_fert_mill --seeds 5
  python scripts/h2h_bench.py surge_mill care_mill --seeds 10 --start-seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment


def extract_cash(final, player: int) -> float:
    obs = final[player].observation
    if not isinstance(obs, dict):
        obs = dict(obs)
    return float(obs["farms"][player].get("money", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head agent benchmark.")
    parser.add_argument("agent_a", help="First agent (submission name)")
    parser.add_argument("agent_b", help="Second agent (submission name)")
    parser.add_argument("--seeds", "-n", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()

    a_wins, b_wins, draws = 0, 0, 0
    a_cash_total, b_cash_total = 0.0, 0.0

    print(f"{'Seed':>5}  {'Seat':>5}  {args.agent_a:>20}  {args.agent_b:>20}  {'Winner':>20}")
    print("-" * 80)

    for i in range(args.seeds):
        seed = args.start_seed + i
        for seat in [0, 1]:
            if seat == 0:
                a = load_agent(args.agent_a, module_suffix=f"_h2h_a{seed}")
                b = load_agent(args.agent_b, module_suffix=f"_h2h_b{seed}")
            else:
                a = load_agent(args.agent_b, module_suffix=f"_h2h_b{seed}s")
                b = load_agent(args.agent_a, module_suffix=f"_h2h_a{seed}s")

            env = Environment(configuration={"episodeSteps": args.steps, "seed": seed}, debug=False)
            final = env.run_env(a, b)

            if seat == 0:
                cash_a = extract_cash(final, 0)
                cash_b = extract_cash(final, 1)
            else:
                cash_a = extract_cash(final, 1)
                cash_b = extract_cash(final, 0)

            a_cash_total += cash_a
            b_cash_total += cash_b

            if cash_a > cash_b:
                a_wins += 1
                winner = args.agent_a
            elif cash_b > cash_a:
                b_wins += 1
                winner = args.agent_b
            else:
                draws += 1
                winner = "DRAW"

            print(f"{seed:5d}  {'P1' if seat==0 else 'P2':>5}  {cash_a:20,.0f}  {cash_b:20,.0f}  {winner:>20}")

    total = a_wins + b_wins + draws
    print("-" * 80)
    print(f"Results: {args.agent_a} wins {a_wins}/{total}, "
          f"{args.agent_b} wins {b_wins}/{total}, draws {draws}/{total}")
    print(f"Mean cash: {args.agent_a}={a_cash_total/total:,.0f}, "
          f"{args.agent_b}={b_cash_total/total:,.0f}, "
          f"Δ={a_cash_total/total - b_cash_total/total:+,.0f}")


if __name__ == "__main__":
    main()
