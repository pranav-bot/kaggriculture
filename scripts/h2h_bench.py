#!/usr/bin/env python3
"""Head-to-head benchmark: run agent_a vs agent_b on multiple seeds.

Powered by `kaggriculture-simulation` (Rust engine) with automatic fallback.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sim_engine import FastSimulation, is_kagg_available
from agent_utils import load_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head agent benchmark.")
    parser.add_argument("agent_a", nargs="?", default="two_team_grandmaster", help="First agent (e.g. two_team_grandmaster)")
    parser.add_argument("agent_b", nargs="?", default="care_mill", help="Second agent (e.g. sovereign_apex, care_mill)")
    parser.add_argument("--seeds", "-n", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--official", action="store_true", help="Force official Python engine")
    args = parser.parse_args()

    use_fast = is_kagg_available() and not args.official
    engine_name = "kaggriculture-simulation (Rust)" if use_fast else "kaggle_environments (Python)"
    print(f"Engine: {engine_name} | {args.agent_a} vs {args.agent_b} ({args.seeds} seeds, {args.seeds * 2} matches)\n")

    a_wins, b_wins, draws = 0, 0, 0
    a_cash_total, b_cash_total = 0.0, 0.0

    print(f"{'Seed':>5}  {'Seat':>5}  {args.agent_a:>20}  {args.agent_b:>20}  {'Winner':>20}")
    print("-" * 80)

    t_start = time.perf_counter()

    def run_matches(sim=None):
        nonlocal a_wins, b_wins, draws, a_cash_total, b_cash_total
        for i in range(args.seeds):
            seed = args.start_seed + i
            for seat in [0, 1]:
                if sim is not None:
                    if seat == 0:
                        cash_a, cash_b, _ = sim.run_match(args.agent_a, args.agent_b, seed=seed)
                    else:
                        cash_b, cash_a, _ = sim.run_match(args.agent_b, args.agent_a, seed=seed)
                else:
                    from kaggriculture.env import Environment
                    if seat == 0:
                        a = load_agent(args.agent_a, module_suffix=f"_h2h_a{seed}")
                        b = load_agent(args.agent_b, module_suffix=f"_h2h_b{seed}")
                    else:
                        a = load_agent(args.agent_b, module_suffix=f"_h2h_b{seed}s")
                        b = load_agent(args.agent_a, module_suffix=f"_h2h_a{seed}s")

                    env = Environment(configuration={"episodeSteps": args.steps, "seed": seed}, debug=False)
                    final = env.run_env(a, b)
                    if seat == 0:
                        cash_a = float(final[0].observation.farms[0]["money"])
                        cash_b = float(final[1].observation.farms[1]["money"])
                    else:
                        cash_a = float(final[1].observation.farms[1]["money"])
                        cash_b = float(final[0].observation.farms[0]["money"])

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

    if use_fast:
        with FastSimulation() as sim:
            run_matches(sim)
    else:
        run_matches(None)

    elapsed = time.perf_counter() - t_start
    total = a_wins + b_wins + draws
    print("-" * 80)
    print(f"Results: {args.agent_a} wins {a_wins}/{total}, "
          f"{args.agent_b} wins {b_wins}/{total}, draws {draws}/{total}")
    print(f"Mean cash: {args.agent_a}={a_cash_total/total:,.0f}, "
          f"{args.agent_b}={b_cash_total/total:,.0f}, "
          f"Δ={a_cash_total/total - b_cash_total/total:+,.0f} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
