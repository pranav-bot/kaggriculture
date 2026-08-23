#!/usr/bin/env python3
"""
Full Season Match Simulator & Benchmark Tester for Kaggriculture.
Tests an agent over a full 30-day season (720 turns) and profiles performance.
"""
import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from kaggle_environments import make


ROOT_DIR = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
BUILD_DIR = ROOT_DIR / "build"


def resolve_agent_target(name_or_path: str) -> str:
    if Path(name_or_path).exists():
        return str(Path(name_or_path).resolve())
    if (SUBMISSIONS_DIR / name_or_path / "main.py").exists():
        return str(SUBMISSIONS_DIR / name_or_path / "main.py")
    if name_or_path in ("random", "pass", "starter"):
        return name_or_path
    if (BUILD_DIR / name_or_path).exists():
        return str(BUILD_DIR / name_or_path)
    return name_or_path


def run_benchmark(agent1_str: str, agent2_str: str, steps: int = 720, render_html: bool = False):
    print("=" * 70)
    print(f"🌾 Kaggriculture Season Simulation ({steps} turns / {steps // 24} days)")
    print(f"   Agent 1: {agent1_str}")
    print(f"   Agent 2: {agent2_str}")
    print("=" * 70)

    target1 = resolve_agent_target(agent1_str)
    target2 = resolve_agent_target(agent2_str)

    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=True)

    t0 = time.time()
    match_steps = env.run([target1, target2])
    total_time = time.time() - t0

    if not match_steps or len(match_steps) == 0:
        print("❌ Match failed to produce steps.")
        return

    final_obs0 = match_steps[-1][0].observation
    final_obs1 = match_steps[-1][1].observation

    p1_money = final_obs0.farms[0]["money"]
    p2_money = final_obs1.farms[1]["money"]

    p1_reward = match_steps[-1][0].reward
    p2_reward = match_steps[-1][1].reward

    print(f"\n🏁 Simulation Completed in {total_time:.2f}s ({total_time / len(match_steps) * 1000:.1f}ms / turn)")
    print("-" * 70)
    print(f"   Player 1 Final Cash: ${p1_money:,.2f}  | Reward: {p1_reward}")
    print(f"   Player 2 Final Cash: ${p2_money:,.2f}  | Reward: {p2_reward}")

    if p1_money > p2_money:
        print(f"\n🏆 WINNER: Player 1 ({agent1_str}) by +${p1_money - p2_money:,.2f}!")
    elif p2_money > p1_money:
        print(f"\n🏆 WINNER: Player 2 ({agent2_str}) by +${p2_money - p1_money:,.2f}!")
    else:
        print("\n🤝 Result: TIE!")

    if render_html:
        html_out = ROOT_DIR / "match_replay.html"
        with open(html_out, "w") as f:
            f.write(env.render(mode="html", width=1000, height=800))
        print(f"\n🎬 Visual Replay saved to: {html_out}")


def main():
    parser = argparse.ArgumentParser(description="Kaggriculture Benchmark & Match Simulator")
    parser.add_argument(
        "--agent1", "-a1",
        default="shop_opportunist",
        help="Agent 1 (wheat_loop, melon_rusher, shop_opportunist, or path to main.py / submission.tar.gz)",
    )
    parser.add_argument(
        "--agent2", "-a2",
        default="random",
        help="Agent 2 (opponent: random, pass, or another agent template)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=720,
        help="Total steps in simulation (default: 720 for full 30-day season)",
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Save match replay to match_replay.html",
    )

    args = parser.parse_args()
    run_benchmark(args.agent1, args.agent2, steps=args.steps, render_html=args.render)


if __name__ == "__main__":
    main()
