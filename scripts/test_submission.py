#!/usr/bin/env python3
"""
Full Season Match Simulator & Benchmark Tester for Kaggriculture.
Tests an agent over a full 30-day season (720 turns) and profiles performance.
Powered by `kaggriculture-simulation` (Rust engine) with automatic fallback.
"""
from __future__ import annotations

import argparse
import sys
import time
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
BUILD_DIR = ROOT_DIR / "build"
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from sim_engine import FastSimulation, is_kagg_available


def assert_profile_limits(
    durations: Iterable[float],
    *,
    expected_turns: int = 720,
    per_turn_limit_s: float = 0.050,
    cumulative_limit_s: float = 5.0,
) -> dict[str, Any]:
    """Assert exact call count and strict per-turn/cumulative latency limits."""
    values = list(durations)
    assert len(values) == expected_turns, (
        f"expected exactly {expected_turns} agent(obs) calls, got {len(values)}"
    )
    worst, total = max(values, default=0.0), sum(values)
    assert all(value <= per_turn_limit_s for value in values), (
        f"agent(obs) exceeded {per_turn_limit_s * 1000:.0f}ms: "
        f"{worst * 1000:.3f}ms"
    )
    assert total <= cumulative_limit_s, (
        f"cumulative agent(obs) time {total:.6f}s exceeds {cumulative_limit_s}s"
    )
    ordered = sorted(values)
    return {
        "turns": len(values),
        "total_s": total,
        "max_turn_s": worst,
        "p50_s": statistics.median(values),
        "p95_s": ordered[max(0, int(len(values) * 0.95) - 1)],
    }


def profile_agent_calls(
    agent: Callable[[Any], Any],
    observations: Iterable[Any],
    *,
    expected_turns: int = 720,
    per_turn_limit_s: float = 0.050,
    cumulative_limit_s: float = 5.0,
) -> dict[str, Any]:
    """Profile a real injected stream of observations (never fabricated here)."""
    durations = []
    for observation in observations:
        started = time.perf_counter()
        agent(observation)
        durations.append(time.perf_counter() - started)
    return assert_profile_limits(
        durations,
        expected_turns=expected_turns,
        per_turn_limit_s=per_turn_limit_s,
        cumulative_limit_s=cumulative_limit_s,
    )


def run_ab_evaluation(
    candidate: str,
    baseline: str,
    *,
    seeds: Iterable[int] = range(100),
    tournament_runner: Optional[Callable[[dict], Mapping[str, Any]]] = None,
    fallback_runner: Optional[
        Callable[[str, str, list[int]], tuple[list[float], list[float]]]
    ] = None,
) -> Mapping[str, Any]:
    """Run a paired 100-seed A/B evaluation using tournament/statistical tools."""
    seed_list = list(seeds)
    if len(seed_list) != 100:
        raise ValueError(f"A/B evaluation requires exactly 100 seeds, got {len(seed_list)}")
    if tournament_runner is None:
        try:
            from kaggsim.tournament import run_tournament
            tournament_runner = run_tournament
        except ImportError:
            pass
    if tournament_runner is not None:
        summary = dict(tournament_runner({
            "name": "submission-ab",
            "candidate": {"name": "candidate", "type": "python", "path": candidate},
            "panel": [{"name": "baseline", "type": "python", "path": baseline}],
            "seats": "both",
            "worlds": {"strategy": "list", "pool": seed_list},
            "workers": 1,
        }))
        out_dir = summary.get("out_dir")
        if out_dir:
            try:
                from kaggsim.tournament import load_results
                from kaggsim.stats import paired_test, score

                rows = load_results(str(Path(out_dir) / "results.jsonl"))
                candidate_scores, baseline_scores = [], []
                for row in rows:
                    agents = row.get("agents", [])
                    scores = row.get("scores")
                    if len(agents) != 2 or not scores or len(scores) != 2:
                        continue
                    if set(agents) != {"candidate", "baseline"}:
                        continue
                    candidate_scores.append(scores[agents.index("candidate")])
                    baseline_scores.append(scores[agents.index("baseline")])
                if candidate_scores:
                    summary["games"] = len(candidate_scores)
                    summary["mcnemar"] = paired_test(
                        candidate_scores, baseline_scores
                    )
                    summary["candidate_scores"] = candidate_scores
                    summary["baseline_scores"] = baseline_scores
            except (ImportError, OSError, ValueError, KeyError):
                pass
        return summary
    if fallback_runner is None:
        raise RuntimeError(
            "kagg tournament is unavailable; inject fallback_runner for a real A/B test"
        )
    a, b = fallback_runner(candidate, baseline, seed_list)
    try:
        from kaggsim.stats import paired_test, score
        return paired_test(
            [score(x, y) for x, y in zip(a, b)],
            [score(y, x) for x, y in zip(a, b)],
        )
    except ImportError:
        return {"candidate": a, "baseline": b}


def resolve_agent_target(name_or_path: str) -> str:
    path = Path(name_or_path)
    if path.is_dir() and (path / "main.py").is_file():
        return str((path / "main.py").resolve())
    if path.is_file():
        return str(path.resolve())
    if path.exists():
        return str(path.resolve())
    if (SUBMISSIONS_DIR / name_or_path / "main.py").exists():
        return str(SUBMISSIONS_DIR / name_or_path / "main.py")
    if name_or_path in ("random", "pass", "starter"):
        return name_or_path
    if (BUILD_DIR / name_or_path).exists():
        return str(BUILD_DIR / name_or_path)
    return name_or_path


def run_benchmark(
    agent1_str: str,
    agent2_str: str,
    steps: int = 720,
    render_html: bool = False,
    use_official: bool = False,
    seed: int = 42,
):
    target1 = resolve_agent_target(agent1_str)
    target2 = resolve_agent_target(agent2_str)

    use_fast = is_kagg_available() and not use_official and not render_html
    engine_name = "kaggriculture-simulation (Rust)" if use_fast else "kaggle_environments (Python)"

    print("=" * 70)
    print(f"🌾 Kaggriculture Season Simulation ({steps} turns / {steps // 24} days)")
    print(f"   Engine:  {engine_name}")
    print(f"   Agent 1: {agent1_str}")
    print(f"   Agent 2: {agent2_str}")
    print("=" * 70)

    if use_fast and steps == 720:
        t0 = time.perf_counter()
        with FastSimulation() as sim:
            p1_money, p2_money, final_st = sim.run_match(target1, target2, seed=seed)
        total_time = time.perf_counter() - t0
        match_len = int(final_st.get("step", steps))

        print(f"\n🏁 Simulation Completed in {total_time:.2f}s ({total_time / match_len * 1000:.2f}ms / turn)")
        print("-" * 70)
        print(f"   Player 1 Final Cash: ${p1_money:,.2f}  | Reward: {p1_money}")
        print(f"   Player 2 Final Cash: ${p2_money:,.2f}  | Reward: {p2_money}")

        if p1_money > p2_money:
            print(f"\n🏆 WINNER: Player 1 ({agent1_str}) by +${p1_money - p2_money:,.2f}!")
        elif p2_money > p1_money:
            print(f"\n🏆 WINNER: Player 2 ({agent2_str}) by +${p2_money - p1_money:,.2f}!")
        else:
            print("\n🤝 Result: TIE!")
        return

    # Official runner fallback
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=True)

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
        default="two_team_grandmaster",
        help="Agent 1 (e.g. two_team_grandmaster, sovereign_apex, care_mill, or path to main.py / submission.tar.gz)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for the episode",
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Save match replay to match_replay.html (forces official Python engine)",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Force official kaggle_environments engine",
    )

    args = parser.parse_args()
    run_benchmark(
        args.agent1,
        args.agent2,
        steps=args.steps,
        render_html=args.render,
        use_official=args.official,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
