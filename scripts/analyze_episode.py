#!/usr/bin/env python3
"""Run one full episode and print JSON telemetry summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment
from kaggriculture.helpers.episode_metrics import EpisodeMetricsRecorder, MetricsWrapper
from kaggriculture.helpers.market_overlays import reset_market_overlay_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run episode and emit JSON metrics.")
    parser.add_argument(
        "--agent",
        "-a",
        default="wheat_loop",
        help="Submission folder name, path to main.py, or built-in opponent name",
    )
    parser.add_argument("--opponent", "-o", default="random")
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()

    reset_market_overlay_state()
    recorder = EpisodeMetricsRecorder()
    agent = load_agent(args.agent)
    wrapped = MetricsWrapper(agent, recorder)

    env = Environment(configuration={"episodeSteps": args.steps})
    final = env.run_env(wrapped, args.opponent)
    obs = env.get_current_state(final, agent1=True)
    if not isinstance(obs, dict):
        obs = dict(obs) if hasattr(obs, "items") else {"farms": [{}]}

    summary = recorder.finalize(obs, steps=args.steps)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
