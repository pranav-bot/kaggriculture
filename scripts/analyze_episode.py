#!/usr/bin/env python3
"""Run one full episode and print JSON telemetry summary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggriculture.actions.controller import ActionController
from kaggriculture.env import Environment
from kaggriculture.env.items import Plants
from kaggriculture.helpers.episode_metrics import EpisodeMetricsRecorder, MetricsWrapper

SUBMISSIONS = ROOT / "submissions"


def load_agent(name: str):
    if name in ("random", "pass", "starter"):
        return name
    path = Path(name)
    if path.is_dir() and (path / "main.py").is_file():
        main_py = path / "main.py"
    elif (SUBMISSIONS / name / "main.py").is_file():
        main_py = SUBMISSIONS / name / "main.py"
    elif path.is_file():
        main_py = path
    else:
        return ActionController(target_crop=Plants.WHEAT)

    spec = importlib.util.spec_from_file_location(f"submission_{main_py.parent.name}", main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load agent from {main_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(main_py.parent))
    spec.loader.exec_module(module)
    if hasattr(module, "controller"):
        return module.controller
    if hasattr(module, "agent"):
        return module.agent
    raise RuntimeError(f"No agent or controller export in {main_py}")


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
