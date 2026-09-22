#!/usr/bin/env python3
"""First-visit Monte Carlo on the daily milk cap.

Each episode is paired. The greedy band plays the seed first. An
epsilon-greedy twin then plays the same opponent. The update is the cash
gap, in thousands of dollars, credited to every state-action the twin
visited. Labor is not learned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment

OUT = ROOT / "submissions" / "rl_sell_mill" / "q_values.json"


def _cash(final, player: int) -> float:
    obs = final[player].observation
    if not isinstance(obs, dict):
        obs = dict(obs)
    return float(obs["farms"][player].get("money", 0))


def _play(learner, opponent, seed: int) -> tuple[float, list]:
    g = learner.__globals__
    g["TRACE"] = []
    g["_LOCK"]["animal"] = None
    g["_DAY"]["day"] = -1
    g["_TREND"]["dawn"] = None
    g["_TREND"]["down"] = False
    if opponent != "pass":
        opponent.__globals__["_LOCK"]["animal"] = None
    env = Environment(configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    final = env.run_env(learner, opponent)
    return _cash(final, 0), list(g["TRACE"])


def main() -> None:
    learner = load_agent("rl_sell_mill", module_suffix="_learn")
    care = load_agent("care_mill", module_suffix="_care")
    g = learner.__globals__
    g["Q"].clear()
    pairs = [(1, "pass"), (1, "care"), (3, "pass"), (3, "care"), (5, "pass"), (5, "care"), (8, "pass"), (8, "care")]
    gaps = []
    for seed, who in pairs:
        opponent = "pass" if who == "pass" else care
        g["MODE"] = "greedy"
        g["EPSILON"] = 0.0
        base, _ = _play(learner, opponent, seed)
        g["MODE"] = "train"
        g["EPSILON"] = 0.45
        cash, trace = _play(learner, opponent, seed)
        adv = (cash - base) / 1000.0
        gaps.append(adv)
        seen = set()
        for state, action in trace:
            key = (state, action)
            if key in seen:
                continue
            seen.add(key)
            old = g["Q"].get(key, 0.0)
            g["Q"][key] = old + 0.5 * (adv - old)
        print(
            f"seed={seed} vs {who:4} greedy={base:8.0f} explore={cash:8.0f} "
            f"gap={adv:+6.2f} decisions={len(trace)}"
        )
    rows = [
        [state[0], state[1], state[2], action, round(value, 3)]
        for (state, action), value in sorted(g["Q"].items())
    ]
    OUT.write_text(json.dumps(rows))
    print(f"mean gap {sum(gaps) / len(gaps):+.2f}  wrote {OUT.name} ({len(rows)} entries)")
    for stage, band, trend, action, value in rows:
        if band >= 2 and trend:
            prior = 2
        elif band >= 2:
            prior = 1
        else:
            prior = 0
        if action != prior and value > 0.5:
            print(
                f"  prefer action {action} over {prior} "
                f"in stage {stage} band {band} trend {trend} ({value:+.2f})"
            )


if __name__ == "__main__":
    main()
