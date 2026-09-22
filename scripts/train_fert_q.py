#!/usr/bin/env python3
"""First-visit Monte Carlo on the daily fertilizer price floor.

Paired episodes: greedy $20 floor, then epsilon-greedy. The cash gap in
thousands of dollars updates every state-action the twin used. Milk labor
is not learned.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment

OUT = ROOT / "submissions" / "rl_fert_mill" / "q_values.json"


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
    if opponent != "pass":
        opponent.__globals__["_LOCK"]["animal"] = None
    env = Environment(configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    final = env.run_env(learner, opponent)
    return _cash(final, 0), list(g["TRACE"])


def main() -> None:
    learner = load_agent("rl_fert_mill", module_suffix="_fl")
    care = load_agent("care_mill", module_suffix="_fc")
    g = learner.__globals__
    g["Q"].clear()
    pairs = [
        (1, "pass"), (1, "care"),
        (3, "pass"), (3, "care"),
        (5, "pass"), (5, "care"),
        (7, "pass"), (7, "care"),
        (8, "pass"), (8, "care"),
    ]
    gaps = []
    for seed, who in pairs:
        opponent = "pass" if who == "pass" else care
        g["MODE"] = "greedy"
        g["EPSILON"] = 0.0
        base, _ = _play(learner, opponent, seed)
        g["MODE"] = "train"
        g["EPSILON"] = 0.4
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
        [state[0], state[1], action, round(value, 3)]
        for (state, action), value in sorted(g["Q"].items())
    ]
    OUT.write_text(json.dumps(rows))
    print(f"mean gap {sum(gaps) / len(gaps):+.2f}  wrote {OUT.name} ({len(rows)} entries)")
    for stage, band, action, value in rows:
        prior = 0 if band == 0 else 1
        if action != prior and value > 0.5:
            print(
                f"  prefer action {action} (floor {g['FLOORS'][action]}) over {prior} "
                f"in stage {stage} band {band} ({value:+.2f})"
            )


if __name__ == "__main__":
    main()
