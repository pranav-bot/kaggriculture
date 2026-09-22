#!/usr/bin/env python3
"""Dawn milk price and cash for one match. Used to size the sell trigger."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment


def wrap(fn, rows, seat):
    def inner(obs):
        if int(obs.get("hour", 0)) == 0:
            price = obs.get("market", {}).get("prices", {}).get("MILK")
            money = obs["farms"][int(obs.get("player", seat))].get("money")
            rows.append((int(obs.get("day", 0)), seat, price, round(float(money))))
        return fn(obs)
    return inner


def main() -> None:
    left_name, right_name, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = []
    left = load_agent(left_name, module_suffix=f"p{seed}a")
    right = load_agent(right_name, module_suffix=f"p{seed}b") if right_name != "pass" else "pass"
    env = Environment(configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run_env(wrap(left, rows, 0), wrap(right, rows, 1) if right_name != "pass" else "pass")
    by_day = {}
    for day, seat, price, money in rows:
        by_day.setdefault(day, {})[seat] = (price, money)
    print(f"{left_name} vs {right_name} seed {seed}")
    for day in sorted(by_day):
        a = by_day[day].get(0, ("-", "-"))
        b = by_day[day].get(1, ("-", "-"))
        print(f"d{day:02d} px={a[0]} cash {left_name}={a[1]} {right_name}={b[1]}")


if __name__ == "__main__":
    main()
