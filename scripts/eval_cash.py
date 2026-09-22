#!/usr/bin/env python3
"""Terminal cash, herd, and milk price for a pair of agents.

Vs `pass` the shop stream depends only on the agent under test, so two
agents are not on the same town. Head-to-head seats share a seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment


def _obs(final, player: int):
    raw = final[player].observation
    return raw if isinstance(raw, dict) else dict(raw)


def _herd(farm) -> dict:
    counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in counts:
                counts[str(tile["animal"])] += 1
    return counts


def play(agent_a: str, agent_b: str, seed: int, suffix: str) -> None:
    a = load_agent(agent_a, module_suffix=suffix + "a") if agent_a not in ("pass", "random") else agent_a
    b = load_agent(agent_b, module_suffix=suffix + "b") if agent_b not in ("pass", "random") else agent_b
    env = Environment(configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    final = env.run_env(a, b)
    for player, name in ((0, agent_a), (1, agent_b)):
        obs = _obs(final, player)
        farm = obs["farms"][player]
        shed = obs.get("private", {}).get("shed", {})
        herd = _herd(farm)
        prices = obs.get("market", {}).get("prices", {})
        shops = obs.get("town", {}).get("unlocked_shops", [])
        print(
            f"seed={seed} seat={player} {name:16} cash={float(farm.get('money', 0)):10.0f} "
            f"cows={herd['COW']:2d} sheep={herd['SHEEP']:2d} geese={herd['GOOSE']:2d} "
            f"milk_px={prices.get('MILK')} wool_px={prices.get('WOOL')} "
            f"egg_px={prices.get('EGG')} fert_px={prices.get('FERTILIZER')} "
            f"shed_m={int(shed.get('MILK', 0))} shed_f={int(shed.get('FERTILIZER', 0))} "
            f"shed_w={int(shed.get('WOOL', 0))} shops={list(shops)}"
        )


def main() -> None:
    args = sys.argv[1:]
    if args and not args[0].isdigit():
        left, right, seeds_raw = args[0], args[1], args[2:]
    else:
        left, right, seeds_raw = "quant_mill", "care_mill", args
    seeds = [int(s) for s in seeds_raw if s.isdigit()] or [3, 5, 7]
    for seed in seeds:
        if "--h2h" in sys.argv:
            play(left, right, seed, f"H{seed}a")
            play(right, left, seed, f"H{seed}b")
        else:
            play(left, "pass", seed, f"L{seed}")
            play(right, "pass", seed, f"R{seed}")
            play(left, right, seed, f"H{seed}a")
            play(right, left, seed, f"H{seed}b")


if __name__ == "__main__":
    main()
