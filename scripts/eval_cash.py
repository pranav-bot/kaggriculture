#!/usr/bin/env python3
"""Terminal cash, herd, and milk price for a pair of agents.

Powered by `kaggriculture-simulation` (Rust engine) with automatic fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sim_engine import FastSimulation, is_kagg_available
from agent_utils import load_agent
from kaggsim.serve import obs_for


def _herd(farm) -> dict:
    counts = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in counts:
                counts[str(tile["animal"])] += 1
    return counts


def play(
    agent_a: str,
    agent_b: str,
    seed: int,
    suffix: str,
    sim: FastSimulation | None = None,
) -> None:
    if sim is not None:
        b0, b1, st = sim.run_match(agent_a, agent_b, seed=seed)
        for player, name in ((0, agent_a), (1, agent_b)):
            obs = obs_for(st, player)
            farm = obs["farms"][player]
            shed = obs.get("private", {}).get("shed", {})
            herd = _herd(farm)
            prices = obs.get("market", {}).get("prices", {})
            shops = obs.get("town", {}).get("unlocked_shops", [])
            print(
                f"seed={seed:2d} seat={player} {name:18} cash={float(farm.get('money', 0)):10.0f} "
                f"cows={herd['COW']:2d} sheep={herd['SHEEP']:2d} geese={herd['GOOSE']:2d} "
                f"milk_px={prices.get('MILK')} wool_px={prices.get('WOOL')} "
                f"egg_px={prices.get('EGG')} fert_px={prices.get('FERTILIZER')} "
                f"shed_m={int(shed.get('MILK', 0))} shed_f={int(shed.get('FERTILIZER', 0))} "
                f"shed_w={int(shed.get('WOOL', 0))} shops={list(shops)}"
            )
    else:
        from kaggriculture.env import Environment

        a = load_agent(agent_a, module_suffix=suffix + "a") if agent_a not in ("pass", "random") else agent_a
        b = load_agent(agent_b, module_suffix=suffix + "b") if agent_b not in ("pass", "random") else agent_b
        env = Environment(configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        final = env.run_env(a, b)
        for player, name in ((0, agent_a), (1, agent_b)):
            raw = final[player].observation
            obs = raw if isinstance(raw, dict) else dict(raw)
            farm = obs["farms"][player]
            shed = obs.get("private", {}).get("shed", {})
            herd = _herd(farm)
            prices = obs.get("market", {}).get("prices", {})
            shops = obs.get("town", {}).get("unlocked_shops", [])
            print(
                f"seed={seed:2d} seat={player} {name:18} cash={float(farm.get('money', 0)):10.0f} "
                f"cows={herd['COW']:2d} sheep={herd['SHEEP']:2d} geese={herd['GOOSE']:2d} "
                f"milk_px={prices.get('MILK')} wool_px={prices.get('WOOL')} "
                f"egg_px={prices.get('EGG')} fert_px={prices.get('FERTILIZER')} "
                f"shed_m={int(shed.get('MILK', 0))} shed_f={int(shed.get('FERTILIZER', 0))} "
                f"shed_w={int(shed.get('WOOL', 0))} shops={list(shops)}"
            )


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--official"]
    use_official = "--official" in sys.argv
    use_fast = is_kagg_available() and not use_official

    if args and not args[0].isdigit():
        left, right, seeds_raw = args[0], args[1], args[2:]
    else:
        left, right, seeds_raw = "two_team_grandmaster", "pass", args
    seeds = [int(s) for s in seeds_raw if s.isdigit()] or [1, 3, 5, 7, 10, 15, 20]

    engine_name = "kaggriculture-simulation (Rust)" if use_fast else "kaggle_environments (Python)"
    print(f"Engine: {engine_name} | Running {len(seeds)} benchmark seeds...\n")

    def run_all(sim=None):
        for seed in seeds:
            if "--h2h" in sys.argv:
                play(left, right, seed, f"H{seed}a", sim=sim)
                play(right, left, seed, f"H{seed}b", sim=sim)
            else:
                play(left, "pass", seed, f"L{seed}", sim=sim)
                if right != "pass":
                    play(right, "pass", seed, f"R{seed}", sim=sim)
                    play(left, right, seed, f"H{seed}a", sim=sim)
                    play(right, left, seed, f"H{seed}b", sim=sim)

    if use_fast:
        with FastSimulation() as sim:
            run_all(sim)
    else:
        run_all(None)


if __name__ == "__main__":
    main()
