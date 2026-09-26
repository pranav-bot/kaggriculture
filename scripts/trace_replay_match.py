#!/usr/bin/env python3
"""Forensic trace of an agent match against a replay opponent.

Powered by `kaggriculture-simulation` (Rust engine) with automatic fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sim_engine import FastSimulation, is_kagg_available
from agent_utils import load_agent
from kaggsim.serve import obs_for


def trace_match(agent_path: str, replay_path: str, use_official: bool = False):
    with open(replay_path) as f:
        d = json.load(f)
    info = d.get("info", {})
    seed = info.get("seed")
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]

    print(f"\n=================================================================")
    print(f"TRACING MATCH: {Path(replay_path).name} | Opponent: {agents[1]}")
    print(f"Engine: {'kaggle_environments (Python)' if use_official or not is_kagg_available() else 'kaggriculture-simulation (Rust)'}")
    print(f"=================================================================")

    if is_kagg_available() and not use_official:
        days_to_check = set(range(0, 30, 2))
        snapshots = {}

        def on_step(step, js, obs_our):
            day = js["day"]
            hour = js["hour"]
            if hour == 1 and day in days_to_check and day not in snapshots:
                obs0 = obs_for(js, 0)
                obs1 = obs_for(js, 1)
                snapshots[day] = (obs0, obs1)

        with FastSimulation() as sim:
            our_m, opp_m, won, final_st = sim.run_replay(
                agent_path, d, our_player=0, on_step=on_step
            )

        for day in sorted(snapshots.keys()):
            obs0, obs1 = snapshots[day]
            f0 = obs0["farms"][0]
            f1 = obs1["farms"][1]
            mkt = obs0.get("market", {}).get("prices", {})

            cows0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            sheep0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
            straw0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")
            wheat0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "WHEAT")
            melon0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "MELON")
            shed0 = obs0.get("private", {}).get("shed", {})

            cows1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            sheep1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
            straw1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")

            print(f"Day {day:02d}:")
            print(f"  OURS: Cash=${f0['money']:6,.0f} | Hands={len(f0.get('hands', [])):2d} | Cows={cows0:2d}, Sheep={sheep0:2d} | Straw={straw0:2d}, Wheat={wheat0:2d}, Melon={melon0:2d} | ShedMilk={shed0.get('MILK',0)}, ShedWool={shed0.get('WOOL',0)}, ShedStraw={shed0.get('STRAWBERRY',0)}, ShedWheat={shed0.get('WHEAT',0)}")
            print(f"  OPP : Cash=${f1['money']:6,.0f} | Hands={len(f1.get('hands', [])):2d} | Cows={cows1:2d}, Sheep={sheep1:2d} | Straw={straw1:2d}")
            print(f"  MKT : Milk=${mkt.get('MILK', 0):.1f} | Wool=${mkt.get('WOOL', 0):.1f} | Straw=${mkt.get('STRAWBERRY', 0):.1f} | Wheat=${mkt.get('WHEAT', 0):.1f}")

        print(f"\nFinal: Ours=${our_m:,.0f} vs Opponent=${opp_m:,.0f} (Diff: ${our_m - opp_m:,.0f})")

    else:
        from kaggle_environments import make
        steps = d.get("steps", [])
        opp_acts = [s[1].get("action") for s in steps]

        def bot_p1(obs):
            step = obs.step
            if step + 1 < len(opp_acts) and opp_acts[step + 1] is not None:
                return opp_acts[step + 1]
            return {"farmer": ["PASS"], "hands": [], "market": []}

        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        agent = load_agent(agent_path, module_suffix=f"trace_{Path(replay_path).stem}")
        res = env.run([agent, bot_p1])

        for day in range(0, 30, 2):
            s_idx = min(day * 24 + 1, len(res) - 1)
            step_entry = res[s_idx]
            obs0 = step_entry[0].observation
            obs1 = step_entry[1].observation
            f0 = obs0["farms"][0]
            f1 = obs1["farms"][1]
            mkt = obs0.get("market", {}).get("prices", {})

            cows0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            sheep0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
            straw0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")
            wheat0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "WHEAT")
            melon0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "MELON")
            shed0 = obs0.get("private", {}).get("shed", {})

            cows1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            sheep1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
            straw1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")

            print(f"Day {day:02d}:")
            print(f"  OURS: Cash=${f0['money']:6,.0f} | Hands={len(f0.get('hands', [])):2d} | Cows={cows0:2d}, Sheep={sheep0:2d} | Straw={straw0:2d}, Wheat={wheat0:2d}, Melon={melon0:2d} | ShedMilk={shed0.get('MILK',0)}, ShedWool={shed0.get('WOOL',0)}, ShedStraw={shed0.get('STRAWBERRY',0)}, ShedWheat={shed0.get('WHEAT',0)}")
            print(f"  OPP : Cash=${f1['money']:6,.0f} | Hands={len(f1.get('hands', [])):2d} | Cows={cows1:2d}, Sheep={sheep1:2d} | Straw={straw1:2d}")
            print(f"  MKT : Milk=${mkt.get('MILK', 0):.1f} | Wool=${mkt.get('WOOL', 0):.1f} | Straw=${mkt.get('STRAWBERRY', 0):.1f} | Wheat=${mkt.get('WHEAT', 0):.1f}")

        p0_final = res[-1][0].observation.farms[0]["money"]
        p1_final = res[-1][1].observation.farms[1]["money"]
        print(f"\nFinal: Ours=${p0_final:,.0f} vs Opponent=${p1_final:,.0f} (Diff: ${p0_final - p1_final:,.0f})")


if __name__ == "__main__":
    rep = sys.argv[1] if len(sys.argv) > 1 else "replays/my_agents/agent_final: Sovereign Apex k+/112619304.json"
    cand = sys.argv[2] if len(sys.argv) > 2 else "two_team_grandmaster"
    official = "--official" in sys.argv
    trace_match(cand, rep, use_official=official)
