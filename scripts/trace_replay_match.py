import json
import sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agent_utils import load_agent

def trace_match(agent_path, replay_path):
    with open(replay_path) as f:
        d = json.load(f)
    info = d.get("info", {})
    seed = info.get("seed")
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]
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

    print(f"\n=================================================================")
    print(f"TRACING MATCH: {Path(replay_path).name} | Opponent: {agents[1]}")
    print(f"=================================================================")

    for day in range(0, 30, 2):
        s_idx = min(day * 24 + 1, len(res) - 1)
        step_entry = res[s_idx]
        obs0 = step_entry[0].observation
        obs1 = step_entry[1].observation
        f0 = obs0["farms"][0]
        f1 = obs1["farms"][1]
        mkt = obs0.get("market", {}).get("prices", {})
        
        # Count animals & crops for P0
        cows0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        sheep0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        straw0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")
        wheat0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "WHEAT")
        melon0 = sum(1 for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "MELON")
        shed0 = obs0.get("private", {}).get("shed", {})
        
        # Count animals & crops for P1
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
    trace_match("scratch_apex_engine.py", rep)
