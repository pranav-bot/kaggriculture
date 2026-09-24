import sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from agent_utils import load_agent

def diagnose(agent_path, seed=1142076532):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    agent = load_agent(agent_path, module_suffix=f"diag_{Path(agent_path).stem}")
    
    # Run against pass bot
    def pass_bot(obs):
        return {"farmer": ["PASS"], "hands": [], "market": []}
        
    print(f"\n=======================================================")
    print(f"Diagnosing {agent_path} on seed {seed}")
    print(f"=======================================================")
    
    res = env.run([agent, pass_bot])
    
    for day in [1, 3, 5, 8, 10, 11, 13, 16, 20, 24, 28, 29]:
        step = min(day * 24 + 1, len(res) - 1)
        obs = res[step][0].observation
        f = obs["farms"][0]
        money = f["money"]
        quads = f["unlocked_quadrants"]
        hands = len(f.get("hands", []))
        
        crops = {}
        cows = 0
        sheep = 0
        weeds = 0
        unwatered = 0
        for row in f["tiles"]:
            for t in row:
                if isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        c = t.get("crop")
                        crops[c] = crops.get(c, 0) + 1
                        if int(t.get("consecutive_unwatered", 0)) > 0:
                            unwatered += 1
                    elif k == "PASTURE":
                        a = t.get("animal")
                        if a == "COW": cows += 1
                        elif a == "SHEEP": sheep += 1
                    elif k == "WEED":
                        weeds += 1
                        
        priv = obs.get("private", {})
        shed = priv.get("shed", {})
        print(f"Day {day:02d}: Cash=${money:7,.0f} | Hands={hands:2d} | Cows={cows:2d} | Sheep={sheep:2d} | Crops={crops} | Weeds={weeds} | Unwatered={unwatered} | Quads={len(quads)} | Shed={shed}")
    final_cash = res[-1][0].observation.farms[0]["money"]
    print(f"FINAL CASH: ${final_cash:,.0f}")
    return final_cash

if __name__ == "__main__":
    p1 = sys.argv[1] if len(sys.argv) > 1 else "scratch_straw_empire.py"
    diagnose(p1)
    if len(sys.argv) > 2:
        diagnose(sys.argv[2])
