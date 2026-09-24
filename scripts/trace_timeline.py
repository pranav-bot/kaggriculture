import json
from pathlib import Path

def trace_agent_timeline(path):
    with open(path) as f:
        d = json.load(f)
    info = d.get("info", {})
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]
    steps = d.get("steps", [])
    
    print(f"\n=======================================================")
    print(f"FILE: {Path(path).name} | Seed: {info.get('seed')}")
    print(f"P0: {agents[0]} vs P1: {agents[1]}")
    print(f"=======================================================")

    days_to_check = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29]
    # Check at Hour 2 (after morning hiring and actions)
    step_indices = [min(day * 24 + 2, len(steps) - 1) for day in days_to_check]
    
    for s_idx in step_indices:
        step_data = steps[s_idx]
        day = s_idx // 24
        hour = s_idx % 24
        print(f"\n--- Day {day:02d} Hour {hour:02d} (Step {s_idx}) ---")
        for p_idx in [0, 1]:
            obs = step_data[p_idx]["observation"]
            f = obs["farms"][p_idx]
            money = f.get("money", 0)
            hands = len(f.get("hands", []))
            quads = f.get("unlocked_quadrants", [])
            tiles = f.get("tiles", [])
            
            cows = 0
            sheep = 0
            crops = {}
            empty = 0
            weeds = 0
            locked = 0
            for row in tiles:
                for t in row:
                    if t is None:
                        empty += 1
                        continue
                    if isinstance(t, str):
                        if t == "LOCKED": locked += 1
                        continue
                    kind = t.get("kind")
                    if kind == "PASTURE":
                        a_type = t.get("animal")
                        if a_type == "COW": cows += 1
                        elif a_type == "SHEEP": sheep += 1
                    elif kind == "PLANT":
                        c_type = t.get("crop")
                        crops[c_type] = crops.get(c_type, 0) + 1
                    elif kind == "WEED":
                        weeds += 1
                    elif kind == "EMPTY":
                        empty += 1
                        
            priv = obs.get("private")
            shed_str = ""
            if priv and "shed" in priv:
                shed_str = f"| Shed={priv['shed']}"
            print(f"  P{p_idx} ({agents[p_idx]}): Cash=${money:7,.0f} | Hands={hands:2d} | Cows={cows:2d} | Sheep={sheep:2d} | Crops={crops} | Empty={empty:2d} | Weeds={weeds:2d} | Quads={len(quads)} {shed_str}")

def main():
    targets = [
        "replays/other_agents/rank1/112542379.json", # Vadim vs Boey
        "replays/other_agents/rank3/112557915.json", # nah id win ($135k) vs 吃白饭的大肥鱼 ($114k)
        "replays/my_agents/agent_final: Sovereign Apex k+/112618133.json", # Clement Ling ($133k)
        "replays/my_agents/agent_final: Sovereign Apex k+/112619304.json", # BenPalmer59 ($103k)
    ]
    for t in targets:
        trace_agent_timeline(t)

if __name__ == "__main__":
    main()
