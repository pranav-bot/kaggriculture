import json
import glob
from pathlib import Path

def analyze_replay(path):
    with open(path) as f:
        d = json.load(f)
    info = d.get("info", {})
    agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]
    steps = d.get("steps", [])
    if not steps:
        return
    last = steps[-1]
    f0 = last[0]["observation"]["farms"][0]
    f1 = last[1]["observation"]["farms"][1]
    
    seed = info.get("seed")
    print(f"\n=== {Path(path).name} ({Path(path).parent.name}) ===")
    print(f"Seed: {seed} | P0: {agents[0]} (${f0['money']:,.0f}) vs P1: {agents[1]} (${f1['money']:,.0f})")
    for p_idx, f_data, name in [(0, f0, agents[0]), (1, f1, agents[1])]:
        animals = f_data.get("animals", [])
        plants = f_data.get("plants", [])
        quads = f_data.get("unlocked_quadrants", [])
        hands = len(f_data.get("hands", []))
        cow_cnt = sum(1 for a in animals if a.get("type") == "COW")
        sheep_cnt = sum(1 for a in animals if a.get("type") == "SHEEP")
        plant_types = {}
        for pl in plants:
            t = pl.get("crop_type") or pl.get("type")
            plant_types[t] = plant_types.get(t, 0) + 1
        shed = f_data.get("shed", {})
        print(f"  {name}: Cash=${f_data['money']:,.0f} | Quads={quads} | Hands={hands} | Cows={cow_cnt} | Sheep={sheep_cnt} | Plants={plant_types} | Shed={shed}")

def main():
    files = sorted(glob.glob("replays/**/*.json", recursive=True))
    for f in files:
        analyze_replay(f)

if __name__ == "__main__":
    main()
