import json

with open("replays/other_agents/rank1/112542379.json") as f:
    d = json.load(f)

print("=== BOEY TIMELINE ===")
for day in [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 20]:
    step = day * 24
    s = d["steps"][step]
    f1 = s[1]["observation"]["farms"][1]
    priv1 = s[1]["observation"]["private"]
    c1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
    s1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
    g1 = sum(1 for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "GOOSE")
    crops1 = {}
    for row in f1["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                cr = t.get("crop")
                crops1[cr] = crops1.get(cr, 0) + 1
    
    m = f1["money"]
    hands = len(f1.get("hands", []))
    quad = f1.get("unlocked_quadrants")
    shed = priv1.get("shed", {})
    print(f"Day {day:02d}: Money=${m:6.0f} | Hands={hands} | Quad={quad} | Cows={c1}, Sheep={s1}, Geese={g1} | Crops={crops1} | Shed={shed}")
