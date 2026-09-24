#!/usr/bin/env python3
import json

for rank_name, path in [
    ("RANK 2", "replays/other_agents/rank2/112526712.json"),
    ("RANK 3", "replays/other_agents/rank3/112538757.json"),
]:
    with open(path) as f:
        replay = json.load(f)

    print("=" * 70)
    print(f"{rank_name} FORENSIC INSPECTION: DAYS 0 to 14")
    print("=" * 70)

    for step in replay["steps"]:
        obs = step[0]["observation"]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        if day > 14:
            break
        if hour == 1:
            farm = obs["farms"][0]
            hands = len(farm.get("hands", []))
            money = farm.get("money", 0)
            tiles = farm.get("tiles", [])
            quadrants = farm.get("unlocked_quadrants", [])
            
            animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
            crops = 0
            for row in tiles:
                for tile in row:
                    if isinstance(tile, dict):
                        if "animal" in tile:
                            animals[tile["animal"]] = animals.get(tile["animal"], 0) + 1
                        elif tile.get("kind") == "PLANT":
                            crops += 1
                            
            action = step[0].get("action", {})
            market = action.get("market", [])
            
            total_anim = sum(animals.values())
            m_types = set(o[0] for o in market if o)
            print(f"Day {day:2d} (H{hour:2d}): Cash=${money:7,.0f} | Hands={hands} | Quad={len(quadrants)} | "
                  f"Anim={total_anim:2d} (C:{animals['COW']} S:{animals['SHEEP']} G:{animals['GOOSE']}) | Crops={crops:2d} | "
                  f"Orders: {list(m_types)}")
