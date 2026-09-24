#!/usr/bin/env python3
import json

with open("replays/other_agents/rank1/112542379.json") as f:
    replay = json.load(f)

print("=" * 70)
print("RANK 1 FORENSIC INSPECTION: DAYS 0 to 14")
print("=" * 70)

for step in replay["steps"]:
    obs = step[0]["observation"]
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    if day > 14:
        break
    if hour == 1:  # check state at dawn after market orders
        farm = obs["farms"][0]
        hands = len(farm.get("hands", []))
        money = farm.get("money", 0)
        tiles = farm.get("tiles", [])
        quadrants = farm.get("unlocked_quadrants", [])
        
        animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
        crops = 0
        pastures = 0
        coops = 0
        for row in tiles:
            for tile in row:
                if isinstance(tile, dict):
                    if "animal" in tile:
                        animals[tile["animal"]] = animals.get(tile["animal"], 0) + 1
                    elif tile.get("kind") == "PASTURE":
                        pastures += 1
                    elif tile.get("kind") == "COOP":
                        coops += 1
                    elif tile.get("kind") == "PLANT":
                        crops += 1
                        
        action = step[0].get("action", {})
        market = action.get("market", [])
        
        total_anim = sum(animals.values())
        print(f"Day {day:2d} (H{hour:2d}): Cash=${money:7,.0f} | Hands={hands} | Quad={len(quadrants)} | "
              f"Anim={total_anim:2d} (C:{animals['COW']} S:{animals['SHEEP']} G:{animals['GOOSE']}) | "
              f"Pastures={pastures:2d} | Crops={crops:2d}")
        if market:
            m_summary = [f"{o[0]} {o[1] if len(o)>1 else ''} {o[2] if len(o)>2 else ''}".strip() for o in market]
            print(f"         Market orders: {m_summary}")
