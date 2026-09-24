#!/usr/bin/env python3
import json
from collections import Counter

for rank_name, path in [
    ("RANK 1", "replays/other_agents/rank1/112542379.json"),
    ("RANK 2", "replays/other_agents/rank2/112526712.json"),
    ("RANK 3", "replays/other_agents/rank3/112538757.json"),
]:
    with open(path) as f:
        replay = json.load(f)

    print(f"\n{'=' * 60}\n{rank_name} CROP BREAKDOWN ACROSS DAYS\n{'=' * 60}")
    for target_day in [2, 5, 8, 12]:
        for step in replay["steps"]:
            obs = step[0]["observation"]
            if obs.get("day", 0) == target_day and obs.get("hour", 0) == 1:
                farm = obs["farms"][0]
                crops = []
                for row in farm.get("tiles", []):
                    for tile in row:
                        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                            crops.append(tile.get("crop"))
                counts = Counter(crops)
                print(f"Day {target_day:2d}: Total Crops={len(crops):2d} | Details: {dict(counts)}")
                break
