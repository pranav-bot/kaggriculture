"""Game Theory Supply: counter opponent supply concentration without sabotage."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, CROPS, Plants


def opponent_supply(obs: Dict[str, Any], item: str) -> int:
    player = int(obs.get("player", 0))
    opponent = 1 - player
    farm = obs.get("farms", [{}])[opponent]
    return sum(
        1
        for row in farm.get("tiles", [])
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == item
    )


def strategic_crop(obs: Dict[str, Any]) -> str:
    prices = obs.get("market", {}).get("prices", {})
    day = int(obs.get("day", 0))
    remaining = 30 - day
    scores = {}
    for name, crop in CROPS.items():
        if crop.time_to_first_yield > remaining:
            continue
        price = float(prices.get(name, crop.base_market_price))
        supply = opponent_supply(obs, name)
        scarcity = 1.0 + min(0.5, supply / 40.0)
        speed = max(0.25, min(1.0, remaining / max(1, crop.time_to_first_yield)))
        scores[name] = price * crop.yield_per_tile_per_day * scarcity * speed - crop.seed_cost
    return max(scores, key=scores.get, default=Plants.MELON)


class GameTheorySupplyController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop=Plants.MELON,
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_feed_animals=False,
            auto_care_animals=False,
            auto_collect_fertilizer=True,
            auto_dig_weeds=True,
            auto_sell=True,
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=2,
            min_sell_margin=0.55,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        day = int(obs.get("day", 0))
        if day < 26:
            self.target_crop = strategic_crop(obs)
        self.auto_expand_land = day < 26
        self.auto_hire_hands = day < 28
        return super().act(obs)


controller = GameTheorySupplyController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
