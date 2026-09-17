"""Equilibrium Harvest: incumbent-style liquidity with opponent-aware crop choice."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, CROPS, Plants


def _score(obs: Dict[str, Any], name: str) -> float:
    crop = CROPS[name]
    day = int(obs.get("day", 0))
    if crop.time_to_first_yield > 30 - day:
        return -1e18
    price = float(obs.get("market", {}).get("prices", {}).get(name, crop.base_market_price))
    shops = obs.get("town", {}).get("unlocked_shops", [])
    demand = obs.get("market", {}).get("inventory", {}).get(name, 0)
    opponent = obs.get("farms", [])[1 - int(obs.get("player", 0))]
    supply = sum(
        1 for row in opponent.get("tiles", []) for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == name
    )
    scarcity = 1.0 + min(0.35, supply / 60.0)
    demand_bonus = 1.0 + 0.03 * len(shops)
    inventory_penalty = 1.0 / (1.0 + max(0, float(demand)) / 100.0)
    return price * crop.yield_per_tile_per_day * scarcity * demand_bonus * inventory_penalty


class EquilibriumHarvestController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop=Plants.MELON,
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_feed_animals=True,
            auto_care_animals=True,
            auto_collect_fertilizer=True,
            auto_dig_weeds=True,
            auto_sell=True,
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=2,
            min_sell_margin=0.7,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        day = int(obs.get("day", 0))
        if day < 26:
            self.target_crop = max(CROPS, key=lambda name: _score(obs, name))
        self.auto_expand_land = day < 26
        self.auto_hire_hands = day < 28
        return super().act(obs)


controller = EquilibriumHarvestController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
