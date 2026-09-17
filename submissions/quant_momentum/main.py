"""Quant Momentum: price momentum and demand-weighted crop selection."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, CROPS, Plants


def demand_pressure(obs: Dict[str, Any], item: str) -> float:
    shops = obs.get("town", {}).get("unlocked_shops", [])
    return float(Actions.calculate_town_daily_consumption(shops).get(item, 0))


def crop_score(obs: Dict[str, Any], name: str) -> float:
    crop = CROPS[name]
    price = float(obs.get("market", {}).get("prices", {}).get(name, crop.base_market_price))
    pressure = demand_pressure(obs, name)
    remaining = max(0, 30 - int(obs.get("day", 0)))
    if crop.time_to_first_yield > remaining:
        return -1e18
    # Favor fast, high-margin crops while giving demand a modest momentum bonus.
    return (price * crop.yield_per_tile_per_day - crop.seed_cost) * (1.0 + 0.12 * pressure)


class QuantMomentumController(ActionController):
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
            max_hires_per_day=3,
            min_sell_margin=0.45,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        day = int(obs.get("day", 0))
        if day < 26:
            scores = {name: crop_score(obs, name) for name in CROPS}
            self.target_crop = max(scores, key=scores.get)
        self.auto_expand_land = day < 26
        self.auto_hire_hands = day < 28
        return super().act(obs)


controller = QuantMomentumController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
