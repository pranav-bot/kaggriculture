"""Shop Velocity: demand-aware production with maximum daily labor throughput."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import (
    ActionController,
    Actions,
    Animals,
    CROPS,
    Plants,
)


class ShopVelocityController(ActionController):
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
            max_hires_per_day=8,
            min_sell_margin=0.25,
        )

    def _adapt(self, obs: Dict[str, Any]) -> None:
        day = int(obs.get("day", 0))
        prices = obs.get("market", {}).get("prices", {})
        shops = obs.get("town", {}).get("unlocked_shops", [])
        demand = Actions.calculate_town_daily_consumption(shops)
        remaining = max(0, 30 - day)
        scores = {}
        for name, crop in CROPS.items():
            if crop.time_to_first_yield > remaining:
                continue
            price = float(prices.get(name, crop.base_market_price))
            demand_bonus = 1.0 + 0.08 * float(demand.get(name, 0))
            ongoing_bonus = 1.25 if crop.ongoing else 1.0
            scores[name] = price * crop.yield_per_tile_per_day * demand_bonus * ongoing_bonus
        if scores:
            self.target_crop = max(scores, key=scores.get)

        if "YARN_STORE" in shops and float(prices.get("WOOL", 0)) >= 180:
            self.target_animal = Animals.SHEEP
        elif "PIZZA_SHOP" in shops or "ICE_CREAM_SHOP" in shops:
            self.target_animal = Animals.COW
        else:
            self.target_animal = None

        self.auto_expand_land = day < 26
        self.auto_hire_hands = day < 28

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._adapt(obs)
        return super().act(obs)


controller = ShopVelocityController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
