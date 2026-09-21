"""Alpha Velocity P0: Market Velocity on library P0 safety (reserve, clamp, room guard)."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, CROPS, Plants


class AlphaVelocityP0Controller(ActionController):
    """Market Velocity crop selection; P0 market/unit safety lives in ActionController."""

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

    def _best_crop(self, obs: Dict[str, Any]) -> Any:
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
            repeat_bonus = 1.0 + (0.35 if crop.ongoing else 0.0)
            scores[name] = price * crop.yield_per_tile_per_day * demand_bonus * repeat_bonus - crop.seed_cost
        return max(scores, key=scores.get, default=Plants.MELON)

    def plan_market_actions(self, farm, private, market, current_day, planned_drop=None, town_shops=None):
        if current_day >= 26:
            self.auto_expand_land = False
            self.auto_hire_hands = False
        return super().plan_market_actions(
            farm,
            private,
            market,
            current_day,
            planned_drop=planned_drop,
            town_shops=town_shops,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if int(obs.get("day", 0)) < 26:
            self.target_crop = self._best_crop(obs)
        return super().act(obs)


controller = AlphaVelocityP0Controller()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
