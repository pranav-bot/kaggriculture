"""
Example 03 — Adaptive crop selection via subclassing ActionController.

Override act() to re-score crops each turn from live market prices and town
shop demand, then delegate field work to the built-in controller.
"""
import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, CROPS, Plants


class AdaptiveCropController(ActionController):
    def __init__(self) -> None:
        super().__init__(
            target_crop=Plants.WHEAT,
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_dig_weeds=True,
            auto_sell=True,
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=2,
            min_sell_margin=0.6,
        )

    def _pick_best_crop(self, obs: Dict[str, Any]) -> str:
        day = int(obs.get("day", 0))
        prices = obs.get("market", {}).get("prices", {})
        shops = obs.get("town", {}).get("unlocked_shops", [])
        demand = Actions.calculate_town_daily_consumption(shops)
        days_left = max(0, 30 - day)

        scores: Dict[str, float] = {}
        for name, crop in CROPS.items():
            if crop.time_to_first_yield > days_left:
                continue
            price = float(prices.get(name, crop.base_market_price))
            demand_bonus = 1.0 + 0.05 * float(demand.get(name, 0))
            scores[name] = price * crop.yield_per_tile_per_day * demand_bonus - crop.seed_cost

        return max(scores, key=scores.get, default=Plants.WHEAT)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.target_crop = self._pick_best_crop(obs)
        return super().act(obs)


controller = AdaptiveCropController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
