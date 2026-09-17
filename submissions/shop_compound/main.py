"""Shop Compound: shop_opportunist economics with bulk seed replenishment."""

import os
import sys
from typing import Any, Dict, List

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import (
    ActionController,
    Actions,
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    Plants,
)


class ShopCompoundController(ActionController):
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

    def _adapt(self, obs: Dict[str, Any]) -> None:
        prices = obs.get("market", {}).get("prices", {})
        day = int(obs.get("day", 0))
        remaining = 30 - day
        scores = {}
        for name, crop in CROPS.items():
            if crop.time_to_first_yield <= remaining:
                scores[name] = float(prices.get(name, crop.base_market_price)) * crop.yield_per_tile_per_day
        if scores:
            self.target_crop = max(scores, key=scores.get)

    def plan_market_actions(self, farm, private, market, current_day):
        orders: List[List[Any]] = []
        money = float(farm.get("money", 0))
        prices = market.get("prices", {})
        shed = private.get("shed", {})
        unlocked = farm.get("unlocked_quadrants", ["NW"])

        if current_day < 26 and len(unlocked) < 4:
            cost = Actions.land_cost(unlocked)
            if cost is not None and money >= cost + 150:
                orders.append(Actions.buy_land())
                money -= cost

        if self.auto_sell:
            for item, count in shed.items():
                base = float(MARKET_PARAMS.get(item, {}).get("base", 0))
                if count > 0 and item not in ANIMALS and (not base or prices.get(item, 0) >= base * self.min_sell_margin):
                    orders.append(Actions.sell(item, count))

        target = CROPS.get(self.target_crop, CROPS["MELON"])
        seeds = int(private.get("seeds", {}).get(self.target_crop, 0))
        vacant = sum(tile is None for row in farm.get("tiles", []) for tile in row)
        buyable = int(max(0, money - 200) // target.seed_cost)
        quantity = min(max(0, vacant - seeds), buyable)
        if current_day < 24 and quantity:
            orders.append(Actions.buy_seed(self.target_crop, quantity))
            money -= quantity * target.seed_cost

        hires = int(farm.get("hires_today", 0))
        for offset in range(max(0, 2 - hires)):
            cost = Actions.hire_cost(hires + offset)
            if money < cost:
                break
            orders.append(Actions.hire())
            money -= cost
        return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._adapt(obs)
        return super().act(obs)


controller = ShopCompoundController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
