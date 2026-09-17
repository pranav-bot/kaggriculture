"""Carrot Compound: fast recurring harvests with conservative liquidity."""

import os
import sys
from typing import Any, List

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, ANIMALS, CROPS, MARKET_PARAMS, Plants


class CarrotCompoundController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop=Plants.CARROT,
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
            max_hires_per_day=4,
            min_sell_margin=0.15,
        )

    def plan_market_actions(self, farm, private, market, current_day) -> List[List[Any]]:
        orders: List[List[Any]] = []
        money = float(farm.get("money", 0))
        shed = private.get("shed", {})
        prices = market.get("prices", {})
        if current_day < 26 and len(farm.get("unlocked_quadrants", [])) < 4:
            cost = Actions.land_cost(farm.get("unlocked_quadrants", ["NW"]))
            if cost is not None and money >= cost + 300:
                orders.append(Actions.buy_land())
                money -= cost
        for item, count in shed.items():
            if count > 0 and item not in ANIMALS:
                base = MARKET_PARAMS.get(item, {}).get("base", 0)
                if not base or prices.get(item, 0) >= base * self.min_sell_margin:
                    orders.append(Actions.sell(item, count))
        seeds = int(private.get("seeds", {}).get("CARROT", 0))
        vacant = sum(tile is None for row in farm.get("tiles", []) for tile in row)
        quantity = min(max(0, vacant - seeds), int(max(0, money - 250) // CROPS["CARROT"].seed_cost))
        if current_day < 28 and quantity:
            orders.append(Actions.buy_seed("CARROT", quantity))
            money -= quantity * CROPS["CARROT"].seed_cost
        hires = int(farm.get("hires_today", 0))
        for offset in range(max(0, 4 - hires)):
            cost = Actions.hire_cost(hires + offset)
            if money < cost:
                break
            orders.append(Actions.hire())
            money -= cost
        return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]


controller = CarrotCompoundController()


def agent(obs):
    return controller.act(obs)
