"""Microbatch Opportunist: incumbent crop selection with controlled sell execution."""

import os
import sys
from typing import Any, Dict, List

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, ANIMALS, CROPS, MARKET_PARAMS, Plants


class MicrobatchOpportunistController(ActionController):
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
            auto_sell=False,
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=2,
            min_sell_margin=0.5,
        )

    def _adapt(self, obs: Dict[str, Any]) -> None:
        prices = obs.get("market", {}).get("prices", {})
        shops = obs.get("town", {}).get("unlocked_shops", [])
        demand = Actions.calculate_town_daily_consumption(shops)
        day = int(obs.get("day", 0))
        scores = {}
        for name, crop in CROPS.items():
            if crop.time_to_first_yield <= 30 - day:
                price = float(prices.get(name, crop.base_market_price))
                margin = price * crop.yield_per_tile_per_day - crop.seed_cost / max(1, crop.time_to_first_yield)
                scores[name] = margin * (1.0 + 0.05 * float(demand.get(name, 0)))
        if scores:
            self.target_crop = max(scores, key=scores.get)

    def plan_market_actions(self, farm, private, market, current_day):
        orders: List[List[Any]] = []
        money = float(farm.get("money", 0))
        prices = market.get("prices", {})
        shed = private.get("shed", {})
        if current_day < 26 and len(farm.get("unlocked_quadrants", [])) < 4:
            cost = Actions.land_cost(farm.get("unlocked_quadrants", ["NW"]))
            if cost is not None and money >= cost + 250:
                orders.append(Actions.buy_land())
                money -= cost

        # Four-unit lots reduce the steepest part of the sell curve while
        # retaining enough throughput to prevent shed overflow.
        for item, count in shed.items():
            if count > 0 and item not in ANIMALS:
                base = float(MARKET_PARAMS.get(item, {}).get("base", 0))
                if not base or prices.get(item, 0) >= base * self.min_sell_margin:
                    orders.append(Actions.sell(item, min(4, int(count))))
                    break

        if current_day < 25:
            target = CROPS[self.target_crop]
            seeds = int(private.get("seeds", {}).get(self.target_crop, 0))
            vacant = sum(tile is None for row in farm.get("tiles", []) for tile in row)
            qty = min(max(0, vacant - seeds), int(max(0, money - 250) // target.seed_cost))
            if qty:
                orders.append(Actions.buy_seed(self.target_crop, qty))
                money -= qty * target.seed_cost

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


controller = MicrobatchOpportunistController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
