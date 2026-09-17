"""Melon Pipeline v2: bootstrap cash before expanding the farm.

The first wave is deliberately smaller than the initial quadrant so daily
workers can be funded until the first harvest. That harvest then finances the
remaining land and a bulk second wave without diverting cash to livestock.
"""

import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, ANIMALS, CROPS, MARKET_PARAMS


class MelonPipelineController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop="MELON",
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=False,
            auto_feed_animals=False,
            auto_care_animals=False,
            auto_collect_fertilizer=False,
            auto_dig_weeds=True,
            auto_sell=False,
            auto_expand_land=False,
            auto_hire_hands=False,
            max_hires_per_day=8,
            min_sell_margin=0.0,
        )

    def plan_market_actions(self, farm, private, market, current_day):
        orders = []
        money = float(farm.get("money", 0))
        shed = private.get("shed", {})
        prices = market.get("prices", {})
        for item, quantity in shed.items():
            if quantity > 0 and item not in ANIMALS:
                orders.append(Actions.sell(item, quantity))
                money += quantity * float(prices.get(item, MARKET_PARAMS.get(item, {}).get("base", 0)))

        if current_day <= 25:
            unlocked = farm.get("unlocked_quadrants", ["NW"])
            land_cost = Actions.land_cost(unlocked)
            if current_day >= 10 and land_cost is not None and money >= land_cost:
                orders.append(Actions.buy_land())
                money -= land_cost

            vacant = sum(
                tile is None
                for row in farm.get("tiles", [])
                for tile in row
            )
            have_seeds = int(private.get("seeds", {}).get("MELON", 0))
            target_seed_count = 20 if current_day == 0 and not farm.get("unlocked_quadrants")[1:] else vacant
            buyable = int(max(0, money) // CROPS["MELON"].seed_cost)
            quantity = min(max(0, target_seed_count - have_seeds), buyable)
            if quantity:
                orders.append(Actions.buy_seed("MELON", quantity))
                money -= quantity * CROPS["MELON"].seed_cost

            hires_today = int(farm.get("hires_today", 0))
            for n in range(hires_today, 8):
                if len(orders) >= Actions.MAX_MARKET_ORDERS_PER_TURN:
                    break
                cost = Actions.hire_cost(n)
                if money < cost:
                    break
                orders.append(Actions.hire())
                money -= cost
        return orders[:Actions.MAX_MARKET_ORDERS_PER_TURN]


controller = MelonPipelineController()


def agent(obs):
    return controller.act(obs)
