"""
Example 04 — Custom market planning.

Override plan_market_actions() to control seed buying, selling, hiring, and
land expansion while keeping the controller's field routing for units.
"""
import os
import sys
from typing import Any, Dict, List

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, Plants


class PhasedMarketController(ActionController):
    """Early game: expand + seed. Mid game: hire hands. Late game: sell only."""

    def __init__(self) -> None:
        super().__init__(
            target_crop=Plants.MELON,
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_dig_weeds=True,
            auto_sell=False,  # we sell manually in plan_market_actions
            auto_expand_land=False,
            auto_hire_hands=False,
            min_sell_margin=0.3,
        )

    def plan_market_actions(
        self,
        farm: dict,
        private: dict,
        market: dict,
        current_day: int,
    ) -> List[List[Any]]:
        orders: List[List[Any]] = []
        money = float(farm.get("money", 0))
        prices = market.get("prices", {})
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        vacant = sum(
            1
            for row in farm.get("tiles", [])
            for tile in row
            if tile is None
        )

        # Phase 1 (days 0–9): buy land and melon seeds aggressively
        if current_day < 10:
            if money >= 1000 and farm.get("unlocked_quadrants", 1) < 4:
                orders.append(Actions.buy_land())
            melon_seeds = seeds.get("MELON", 0)
            if money >= 80 and melon_seeds < vacant:
                buy_qty = min(10, vacant - melon_seeds, int(money // 80))
                if buy_qty > 0:
                    orders.append(Actions.buy_seed(Plants.MELON, buy_qty))

        # Phase 2 (days 10–24): hire up to 6 hands per day when affordable
        elif current_day < 25:
            hires_today = farm.get("hires_today", 0)
            while hires_today < 6 and money >= 50 * (hires_today + 1):
                orders.append(Actions.hire())
                hires_today += 1

        # All phases: sell melon when price is strong or shed is nearly full
        melon_price = float(prices.get("MELON", 250))
        melon_in_shed = int(shed.get("MELON", 0))
        if melon_in_shed > 0 and (melon_price >= 200 or melon_in_shed >= 80):
            orders.append(Actions.sell(Plants.MELON, min(melon_in_shed, 20)))

        return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]


controller = PhasedMarketController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
