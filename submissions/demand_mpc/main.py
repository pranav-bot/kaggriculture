"""Demand MPC: Alpha Velocity P1 with calendar-aware sell/hold."""

import os
import sys
from typing import Any, Dict, List, Optional

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants
from kaggriculture.actions.actions import Actions
from kaggriculture.helpers.sell_mpc import plan_sell_horizon


class DemandMpcController(ActionController):
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
            max_hires_per_day=12,
            min_sell_margin=0.25,
            market_policy="alpha_p1",
            enable_terminal_return=True,
            enable_alpha_planting=True,
        )
        self._last_obs: Dict[str, Any] = {}

    def plan_market_actions(
        self,
        farm,
        private,
        market,
        current_day,
        planned_drop=None,
        town_shops: Optional[List[str]] = None,
    ):
        if current_day >= 26:
            self.auto_expand_land = False
            self.auto_hire_hands = False

        orders = super().plan_market_actions(
            farm,
            private,
            market,
            current_day,
            planned_drop=planned_drop,
            town_shops=town_shops,
        )
        non_sells = [
            order
            for order in orders
            if not (isinstance(order, list) and order and str(order[0]).upper() == "SELL")
        ]
        slots_left = max(0, Actions.MAX_MARKET_ORDERS_PER_TURN - len(non_sells))
        if slots_left <= 0:
            return non_sells[: Actions.MAX_MARKET_ORDERS_PER_TURN]

        mpc_sells = plan_sell_horizon(
            self._last_obs,
            private.get("shed", {}),
            planned_drop=planned_drop,
            max_orders=slots_left,
            horizon=12,
            current_day=current_day,
            min_sell_margin=self.min_sell_margin,
        )
        combined = non_sells + mpc_sells
        return combined[: Actions.MAX_MARKET_ORDERS_PER_TURN]

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._last_obs = obs
        return super().act(obs)


controller = DemandMpcController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
