"""Alpha Velocity P1: P0 safety + Architecture Alpha mix, labor, and sell policy."""

import os
import sys
from typing import Any, Dict, List, Optional

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants


class AlphaVelocityP1Controller(ActionController):
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
        return super().plan_market_actions(
            farm,
            private,
            market,
            current_day,
            planned_drop=planned_drop,
            town_shops=town_shops,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return super().act(obs)


controller = AlphaVelocityP1Controller()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
