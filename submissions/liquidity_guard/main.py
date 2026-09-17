"""Liquidity Guard: phase-gated production with explicit terminal cash protection."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants


class LiquidityGuardController(ActionController):
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
            max_hires_per_day=2,
            min_sell_margin=0.25,
        )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        day = int(obs.get("day", 0))
        self.auto_expand_land = day < 24
        self.auto_hire_hands = day < 27
        # Melons are the stable bootstrap; switch to fast crops only near cutoff.
        self.target_crop = Plants.MELON if day < 20 else Plants.CARROT
        return super().act(obs)


controller = LiquidityGuardController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
