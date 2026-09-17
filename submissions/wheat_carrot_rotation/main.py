"""Wheat-Carrot Rotation: cheap bootstrap followed by rapid crop turnover."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants


class RotationController(ActionController):
    def __init__(self):
        super().__init__(
            target_crop=Plants.WHEAT,
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
            max_hires_per_day=3,
            min_sell_margin=0.1,
        )

    def act(self, obs: Dict[str, Any]):
        day = int(obs.get("day", 0))
        self.target_crop = Plants.WHEAT if day < 6 else Plants.CARROT
        self.auto_expand_land = day < 26
        self.auto_hire_hands = day < 28
        return super().act(obs)


controller = RotationController()


def agent(obs: Dict[str, Any]):
    return controller.act(obs)
