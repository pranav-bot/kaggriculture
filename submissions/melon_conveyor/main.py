"""Aggressive full-board melon production candidate.

This candidate deliberately favors throughput over diversification: unlock land,
hire enough daily workers to service the board, plant melons continuously, and
sell harvested inventory before the shed reaches capacity.
"""

import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants


controller = ActionController(
    target_crop=Plants.MELON,
    auto_water=True,
    auto_harvest=True,
    auto_fertilize=True,
    auto_feed_animals=False,
    auto_care_animals=False,
    auto_collect_fertilizer=False,
    auto_dig_weeds=True,
    auto_sell=True,
    auto_expand_land=True,
    auto_hire_hands=True,
    max_hires_per_day=8,
    min_sell_margin=0.0,
)


def agent(obs):
    """Kaggle entrypoint for the aggressive melon conveyor."""
    return controller.act(obs)
