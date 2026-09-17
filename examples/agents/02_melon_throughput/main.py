"""
Example 02 — Melon throughput with hired hands.

Shows how controller flags unlock higher throughput without custom logic:
  - Switch crop target to melon (higher peak value)
  - Enable fertilizer, land expansion, and daily farm-hand hiring
  - Lower sell margin to liquidate inventory faster
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
    auto_dig_weeds=True,
    auto_sell=True,
    auto_expand_land=True,
    auto_hire_hands=True,
    max_hires_per_day=4,
    min_sell_margin=0.5,
)


def agent(obs):
    return controller.act(obs)
