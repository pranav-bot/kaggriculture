"""
Melon Rusher Strategic Agent for Kaggriculture.
Focuses on high-value Melon cultivation ($250 base), fertilizer application,
and timed market sales to maximize profit.
"""
import os
import sys

# Ensure local bundled packages are importable
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import Actions, ActionController, Observation, Plants

# Initialize controller configured for Melon high-yield cultivation
controller = ActionController(
    target_crop=Plants.MELON,
    auto_water=True,
    auto_harvest=True,
    auto_fertilize=True,
    auto_sell=True,
    auto_expand_land=True,
    auto_hire_hands=True,
    max_hires_per_day=1,
    auto_dig_weeds=True,
    min_sell_margin=0.6,
)


def agent(obs):
    """Kaggle entrypoint for Melon Rusher agent."""
    return controller.act(obs)
