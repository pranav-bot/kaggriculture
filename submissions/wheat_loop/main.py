"""
Wheat Loop Starter Agent for Kaggriculture.
Focuses on fast 4-day wheat turnarounds, reliable cash flow, and automatic shed drops.
"""
import os
import sys

# Ensure local bundled packages (e.g. kaggriculture) are importable
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import Actions, ActionController, Observation, Plants

# Initialize controller configured for wheat production
controller = ActionController(
    target_crop=Plants.WHEAT,
    auto_water=True,
    auto_harvest=True,
    auto_sell=True,
    auto_expand_land=True,
    auto_dig_weeds=True,
    min_sell_margin=0.8,
)


def agent(obs):
    """Kaggle entrypoint for Wheat Loop agent."""
    return controller.act(obs)
