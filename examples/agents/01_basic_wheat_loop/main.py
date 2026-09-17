"""
Example 01 — Basic Wheat Loop (starter submission).

The smallest viable competition agent:
  - Uses ActionController with sensible defaults
  - Exposes agent(obs) as the Kaggle entrypoint
  - Copies this file to submissions/my_agent/main.py to iterate
"""
import os
import sys

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Plants

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
    """Kaggle entrypoint — called once per turn with the full observation dict."""
    return controller.act(obs)
