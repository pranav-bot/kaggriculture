"""Architecture Delta: market-velocity controller with ranked sell orders."""

import os
import sys
from typing import Any, Dict

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # The standoff loader imports this file by path, so the repository root is
    # not necessarily present on sys.path for sibling submission imports.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture.helpers.sell_ranking import rank_sell_slots
from submissions.market_velocity.main import MarketVelocityController


class DeltaRankedVelocityController(MarketVelocityController):
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action = super().act(obs)
        action["market"] = rank_sell_slots(
            action.get("market", []),
            obs.get("market", {}),
            obs.get("town", {}).get("unlocked_shops", []),
        )
        return action


controller = DeltaRankedVelocityController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
