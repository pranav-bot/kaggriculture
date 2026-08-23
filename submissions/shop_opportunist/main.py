"""
Shop Opportunist Adaptive Agent for Kaggriculture.
Dynamically inspects unlocked town shops, identifies high-demand/scarcity price spikes,
and adapts crop/livestock production to maximize profit.
"""
import os
import sys
from typing import Any, Dict

# Ensure local bundled packages are importable
if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import (
    Actions,
    ActionController,
    Observation,
    Plants,
    Animals,
    Products,
    CROPS,
    ANIMALS,
)


class AdaptiveOpportunistController(ActionController):
    """
    Subclasses ActionController to dynamically evaluate town demand and live market prices,
    switching target crops or livestock to exploit high-margin market scarcity.
    """

    def __init__(self, **kwargs):
        super().__init__(
            target_crop=Plants.WHEAT,
            auto_water=True,
            auto_harvest=True,
            auto_fertilize=True,
            auto_feed_animals=True,
            auto_care_animals=True,
            auto_collect_fertilizer=True,
            auto_dig_weeds=True,
            auto_sell=True,
            auto_expand_land=True,
            auto_hire_hands=True,
            max_hires_per_day=2,
            min_sell_margin=0.7,
            **kwargs,
        )

    def adapt_strategy_to_market(self, obs: Dict[str, Any]) -> None:
        """Adapts target crop and animal based on live prices and town shop consumption."""
        town = obs.get("town", {})
        unlocked_shops = town.get("unlocked_shops", [])
        market = obs.get("market", {})
        prices = market.get("prices", {})
        current_day = obs.get("day", 0)

        # 1. Calculate town consumption pressure
        daily_drain = Actions.calculate_town_daily_consumption(unlocked_shops)

        # 2. Score potential crops based on profit potential and current prices
        crop_scores = {}
        for crop_name, crop_cfg in CROPS.items():
            current_price = prices.get(crop_name, crop_cfg.base_market_price)
            drain = daily_drain.get(crop_name, 0)
            
            # Days remaining constraint: don't plant late if time_to_max_yield exceeds remaining season
            days_left = 30 - current_day
            if crop_cfg.time_to_first_yield > days_left:
                continue

            # Profit per day metric
            margin = current_price * crop_cfg.yield_per_tile_per_day - (crop_cfg.seed_cost / max(1, crop_cfg.time_to_max_yield))
            demand_bonus = 1.0 + (drain * 0.05)
            crop_scores[crop_name] = margin * demand_bonus

        if crop_scores:
            best_crop = max(crop_scores.items(), key=lambda kv: kv[1])[0]
            self.target_crop = best_crop

        # 3. Opportunistic animal purchase if lucrative
        if "YARN_STORE" in unlocked_shops and prices.get("WOOL", 200) >= 180:
            self.target_animal = Animals.SHEEP
        elif "BAKERY" in unlocked_shops or "BRUNCH_SPOT" in unlocked_shops:
            self.target_animal = Animals.GOOSE
        elif "PIZZA_SHOP" in unlocked_shops or "ICE_CREAM_SHOP" in unlocked_shops:
            self.target_animal = Animals.COW
        else:
            self.target_animal = None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Updates market awareness before planning actions."""
        self.adapt_strategy_to_market(obs)
        return super().act(obs)


# Instantiate the adaptive controller
opportunist_controller = AdaptiveOpportunistController()


def agent(obs):
    """Kaggle entrypoint for Shop Opportunist agent."""
    return opportunist_controller.act(obs)
