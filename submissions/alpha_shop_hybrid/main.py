"""Alpha Shop Hybrid: P0+P1 stack, MV+shop crop scoring, P2 sell microstructure."""

import os
import sys
from typing import Any, Dict, List, Optional

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, CROPS, Plants
from kaggriculture.helpers.phase_brain import (
    CROP_ORDER,
    crop_counts,
    desired_mix,
    last_profitable_start_day,
    pick_plant_crop,
)


class AlphaShopHybridController(ActionController):
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
            enable_market_microstructure=True,
        )

    @staticmethod
    def _market_velocity_scores(obs: Dict[str, Any]) -> Dict[str, float]:
        day = int(obs.get("day", 0))
        prices = obs.get("market", {}).get("prices", {})
        shops = obs.get("town", {}).get("unlocked_shops", [])
        demand = Actions.calculate_town_daily_consumption(shops)
        remaining = max(0, 30 - day)
        scores: Dict[str, float] = {}
        for name, crop in CROPS.items():
            if crop.time_to_first_yield > remaining:
                continue
            price = float(prices.get(name, crop.base_market_price))
            demand_bonus = 1.0 + 0.08 * float(demand.get(name, 0))
            repeat_bonus = 1.0 + (0.35 if crop.ongoing else 0.0)
            scores[name] = price * crop.yield_per_tile_per_day * demand_bonus * repeat_bonus - crop.seed_cost
        return scores

    @staticmethod
    def _shop_crop_scores(obs: Dict[str, Any]) -> Dict[str, float]:
        shops = obs.get("town", {}).get("unlocked_shops", [])
        prices = obs.get("market", {}).get("prices", {})
        day = int(obs.get("day", 0))
        daily_drain = Actions.calculate_town_daily_consumption(shops)
        scores: Dict[str, float] = {}
        for crop_name, crop_cfg in CROPS.items():
            days_left = 30 - day
            if crop_cfg.time_to_first_yield > days_left:
                continue
            current_price = float(prices.get(crop_name, crop_cfg.base_market_price))
            drain = float(daily_drain.get(crop_name, 0))
            margin = current_price * crop_cfg.yield_per_tile_per_day - (
                crop_cfg.seed_cost / max(1, crop_cfg.time_to_max_yield)
            )
            scores[crop_name] = margin * (1.0 + drain * 0.05)
        return scores

    def _hybrid_crop_scores(self, obs: Dict[str, Any]) -> Dict[str, float]:
        mv = self._market_velocity_scores(obs)
        shop = self._shop_crop_scores(obs)
        keys = set(mv) | set(shop)
        return {crop: mv.get(crop, 0.0) + shop.get(crop, 0.0) for crop in keys}

    def choose_plant_crop(
        self,
        available_seeds: Dict[str, int],
        obs: dict,
        farm: dict,
        private: dict,
    ) -> Optional[str]:
        day = int(obs.get("day", 0))
        shops = (obs.get("town") or {}).get("unlocked_shops", [])
        targets = desired_mix(day, shops)
        active = crop_counts(farm)
        deficits = {
            crop: max(0, targets.get(crop, 0) - active.get(crop, 0))
            for crop in CROP_ORDER
            if day <= last_profitable_start_day(crop)
        }
        choices = [
            crop
            for crop in CROP_ORDER
            if deficits.get(crop, 0) > 0 and int(available_seeds.get(crop, 0)) > 0
        ]
        if not choices:
            return pick_plant_crop(available_seeds, day, farm, shops)
        scores = self._hybrid_crop_scores(obs)
        return max(choices, key=lambda crop: (scores.get(crop, 0.0), -deficits[crop] / max(1, targets.get(crop, 1))))

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
        if int(obs.get("day", 0)) < 26:
            scores = self._hybrid_crop_scores(obs)
            if scores:
                self.target_crop = max(scores, key=scores.get)
        return super().act(obs)


controller = AlphaShopHybridController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
