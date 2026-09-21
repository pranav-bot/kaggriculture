"""Regime counter: alpha_shop_hybrid + opponent regime → phase_brain adjustments."""

import os
import sys
from typing import Any, Dict, List, Optional

if "__file__" in globals():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, os.getcwd())

from kaggriculture import ActionController, Actions, CROPS, Plants
from kaggriculture.helpers.market_overlays import collision_guard
from kaggriculture.helpers.opponent import clone_like
from kaggriculture.helpers.phase_brain import (
    crop_counts,
    last_profitable_start_day,
    pick_plant_crop,
)
from kaggriculture.helpers.regime_counter import (
    RegimeKnobs,
    RegimeState,
    analyze_regime,
    desired_mix_with_regime,
    sell_delay_items,
    target_hired_hands_with_regime,
)
from kaggriculture.helpers.sell_ranking import rank_sell_slots


class RegimeCounterController(ActionController):
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
        self._regime: RegimeState | None = None
        self._last_obs: Dict[str, Any] = {}

    @property
    def _knobs(self) -> RegimeKnobs:
        if self._regime is None:
            return analyze_regime({}).knobs
        return self._regime.knobs

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
        targets = desired_mix_with_regime(day, shops, self._knobs)
        active = crop_counts(farm)
        priority = self._knobs.plant_priority
        deficits = {
            crop: max(0, targets.get(crop, 0) - active.get(crop, 0))
            for crop in priority
            if day <= last_profitable_start_day(crop)
        }
        choices = [
            crop
            for crop in priority
            if deficits.get(crop, 0) > 0 and int(available_seeds.get(crop, 0)) > 0
        ]
        if not choices:
            return pick_plant_crop(available_seeds, day, farm, shops)
        scores = self._hybrid_crop_scores(obs)
        return max(
            choices,
            key=lambda crop: (scores.get(crop, 0.0), -deficits[crop] / max(1, targets.get(crop, 1))),
        )

    def _apply_sell_delays(
        self,
        orders: List[List[Any]],
        obs: Dict[str, Any],
        private: Dict[str, Any],
    ) -> List[List[Any]]:
        if self._regime is None:
            return orders
        shed = private.get("shed", {}) or {}
        shed_total = sum(int(v or 0) for v in shed.values())
        if shed_total >= 82:
            return orders
        delayed = sell_delay_items(
            self._knobs,
            self._regime.profile,
            clone_like_opponent=clone_like(obs),
        )
        if not delayed:
            return orders
        kept: List[List[Any]] = []
        for order in orders:
            if (
                isinstance(order, list)
                and order
                and str(order[0]).upper() == "SELL"
                and str(order[1]).upper() in delayed
            ):
                continue
            kept.append(order)
        return kept

    def _boost_hires_for_expander(
        self,
        orders: List[List[Any]],
        farm: Dict[str, Any],
        current_day: int,
    ) -> List[List[Any]]:
        if not self._knobs.expander_match_pace or current_day >= 26:
            return orders
        if not self.auto_hire_hands:
            return orders
        target = target_hired_hands_with_regime(farm.get("unlocked_quadrants"), self._knobs)
        current_hands = len(farm.get("hands", []) or [])
        hires_today = int(farm.get("hires_today", current_hands))
        money = float(farm.get("money", 0))
        reserve = self.operating_reserve
        while current_hands < target and len(orders) < Actions.MAX_MARKET_ORDERS_PER_TURN:
            cost = Actions.hire_cost(hires_today)
            if money < cost + reserve:
                break
            if any(isinstance(o, list) and o and o[0] == "HIRE" for o in orders):
                break
            orders.append(Actions.hire())
            money -= cost
            current_hands += 1
            hires_today += 1
        return orders

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
        orders = super().plan_market_actions(
            farm,
            private,
            market,
            current_day,
            planned_drop=planned_drop,
            town_shops=town_shops,
        )
        orders = self._apply_sell_delays(orders, self._last_obs, private)
        orders = self._boost_hires_for_expander(orders, farm, current_day)
        return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]

    def _postprocess_market(
        self,
        obs: dict,
        market_act: List[List[Any]],
        unit_actions: List[List[Any]],
        farmer_act: List[Any],
        hands_act: List[List[Any]],
        town_shops: List[str],
    ) -> List[List[Any]]:
        if not self.enable_market_microstructure:
            return market_act
        market_info = obs.get("market", {})
        market_act = rank_sell_slots(market_act, market_info, town_shops)
        if self._knobs.collision_aggressive or clone_like(obs):
            action = collision_guard(
                obs,
                {"farmer": farmer_act, "hands": hands_act, "market": market_act},
            )
            return action.get("market", market_act)
        return market_act

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._last_obs = obs
        self._regime = analyze_regime(obs)
        if int(obs.get("day", 0)) < 26:
            scores = self._hybrid_crop_scores(obs)
            if scores:
                self.target_crop = max(scores, key=scores.get)
        return super().act(obs)


controller = RegimeCounterController()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return controller.act(obs)
