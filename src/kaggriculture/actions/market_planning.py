"""Market-order planning for :class:`ActionController`."""

from typing import Any, Dict, List

from kaggriculture.actions.actions import ANIMALS, CROPS, Actions
from kaggriculture.env.items import MARKET_PARAMS
from kaggriculture.helpers.phase_brain import (
    CROP_ORDER,
    alpha_sale_quantity,
    seed_deficits,
    target_hired_hands,
)

OPERATING_RESERVE = 100


def _plan_alpha_p1_market(
    *,
    auto_expand_land: bool,
    auto_hire_hands: bool,
    auto_sell: bool,
    min_sell_margin: float,
    farm: Dict[str, Any],
    private: Dict[str, Any],
    market: Dict[str, Any],
    operating_reserve: int,
    planned_drop: Dict[str, int],
    current_day: int,
    town_shops: List[str] | None,
) -> List[List[Any]]:
    orders: List[List[Any]] = []
    money = float(farm.get("money", 0))
    unlocked = farm.get("unlocked_quadrants", ["NW"])
    shed = private.get("shed", {})
    drop = planned_drop or {}
    prices = market.get("prices", {})
    shed_total = sum(int(shed.get(item, 0) or 0) + int(drop.get(item, 0) or 0) for item in set(shed) | set(drop))

    stop_growth = current_day >= 26

    if auto_expand_land and not stop_growth and len(unlocked) < 4:
        cost = Actions.land_cost(unlocked)
        if cost is not None and money >= cost + operating_reserve:
            orders.append(Actions.buy_land())
            money -= cost

    if auto_hire_hands and not stop_growth:
        target = target_hired_hands(unlocked)
        current_hands = len(farm.get("hands", []) or [])
        hires_today = int(farm.get("hires_today", current_hands))
        while current_hands < target and len(orders) < Actions.MAX_MARKET_ORDERS_PER_TURN:
            cost = Actions.hire_cost(hires_today)
            if money < cost + operating_reserve:
                break
            orders.append(Actions.hire())
            money -= cost
            current_hands += 1
            hires_today += 1

    if auto_sell:
        sell_items = set(shed.keys()) | set(drop.keys())
        for item in sorted(sell_items, key=lambda p: (-int(prices.get(p, 0) or 0), p)):
            count = int(shed.get(item, 0)) + int(drop.get(item, 0))
            if count <= 0 or item in ANIMALS:
                continue
            base_price = MARKET_PARAMS.get(item, {}).get("base", 0)
            current_price = prices.get(item, 0)
            if base_price and current_price < base_price * min_sell_margin and shed_total < 82:
                continue
            quantity = alpha_sale_quantity(item, count, current_day, shed_total)
            if quantity > 0:
                orders.append(Actions.sell(item, quantity))

    if not stop_growth:
        deficits = seed_deficits(farm, private.get("seeds", {}), current_day, town_shops)
        for crop in CROP_ORDER:
            quantity = int(deficits.get(crop, 0))
            if quantity <= 0 or len(orders) >= Actions.MAX_MARKET_ORDERS_PER_TURN:
                continue
            unit_cost = int(CROPS[crop].seed_cost)
            affordable = max(0, int((money - operating_reserve) // unit_cost))
            quantity = min(quantity, affordable)
            if quantity <= 0:
                continue
            orders.append(Actions.buy_seed(crop, quantity))
            money -= quantity * unit_cost

    return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]


def plan_market_actions(
    *,
    auto_expand_land: bool,
    auto_hire_hands: bool,
    auto_sell: bool,
    target_animal: str | None,
    target_crop: str,
    max_hires_per_day: int,
    min_sell_margin: float,
    farm: Dict[str, Any],
    private: Dict[str, Any],
    market: Dict[str, Any],
    operating_reserve: int = OPERATING_RESERVE,
    planned_drop: Dict[str, int] | None = None,
    current_day: int = 0,
    town_shops: List[str] | None = None,
    alpha_p1: bool = False,
) -> List[List[Any]]:
    """Build legal market orders using a conservative cash projection."""
    drop = planned_drop or {}
    if alpha_p1:
        return _plan_alpha_p1_market(
            auto_expand_land=auto_expand_land,
            auto_hire_hands=auto_hire_hands,
            auto_sell=auto_sell,
            min_sell_margin=min_sell_margin,
            farm=farm,
            private=private,
            market=market,
            operating_reserve=operating_reserve,
            planned_drop=drop,
            current_day=current_day,
            town_shops=town_shops,
        )

    orders: List[List[Any]] = []
    money = float(farm.get("money", 0))
    unlocked = farm.get("unlocked_quadrants", ["NW"])
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    prices = market.get("prices", {})

    if auto_expand_land and len(unlocked) < 4:
        cost = Actions.land_cost(unlocked)
        if cost is not None and money >= cost + operating_reserve:
            orders.append(Actions.buy_land())
            money -= cost

    if auto_hire_hands:
        hires_today = int(farm.get("hires_today", 0))
        remaining_hires = max(0, max_hires_per_day - hires_today)
        for offset in range(remaining_hires):
            cost = Actions.hire_cost(hires_today + offset)
            if money < cost + operating_reserve:
                break
            orders.append(Actions.hire())
            money -= cost

    if auto_sell:
        sell_items = set(shed.keys()) | set(drop.keys())
        for item in sell_items:
            count = int(shed.get(item, 0)) + int(drop.get(item, 0))
            if count <= 0 or item in ANIMALS:
                continue
            base_price = MARKET_PARAMS.get(item, {}).get("base", 0)
            current_price = prices.get(item, 0)
            if base_price and current_price >= base_price * min_sell_margin:
                orders.append(Actions.sell(item, count))

    target_crop_cfg = CROPS.get(target_crop, CROPS["WHEAT"])
    seed_count = seeds.get(target_crop, 0)
    if seed_count < 5 and money >= target_crop_cfg.seed_cost + operating_reserve:
        affordable = max(0, int((money - operating_reserve) // target_crop_cfg.seed_cost))
        quantity = min(5 - seed_count, affordable)
        if quantity > 0:
            orders.append(Actions.buy_seed(target_crop, quantity))
            money -= quantity * target_crop_cfg.seed_cost

    if target_animal and target_animal in ANIMALS:
        animal_cfg = ANIMALS[target_animal]
        if shed.get(target_animal, 0) == 0 and money >= animal_cfg.cost + operating_reserve:
            orders.append(Actions.buy_animal(target_animal, 1))
            money -= animal_cfg.cost

    return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]
