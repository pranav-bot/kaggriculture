"""Market-order planning for :class:`ActionController`."""

from typing import Any, Dict, List

from kaggriculture.actions.actions import ANIMALS, CROPS, Actions
from kaggriculture.env.items import MARKET_PARAMS


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
) -> List[List[Any]]:
    """Build legal market orders using a conservative cash projection."""
    orders: List[List[Any]] = []
    money = float(farm.get("money", 0))
    unlocked = farm.get("unlocked_quadrants", ["NW"])
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    prices = market.get("prices", {})

    if auto_expand_land and len(unlocked) < 4:
        cost = Actions.land_cost(unlocked)
        if cost is not None and money >= cost:
            orders.append(Actions.buy_land())
            money -= cost

    if auto_hire_hands:
        hires_today = int(farm.get("hires_today", 0))
        remaining_hires = max(0, max_hires_per_day - hires_today)
        for offset in range(remaining_hires):
            cost = Actions.hire_cost(hires_today + offset)
            if money < cost:
                break
            orders.append(Actions.hire())
            money -= cost

    if auto_sell:
        for item, count in shed.items():
            if count <= 0 or item in ANIMALS:
                continue
            base_price = MARKET_PARAMS.get(item, {}).get("base", 0)
            current_price = prices.get(item, 0)
            if base_price and current_price >= base_price * min_sell_margin:
                orders.append(Actions.sell(item, count))

    target_crop_cfg = CROPS.get(target_crop, CROPS["WHEAT"])
    seed_count = seeds.get(target_crop, 0)
    if seed_count < 5 and money >= target_crop_cfg.seed_cost:
        quantity = min(5 - seed_count, int(money // target_crop_cfg.seed_cost))
        if quantity > 0:
            orders.append(Actions.buy_seed(target_crop, quantity))
            money -= quantity * target_crop_cfg.seed_cost

    if target_animal and target_animal in ANIMALS:
        animal_cfg = ANIMALS[target_animal]
        if shed.get(target_animal, 0) == 0 and money >= animal_cfg.cost:
            orders.append(Actions.buy_animal(target_animal, 1))
            money -= animal_cfg.cost

    return orders[: Actions.MAX_MARKET_ORDERS_PER_TURN]
