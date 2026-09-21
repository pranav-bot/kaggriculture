from __future__ import annotations
from typing import Any

from mechanics import CROPS, LAND_PRICES, MAX_MARKET_ORDERS, PRODUCTS, hire_cost
from production import seed_deficits
from sale_sim import simulate_sale
from state import GameState

PREMIUM_PRODUCTS = {"STRAWBERRY", "MELON", "MILK", "WOOL"}


def _sale_quantity(item: str, amount: int, state: GameState, config: dict[str, Any]) -> int:
    if amount <= 0:
        return 0
    terminal = state.day >= int(config["terminal_day"])
    if terminal:
        return amount
    mode = config.get("sell_mode", "immediate")
    if mode == "delayed" and state.hour < 18 and sum(int(v) for v in state.shed.values()) < 80:
        return 0
    if mode == "batched" and item in PREMIUM_PRODUCTS:
        return min(amount, int(config.get("premium_batch", 8)))
    return amount


def make_market_orders(state: GameState, config: dict[str, Any], planned_drop: dict[str, int] | None = None) -> list[list[Any]]:
    planned_drop = planned_drop or {}
    orders: list[list[Any]] = []
    projected_cash = float(state.money)

    available = {item: int(state.shed.get(item, 0)) + int(planned_drop.get(item, 0)) for item in PRODUCTS}
    for item in sorted(PRODUCTS, key=lambda p: (-state.market_price(p), p)):
        quantity = _sale_quantity(item, available[item], state, config)
        if quantity <= 0 or len(orders) >= MAX_MARKET_ORDERS:
            continue
        orders.append(["SELL", item, quantity])
        projected_cash += simulate_sale(item, state.market_inventory(item), quantity).revenue

    planned_unlocked = state.unlocked_count
    target_unlocked = int(config.get("target_unlocked", 1))
    if planned_unlocked < target_unlocked and state.day <= 2 and len(orders) < MAX_MARKET_ORDERS:
        price = LAND_PRICES[planned_unlocked - 1]
        if projected_cash >= price + int(config.get("operating_reserve", 0)):
            orders.append(["BUY_LAND"])
            projected_cash -= price
            planned_unlocked += 1

    deficits = seed_deficits(state, config)
    for crop in config.get("crop_order", []):
        quantity = int(deficits.get(crop, 0))
        if quantity <= 0 or len(orders) >= MAX_MARKET_ORDERS:
            continue
        unit_cost = int(CROPS[crop]["seed"])
        reserve = int(config.get("operating_reserve", 0))
        affordable = max(0, int((projected_cash - reserve) // unit_cost))
        quantity = min(quantity, affordable)
        if quantity <= 0:
            continue
        orders.append(["BUY_SEED", crop, quantity])
        projected_cash -= quantity * unit_cost

    hands_by_unlocked = config.get("hands_by_unlocked", {})
    target_hands = int(hands_by_unlocked.get(planned_unlocked, hands_by_unlocked.get(str(planned_unlocked), 0)))
    current_hands = len(state.units()) - 1
    hires_today = int(state.farm.get("hires_today", current_hands)) if isinstance(state.farm, dict) else current_hands
    while current_hands < target_hands and len(orders) < MAX_MARKET_ORDERS:
        cost = hire_cost(hires_today)
        if projected_cash < cost + int(config.get("operating_reserve", 0)):
            break
        orders.append(["HIRE"])
        projected_cash -= cost
        current_hands += 1
        hires_today += 1
    return orders[:MAX_MARKET_ORDERS]
