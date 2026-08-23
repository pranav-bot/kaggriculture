"""
Market Prediction & Concurrency Helpers for Kaggriculture.

Provides:
- Demand Cycle Predictor: Pinpoints exact future turns of town shop and town center consumption.
- Slippage Simulator: Models concurrent unit-by-unit execution and estimates true average prices.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from kaggriculture.actions.actions import Actions
from kaggriculture.env.items import (
    MARKET_I0,
    PRICE_FLOOR,
    PRODUCTS_LIST,
    TOWN_CENTER_PRODUCTS,
    TOWN_SHOP_SELL_INTERVAL,
    TOWN_CENTER_SELL_INTERVAL,
    TURNS_PER_DAY,
    calculate_shop_turn_consumption,
)


# ==============================================================================
# 1. Demand Cycle Predictor
# ==============================================================================

@dataclass
class DemandTick:
    turn_offset: int
    global_step: int
    is_shop_tick: bool
    is_town_center_tick: bool
    consumed_products: Dict[str, int]


def predict_upcoming_consumption_ticks(
    unlocked_shops: List[str],
    current_step: int,
    lookahead_turns: int = 24,
) -> List[DemandTick]:
    """
    Predicts exact future steps when the town center and unlocked shops will consume products.
    """
    ticks: List[DemandTick] = []
    
    # Calculate single shop tick demand across all shop instances
    shop_tick_demand = {p: 0 for p in PRODUCTS_LIST}
    for shop in unlocked_shops:
        consump = calculate_shop_turn_consumption(shop)
        for p, qty in consump.items():
            shop_tick_demand[p] += qty

    for offset in range(1, lookahead_turns + 1):
        step = current_step + offset
        is_shop = (step % TOWN_SHOP_SELL_INTERVAL == 0) and len(unlocked_shops) > 0
        is_tc = (step % TOWN_CENTER_SELL_INTERVAL == 0)

        if not (is_shop or is_tc):
            continue

        turn_consumed = {p: 0 for p in PRODUCTS_LIST}
        if is_shop:
            for p, qty in shop_tick_demand.items():
                turn_consumed[p] += qty
        if is_tc:
            for p in TOWN_CENTER_PRODUCTS:
                turn_consumed[p] += 1

        # Only record if products are consumed
        if any(v > 0 for v in turn_consumed.values()):
            ticks.append(DemandTick(
                turn_offset=offset,
                global_step=step,
                is_shop_tick=is_shop,
                is_town_center_tick=is_tc,
                consumed_products={p: v for p, v in turn_consumed.items() if v > 0},
            ))

    return ticks


def find_next_consumption_window(
    unlocked_shops: List[str],
    current_step: int,
    product: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Finds the turns remaining until the next town consumption event,
    and returns (turns_until_event, units_drained).
    """
    upcoming = predict_upcoming_consumption_ticks(unlocked_shops, current_step, lookahead_turns=24)
    target = product.upper() if product else None

    for tick in upcoming:
        if target is None:
            total_drained = sum(tick.consumed_products.values())
            return (tick.turn_offset, total_drained)
        elif target in tick.consumed_products:
            return (tick.turn_offset, tick.consumed_products[target])

    # Fallback to next standard shop tick
    offset = TOWN_SHOP_SELL_INTERVAL - (current_step % TOWN_SHOP_SELL_INTERVAL)
    return (offset if offset > 0 else TOWN_SHOP_SELL_INTERVAL, 0)


def estimate_market_drain_over_horizon(
    unlocked_shops: List[str],
    start_step: int,
    num_turns: int = 24,
) -> Dict[str, int]:
    """
    Calculates total expected units drained from market across all products over `num_turns`.
    """
    ticks = predict_upcoming_consumption_ticks(unlocked_shops, start_step, lookahead_turns=num_turns)
    total_drain = {p: 0 for p in PRODUCTS_LIST}
    for tick in ticks:
        for p, qty in tick.consumed_products.items():
            total_drain[p] = total_drain.get(p, 0) + qty
    return total_drain


# ==============================================================================
# 2. Slippage Simulator
# ==============================================================================

@dataclass
class SlippageResult:
    item: str
    quantity: int
    total_revenue: int
    average_price: float
    starting_price: int
    ending_price: int
    price_impact: int
    final_market_inventory: int
    unit_prices: List[int] = field(default_factory=list)


def simulate_sell_slippage(
    item: str,
    quantity: int,
    current_market_inv: int = MARKET_I0,
    opponent_simultaneous_sell: int = 0,
    params: Optional[Dict[str, Any]] = None,
) -> SlippageResult:
    """
    Simulates the concurrent unit-by-unit execution queue of a bulk SELL order against
    the dynamic price curve, modeling the exact price slippage.
    
    If sell price hits $1 floor, units sold do not increment market inventory.
    """
    item = item.upper()
    if quantity <= 0:
        p = Actions.market_price(item, current_market_inv, params)
        return SlippageResult(
            item=item,
            quantity=0,
            total_revenue=0,
            average_price=float(p),
            starting_price=p,
            ending_price=p,
            price_impact=0,
            final_market_inventory=current_market_inv,
            unit_prices=[],
        )

    inv = current_market_inv
    unit_prices = []
    my_units_left = quantity
    opp_units_left = opponent_simultaneous_sell
    start_price = Actions.market_price(item, inv, params)

    while my_units_left > 0:
        # Quote sell price at pre-sell inventory
        p = Actions.market_price(item, inv, params)
        unit_prices.append(p)
        my_units_left -= 1

        # Opponent sell in lockstep if concurrent
        opp_sold = 1 if opp_units_left > 0 else 0
        if opp_sold:
            opp_units_left -= 1

        # Update market inventory: at $1 price floor, inventory does not increase
        if p > PRICE_FLOOR:
            inv += 1 + opp_sold

    end_price = unit_prices[-1] if unit_prices else start_price
    total_rev = sum(unit_prices)
    avg_price = total_rev / float(quantity)

    return SlippageResult(
        item=item,
        quantity=quantity,
        total_revenue=total_rev,
        average_price=avg_price,
        starting_price=start_price,
        ending_price=end_price,
        price_impact=start_price - end_price,
        final_market_inventory=inv,
        unit_prices=unit_prices,
    )


def simulate_buy_slippage(
    item: str,
    quantity: int,
    current_market_inv: int = MARKET_I0,
    opponent_simultaneous_buy: int = 0,
    params: Optional[Dict[str, Any]] = None,
) -> SlippageResult:
    """
    Simulates the unit-by-unit BUY_PRODUCT execution queue (WHEAT or FERTILIZER),
    quoted at post-buy inventory.
    """
    item = item.upper()
    if quantity <= 0:
        p = Actions.market_price(item, current_market_inv, params)
        return SlippageResult(
            item=item,
            quantity=0,
            total_revenue=0,
            average_price=float(p),
            starting_price=p,
            ending_price=p,
            price_impact=0,
            final_market_inventory=current_market_inv,
            unit_prices=[],
        )

    inv = current_market_inv
    unit_prices = []
    my_units_left = quantity
    opp_units_left = opponent_simultaneous_buy
    start_price = Actions.market_price(item, inv - 1, params)

    while my_units_left > 0:
        opp_bought = 1 if opp_units_left > 0 else 0
        if opp_bought:
            opp_units_left -= 1

        # Buy price quoted at post-buy inventory
        inv -= (1 + opp_bought)
        p = Actions.market_price(item, inv, params)
        unit_prices.append(p)
        my_units_left -= 1

    end_price = unit_prices[-1]
    total_cost = sum(unit_prices)
    avg_price = total_cost / float(quantity)

    return SlippageResult(
        item=item,
        quantity=quantity,
        total_revenue=total_cost,  # Represents total cost for purchases
        average_price=avg_price,
        starting_price=start_price,
        ending_price=end_price,
        price_impact=end_price - start_price,
        final_market_inventory=inv,
        unit_prices=unit_prices,
    )
