"""Demand-calendar sell/hold MPC (research/05 innovation 2)."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from kaggriculture.actions.actions import ANIMALS, Actions
from kaggriculture.env.items import MARKET_I0, MARKET_PARAMS
from kaggriculture.helpers.market_prediction import (
    predict_upcoming_consumption_ticks,
    simulate_sell_slippage,
)
from kaggriculture.helpers.phase_brain import alpha_sale_quantity
from kaggriculture.helpers.sell_ranking import impact_score

DEFAULT_HORIZON = 12
DEFAULT_WAIT_TURNS = 2
SHED_WAIT_CAP = 85


def _combined_stock(
    shed: Mapping[str, int],
    planned_drop: Optional[Mapping[str, int]] = None,
) -> dict[str, int]:
    drop = planned_drop or {}
    items = set(shed.keys()) | set(drop.keys())
    return {
        str(item): int(shed.get(item, 0) or 0) + int(drop.get(item, 0) or 0)
        for item in items
    }


def _shed_total(stock: Mapping[str, int]) -> int:
    return int(sum(max(0, int(v)) for v in stock.values()))


def turns_until_product_consumption(
    unlocked_shops: Sequence[str],
    current_step: int,
    product: str,
    horizon: int = DEFAULT_HORIZON,
) -> Tuple[Optional[int], int]:
    """Return (turns_until_event, units_drained) for *product*, or (None, 0)."""
    item = str(product).upper()
    for tick in predict_upcoming_consumption_ticks(
        list(unlocked_shops),
        current_step,
        lookahead_turns=horizon,
    ):
        drained = int(tick.consumed_products.get(item, 0) or 0)
        if drained > 0:
            return tick.turn_offset, drained
    return None, 0


def _inv_after_drain(current_inv: int, drained: int) -> int:
    return max(0, int(current_inv) - int(drained))


def sell_now_vs_wait_revenue(
    item: str,
    quantity: int,
    market_inv: int,
    drained_before_sell: int,
) -> Tuple[int, int]:
    """Return (revenue_if_sell_now, revenue_if_wait_for_drain)."""
    if quantity <= 0:
        return 0, 0
    now = simulate_sell_slippage(item, quantity, market_inv)
    later_inv = _inv_after_drain(market_inv, drained_before_sell)
    later = simulate_sell_slippage(item, quantity, later_inv)
    return now.total_revenue, later.total_revenue


def should_delay_for_demand(
    *,
    shed_total: int,
    turns_until: Optional[int],
    drained: int,
    revenue_now: int,
    revenue_wait: int,
    wait_turns: int = DEFAULT_WAIT_TURNS,
    shed_cap: int = SHED_WAIT_CAP,
) -> bool:
    if shed_total >= shed_cap or turns_until is None or drained <= 0:
        return False
    if turns_until > wait_turns:
        return False
    return revenue_wait > revenue_now


def plan_sell_horizon(
    obs: Mapping[str, Any],
    shed: Mapping[str, int],
    *,
    planned_drop: Optional[Mapping[str, int]] = None,
    max_orders: int = 10,
    horizon: int = DEFAULT_HORIZON,
    wait_turns: int = DEFAULT_WAIT_TURNS,
    shed_cap: int = SHED_WAIT_CAP,
    current_day: Optional[int] = None,
    min_sell_margin: float = 0.25,
) -> List[List[Any]]:
    """
    Greedy sell plan: per stocked product, sell now vs wait using slippage sim and
    the next consumption tick within *horizon* (wait when drain ≤ *wait_turns* and
    shed_total < *shed_cap* and waiting improves revenue).
    """
    stock = _combined_stock(shed, planned_drop)
    shed_total = _shed_total(stock)
    day = int(current_day if current_day is not None else obs.get("day", 0) or 0)
    step = int(obs.get("step", 0) or 0)
    market = obs.get("market") or {}
    prices = market.get("prices") or {}
    inventories = market.get("inventory") or {}
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])

    candidates: List[Tuple[float, str, int]] = []

    for item, count in stock.items():
        if count <= 0 or item in ANIMALS:
            continue
        base_price = MARKET_PARAMS.get(item, {}).get("base", 0)
        current_price = prices.get(item, 0)
        if base_price and current_price < base_price * min_sell_margin and shed_total < 82:
            continue

        quantity = alpha_sale_quantity(item, count, day, shed_total)
        if quantity <= 0:
            continue

        market_inv = int(inventories.get(item, MARKET_I0))
        turns_until, drained = turns_until_product_consumption(shops, step, item, horizon)
        rev_now, rev_wait = sell_now_vs_wait_revenue(item, quantity, market_inv, drained)

        if should_delay_for_demand(
            shed_total=shed_total,
            turns_until=turns_until,
            drained=drained,
            revenue_now=rev_now,
            revenue_wait=rev_wait,
            wait_turns=wait_turns,
            shed_cap=shed_cap,
        ):
            continue

        score = float(impact_score(Actions.sell(item, quantity), market, shops))
        candidates.append((score, item, quantity))

    candidates.sort(key=lambda row: (-row[0], row[1]))
    orders: List[List[Any]] = []
    for _, item, quantity in candidates[: max(0, max_orders)]:
        orders.append(Actions.sell(item, quantity))
    return orders
