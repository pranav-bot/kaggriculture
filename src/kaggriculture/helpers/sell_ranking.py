"""Market sell ordering and conservative order-repair helpers."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kaggriculture.env.items import MARKET_PARAMS


def _item_name(order: Sequence[Any]) -> str:
    return str(order[1]).upper() if len(order) > 1 else ""


def demand_per_day(
    item: str,
    unlocked_shops: Iterable[str] | None = None,
) -> float:
    """Return the town's estimated daily demand for *item*.

    Keeping the import local avoids making the market-parameter module depend on
    the action model during submission packaging.
    """
    from kaggriculture.env.items import calculate_town_daily_consumption

    demand = calculate_town_daily_consumption(list(unlocked_shops or []))
    return float(demand.get(str(item).upper(), 0))


def impact_score(
    order: Sequence[Any],
    market: Mapping[str, Any] | None = None,
    unlocked_shops: Iterable[str] | None = None,
) -> float:
    """Score the market impact of a SELL order.

    Larger lots, higher-value goods, and goods with less daily demand/liquidity
    are prioritized.  The score is intentionally monotonic in quantity so a
    larger same-item order is always sold first.
    """
    if len(order) < 3 or str(order[0]).upper() != "SELL":
        return float("-inf")
    item = _item_name(order)
    quantity = max(0.0, float(order[2]))
    params = MARKET_PARAMS.get(item, {})
    price = float((market or {}).get("prices", {}).get(item, params.get("base", 0)))
    turnover = max(float(params.get("T", 1)), 1.0)
    demand = demand_per_day(item, unlocked_shops)
    scarcity = turnover / max(demand, 1.0)
    return quantity * max(price, 0.0) * (1.0 + scarcity / turnover)


def order_score(
    order: Sequence[Any],
    market: Mapping[str, Any] | None = None,
    unlocked_shops: Iterable[str] | None = None,
) -> float:
    """Return the ranking score for an order, with demand as a tie-break signal."""
    return impact_score(order, market, unlocked_shops) + demand_per_day(
        _item_name(order), unlocked_shops
    ) * 1e-6


def rank_sell_slots(
    orders: Sequence[Sequence[Any]],
    market: Mapping[str, Any] | None = None,
    unlocked_shops: Iterable[str] | None = None,
) -> list[list[Any]]:
    """Rank SELL orders while retaining the positions of all other orders."""
    ranked_sells = sorted(
        (list(order) for order in orders if len(order) and str(order[0]).upper() == "SELL"),
        key=lambda order: order_score(order, market, unlocked_shops),
        reverse=True,
    )
    sell_index = iter(ranked_sells)
    result: list[list[Any]] = []
    for order in orders:
        result.append(next(sell_index) if len(order) and str(order[0]).upper() == "SELL" else list(order))
    return result


def projected_shed(
    shed: Mapping[str, Any],
    orders: Sequence[Sequence[Any]],
) -> dict[str, int]:
    """Project shed quantities after applying SELL orders, never below zero."""
    projected = {str(item).upper(): max(0, int(quantity)) for item, quantity in shed.items()}
    for order in orders:
        if len(order) >= 3 and str(order[0]).upper() == "SELL":
            item = _item_name(order)
            projected[item] = max(0, projected.get(item, 0) - max(0, int(order[2])))
    return projected


def clamp_sells(
    orders: Sequence[Sequence[Any]],
    shed: Mapping[str, Any],
) -> list[list[Any]]:
    """Clamp each SELL quantity to the remaining item quantity in the shed."""
    remaining = {str(item).upper(): max(0, int(quantity)) for item, quantity in shed.items()}
    repaired: list[list[Any]] = []
    for order in orders:
        if len(order) >= 3 and str(order[0]).upper() == "SELL":
            item = _item_name(order)
            quantity = min(max(0, int(order[2])), remaining.get(item, 0))
            remaining[item] = remaining.get(item, 0) - quantity
            if quantity:
                repaired.append([order[0], item, quantity])
        else:
            repaired.append(list(order))
    return repaired


def room_guard_99(
    shed: Mapping[str, Any],
    orders: Sequence[Sequence[Any]],
    capacity: int = 100,
) -> list[list[Any]]:
    """Drop non-essential BUY orders when projected shed occupancy reaches 99."""
    occupancy = sum(max(0, int(quantity)) for quantity in shed.values())
    if occupancy >= capacity - 1:
        return [list(order) for order in orders if str(order[0]).upper() == "SELL"]
    return [list(order) for order in orders]

