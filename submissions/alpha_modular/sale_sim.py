from __future__ import annotations
from dataclasses import dataclass

from .mechanics import PRICE_FLOOR, market_price


@dataclass(frozen=True)
class SaleResult:
    revenue: int
    inventory: int
    units: int
    prices: tuple[int, ...]


def simulate_sale(item: str, inventory: int, quantity: int) -> SaleResult:
    revenue = 0
    prices: list[int] = []
    units = 0
    for _ in range(max(0, int(quantity))):
        price = market_price(item, inventory)
        prices.append(price)
        revenue += price
        units += 1
        if price > PRICE_FLOOR:
            inventory += 1
    return SaleResult(revenue=revenue, inventory=inventory, units=units, prices=tuple(prices))
