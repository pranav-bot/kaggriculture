"""Pinned environment mechanics used by planning (no env import required)."""
from __future__ import annotations

import math
from typing import Any, Mapping

EPISODE_STEPS = 720
TURNS_PER_DAY = 24
BOARD_SIZE = 10
SHED_CAPACITY = 100
MAX_MARKET_ORDERS = 10
PRICE_FLOOR = 1
MARKET_I0 = 10_000

CROPS: dict[str, dict[str, Any]] = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS: dict[str, dict[str, Any]] = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")

MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT": {"base": 25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT": {"base": 35, "I0": MARKET_I0, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO": {"base": 60, "I0": MARKET_I0, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON": {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG": {"base": 50, "I0": MARKET_I0, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK": {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL": {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

SHOPS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

LAND_PRICES = (1_000, 2_000, 4_000)


def read(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def shape(name: str, x: float, anchor: float | None = None) -> float:
    x = max(0.0, x)
    if name == "linear":
        return x
    if name == "sq":
        return x * x
    if name == "sqrt":
        return math.sqrt(x)
    if name == "log":
        return math.log1p(x)
    if name == "hinge":
        if not anchor or anchor <= 0:
            return x
        u = x / anchor
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(item: str, inventory: int, params: Mapping[str, Mapping[str, Any]] | None = None) -> int:
    p = (params or MARKET_PARAMS)[item]
    base, initial, anchor = p["base"], p["I0"], p["T"]
    if inventory < initial:
        func, target, sign = p["below_func"], p["below_target"], 1
    else:
        func, target, sign = p["above_func"], p["above_target"], -1
    amplitude = target * base / shape(func, anchor, anchor)
    price = base + sign * amplitude * shape(func, abs(inventory - initial), anchor)
    return max(PRICE_FLOOR, int(round(price)))


def shed_access_tiles(board_size: int = BOARD_SIZE) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def is_shed_adjacent(position, board_size: int = BOARD_SIZE) -> bool:
    return tuple(position) in shed_access_tiles(board_size)


def hire_cost(number_already_hired: int) -> int:
    a, b = 1, 1
    for _ in range(number_already_hired):
        a, b = b, a + b
    return a


def harvest_age(crop: str) -> int:
    if crop == "MELON":
        return 10
    return int(CROPS[crop]["max_yield_day"])


def expected_unfertilized_yield(crop: str) -> int:
    if crop == "WHEAT":
        return 4
    if crop == "CARROT":
        return 3
    if crop == "MELON":
        return 6
    return int(CROPS[crop]["max_yield"])


def last_profitable_start_day(crop: str, season_days: int = 30) -> int:
    data = CROPS[crop]
    grow = int(data["first_yield_day"]) if data["ongoing"] else harvest_age(crop)
    return season_days - 1 - grow
