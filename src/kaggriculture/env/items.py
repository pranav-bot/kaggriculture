import math
from enum import StrEnum
from typing import Dict, List, Optional, Tuple, Any, Union


class YieldType(StrEnum):
    ONE_TIME = "One-time"
    ONGOING = "Ongoing"


class Plants(StrEnum):
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"


class Animals(StrEnum):
    GOOSE = "GOOSE"
    COW = "COW"
    SHEEP = "SHEEP"


class Products(StrEnum):
    WHEAT = "WHEAT"
    CARROT = "CARROT"
    TOMATO = "TOMATO"
    STRAWBERRY = "STRAWBERRY"
    MELON = "MELON"
    EGG = "EGG"
    MILK = "MILK"
    WOOL = "WOOL"
    FERTILIZER = "FERTILIZER"


class Structures(StrEnum):
    COOP = "COOP"
    PASTURE = "PASTURE"


class Quadrants(StrEnum):
    NW = "NW"
    NE = "NE"
    SW = "SW"
    SE = "SE"


class TileKind(StrEnum):
    PLANT = "PLANT"
    WEED = "WEED"
    COOP = "COOP"
    PASTURE = "PASTURE"


# --------------------------------------------------------------------------
# Map & Quadrant Helpers
# --------------------------------------------------------------------------

def get_quadrant_bounds(quadrant: str, board_size: int = 10) -> Tuple[int, int, int, int]:
    """Returns (x_min, x_max, y_min, y_max) for the specified quadrant."""
    half = board_size // 2
    q = quadrant.upper()
    if q == "NW": return (0, half, 0, half)
    if q == "NE": return (half, board_size, 0, half)
    if q == "SW": return (0, half, half, board_size)
    if q == "SE": return (half, board_size, half, board_size)
    raise ValueError(f"Unknown quadrant: {quadrant}")


def quadrant_of(x: int, y: int, board_size: int = 10) -> str:
    """Determines which quadrant a grid coordinate belongs to ('NW', 'NE', 'SW', 'SE')."""
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access_tiles(board_size: int = 10) -> List[Tuple[int, int]]:
    """Four inner-corner tiles orthogonally adjacent to the central shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def is_shed_adjacent(pos: Tuple[int, int], board_size: int = 10) -> bool:
    """Returns True if pos is orthogonally adjacent to the central shed."""
    return tuple(pos) in set(shed_access_tiles(board_size))


def spawn_hand_position(farm: Dict[str, Any], board_size: int = 10) -> List[int]:
    """
    Determines spawn position for a newly hired hand:
    First free shed-access tile (NWSE order); ties broken by min occupants.
    """
    tiles = shed_access_tiles(board_size)
    occupants = {tile: 0 for tile in tiles}
    all_pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
    for pos in all_pos:
        if pos in occupants:
            occupants[pos] += 1
    best = sorted(occupants.items(), key=lambda kv: (kv[1], tiles.index(kv[0])))
    return list(best[0][0])


# --------------------------------------------------------------------------
# Town Shops and Market Demand Constants
# --------------------------------------------------------------------------

SHOPS: Dict[str, List[str]] = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],                          # Single-product -> 2x consumption
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],                        # Single-product -> 2x consumption
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

TOWN_CENTER_PRODUCTS: List[str] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"
]

PRODUCTS_LIST: List[str] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"
]

MAX_SHOP_INSTANCES = 8
TOWN_SHOP_UNLOCK_INTERVAL = 3  # Days
TOWN_SHOP_SELL_INTERVAL = 4    # Turns (every 4 turns = 6 times/day)
TOWN_CENTER_SELL_INTERVAL = 24 # Turns (once per day)


def calculate_shop_turn_consumption(shop_name: str) -> Dict[str, int]:
    """Calculates product consumption for one instance of a shop per sell tick."""
    products = SHOPS.get(shop_name.upper(), [])
    multiplier = 2 if len(products) == 1 else 1
    return {p: multiplier for p in products}


def calculate_town_daily_consumption(unlocked_shops: List[str]) -> Dict[str, int]:
    """
    Calculates total units consumed from the market per day (24 turns)
    by all currently unlocked shop instances + town center.
    """
    daily = {p: 0 for p in PRODUCTS_LIST}
    for p in TOWN_CENTER_PRODUCTS:
        daily[p] += 1

    ticks_per_day = 24 // TOWN_SHOP_SELL_INTERVAL
    for shop in unlocked_shops:
        consump = calculate_shop_turn_consumption(shop)
        for p, qty in consump.items():
            daily[p] = daily.get(p, 0) + qty * ticks_per_day

    return daily


# --------------------------------------------------------------------------
# Market Pricing Curve & Math Engine
# --------------------------------------------------------------------------

MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0

MARKET_PARAMS: Dict[str, Dict[str, Any]] = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def shape_func(func: str, x: float, T: Optional[float] = None) -> float:
    """Evaluates mathematical pricing shape function."""
    x = max(0.0, float(x))
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / float(T)
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def resolve_market_params(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Merges per-resource sparse overrides onto MARKET_PARAMS defaults."""
    resolved = {item: dict(p) for item, p in MARKET_PARAMS.items()}
    if not overrides:
        return resolved
    for item, patch in overrides.items():
        if item in resolved and isinstance(patch, dict):
            resolved[item].update(patch)
    return resolved


def market_price(item: Union[str, Products], inventory: int, params: Optional[Dict[str, Any]] = None) -> int:
    """
    Calculates exact dynamic market price for an item given its market inventory:
    price(inv) = base + sign * amp * f(|inv - I0|)
    Floored at $1 and rounded to the nearest integer.
    """
    key = str(item).upper()
    p = (params or MARKET_PARAMS)[key]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]

    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / shape_func(f, T, T)
        raw_price = base + amp * shape_func(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / shape_func(f, T, T)
        raw_price = base - amp * shape_func(f, inventory - I0, T)

    return max(PRICE_FLOOR, int(round(raw_price)))


# --------------------------------------------------------------------------
# Raw Environment Data Definitions
# --------------------------------------------------------------------------

CROPS_DATA: Dict[str, Dict[str, Any]] = {
    "WHEAT": {
        "seed": 10,
        "base_market_price": 25,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "time_to_first_yield": 2,
        "time_to_max_yield": 4,
        "interval": 0,
        "max_yield": 6,
        "unfertilized_max_yield": 4,
        "fertilized_max_yield": 6,
        "yield_type": YieldType.ONE_TIME,
        "ongoing": False,
        "action_cost": 1,
        "yield_per_tile_per_day": 0.80,
        "subsequent_yields": "none",
    },
    "CARROT": {
        "seed": 20,
        "base_market_price": 35,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "time_to_first_yield": 2,
        "time_to_max_yield": 3,
        "interval": 0,
        "max_yield": 4,
        "unfertilized_max_yield": 3,
        "fertilized_max_yield": 4,
        "yield_type": YieldType.ONE_TIME,
        "ongoing": False,
        "action_cost": 1,
        "yield_per_tile_per_day": 0.75,
        "subsequent_yields": "none",
    },
    "TOMATO": {
        "seed": 50,
        "base_market_price": 60,
        "first_yield_day": 8,
        "max_yield_day": 11,
        "time_to_first_yield": 8,
        "time_to_max_yield": 11,
        "interval": 1,
        "max_yield": 4,
        "unfertilized_max_yield": 4,
        "fertilized_max_yield": 4,
        "yield_type": YieldType.ONGOING,
        "ongoing": True,
        "action_cost": 1,
        "yield_per_tile_per_day": 0.33,
        "subsequent_yields": "every day ×4",
    },
    "STRAWBERRY": {
        "seed": 100,
        "base_market_price": 120,
        "first_yield_day": 10,
        "max_yield_day": 16,
        "time_to_first_yield": 10,
        "time_to_max_yield": 16,
        "interval": 2,
        "max_yield": 4,
        "unfertilized_max_yield": 4,
        "fertilized_max_yield": 4,
        "yield_type": YieldType.ONGOING,
        "ongoing": True,
        "action_cost": 1,
        "yield_per_tile_per_day": 0.24,
        "subsequent_yields": "every other day ×4",
    },
    "MELON": {
        "seed": 80,
        "base_market_price": 250,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "time_to_first_yield": 10,
        "time_to_max_yield": 10,
        "interval": 0,
        "max_yield": 6,
        "unfertilized_max_yield": 6,
        "fertilized_max_yield": 6,
        "yield_type": YieldType.ONE_TIME,
        "ongoing": False,
        "action_cost": 1,
        "yield_per_tile_per_day": 0.55,
        "subsequent_yields": "none",
    },
}

ANIMALS_DATA: Dict[str, Dict[str, Any]] = {
    "GOOSE": {
        "cost": 300,
        "base_market_price": 50,
        "structure": Structures.COOP,
        "first_yield_day": 4,
        "time_to_first_yield": 4,
        "time_to_max_yield": None,
        "interval": 1,
        "max_held": 4,
        "product": Products.EGG,
        "action_cost": "1 + 1 (build coop)",
        "yield_per_tile_per_day": 1.00,
        "subsequent_yields": "every day, indefinitely",
        "feed_item": Products.WHEAT,
    },
    "COW": {
        "cost": 400,
        "base_market_price": 160,
        "structure": Structures.PASTURE,
        "first_yield_day": 8,
        "time_to_first_yield": 8,
        "time_to_max_yield": None,
        "interval": 2,
        "max_held": 6,
        "product": Products.MILK,
        "action_cost": "1 + 1 (build pasture)",
        "yield_per_tile_per_day": 0.50,
        "subsequent_yields": "every two days, indefinitely",
        "feed_item": Products.WHEAT,
    },
    "SHEEP": {
        "cost": 500,
        "base_market_price": 200,
        "structure": Structures.PASTURE,
        "first_yield_day": 6,
        "time_to_first_yield": 6,
        "time_to_max_yield": None,
        "interval": 3,
        "max_held": 6,
        "product": Products.WOOL,
        "action_cost": "1 + 1 (build pasture)",
        "yield_per_tile_per_day": 0.33,
        "subsequent_yields": "every three days, indefinitely",
        "feed_item": Products.WHEAT,
    },
}

FERTILIZER_DATA: Dict[str, Any] = {
    "cost": 100,
    "action_cost": 1,
    "duration_days": 3,
    "base_market_price": 100,
}

LAND_EXPANSION_DATA = [
    {"quadrant": Quadrants.NE, "cost": 1000},
    {"quadrant": Quadrants.SW, "cost": 2000},
    {"quadrant": Quadrants.SE, "cost": 4000},
]