from enum import StrEnum
from typing import Dict, Any


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


# Raw environment definitions
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