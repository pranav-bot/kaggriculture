from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from kaggriculture.env.items import (
    Plants,
    Animals,
    Products,
    Structures,
    Quadrants,
    YieldType,
    CROPS_DATA,
    ANIMALS_DATA,
    FERTILIZER_DATA,
    LAND_EXPANSION_DATA,
)


# ==============================================================================
# Crop Domain Models & Specific Plant Functionality
# ==============================================================================

@dataclass(frozen=True)
class CropConfig:
    """Configuration and behavioral analytics for a crop type."""
    name: str
    yield_type: YieldType
    seed_cost: int
    base_market_price: int
    time_to_first_yield: int
    time_to_max_yield: int
    first_yield_day: int
    max_yield_day: int
    interval: int
    max_yield: int
    unfertilized_max_yield: int
    fertilized_max_yield: int
    action_cost: int
    yield_per_tile_per_day: float
    subsequent_yields: str
    ongoing: bool

    @property
    def bonus_window(self) -> Tuple[int, int]:
        """
        Returns (start_day, end_day) for the bonus watering window.
        For one-time crops, window starts at ceil(max_yield_day / 2) = (max_yield_day + 1) // 2.
        For Melon, bonus window is ages 6–12 (though cap of 6 is reached at age 10 unfertilized, 8 fertilized).
        """
        if self.name == "MELON":
            return (6, 12)
        if not self.ongoing:
            return ((self.max_yield_day + 1) // 2, self.max_yield_day)
        return (self.first_yield_day, self.max_yield_day)

    def is_bonus_window(self, age_days: int) -> bool:
        """Checks if a given age (in days) falls within the bonus watering window."""
        start, end = self.bonus_window
        return start <= age_days <= end

    def is_harvestable(self, planted_day: int, current_day: int, yield_units: int = 1) -> bool:
        """Checks if the crop can be harvested on current_day."""
        age_days = current_day - planted_day
        return age_days >= self.first_yield_day and yield_units > 0

    def is_optimal_harvest_age(self, planted_day: int, current_day: int, fertilized: bool = False) -> bool:
        """
        Checks if the crop has reached its peak yield under daily watering:
        - Wheat: age 4
        - Carrot: age 3
        - Melon: age 10 (or age 10 when fertilized, reaching cap at age 8 and first harvestable at age 10)
        - Tomato: age >= 8 (ongoing, harvested as scheduled)
        - Strawberry: age >= 10 (ongoing, harvested as scheduled)
        """
        age_days = current_day - planted_day
        if self.name == "MELON":
            return age_days >= self.first_yield_day
        if self.ongoing:
            return age_days >= self.first_yield_day
        return age_days >= self.max_yield_day

    def needs_watering(self, tile: dict) -> bool:
        """Returns True if the plant tile needs watering today."""
        return not tile.get("watered_today", False)

    def in_danger_of_weed(self, tile: dict) -> bool:
        """
        Returns True if the plant tile is in danger of decaying into a weed
        (i.e. missed 1 day already and has not been watered today).
        """
        return tile.get("consecutive_unwatered", 0) >= 1 and not tile.get("watered_today", False)

    def decay_start_step(self, planted_day: int, turns_per_day: int = 24) -> int:
        """Returns the step at which yield decay begins."""
        if not self.ongoing:
            return (planted_day + self.max_yield_day + 1) * turns_per_day
        last_prod_age = self.first_yield_day + (self.max_yield - 1) * self.interval
        return (planted_day + last_prod_age + 2) * turns_per_day

    def is_decaying(self, tile: dict, current_step: int) -> bool:
        """Checks if the plant has entered its decay phase."""
        mls = tile.get("max_lifespan_step", -1)
        return mls >= 0 and current_step >= mls

    def turns_until_decay(self, tile: dict, current_step: int) -> Optional[int]:
        """Returns the number of turns until plant decay begins, or None if ongoing and not capped."""
        mls = tile.get("max_lifespan_step", -1)
        if mls < 0:
            return None
        return max(0, mls - current_step)

    def simulate_decay(self, current_yield: int, max_lifespan_step: int, current_step: int) -> Tuple[int, bool]:
        """
        Simulates step decay: yield drops by 1 every other step once max_lifespan_step is reached.
        Returns (remaining_yield, is_weed).
        """
        if max_lifespan_step < 0 or current_step < max_lifespan_step:
            return (current_yield, False)
        decay_ticks = (current_step - max_lifespan_step) // 2 + 1
        remaining = max(0, current_yield - decay_ticks)
        return (remaining, remaining <= 0)

    def accumulated_yield_units(
        self,
        planted_day: int,
        current_day: int,
        watered_days: Set[int],
        fertilized_until_day: int = -1,
    ) -> int:
        """
        Calculates the internal yield units accumulated on the plant tile up to current_day.
        """
        if not self.ongoing:
            units = 1
            start, end = self.bonus_window
            for day in range(planted_day + start, planted_day + end + 1):
                if day > current_day:
                    break
                if day in watered_days:
                    bonus = 2 if fertilized_until_day >= day else 1
                    units = min(self.max_yield, units + bonus)
            return units
        else:
            units = 0
            production_count = 0
            for day in range(planted_day + self.first_yield_day, current_day + 1):
                days_since_first = day - (planted_day + self.first_yield_day)
                if days_since_first % self.interval == 0:
                    production_count += 1
                    if production_count <= self.max_yield:
                        was_watered = day in watered_days
                        fertilized = was_watered and fertilized_until_day >= day
                        units = min(self.max_yield, units + (2 if fertilized else 1))
            return units

    def calculate_yield(
        self,
        planted_day: int,
        current_day: int,
        watered_days: Set[int],
        fertilized_until_day: int = -1,
    ) -> int:
        """
        Simulates the harvestable yield on current_day (0 if immature before first_yield_day).
        """
        age_days = current_day - planted_day
        if age_days < self.first_yield_day:
            return 0
        return self.accumulated_yield_units(planted_day, current_day, watered_days, fertilized_until_day)


class WheatCrop(CropConfig):
    """
    Wheat: One-time yield crop.
    - Seed Cost: 10
    - Base Market Price: 25
    - Time to First Yield: 2 days
    - Time to Max Yield: 4 days
    - Max Yield: 6 (4 unfertilized)
    - Action Cost: 1
    - Yield / tile / day: 0.80
    - Bonus window: ages 2–4
    """
    def __init__(self):
        super().__init__(
            name=Plants.WHEAT,
            yield_type=YieldType.ONE_TIME,
            seed_cost=10,
            base_market_price=25,
            time_to_first_yield=2,
            time_to_max_yield=4,
            first_yield_day=2,
            max_yield_day=4,
            interval=0,
            max_yield=6,
            unfertilized_max_yield=4,
            fertilized_max_yield=6,
            action_cost=1,
            yield_per_tile_per_day=0.80,
            subsequent_yields="none",
            ongoing=False,
        )


class CarrotCrop(CropConfig):
    """
    Carrot: One-time yield crop.
    - Seed Cost: 20
    - Base Market Price: 35
    - Time to First Yield: 2 days
    - Time to Max Yield: 3 days
    - Max Yield: 4 (3 unfertilized)
    - Action Cost: 1
    - Yield / tile / day: 0.75
    - Bonus window: ages 2–3
    """
    def __init__(self):
        super().__init__(
            name=Plants.CARROT,
            yield_type=YieldType.ONE_TIME,
            seed_cost=20,
            base_market_price=35,
            time_to_first_yield=2,
            time_to_max_yield=3,
            first_yield_day=2,
            max_yield_day=3,
            interval=0,
            max_yield=4,
            unfertilized_max_yield=3,
            fertilized_max_yield=4,
            action_cost=1,
            yield_per_tile_per_day=0.75,
            subsequent_yields="none",
            ongoing=False,
        )


class TomatoCrop(CropConfig):
    """
    Tomato: Ongoing yield crop (capped at 4 scheduled yields).
    - Seed Cost: 50
    - Base Market Price: 60
    - Time to First Yield: 8 days
    - Time to Max Yield: 11 days
    - Subsequent Yields: every day ×4 (ages 8, 9, 10, 11)
    - Max Yield: 4 total scheduled yields
    - Action Cost: 1
    - Yield / tile / day: 0.33
    - Decays into weed starting 1 day after 4th production (age 12).
    """
    def __init__(self):
        super().__init__(
            name=Plants.TOMATO,
            yield_type=YieldType.ONGOING,
            seed_cost=50,
            base_market_price=60,
            time_to_first_yield=8,
            time_to_max_yield=11,
            first_yield_day=8,
            max_yield_day=11,
            interval=1,
            max_yield=4,
            unfertilized_max_yield=4,
            fertilized_max_yield=4,
            action_cost=1,
            yield_per_tile_per_day=0.33,
            subsequent_yields="every day ×4",
            ongoing=True,
        )


class StrawberryCrop(CropConfig):
    """
    Strawberry: Ongoing yield crop (capped at 4 scheduled yields).
    - Seed Cost: 100
    - Base Market Price: 120
    - Time to First Yield: 10 days
    - Time to Max Yield: 16 days
    - Subsequent Yields: every other day ×4 (ages 10, 12, 14, 16)
    - Max Yield: 4 total scheduled yields
    - Action Cost: 1
    - Yield / tile / day: 0.24
    - Decays into weed starting 1 day after 4th production (age 17).
    """
    def __init__(self):
        super().__init__(
            name=Plants.STRAWBERRY,
            yield_type=YieldType.ONGOING,
            seed_cost=100,
            base_market_price=120,
            time_to_first_yield=10,
            time_to_max_yield=16,
            first_yield_day=10,
            max_yield_day=16,
            interval=2,
            max_yield=4,
            unfertilized_max_yield=4,
            fertilized_max_yield=4,
            action_cost=1,
            yield_per_tile_per_day=0.24,
            subsequent_yields="every other day ×4",
            ongoing=True,
        )


class MelonCrop(CropConfig):
    """
    Melon: High-value one-time yield crop.
    - Seed Cost: 80
    - Base Market Price: 250
    - Time to First Yield: 10 days
    - Time to Max Yield: 10 days (unfertilized peaks at age 10; fertilized reaches cap 6 at age 8)
    - Max Yield: 6
    - Action Cost: 1
    - Yield / tile / day: 0.55
    - Bonus window: ages 6–12 (ages 11–12 add nothing once capped at 6)
    """
    def __init__(self):
        super().__init__(
            name=Plants.MELON,
            yield_type=YieldType.ONE_TIME,
            seed_cost=80,
            base_market_price=250,
            time_to_first_yield=10,
            time_to_max_yield=10,
            first_yield_day=10,
            max_yield_day=12,  # engine max_yield_day is 12 (bonus window 6..12)
            interval=0,
            max_yield=6,
            unfertilized_max_yield=6,
            fertilized_max_yield=6,
            action_cost=1,
            yield_per_tile_per_day=0.55,
            subsequent_yields="none",
            ongoing=False,
        )


# ==============================================================================
# Animal Domain Models & Specific Livestock Functionality
# ==============================================================================

@dataclass(frozen=True)
class AnimalConfig:
    """Configuration and behavioral analytics for an animal type."""
    name: str
    cost: int
    base_market_price: int
    structure: str
    product: str
    first_yield_day: int
    interval: int
    max_held: int
    action_cost: str
    yield_per_tile_per_day: float
    subsequent_yields: str
    feed_item: str = "WHEAT"

    def needs_feed(self, tile: dict) -> bool:
        """Returns True if the animal has not been fed wheat today."""
        return not tile.get("fed_today", False)

    def in_danger_of_escape(self, tile: dict) -> bool:
        """Returns True if the animal missed 1 day of feeding and hasn't been fed today."""
        return tile.get("consecutive_unfed", 0) >= 1 and not tile.get("fed_today", False)

    def is_production_day(self, placed_day: int, current_day: int) -> bool:
        """Checks if current_day is a scheduled production day for the animal."""
        days_since_first = current_day - placed_day - self.first_yield_day
        return days_since_first >= 0 and days_since_first % self.interval == 0

    def needs_care(self, tile: dict) -> bool:
        """Returns True if the animal has not been cared for today."""
        return not tile.get("cared_today", False)

    def has_fertilizer(self, tile: dict) -> bool:
        """Returns True if the animal has uncollected fertilizer available."""
        return tile.get("fertilizer_available", False)

    def is_harvestable(self, tile: dict) -> bool:
        """Returns True if the animal has unharvested product on the tile."""
        return tile.get("yield_units", 0) > 0


class GooseAnimal(AnimalConfig):
    """
    Goose / Egg:
    - Cost: 300
    - Base Market Price: 50
    - Structure: COOP
    - Product: EGG
    - Time to First Yield: 4 days
    - Interval: 1 (every day, indefinitely)
    - Max Held: 4
    - Action Cost: 1 + 1 (build coop)
    - Yield / tile / day: 1.00
    - Feed: WHEAT
    """
    def __init__(self):
        super().__init__(
            name=Animals.GOOSE,
            cost=300,
            base_market_price=50,
            structure=Structures.COOP,
            product=Products.EGG,
            first_yield_day=4,
            interval=1,
            max_held=4,
            action_cost="1 + 1 (build coop)",
            yield_per_tile_per_day=1.00,
            subsequent_yields="every day, indefinitely",
            feed_item=Products.WHEAT,
        )


class CowAnimal(AnimalConfig):
    """
    Cow / Milk:
    - Cost: 400
    - Base Market Price: 160
    - Structure: PASTURE
    - Product: MILK
    - Time to First Yield: 8 days
    - Interval: 2 (every two days, indefinitely)
    - Max Held: 6
    - Action Cost: 1 + 1 (build pasture)
    - Yield / tile / day: 0.50
    - Feed: WHEAT
    """
    def __init__(self):
        super().__init__(
            name=Animals.COW,
            cost=400,
            base_market_price=160,
            structure=Structures.PASTURE,
            product=Products.MILK,
            first_yield_day=8,
            interval=2,
            max_held=6,
            action_cost="1 + 1 (build pasture)",
            yield_per_tile_per_day=0.50,
            subsequent_yields="every two days, indefinitely",
            feed_item=Products.WHEAT,
        )


class SheepAnimal(AnimalConfig):
    """
    Sheep / Wool:
    - Cost: 500
    - Base Market Price: 200
    - Structure: PASTURE
    - Product: WOOL
    - Time to First Yield: 6 days
    - Interval: 3 (every three days, indefinitely)
    - Max Held: 6
    - Action Cost: 1 + 1 (build pasture)
    - Yield / tile / day: 0.33
    - Feed: WHEAT
    """
    def __init__(self):
        super().__init__(
            name=Animals.SHEEP,
            cost=500,
            base_market_price=200,
            structure=Structures.PASTURE,
            product=Products.WOOL,
            first_yield_day=6,
            interval=3,
            max_held=6,
            action_cost="1 + 1 (build pasture)",
            yield_per_tile_per_day=0.33,
            subsequent_yields="every three days, indefinitely",
            feed_item=Products.WHEAT,
        )


@dataclass(frozen=True)
class FertilizerConfig:
    """Fertilizer details: cost 100, active for 3 days, doubles watering/growth bonus."""
    cost: int = 100
    action_cost: int = 1
    duration_days: int = 3
    base_market_price: int = 100


# Concrete singletons / registries
Wheat = WheatCrop()
Carrot = CarrotCrop()
Tomato = TomatoCrop()
Strawberry = StrawberryCrop()
Melon = MelonCrop()

Goose = GooseAnimal()
Cow = CowAnimal()
Sheep = SheepAnimal()

Fertilizer = FertilizerConfig()

CROPS: Dict[str, CropConfig] = {
    "WHEAT": Wheat,
    "CARROT": Carrot,
    "TOMATO": Tomato,
    "STRAWBERRY": Strawberry,
    "MELON": Melon,
}

ANIMALS: Dict[str, AnimalConfig] = {
    "GOOSE": Goose,
    "COW": Cow,
    "SHEEP": Sheep,
}


def get_crop(name: Union[str, Plants]) -> CropConfig:
    """Retrieves the CropConfig object for a given crop name."""
    key = str(name).upper()
    if key not in CROPS:
        raise KeyError(f"Unknown crop: {name}. Valid crops are: {list(CROPS.keys())}")
    return CROPS[key]


def get_animal(name: Union[str, Animals]) -> AnimalConfig:
    """Retrieves the AnimalConfig object for a given animal name."""
    key = str(name).upper()
    if key not in ANIMALS:
        raise KeyError(f"Unknown animal: {name}. Valid animals are: {list(ANIMALS.keys())}")
    return ANIMALS[key]


# ==============================================================================
# Actions Definition, Action Factory, and Feasibility Predicates
# ==============================================================================

class Actions:
    """
    Standard Action Constants, Action Creators, and Rule Predicates for Kaggriculture.
    """
    # Directions / Movement
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"
    PASS = "PASS"

    # Crop / Field Actions
    WATER = "WATER"
    HARVEST = "HARVEST"
    FERTILIZE = "FERTILIZE"
    DIG = "DIG"
    PLANT = "PLANT"

    # Animal / Structure Actions
    BUILD_COOP = "BUILD_COOP"
    BUILD_PASTURE = "BUILD_PASTURE"
    FEED = "FEED"
    COLLECT_FERTILIZER = "COLLECT_FERTILIZER"
    CARE = "CARE"

    # Shed / Inventory Actions
    DROP = "DROP"
    PICKUP = "PICKUP"
    PLACE = "PLACE"

    # Market Actions
    BUY_SEED = "BUY_SEED"
    BUY_ANIMAL = "BUY_ANIMAL"
    BUY_PRODUCT = "BUY_PRODUCT"
    SELL = "SELL"
    HIRE = "HIRE"
    BUY_LAND = "BUY_LAND"

    # Engine Constants
    MAX_MARKET_ORDERS_PER_TURN = 10
    LAND_EXPANSION_ORDER = ["NE", "SW", "SE"]
    LAND_EXPANSION_PRICES = [1000, 2000, 4000]

    # Registered Crop & Animal Helpers
    CROPS = CROPS
    ANIMALS = ANIMALS
    Wheat = Wheat
    Carrot = Carrot
    Tomato = Tomato
    Strawberry = Strawberry
    Melon = Melon
    Goose = Goose
    Cow = Cow
    Sheep = Sheep
    Fertilizer = Fertilizer

    # --------------------------------------------------------------------------
    # Farmer / Hand Unit Action Creators
    # --------------------------------------------------------------------------

    @staticmethod
    def move(direction: str) -> List[str]:
        return [direction.upper()]

    @staticmethod
    def north() -> List[str]:
        return ["NORTH"]

    @staticmethod
    def south() -> List[str]:
        return ["SOUTH"]

    @staticmethod
    def east() -> List[str]:
        return ["EAST"]

    @staticmethod
    def west() -> List[str]:
        return ["WEST"]

    @staticmethod
    def pass_action() -> List[str]:
        return ["PASS"]

    @staticmethod
    def plant(crop_name: Union[str, Plants]) -> List[str]:
        return ["PLANT", str(crop_name).upper()]

    @staticmethod
    def plant_wheat() -> List[str]:
        return ["PLANT", "WHEAT"]

    @staticmethod
    def plant_carrot() -> List[str]:
        return ["PLANT", "CARROT"]

    @staticmethod
    def plant_tomato() -> List[str]:
        return ["PLANT", "TOMATO"]

    @staticmethod
    def plant_strawberry() -> List[str]:
        return ["PLANT", "STRAWBERRY"]

    @staticmethod
    def plant_melon() -> List[str]:
        return ["PLANT", "MELON"]

    @staticmethod
    def water() -> List[str]:
        return ["WATER"]

    @staticmethod
    def harvest() -> List[str]:
        return ["HARVEST"]

    @staticmethod
    def fertilize() -> List[str]:
        return ["FERTILIZE"]

    @staticmethod
    def dig() -> List[str]:
        return ["DIG"]

    @staticmethod
    def build_coop() -> List[str]:
        return ["BUILD_COOP"]

    @staticmethod
    def build_pasture() -> List[str]:
        return ["BUILD_PASTURE"]

    @staticmethod
    def feed() -> List[str]:
        return ["FEED"]

    @staticmethod
    def collect_fertilizer() -> List[str]:
        return ["COLLECT_FERTILIZER"]

    @staticmethod
    def care() -> List[str]:
        return ["CARE"]

    @staticmethod
    def drop() -> List[str]:
        return ["DROP"]

    @staticmethod
    def pickup(item_name: str, quantity: int = 1) -> List[Any]:
        return ["PICKUP", item_name.upper(), int(quantity)]

    @staticmethod
    def place(item_name: str, quantity: int = 1) -> List[Any]:
        return ["PLACE", item_name.upper(), int(quantity)]

    # --------------------------------------------------------------------------
    # Market Action Creators
    # --------------------------------------------------------------------------

    @staticmethod
    def buy_seed(crop_name: Union[str, Plants], quantity: int = 1) -> List[Any]:
        return ["BUY_SEED", str(crop_name).upper(), int(quantity)]

    @staticmethod
    def buy_animal(animal_name: Union[str, Animals], quantity: int = 1) -> List[Any]:
        return ["BUY_ANIMAL", str(animal_name).upper(), int(quantity)]

    @staticmethod
    def buy_product(item_name: Union[str, Products], quantity: int = 1) -> List[Any]:
        return ["BUY_PRODUCT", str(item_name).upper(), int(quantity)]

    @staticmethod
    def sell(item_name: Union[str, Products], quantity: int = 1) -> List[Any]:
        return ["SELL", str(item_name).upper(), int(quantity)]

    @staticmethod
    def hire() -> List[str]:
        return ["HIRE"]

    @staticmethod
    def buy_land() -> List[str]:
        return ["BUY_LAND"]

    # --------------------------------------------------------------------------
    # Market Calculation & Cost Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def fib(n: int) -> int:
        """Indexed so fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, fib(5)=8, fib(6)=13..."""
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    @staticmethod
    def hire_cost(hires_already_today: int, mult: int = 1) -> int:
        """Returns the cost to hire the next farm hand today."""
        return mult * Actions.fib(hires_already_today)

    @staticmethod
    def land_cost(unlocked_quadrants: Union[List[str], int]) -> Optional[int]:
        """
        Returns the cost to purchase the next 5x5 land quadrant:
        - 1st expansion (NE): $1,000
        - 2nd expansion (SW): $2,000
        - 3rd expansion (SE): $4,000
        """
        count = len(unlocked_quadrants) if isinstance(unlocked_quadrants, list) else int(unlocked_quadrants)
        extra_unlocked = count - 1  # NW is unlocked by default
        if 0 <= extra_unlocked < len(Actions.LAND_EXPANSION_PRICES):
            return Actions.LAND_EXPANSION_PRICES[extra_unlocked]
        return None

    @staticmethod
    def next_quadrant(unlocked_quadrants: List[str]) -> Optional[str]:
        """Returns the next quadrant that will be unlocked (NE -> SW -> SE)."""
        extra_unlocked = len(unlocked_quadrants) - 1
        if 0 <= extra_unlocked < len(Actions.LAND_EXPANSION_ORDER):
            return Actions.LAND_EXPANSION_ORDER[extra_unlocked]
        return None

    # --------------------------------------------------------------------------
    # Map & Quadrant Utilities
    # --------------------------------------------------------------------------

    @staticmethod
    def quadrant_of(x: int, y: int, board_size: int = 10) -> str:
        """Determines which quadrant a grid coordinate belongs to ('NW', 'NE', 'SW', 'SE')."""
        half = board_size // 2
        return ("N" if y < half else "S") + ("W" if x < half else "E")

    @staticmethod
    def get_quadrant_bounds(quadrant: str, board_size: int = 10) -> Tuple[int, int, int, int]:
        """Returns (x_min, x_max, y_min, y_max) for the specified quadrant."""
        half = board_size // 2
        q = quadrant.upper()
        if q == "NW": return (0, half, 0, half)
        if q == "NE": return (half, board_size, 0, half)
        if q == "SW": return (0, half, half, board_size)
        if q == "SE": return (half, board_size, half, board_size)
        raise ValueError(f"Unknown quadrant: {quadrant}")

    @staticmethod
    def is_tile_unlocked(x: int, y: int, unlocked_quadrants: List[str], board_size: int = 10) -> bool:
        """Checks whether the tile at (x, y) belongs to an unlocked quadrant."""
        q = Actions.quadrant_of(x, y, board_size)
        return q in unlocked_quadrants

    @staticmethod
    def shed_access_tiles(board_size: int = 10) -> List[Tuple[int, int]]:
        """Four inner-corner tiles orthogonally adjacent to the central shed, in NWSE order."""
        half = board_size // 2
        return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]

    @staticmethod
    def is_shed_adjacent(pos: Tuple[int, int], board_size: int = 10) -> bool:
        """Returns True if pos is orthogonally adjacent to the central shed."""
        return tuple(pos) in set(Actions.shed_access_tiles(board_size))

    @staticmethod
    def default_spawn(board_size: int = 10) -> Tuple[int, int]:
        """First free shed-access tile in the NW quadrant (default: (4,4) for boardSize=10)."""
        half = board_size // 2
        return (half - 1, half - 1)

    # --------------------------------------------------------------------------
    # Action Feasibility & Rule Predicates
    # --------------------------------------------------------------------------

    @staticmethod
    def can_move(from_pos: Tuple[int, int], direction: str, board_size: int = 10) -> bool:
        """Checks if moving in direction stays within board bounds."""
        fx, fy = from_pos
        moves = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
        if direction.upper() not in moves:
            return False
        dx, dy = moves[direction.upper()]
        nx, ny = fx + dx, fy + dy
        return 0 <= nx < board_size and 0 <= ny < board_size

    @staticmethod
    def can_plant(tile: Any, crop_name: str, available_seeds: Dict[str, int]) -> bool:
        """Checks if a plant action is valid on the current tile."""
        if tile == "LOCKED" or tile is not None:
            return False
        return crop_name.upper() in CROPS and available_seeds.get(crop_name.upper(), 0) > 0

    @staticmethod
    def can_water(tile: Any) -> bool:
        """Checks if watering can be performed on tile."""
        if tile == "LOCKED" or not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            return False
        return not tile.get("watered_today", False)

    @staticmethod
    def can_harvest(tile: Any, current_day: int) -> bool:
        """Checks if harvesting can be performed on tile."""
        if tile == "LOCKED" or not isinstance(tile, dict):
            return False
        if tile.get("yield_units", 0) <= 0:
            return False
        if tile.get("kind") == "PLANT":
            crop_cfg = CROPS.get(tile.get("crop"))
            if not crop_cfg:
                return False
            return current_day - tile.get("planted_day", 0) >= crop_cfg.first_yield_day
        if "animal" in tile:
            return True
        return False

    @staticmethod
    def can_fertilize(tile: Any, unit_inventory: Dict[str, int]) -> bool:
        """Checks if fertilizing can be performed on tile."""
        if tile == "LOCKED" or not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            return False
        return unit_inventory.get("FERTILIZER", 0) > 0

    @staticmethod
    def can_dig(tile: Any) -> bool:
        """Checks if digging can be performed on tile (plants, weeds, or empty coops/pastures)."""
        if tile == "LOCKED" or tile is None:
            return False
        # Cannot dig structure with animal on it
        if isinstance(tile, dict) and "animal" in tile:
            return False
        return True

    @staticmethod
    def can_build_coop(tile: Any) -> bool:
        """Checks if BUILD_COOP can be executed on tile."""
        return tile is None

    @staticmethod
    def can_build_pasture(tile: Any) -> bool:
        """Checks if BUILD_PASTURE can be executed on tile."""
        return tile is None

    @staticmethod
    def can_place_animal(tile: Any, animal_name: str, unit_inventory: Dict[str, int]) -> bool:
        """Checks if PLACE animal on a structure is valid."""
        animal_name = animal_name.upper()
        if animal_name not in ANIMALS:
            return False
        if unit_inventory.get(animal_name, 0) <= 0:
            return False
        if not isinstance(tile, dict):
            return False
        matching_structure = ANIMALS[animal_name].structure
        return tile.get("kind") == matching_structure and "animal" not in tile

    @staticmethod
    def can_feed(tile: Any, unit_inventory: Dict[str, int]) -> bool:
        """Checks if FEED animal with wheat is valid."""
        if tile == "LOCKED" or not isinstance(tile, dict) or "animal" not in tile:
            return False
        return not tile.get("fed_today", False) and unit_inventory.get("WHEAT", 0) > 0

    @staticmethod
    def can_care(tile: Any) -> bool:
        """Checks if CARE for animal is valid."""
        if tile == "LOCKED" or not isinstance(tile, dict) or "animal" not in tile:
            return False
        return not tile.get("cared_today", False)

    @staticmethod
    def can_collect_fertilizer(tile: Any) -> bool:
        """Checks if COLLECT_FERTILIZER from animal is valid."""
        if tile == "LOCKED" or not isinstance(tile, dict) or "animal" not in tile:
            return False
        return tile.get("fertilizer_available", False)

    @staticmethod
    def can_drop(pos: Tuple[int, int], board_size: int = 10) -> bool:
        """Checks if DROP into shed is valid from pos."""
        return Actions.is_shed_adjacent(pos, board_size)

    @staticmethod
    def can_pickup(pos: Tuple[int, int], item_name: str, shed: Dict[str, int], board_size: int = 10) -> bool:
        """Checks if PICKUP from shed is valid from pos."""
        if not Actions.is_shed_adjacent(pos, board_size):
            return False
        return shed.get(item_name.upper(), 0) > 0

    @staticmethod
    def can_buy_land(money: float, unlocked_quadrants: List[str]) -> bool:
        """Checks if the player has enough money to buy the next land quadrant."""
        cost = Actions.land_cost(unlocked_quadrants)
        return cost is not None and money >= cost

    @staticmethod
    def can_hire(money: float, hires_already_today: int, mult: int = 1) -> bool:
        """Checks if the player has enough money to hire another farm hand today."""
        cost = Actions.hire_cost(hires_already_today, mult)
        return money >= cost

    @staticmethod
    def can_buy_seed(money: float, crop_name: str, quantity: int = 1) -> bool:
        """Checks if the player has enough money to buy seeds."""
        crop_cfg = CROPS.get(crop_name.upper())
        if not crop_cfg or quantity <= 0:
            return False
        return money >= crop_cfg.seed_cost * quantity

    @staticmethod
    def can_buy_animal(money: float, animal_name: str, shed: Dict[str, int], shed_capacity: int = 100, quantity: int = 1) -> bool:
        """Checks if the player can purchase an animal (money + shed capacity)."""
        animal_cfg = ANIMALS.get(animal_name.upper())
        if not animal_cfg or quantity <= 0:
            return False
        if money < animal_cfg.cost * quantity:
            return False
        current_shed_items = sum(shed.values())
        return current_shed_items + quantity <= shed_capacity

    @staticmethod
    def can_buy_product(money: float, item_name: str, current_price: int, shed: Dict[str, int], shed_capacity: int = 100, quantity: int = 1) -> bool:
        """Checks if the player can buy WHEAT or FERTILIZER from market."""
        if item_name.upper() not in ("WHEAT", "FERTILIZER") or quantity <= 0:
            return False
        if money < current_price * quantity:
            return False
        current_shed_items = sum(shed.values())
        return current_shed_items + quantity <= shed_capacity

    @staticmethod
    def can_sell(item_name: str, shed: Dict[str, int], quantity: int = 1) -> bool:
        """Checks if the player has enough product in shed to sell."""
        if quantity <= 0:
            return False
        return shed.get(item_name.upper(), 0) >= quantity

    # --------------------------------------------------------------------------
    # End-Of-Day Lifecycle Simulators
    # --------------------------------------------------------------------------

    @staticmethod
    def simulate_shed_drop(
        shed: Dict[str, int],
        inventories: List[Dict[str, int]],
        capacity: int = 100,
    ) -> Tuple[Dict[str, int], List[Dict[str, int]], int]:
        """
        Simulates end-of-day automatic inventory drop into the shed.
        Items beyond capacity are discarded. Seeds are not dropped.
        Returns (new_shed, new_inventories, total_discarded).
        """
        new_shed = dict(shed)
        new_inventories = [dict(inv) for inv in inventories]
        total_discarded = 0

        for inv in new_inventories:
            for item, n in list(inv.items()):
                if n <= 0:
                    del inv[item]
                    continue
                current_total = sum(new_shed.values())
                room = max(0, capacity - current_total)
                take = min(n, room)
                if take > 0:
                    new_shed[item] = new_shed.get(item, 0) + take
                discarded = n - take
                total_discarded += discarded
                del inv[item]

        return (new_shed, new_inventories, total_discarded)

    @staticmethod
    def simulate_weed_spawns(
        farm_tiles: List[List[Any]],
        weed_chance: float = 0.005,
        rng: Optional[random.Random] = None,
    ) -> List[List[Any]]:
        """Simulates random weed spawning on empty unlocked tiles (tile is None)."""
        rng = rng or random.Random()
        new_tiles = [list(row) for row in farm_tiles]
        board_size = len(new_tiles)
        for y in range(board_size):
            for x in range(board_size):
                if new_tiles[y][x] is None and rng.random() < weed_chance:
                    new_tiles[y][x] = {"kind": "WEED"}
        return new_tiles

    @staticmethod
    def simulate_end_of_day_plant(
        tile: dict,
        was_watered: bool,
        current_day: int,
        turns_per_day: int = 24,
    ) -> dict:
        """
        Simulates the end-of-day refresh on a plant tile:
        - consecutive_unwatered updates (2 missed refreshes -> WEED)
        - ongoing crops scheduled yield increment (doubled if fertilized and watered)
        """
        next_tile = dict(tile)
        if was_watered:
            next_tile["consecutive_unwatered"] = 0
        else:
            next_tile["consecutive_unwatered"] = next_tile.get("consecutive_unwatered", 0) + 1

        next_tile["watered_today"] = False

        if next_tile["consecutive_unwatered"] >= 2:
            return {"kind": "WEED"}

        crop_name = next_tile.get("crop")
        crop_cfg = CROPS.get(crop_name)
        if not crop_cfg or not crop_cfg.ongoing:
            return next_tile

        next_day = current_day + 1
        days_since_first = next_day - next_tile["planted_day"] - crop_cfg.first_yield_day
        if days_since_first >= 0 and days_since_first % crop_cfg.interval == 0:
            prod_count = days_since_first // crop_cfg.interval + 1
            if prod_count <= crop_cfg.max_yield:
                fertilized = was_watered and next_tile.get("fertilized_until_day", -1) >= current_day
                add_units = 2 if fertilized else 1
                next_tile["yield_units"] = min(crop_cfg.max_yield, next_tile.get("yield_units", 0) + add_units)
                if prod_count == crop_cfg.max_yield:
                    next_tile["max_lifespan_step"] = (next_day + 1) * turns_per_day

        return next_tile

    @staticmethod
    def simulate_end_of_day_animal(
        tile: dict,
        was_fed: bool,
        was_cared: bool,
        current_day: int,
    ) -> dict:
        """
        Simulates the end-of-day refresh on an animal tile:
        - consecutive_unfed updates (2 missed refreshes -> animal escapes)
        - scheduled yield and care bonus banking
        - fertilizer production
        """
        next_tile = dict(tile)
        anim_name = next_tile.get("animal")
        anim_cfg = ANIMALS.get(anim_name)
        if not anim_cfg:
            return next_tile

        if was_fed:
            next_tile["consecutive_unfed"] = 0
        else:
            next_tile["consecutive_unfed"] = next_tile.get("consecutive_unfed", 0) + 1

        if next_tile["consecutive_unfed"] >= 2:
            # Animal escapes, structure remains
            return {"kind": anim_cfg.structure}

        next_day = current_day + 1
        days_since_first = next_day - next_tile["placed_day"] - anim_cfg.first_yield_day
        if days_since_first >= 0 and days_since_first % anim_cfg.interval == 0:
            base = 1
            care_bonus = next_tile.pop("pending_care_bonus", 0) if was_fed else 0
            next_tile["yield_units"] = min(anim_cfg.max_held, next_tile.get("yield_units", 0) + base + care_bonus)
            next_tile["pending_care_bonus"] = 0

        if was_cared and was_fed:
            next_tile["pending_care_bonus"] = next_tile.get("pending_care_bonus", 0) + 1

        next_tile["fertilizer_available"] = True
        next_tile["fed_today"] = False
        next_tile["cared_today"] = False
        return next_tile

    # --------------------------------------------------------------------------
    # Convenience Plant/Animal Lookup
    # --------------------------------------------------------------------------

    @staticmethod
    def get_crop(crop_name: Union[str, Plants]) -> CropConfig:
        return get_crop(crop_name)

    @staticmethod
    def get_animal(animal_name: Union[str, Animals]) -> AnimalConfig:
        return get_animal(animal_name)