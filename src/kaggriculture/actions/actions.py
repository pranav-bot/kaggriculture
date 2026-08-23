from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from kaggriculture.env.items import (
    Plants,
    Animals,
    Products,
    Structures,
    YieldType,
    CROPS_DATA,
    ANIMALS_DATA,
    FERTILIZER_DATA,
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
    # Action Feasibility & Rule Predicates
    # --------------------------------------------------------------------------

    @staticmethod
    def is_shed_adjacent(pos: Tuple[int, int], board_size: int = 10) -> bool:
        """Returns True if pos is orthogonally adjacent to the central shed."""
        half = board_size // 2
        return tuple(pos) in {
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        }

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

    # --------------------------------------------------------------------------
    # Convenience Plant/Animal Lookup
    # --------------------------------------------------------------------------

    @staticmethod
    def get_crop(crop_name: Union[str, Plants]) -> CropConfig:
        return get_crop(crop_name)

    @staticmethod
    def get_animal(animal_name: Union[str, Animals]) -> AnimalConfig:
        return get_animal(animal_name)