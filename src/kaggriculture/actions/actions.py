"""
Crop/Animal domain models, action factory, feasibility predicates, and lifecycle simulators.
"""
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from kaggriculture.env.items import (
    Plants, Animals, Products, Structures, Quadrants, YieldType,
    CROPS_DATA, ANIMALS_DATA, FERTILIZER_DATA, LAND_EXPANSION_DATA,
    SHOPS, TOWN_CENTER_PRODUCTS, PRODUCTS_LIST,
    MAX_SHOP_INSTANCES, TOWN_SHOP_UNLOCK_INTERVAL, TOWN_SHOP_SELL_INTERVAL,
    TOWN_CENTER_SELL_INTERVAL,
    MARKET_I0, PRICE_FLOOR, HINGE_GAIN, MARKET_PARAMS,
    EPISODE_STEPS, BOARD_SIZE, STARTING_MONEY, MAX_MARKET_ORDERS_PER_TURN,
    TURNS_PER_DAY, SHED_CAPACITY, WEED_SPAWN_CHANCE, FARM_HAND_COST_MULT,
    shape_func, resolve_market_params, market_price,
    get_quadrant_bounds, quadrant_of, shed_access_tiles, is_shed_adjacent,
    spawn_hand_position,
    calculate_shop_turn_consumption, calculate_town_daily_consumption,
)


# ==============================================================================
# Crop Domain Model
# ==============================================================================

@dataclass(frozen=True)
class CropConfig:
    """Configuration for a crop type with yield analytics."""
    name: str
    yield_type: YieldType
    seed_cost: int
    base_market_price: int
    time_to_first_yield: int  # Alias for first_yield_day
    time_to_max_yield: int    # Age at which yield stops increasing under daily watering
    first_yield_day: int      # First day a HARVEST can yield produce
    max_yield_day: int        # Last day of bonus window (for decay calc on one-time crops)
    interval: int             # 0 for one-time, >0 for ongoing production interval
    max_yield: int            # Max yield units (fertilized cap for one-time, prod count for ongoing)
    unfertilized_max_yield: int
    fertilized_max_yield: int
    action_cost: int
    yield_per_tile_per_day: float
    subsequent_yields: str
    ongoing: bool

    @property
    def bonus_window(self) -> Tuple[int, int]:
        """(start_day, end_day) for the watering bonus window.
        Melon: ages 6–12. Wheat: ages 3–4. Carrot: ages 2–3.
        For one-time: start = ceil(time_to_max_yield / 2).
        """
        if self.name == "MELON":
            return (6, 12)
        if not self.ongoing:
            return ((self.time_to_max_yield + 1) // 2, self.time_to_max_yield)
        return (self.first_yield_day, self.max_yield_day)

    def is_bonus_window(self, age_days: int) -> bool:
        s, e = self.bonus_window
        return s <= age_days <= e

    def is_harvestable(self, planted_day: int, current_day: int, yield_units: int = 1) -> bool:
        return (current_day - planted_day) >= self.first_yield_day and yield_units > 0

    def is_optimal_harvest_age(self, planted_day: int, current_day: int) -> bool:
        age = current_day - planted_day
        return age >= self.first_yield_day if self.ongoing else age >= self.time_to_max_yield

    def needs_watering(self, tile: dict) -> bool:
        return not tile.get("watered_today", False)

    def in_danger_of_weed(self, tile: dict) -> bool:
        """True if missed 1+ days and not watered today → becomes weed at end-of-day."""
        return tile.get("consecutive_unwatered", 0) >= 1 and not tile.get("watered_today", False)

    def decay_start_step(self, planted_day: int, turns_per_day: int = TURNS_PER_DAY) -> int:
        """Step at which yield decay begins.
        One-time: 1 day after time_to_max_yield.
        Ongoing: 1 day after last scheduled production fires.
        """
        if not self.ongoing:
            return (planted_day + self.time_to_max_yield + 1) * turns_per_day
        last_prod_age = self.first_yield_day + (self.max_yield - 1) * self.interval
        return (planted_day + last_prod_age + 1) * turns_per_day

    def simulate_decay(self, current_yield: int, max_lifespan_step: int, current_step: int) -> Tuple[int, bool]:
        """Yield drops by 1 every other step once decay starts. Returns (remaining, is_weed)."""
        if max_lifespan_step < 0 or current_step < max_lifespan_step:
            return (current_yield, False)
        decay_ticks = (current_step - max_lifespan_step) // 2 + 1
        remaining = max(0, current_yield - decay_ticks)
        return (remaining, remaining <= 0)

    def accumulated_yield_units(
        self, planted_day: int, current_day: int,
        watered_days: Set[int], fertilized_until_day: int = -1,
    ) -> int:
        """Calculate yield units accumulated through the bonus/production window."""
        if not self.ongoing:
            # One-time: base 1 + 1 per watered bonus-window day (2 if fertilized)
            units = 1
            start, end = self.bonus_window
            for day in range(planted_day + start, min(current_day, planted_day + end) + 1):
                if day in watered_days:
                    bonus = 2 if fertilized_until_day >= day else 1
                    units = min(self.max_yield, units + bonus)
            return units
        else:
            # Ongoing: scheduled productions, base 1 each (2 if fertilized+watered)
            units = prod_count = 0
            for day in range(planted_day + self.first_yield_day, current_day + 1):
                if (day - planted_day - self.first_yield_day) % self.interval == 0:
                    prod_count += 1
                    if prod_count <= self.max_yield:
                        fert = (day in watered_days) and fertilized_until_day >= day
                        units = min(self.max_yield, units + (2 if fert else 1))
            return units

    def calculate_yield(
        self, planted_day: int, current_day: int,
        watered_days: Set[int], fertilized_until_day: int = -1,
    ) -> int:
        """Harvestable yield on current_day (0 if immature)."""
        if (current_day - planted_day) < self.first_yield_day:
            return 0
        return self.accumulated_yield_units(planted_day, current_day, watered_days, fertilized_until_day)


# Concrete crop configs
Wheat = CropConfig("WHEAT", YieldType.ONE_TIME, 10, 25, 2, 4, 2, 4, 0, 6, 4, 6, 1, 0.80, "none", False)
Carrot = CropConfig("CARROT", YieldType.ONE_TIME, 20, 35, 2, 3, 2, 3, 0, 4, 3, 4, 1, 0.75, "none", False)
Tomato = CropConfig("TOMATO", YieldType.ONGOING, 50, 60, 8, 11, 8, 11, 1, 4, 4, 4, 1, 0.33, "every day ×4", True)
Strawberry = CropConfig("STRAWBERRY", YieldType.ONGOING, 100, 120, 10, 16, 10, 16, 2, 4, 4, 4, 1, 0.24, "every other day ×4", True)
Melon = CropConfig("MELON", YieldType.ONE_TIME, 80, 250, 10, 10, 10, 12, 0, 6, 6, 6, 1, 0.55, "none", False)

CROPS: Dict[str, CropConfig] = {
    "WHEAT": Wheat, "CARROT": Carrot, "TOMATO": Tomato,
    "STRAWBERRY": Strawberry, "MELON": Melon,
}


# ==============================================================================
# Animal Domain Model
# ==============================================================================

@dataclass(frozen=True)
class AnimalConfig:
    """Configuration for an animal type with care/production analytics."""
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
        return not tile.get("fed_today", False)

    def in_danger_of_escape(self, tile: dict) -> bool:
        return tile.get("consecutive_unfed", 0) >= 1 and not tile.get("fed_today", False)

    def is_production_day(self, placed_day: int, current_day: int) -> bool:
        d = current_day - placed_day - self.first_yield_day
        return d >= 0 and d % self.interval == 0

    def needs_care(self, tile: dict) -> bool:
        return not tile.get("cared_today", False)

    def has_fertilizer(self, tile: dict) -> bool:
        return tile.get("fertilizer_available", False)

    def is_harvestable(self, tile: dict) -> bool:
        return tile.get("yield_units", 0) > 0


Goose = AnimalConfig("GOOSE", 300, 50, Structures.COOP, Products.EGG, 4, 1, 4, "1 + 1 (build coop)", 1.00, "every day, indefinitely", Products.WHEAT)
Cow = AnimalConfig("COW", 400, 160, Structures.PASTURE, Products.MILK, 8, 2, 6, "1 + 1 (build pasture)", 0.50, "every two days, indefinitely", Products.WHEAT)
Sheep = AnimalConfig("SHEEP", 500, 200, Structures.PASTURE, Products.WOOL, 6, 3, 6, "1 + 1 (build pasture)", 0.33, "every three days, indefinitely", Products.WHEAT)

ANIMALS: Dict[str, AnimalConfig] = {"GOOSE": Goose, "COW": Cow, "SHEEP": Sheep}


@dataclass(frozen=True)
class FertilizerConfig:
    """Fertilizer: cost 100, active for 3 days, doubles watering/growth bonus."""
    cost: int = 100
    action_cost: int = 1
    duration_days: int = 3
    base_market_price: int = 100

Fertilizer = FertilizerConfig()


def get_crop(name: Union[str, Plants]) -> CropConfig:
    key = str(name).upper()
    if key not in CROPS:
        raise KeyError(f"Unknown crop: {name}. Valid: {list(CROPS.keys())}")
    return CROPS[key]

def get_animal(name: Union[str, Animals]) -> AnimalConfig:
    key = str(name).upper()
    if key not in ANIMALS:
        raise KeyError(f"Unknown animal: {name}. Valid: {list(ANIMALS.keys())}")
    return ANIMALS[key]


# ==============================================================================
# Actions: Constants, Creators, Feasibility Predicates, Lifecycle Simulators
# ==============================================================================

class Actions:
    """Action constants, creators, rule predicates, pricing, and turn simulators."""

    # --- Directions ---
    NORTH = "NORTH"; SOUTH = "SOUTH"; EAST = "EAST"; WEST = "WEST"; PASS = "PASS"
    _DELTAS = {"NORTH": (0,-1), "SOUTH": (0,1), "EAST": (1,0), "WEST": (-1,0)}

    # --- Tile Actions ---
    WATER = "WATER"; HARVEST = "HARVEST"; FERTILIZE = "FERTILIZE"
    DIG = "DIG"; PLANT = "PLANT"

    # --- Structure / Animal Actions ---
    BUILD_COOP = "BUILD_COOP"; BUILD_PASTURE = "BUILD_PASTURE"
    FEED = "FEED"; COLLECT_FERTILIZER = "COLLECT_FERTILIZER"; CARE = "CARE"

    # --- Shed / Inventory Actions ---
    DROP = "DROP"; PICKUP = "PICKUP"; PLACE = "PLACE"

    # --- Market Actions ---
    BUY_SEED = "BUY_SEED"; BUY_ANIMAL = "BUY_ANIMAL"; BUY_PRODUCT = "BUY_PRODUCT"
    SELL = "SELL"; HIRE = "HIRE"; BUY_LAND = "BUY_LAND"

    # --- Engine Constants ---
    MAX_MARKET_ORDERS_PER_TURN = MAX_MARKET_ORDERS_PER_TURN
    LAND_EXPANSION_ORDER = ["NE", "SW", "SE"]
    LAND_EXPANSION_PRICES = [1000, 2000, 4000]
    EPISODE_STEPS = EPISODE_STEPS
    BOARD_SIZE = BOARD_SIZE
    STARTING_MONEY = STARTING_MONEY
    TURNS_PER_DAY = TURNS_PER_DAY
    SHED_CAPACITY = SHED_CAPACITY
    WEED_SPAWN_CHANCE = WEED_SPAWN_CHANCE

    # --- Town & Market ---
    SHOPS = SHOPS; TOWN_CENTER_PRODUCTS = TOWN_CENTER_PRODUCTS
    PRODUCTS_LIST = PRODUCTS_LIST
    MAX_SHOP_INSTANCES = MAX_SHOP_INSTANCES
    TOWN_SHOP_UNLOCK_INTERVAL = TOWN_SHOP_UNLOCK_INTERVAL
    TOWN_SHOP_SELL_INTERVAL = TOWN_SHOP_SELL_INTERVAL
    TOWN_CENTER_SELL_INTERVAL = TOWN_CENTER_SELL_INTERVAL
    MARKET_I0 = MARKET_I0; PRICE_FLOOR = PRICE_FLOOR
    HINGE_GAIN = HINGE_GAIN; MARKET_PARAMS = MARKET_PARAMS

    # --- Registered Configs ---
    CROPS = CROPS; ANIMALS = ANIMALS
    Wheat = Wheat; Carrot = Carrot; Tomato = Tomato
    Strawberry = Strawberry; Melon = Melon
    Goose = Goose; Cow = Cow; Sheep = Sheep; Fertilizer = Fertilizer

    # -------------------------------------------------------------------------
    # Unit Action Creators
    # -------------------------------------------------------------------------
    @staticmethod
    def move(d: str) -> List[str]: return [d.upper()]
    @staticmethod
    def north() -> List[str]: return ["NORTH"]
    @staticmethod
    def south() -> List[str]: return ["SOUTH"]
    @staticmethod
    def east() -> List[str]: return ["EAST"]
    @staticmethod
    def west() -> List[str]: return ["WEST"]
    @staticmethod
    def pass_action() -> List[str]: return ["PASS"]
    @staticmethod
    def plant(crop: Union[str, Plants]) -> List[str]: return ["PLANT", str(crop).upper()]
    @staticmethod
    def plant_wheat() -> List[str]: return ["PLANT", "WHEAT"]
    @staticmethod
    def plant_carrot() -> List[str]: return ["PLANT", "CARROT"]
    @staticmethod
    def plant_tomato() -> List[str]: return ["PLANT", "TOMATO"]
    @staticmethod
    def plant_strawberry() -> List[str]: return ["PLANT", "STRAWBERRY"]
    @staticmethod
    def plant_melon() -> List[str]: return ["PLANT", "MELON"]
    @staticmethod
    def water() -> List[str]: return ["WATER"]
    @staticmethod
    def harvest() -> List[str]: return ["HARVEST"]
    @staticmethod
    def fertilize() -> List[str]: return ["FERTILIZE"]
    @staticmethod
    def dig() -> List[str]: return ["DIG"]
    @staticmethod
    def build_coop() -> List[str]: return ["BUILD_COOP"]
    @staticmethod
    def build_pasture() -> List[str]: return ["BUILD_PASTURE"]
    @staticmethod
    def feed() -> List[str]: return ["FEED"]
    @staticmethod
    def collect_fertilizer() -> List[str]: return ["COLLECT_FERTILIZER"]
    @staticmethod
    def care() -> List[str]: return ["CARE"]
    @staticmethod
    def drop() -> List[str]: return ["DROP"]
    @staticmethod
    def pickup(item: str, n: int = 1) -> List[Any]: return ["PICKUP", item.upper(), int(n)]
    @staticmethod
    def place(item: str, n: int = 1) -> List[Any]: return ["PLACE", item.upper(), int(n)]

    # -------------------------------------------------------------------------
    # Market Action Creators
    # -------------------------------------------------------------------------
    @staticmethod
    def buy_seed(crop: Union[str, Plants], n: int = 1) -> List[Any]: return ["BUY_SEED", str(crop).upper(), int(n)]
    @staticmethod
    def buy_animal(animal: Union[str, Animals], n: int = 1) -> List[Any]: return ["BUY_ANIMAL", str(animal).upper(), int(n)]
    @staticmethod
    def buy_product(item: Union[str, Products], n: int = 1) -> List[Any]: return ["BUY_PRODUCT", str(item).upper(), int(n)]
    @staticmethod
    def sell(item: Union[str, Products], n: int = 1) -> List[Any]: return ["SELL", str(item).upper(), int(n)]
    @staticmethod
    def hire() -> List[str]: return ["HIRE"]
    @staticmethod
    def buy_land() -> List[str]: return ["BUY_LAND"]

    # -------------------------------------------------------------------------
    # Pricing
    # -------------------------------------------------------------------------
    @staticmethod
    def market_price(item: Union[str, Products], inventory: int, params: Optional[Dict[str, Any]] = None) -> int:
        return market_price(item, inventory, params)

    @staticmethod
    def resolve_market_params(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        return resolve_market_params(overrides)

    @staticmethod
    def shape_func(func: str, x: float, T: Optional[float] = None) -> float:
        return shape_func(func, x, T)

    @staticmethod
    def is_premium_resource(item: Union[str, Products]) -> bool:
        """True if base > $100 (strawberry, melon, milk, wool)."""
        return MARKET_PARAMS.get(str(item).upper(), {}).get("base", 0) > 100

    @staticmethod
    def predict_price_impact(item: Union[str, Products], current_inv: int, units_sold: int, params: Optional[Dict[str, Any]] = None) -> int:
        """Price after selling units_sold into market."""
        return market_price(item, current_inv + units_sold, params)

    # -------------------------------------------------------------------------
    # Economy Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def fib(n: int) -> int:
        """fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ..."""
        a, b = 1, 1
        for _ in range(n): a, b = b, a + b
        return a

    @staticmethod
    def hire_cost(hires_today: int, mult: int = FARM_HAND_COST_MULT) -> int:
        return mult * Actions.fib(hires_today)

    @staticmethod
    def land_cost(unlocked: Union[List[str], int]) -> Optional[int]:
        """Cost for next 5×5 quadrant: $1k, $2k, $4k (None if all bought)."""
        idx = (len(unlocked) if isinstance(unlocked, list) else int(unlocked)) - 1
        return Actions.LAND_EXPANSION_PRICES[idx] if 0 <= idx < 3 else None

    @staticmethod
    def next_quadrant(unlocked: List[str]) -> Optional[str]:
        idx = len(unlocked) - 1
        return Actions.LAND_EXPANSION_ORDER[idx] if 0 <= idx < 3 else None

    @staticmethod
    def calculate_shop_turn_consumption(shop: str) -> Dict[str, int]:
        return calculate_shop_turn_consumption(shop)

    @staticmethod
    def calculate_town_daily_consumption(shops: List[str]) -> Dict[str, int]:
        return calculate_town_daily_consumption(shops)

    @staticmethod
    def get_shop_demands(shop: str) -> List[str]:
        return SHOPS.get(shop.upper(), [])

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------
    @staticmethod
    def quadrant_of(x: int, y: int, bs: int = BOARD_SIZE) -> str:
        return quadrant_of(x, y, bs)

    @staticmethod
    def get_quadrant_bounds(q: str, bs: int = BOARD_SIZE) -> Tuple[int, int, int, int]:
        return get_quadrant_bounds(q, bs)

    @staticmethod
    def is_tile_unlocked(x: int, y: int, unlocked: List[str], bs: int = BOARD_SIZE) -> bool:
        return quadrant_of(x, y, bs) in unlocked

    @staticmethod
    def shed_access_tiles(bs: int = BOARD_SIZE) -> List[Tuple[int, int]]:
        return shed_access_tiles(bs)

    @staticmethod
    def is_shed_adjacent(pos: Tuple[int, int], bs: int = BOARD_SIZE) -> bool:
        return is_shed_adjacent(pos, bs)

    @staticmethod
    def default_spawn(bs: int = BOARD_SIZE) -> Tuple[int, int]:
        h = bs // 2
        return (h - 1, h - 1)

    @staticmethod
    def spawn_hand(farm: Dict[str, Any], bs: int = BOARD_SIZE) -> List[int]:
        return spawn_hand_position(farm, bs)

    # -------------------------------------------------------------------------
    # Feasibility Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def can_move(pos: Tuple[int, int], direction: str, bs: int = BOARD_SIZE) -> bool:
        d = Actions._DELTAS.get(direction.upper())
        if not d: return False
        nx, ny = pos[0] + d[0], pos[1] + d[1]
        return 0 <= nx < bs and 0 <= ny < bs

    @staticmethod
    def can_plant(tile: Any, crop: str, seeds: Dict[str, int]) -> bool:
        """Valid if tile is empty (None) and seeds available. LOCKED or occupied → False."""
        return tile is None and crop.upper() in CROPS and seeds.get(crop.upper(), 0) > 0

    @staticmethod
    def can_water(tile: Any) -> bool:
        """True if tile is an unwatered plant. (Subsequent waterings are no-ops in the engine.)"""
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT": return False
        return not tile.get("watered_today", False)

    @staticmethod
    def can_harvest(tile: Any, current_day: int) -> bool:
        if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0: return False
        if tile.get("kind") == "PLANT":
            cfg = CROPS.get(tile.get("crop"))
            return cfg is not None and (current_day - tile.get("planted_day", 0)) >= cfg.first_yield_day
        return tile.get("animal") is not None  # Animal harvest

    @staticmethod
    def can_fertilize(tile: Any, inv: Dict[str, int]) -> bool:
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT": return False
        return inv.get("FERTILIZER", 0) > 0

    @staticmethod
    def can_dig(tile: Any) -> bool:
        """Can dig plants, weeds, and empty structures. Cannot dig structure with animal."""
        if tile is None or tile == "LOCKED": return False
        if isinstance(tile, dict) and tile.get("animal") is not None: return False
        return True

    @staticmethod
    def can_build_coop(tile: Any) -> bool:
        return tile is None

    @staticmethod
    def can_build_pasture(tile: Any) -> bool:
        return tile is None

    @staticmethod
    def can_place_animal(tile: Any, animal: str, inv: Dict[str, int]) -> bool:
        animal = animal.upper()
        cfg = ANIMALS.get(animal)
        if not cfg or inv.get(animal, 0) <= 0: return False
        return isinstance(tile, dict) and tile.get("kind") == cfg.structure and tile.get("animal") is None

    @staticmethod
    def can_feed(tile: Any, inv: Dict[str, int]) -> bool:
        if not isinstance(tile, dict) or tile.get("animal") is None: return False
        return not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0

    @staticmethod
    def can_care(tile: Any) -> bool:
        if not isinstance(tile, dict) or tile.get("animal") is None: return False
        return not tile.get("cared_today", False)

    @staticmethod
    def can_collect_fertilizer(tile: Any) -> bool:
        if not isinstance(tile, dict) or tile.get("animal") is None: return False
        return tile.get("fertilizer_available", False)

    @staticmethod
    def can_drop(pos: Tuple[int, int], bs: int = BOARD_SIZE) -> bool:
        return is_shed_adjacent(pos, bs)

    @staticmethod
    def can_pickup(pos: Tuple[int, int], item: str, shed: Dict[str, int], bs: int = BOARD_SIZE) -> bool:
        return is_shed_adjacent(pos, bs) and shed.get(item.upper(), 0) > 0

    @staticmethod
    def can_buy_land(money: float, unlocked: List[str]) -> bool:
        c = Actions.land_cost(unlocked)
        return c is not None and money >= c

    @staticmethod
    def can_hire(money: float, hires_today: int, mult: int = FARM_HAND_COST_MULT) -> bool:
        return money >= Actions.hire_cost(hires_today, mult)

    @staticmethod
    def can_buy_seed(money: float, crop: str, qty: int = 1) -> bool:
        cfg = CROPS.get(crop.upper())
        return cfg is not None and qty > 0 and money >= cfg.seed_cost * qty

    @staticmethod
    def can_buy_animal(money: float, animal: str, shed: Dict[str, int], cap: int = SHED_CAPACITY, qty: int = 1) -> bool:
        cfg = ANIMALS.get(animal.upper())
        if not cfg or qty <= 0 or money < cfg.cost * qty: return False
        return sum(shed.values()) + qty <= cap

    @staticmethod
    def can_buy_product(money: float, item: str, price: int, shed: Dict[str, int], cap: int = SHED_CAPACITY, qty: int = 1) -> bool:
        """Only WHEAT and FERTILIZER can be bought from market via BUY_PRODUCT."""
        if item.upper() not in ("WHEAT", "FERTILIZER") or qty <= 0: return False
        return money >= price * qty and sum(shed.values()) + qty <= cap

    @staticmethod
    def can_sell(item: str, shed: Dict[str, int], qty: int = 1) -> bool:
        return qty > 0 and shed.get(item.upper(), 0) >= qty

    # -------------------------------------------------------------------------
    # End-of-Day Lifecycle Simulators
    # -------------------------------------------------------------------------

    @staticmethod
    def simulate_shed_drop(
        shed: Dict[str, int], inventories: List[Dict[str, int]], capacity: int = SHED_CAPACITY,
    ) -> Tuple[Dict[str, int], List[Dict[str, int]], int]:
        """End-of-day: dump all unit inventories into shed. Overflow discarded."""
        new_shed = dict(shed)
        new_invs = [dict(inv) for inv in inventories]
        discarded = 0
        for inv in new_invs:
            for item, n in list(inv.items()):
                if n <= 0: del inv[item]; continue
                room = max(0, capacity - sum(new_shed.values()))
                take = min(n, room)
                if take > 0: new_shed[item] = new_shed.get(item, 0) + take
                discarded += n - take
                del inv[item]
        return (new_shed, new_invs, discarded)

    @staticmethod
    def simulate_weed_spawns(
        tiles: List[List[Any]], chance: float = WEED_SPAWN_CHANCE, rng: Optional[random.Random] = None,
    ) -> List[List[Any]]:
        """Spawn weeds on empty unlocked tiles (None) with given probability."""
        rng = rng or random.Random()
        out = [list(row) for row in tiles]
        for y in range(len(out)):
            for x in range(len(out[y])):
                if out[y][x] is None and rng.random() < chance:
                    out[y][x] = {"kind": "WEED"}
        return out

    @staticmethod
    def simulate_end_of_day_plant(tile: dict, was_watered: bool, current_day: int, turns_per_day: int = TURNS_PER_DAY) -> dict:
        """End-of-day plant refresh: unwatered counter, weed decay, one-time bonus yield, ongoing production."""
        t = dict(tile)

        # Watering counter
        t["consecutive_unwatered"] = 0 if was_watered else t.get("consecutive_unwatered", 0) + 1
        t["watered_today"] = False
        if t["consecutive_unwatered"] >= 2:
            return {"kind": "WEED"}

        crop = CROPS.get(t.get("crop"))
        if not crop:
            return t

        next_day = current_day + 1
        age_next = next_day - t["planted_day"]

        if not crop.ongoing:
            # One-time crops: bonus watering adds yield during window
            if was_watered and crop.is_bonus_window(age_next - 1):
                # age_next - 1 == current_day's age (today's watering)
                fertilized = t.get("fertilized_until_day", -1) >= current_day
                add = 2 if fertilized else 1
                t["yield_units"] = min(crop.max_yield, t.get("yield_units", 1) + add)
            # Decay: one day after time_to_max_yield
            if age_next == crop.time_to_max_yield + 1:
                t["max_lifespan_step"] = next_day * turns_per_day
        else:
            # Ongoing: scheduled production
            d = age_next - crop.first_yield_day
            if d >= 0 and d % crop.interval == 0:
                prod_count = d // crop.interval + 1
                if prod_count <= crop.max_yield:
                    fertilized = was_watered and t.get("fertilized_until_day", -1) >= current_day
                    t["yield_units"] = min(crop.max_yield, t.get("yield_units", 0) + (2 if fertilized else 1))
                    if prod_count == crop.max_yield:
                        t["max_lifespan_step"] = (next_day + 1) * turns_per_day

        return t

    @staticmethod
    def simulate_end_of_day_animal(tile: dict, was_fed: bool, was_cared: bool, current_day: int) -> dict:
        """End-of-day animal refresh: feeding counter, escape, production, care bonus, fertilizer."""
        t = dict(tile)
        cfg = ANIMALS.get(t.get("animal"))
        if not cfg:
            return t

        # Feeding counter
        t["consecutive_unfed"] = 0 if was_fed else t.get("consecutive_unfed", 0) + 1
        if t["consecutive_unfed"] >= 2:
            return {"kind": cfg.structure}  # Animal escapes, structure remains

        # Scheduled production
        next_day = current_day + 1
        d = next_day - t["placed_day"] - cfg.first_yield_day
        if d >= 0 and d % cfg.interval == 0:
            # Base 1 always produced; care bonus only if fed
            care_bonus = t.pop("pending_care_bonus", 0) if was_fed else 0
            t["yield_units"] = min(cfg.max_held, t.get("yield_units", 0) + 1 + care_bonus)
            t["pending_care_bonus"] = 0

        # Bank care bonus: only if both fed AND cared
        if was_fed and was_cared:
            t["pending_care_bonus"] = t.get("pending_care_bonus", 0) + 1

        # Every surviving animal makes 1 fertilizer available at end-of-day
        t["fertilizer_available"] = True
        t["fed_today"] = False
        t["cared_today"] = False
        return t

    # -------------------------------------------------------------------------
    # Convenience Lookups
    # -------------------------------------------------------------------------
    @staticmethod
    def get_crop(name: Union[str, Plants]) -> CropConfig: return get_crop(name)
    @staticmethod
    def get_animal(name: Union[str, Animals]) -> AnimalConfig: return get_animal(name)