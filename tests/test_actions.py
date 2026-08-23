import random
import pytest
from kaggriculture.actions import (
    Actions, Wheat, Carrot, Tomato, Strawberry, Melon,
    Goose, Cow, Sheep, Fertilizer, CROPS, ANIMALS, get_crop, get_animal,
)
from kaggriculture.env.items import Plants, Animals, Products, Structures, YieldType, market_price


# ==========================================================================
# Market Pricing Curves — exact table verification
# ==========================================================================

def test_market_price_curve_table():
    I0 = 10_000
    mp = Actions.market_price

    # Wheat: Base 25, T 400, below sqrt/0.80, above log/0.20
    assert mp("WHEAT", I0) == 25
    assert mp("WHEAT", I0 - 400) == 45
    assert mp("WHEAT", I0 + 400) == 20
    assert mp("WHEAT", I0 + 800) == 19

    # Carrot: Base 35, T 450, below hinge/1.00, above sqrt/0.70
    assert mp("CARROT", I0) == 35
    assert mp("CARROT", I0 - 450) == 70
    assert mp("CARROT", I0 + 450) in (10, 11)
    assert mp("CARROT", I0 + 900) == 1

    # Tomato: Base 60, T 200, below hinge/0.40, above sqrt/0.60
    assert mp("TOMATO", I0) == 60
    assert mp("TOMATO", I0 - 200) == 84
    assert mp("TOMATO", I0 + 200) == 24
    assert mp("TOMATO", I0 + 400) == 9

    # Strawberry: Base 120, T 100, below sqrt/0.70, above linear/1.60
    assert mp("STRAWBERRY", I0) == 120
    assert mp("STRAWBERRY", I0 - 100) == 204
    assert mp("STRAWBERRY", I0 + 100) == 1
    assert mp("STRAWBERRY", I0 + 200) == 1

    # Melon: Base 250, T 300, below log/0.20, above sq/3.60
    assert mp("MELON", I0) == 250
    assert mp("MELON", I0 - 300) == 300
    assert mp("MELON", I0 + 300) == 1
    assert mp("MELON", I0 + 600) == 1

    # Egg: Base 50, T 332, below hinge/0.40, above log/0.20
    assert mp("EGG", I0) == 50
    assert mp("EGG", I0 - 332) == 70
    assert mp("EGG", I0 + 332) == 40
    assert mp("EGG", I0 + 664) == 39

    # Milk: Base 160, T 122, below sqrt/0.60, above linear/1.60
    assert mp("MILK", I0) == 160
    assert mp("MILK", I0 - 122) == 256
    assert mp("MILK", I0 + 122) == 1
    assert mp("MILK", I0 + 244) == 1

    # Wool: Base 200, T 105, below log/0.20, above sq/3.20
    assert mp("WOOL", I0) == 200
    assert mp("WOOL", I0 - 105) == 240
    assert mp("WOOL", I0 + 105) == 1
    assert mp("WOOL", I0 + 210) == 1

    # Fertilizer: Base 100, T 200, below linear/0.40, above linear/0.40
    assert mp("FERTILIZER", I0) == 100
    assert mp("FERTILIZER", I0 - 200) == 140
    assert mp("FERTILIZER", I0 + 200) == 60
    assert mp("FERTILIZER", I0 + 400) == 20


def test_pricing_helpers():
    assert Actions.is_premium_resource("STRAWBERRY") is True
    assert Actions.is_premium_resource("MELON") is True
    assert Actions.is_premium_resource("WHEAT") is False

    overrides = {"WOOL": {"above_target": 0.95}}
    resolved = Actions.resolve_market_params(overrides)
    assert resolved["WOOL"]["above_target"] == 0.95
    assert resolved["WHEAT"]["above_target"] == 0.20

    assert Actions.predict_price_impact("WHEAT", 10000, 400) == 20


# ==========================================================================
# Spawn & Town
# ==========================================================================

def test_hand_spawn():
    farm = {"farmer": [4, 4], "hands": []}
    assert Actions.spawn_hand(farm) == [5, 4]
    farm["hands"].append([5, 4])
    assert Actions.spawn_hand(farm) == [4, 5]
    farm["hands"].append([4, 5])
    assert Actions.spawn_hand(farm) == [5, 5]
    farm["hands"].append([5, 5])
    assert Actions.spawn_hand(farm) == [4, 4]


def test_town_shops():
    assert Actions.get_shop_demands("BAKERY") == ["EGG", "WHEAT"]
    assert Actions.get_shop_demands("YARN_STORE") == ["WOOL"]

    assert Actions.calculate_shop_turn_consumption("BAKERY") == {"EGG": 1, "WHEAT": 1}
    assert Actions.calculate_shop_turn_consumption("YARN_STORE") == {"WOOL": 2}
    assert Actions.calculate_shop_turn_consumption("PET_CAFE") == {"CARROT": 2}

    daily = Actions.calculate_town_daily_consumption(["BAKERY", "YARN_STORE"])
    assert daily["WHEAT"] == 7    # 1 town center + 6 bakery
    assert daily["EGG"] == 7      # 1 town center + 6 bakery
    assert daily["WOOL"] == 13    # 1 town center + 12 yarn store (2×6)
    assert daily["FERTILIZER"] == 0


# ==========================================================================
# Geometry
# ==========================================================================

def test_quadrants():
    assert Actions.quadrant_of(0, 0) == "NW"
    assert Actions.quadrant_of(5, 0) == "NE"
    assert Actions.quadrant_of(0, 5) == "SW"
    assert Actions.quadrant_of(5, 5) == "SE"

    assert Actions.get_quadrant_bounds("NW") == (0, 5, 0, 5)
    assert Actions.get_quadrant_bounds("SE") == (5, 10, 5, 10)

    assert Actions.is_tile_unlocked(2, 2, ["NW"]) is True
    assert Actions.is_tile_unlocked(7, 2, ["NW"]) is False
    assert Actions.is_tile_unlocked(7, 2, ["NW", "NE"]) is True

    assert Actions.shed_access_tiles() == [(4, 4), (5, 4), (4, 5), (5, 5)]
    assert Actions.default_spawn() == (4, 4)


# ==========================================================================
# Shed Drop & Weed Spawning
# ==========================================================================

def test_shed_drop():
    shed = {"WHEAT": 90}
    invs = [{"MELON": 15}, {"CARROT": 5}]
    new_shed, new_invs, discarded = Actions.simulate_shed_drop(shed, invs, capacity=100)
    assert sum(new_shed.values()) == 100
    assert discarded == 10


def test_weed_spawning():
    tiles = [[None, None, "LOCKED"], [None, {"kind": "PLANT"}, "LOCKED"], ["LOCKED"] * 3]
    spawned = Actions.simulate_weed_spawns(tiles, chance=1.0, rng=random.Random(42))
    assert spawned[0][0] == {"kind": "WEED"}
    assert spawned[1][1] == {"kind": "PLANT"}
    assert spawned[0][2] == "LOCKED"


# ==========================================================================
# Movement & Action Creators
# ==========================================================================

def test_movement():
    assert Actions.north() == ["NORTH"]
    assert Actions.move("south") == ["SOUTH"]
    assert Actions.can_move((0, 0), "NORTH") is False
    assert Actions.can_move((0, 0), "SOUTH") is True
    assert Actions.can_move((9, 9), "EAST") is False


def test_action_creators():
    assert Actions.plant("wheat") == ["PLANT", "WHEAT"]
    assert Actions.pickup("WHEAT", 5) == ["PICKUP", "WHEAT", 5]
    assert Actions.buy_seed("MELON", 3) == ["BUY_SEED", "MELON", 3]
    assert Actions.sell("WOOL", 2) == ["SELL", "WOOL", 2]
    assert Actions.hire() == ["HIRE"]
    assert Actions.buy_land() == ["BUY_LAND"]


# ==========================================================================
# Feasibility Predicates
# ==========================================================================

def test_plant_feasibility():
    assert Actions.can_plant(None, "WHEAT", {"WHEAT": 2}) is True
    assert Actions.can_plant(None, "WHEAT", {"WHEAT": 0}) is False
    assert Actions.can_plant("LOCKED", "WHEAT", {"WHEAT": 2}) is False
    assert Actions.can_plant({"kind": "WEED"}, "WHEAT", {"WHEAT": 2}) is False


def test_water_harvest_fertilize():
    plant = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False}
    assert Actions.can_water(plant) is True
    assert Actions.can_water({"kind": "PLANT", "crop": "WHEAT", "watered_today": True}) is False
    assert Actions.can_water(None) is False

    ripe = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 4}
    assert Actions.can_harvest(ripe, current_day=4) is True
    assert Actions.can_harvest({"kind": "PLANT", "crop": "WHEAT", "planted_day": 3, "yield_units": 0}, current_day=3) is False

    assert Actions.can_fertilize(plant, {"FERTILIZER": 1}) is True
    assert Actions.can_fertilize(plant, {"FERTILIZER": 0}) is False


def test_dig_feasibility():
    """DIG: can dig weeds, plants, empty structures. Cannot dig structure with animal."""
    assert Actions.can_dig({"kind": "WEED"}) is True
    assert Actions.can_dig({"kind": "PLANT"}) is True
    assert Actions.can_dig({"kind": "COOP"}) is True                         # Empty structure
    assert Actions.can_dig({"kind": "COOP", "animal": "GOOSE"}) is False     # Occupied
    assert Actions.can_dig({"kind": "COOP", "animal": None}) is True         # Unoccupied (None animal)
    assert Actions.can_dig(None) is False
    assert Actions.can_dig("LOCKED") is False


def test_animal_feasibility():
    empty_coop = {"kind": "COOP", "animal": None}
    occupied = {"kind": "COOP", "animal": "GOOSE", "fed_today": False, "cared_today": False, "yield_units": 2, "fertilizer_available": True}

    assert Actions.can_place_animal(empty_coop, "GOOSE", {"GOOSE": 1}) is True
    assert Actions.can_place_animal(empty_coop, "COW", {"COW": 1}) is False  # Wrong structure
    assert Actions.can_feed(occupied, {"WHEAT": 1}) is True
    assert Actions.can_feed(occupied, {"WHEAT": 0}) is False
    assert Actions.can_care(occupied) is True
    assert Actions.can_harvest(occupied, current_day=99) is True
    assert Actions.can_collect_fertilizer(occupied) is True


def test_market_feasibility():
    assert [Actions.hire_cost(i) for i in range(8)] == [1, 1, 2, 3, 5, 8, 13, 21]
    assert Actions.land_cost(["NW"]) == 1000
    assert Actions.land_cost(["NW", "NE"]) == 2000
    assert Actions.land_cost(["NW", "NE", "SW"]) == 4000
    assert Actions.land_cost(["NW", "NE", "SW", "SE"]) is None

    assert Actions.next_quadrant(["NW"]) == "NE"
    assert Actions.next_quadrant(["NW", "NE", "SW", "SE"]) is None

    assert Actions.can_buy_land(1500, ["NW"]) is True
    assert Actions.can_buy_land(500, ["NW"]) is False
    assert Actions.can_buy_seed(100, "WHEAT", 5) is True
    assert Actions.can_buy_seed(30, "WHEAT", 5) is False

    shed = {"WHEAT": 20}
    assert Actions.can_sell("WHEAT", shed, 15) is True
    assert Actions.can_sell("WHEAT", shed, 25) is False


# ==========================================================================
# End-of-Day Lifecycle
# ==========================================================================

def test_plant_weed_decay():
    """New seed planted and left unwatered becomes weed (consecutive_unwatered starts at 1)."""
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "watered_today": False, "consecutive_unwatered": 1, "yield_units": 1}

    weed = Actions.simulate_end_of_day_plant(tile, was_watered=False, current_day=0)
    assert weed == {"kind": "WEED"}

    saved = Actions.simulate_end_of_day_plant(tile, was_watered=True, current_day=0)
    assert saved["kind"] == "PLANT"
    assert saved["consecutive_unwatered"] == 0


def test_one_time_crop_bonus_yield():
    """Wheat bonus window: ages 2–4. Watering during window adds +1 yield per day."""
    # Wheat planted day 0, bonus window = (2, 4)
    # At end-of-day for current_day=1: next_day=2, age_next=2, age_next-1=1, NOT in window → no bonus
    # At end-of-day for current_day=2: next_day=3, age_next=3, age_next-1=2, IN window → +1
    # At end-of-day for current_day=3: next_day=4, age_next=4, age_next-1=3, IN window → +1
    # At end-of-day for current_day=4: next_day=5, age_next=5, age_next-1=4, IN window → +1
    #                                  Also: age_next == time_to_max_yield + 1 = 5, max_lifespan set

    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "watered_today": True, "consecutive_unwatered": 0,
            "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1}

    # Day 1: age_next-1 = 1, not in window (2,4)
    t = Actions.simulate_end_of_day_plant(tile, was_watered=True, current_day=1)
    assert t["yield_units"] == 1  # No bonus yet

    # Day 2: age_next-1 = 2, in window → yield 1 → 2
    t["watered_today"] = True
    t = Actions.simulate_end_of_day_plant(t, was_watered=True, current_day=2)
    assert t["yield_units"] == 2

    # Day 3: age_next-1 = 3, in window → yield 2 → 3
    t["watered_today"] = True
    t = Actions.simulate_end_of_day_plant(t, was_watered=True, current_day=3)
    assert t["yield_units"] == 3

    # Day 4: age_next-1 = 4, in window → yield 3 → 4
    # Also: age_next == 5 == time_to_max_yield + 1 → max_lifespan_step set
    t["watered_today"] = True
    t = Actions.simulate_end_of_day_plant(t, was_watered=True, current_day=4)
    assert t["yield_units"] == 4  # Unfertilized max = 4 ✓
    assert t["max_lifespan_step"] == 5 * 24  # Decay begins day 5

    # Day 5: age_next-1 = 5, NOT in window (2,4) → no bonus
    t["watered_today"] = True
    t = Actions.simulate_end_of_day_plant(t, was_watered=True, current_day=5)
    assert t["yield_units"] == 4  # Stays at 4


def test_one_time_crop_fertilized_yield():
    """Fertilized wheat: bonus window gives +2/day instead of +1."""
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "watered_today": True, "consecutive_unwatered": 0,
            "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": 10}

    # Day 2: +2 → yield 1 → 3
    t = Actions.simulate_end_of_day_plant(tile, was_watered=True, current_day=2)
    assert t["yield_units"] == 3

    # Day 3: +2 → yield 3 → 5
    t["watered_today"] = True
    t = Actions.simulate_end_of_day_plant(t, was_watered=True, current_day=3)
    assert t["yield_units"] == 5

    # Day 4: +2 → yield 5 → min(6, 7) = 6 (capped at max_yield)
    t["watered_today"] = True
    t = Actions.simulate_end_of_day_plant(t, was_watered=True, current_day=4)
    assert t["yield_units"] == 6  # Fertilized max = 6 ✓


def test_animal_escape():
    tile = {"kind": "COOP", "animal": "GOOSE", "placed_day": 0,
            "yield_units": 0, "consecutive_unfed": 0,
            "fed_today": False, "cared_today": False,
            "fertilizer_available": False, "pending_care_bonus": 0}

    a1 = Actions.simulate_end_of_day_animal(tile, was_fed=False, was_cared=False, current_day=0)
    assert a1["consecutive_unfed"] == 1
    assert "animal" in a1

    a2 = Actions.simulate_end_of_day_animal(a1, was_fed=False, was_cared=False, current_day=1)
    assert a2 == {"kind": "COOP"}  # Escaped, structure remains


def test_animal_care_bonus():
    """Care banking: fed+cared → bonus banked; paid out on production day only if fed."""
    # Goose: placed day 0, first_yield_day=4, interval=1
    # Production days: next_day=4 → d=0, next_day=5 → d=1, etc.
    # So production fires at end-of-day for current_day=3 (next_day=4), current_day=4 (next_day=5), etc.

    tile = {"kind": "COOP", "animal": "GOOSE", "placed_day": 0,
            "yield_units": 0, "consecutive_unfed": 0,
            "fed_today": True, "cared_today": True,
            "fertilizer_available": False, "pending_care_bonus": 0}

    # Day 0: fed+cared, not production day (next_day=1, d=1-0-4=-3, not >= 0)
    t = Actions.simulate_end_of_day_animal(tile, was_fed=True, was_cared=True, current_day=0)
    assert t["pending_care_bonus"] == 1
    assert t["yield_units"] == 0

    # Day 1: fed+cared, not production (next_day=2, d=-2)
    t = Actions.simulate_end_of_day_animal(t, was_fed=True, was_cared=True, current_day=1)
    assert t["pending_care_bonus"] == 2
    assert t["yield_units"] == 0

    # Day 2: fed+cared, not production (next_day=3, d=-1)
    t = Actions.simulate_end_of_day_animal(t, was_fed=True, was_cared=True, current_day=2)
    assert t["pending_care_bonus"] == 3
    assert t["yield_units"] == 0

    # Day 3: fed+cared, PRODUCTION (next_day=4, d=0, 0%1==0)
    # Base 1 + care_bonus 3 = 4 yield_units (capped at max_held=4)
    t = Actions.simulate_end_of_day_animal(t, was_fed=True, was_cared=True, current_day=3)
    assert t["yield_units"] == 4  # base 1 + 3 care bonus
    assert t["pending_care_bonus"] == 1  # re-banked this day's care


def test_decay_simulation():
    mls = 5 * 24  # 120
    assert Wheat.simulate_decay(4, mls, 119) == (4, False)
    assert Wheat.simulate_decay(4, mls, 120) == (3, False)
    assert Wheat.simulate_decay(4, mls, 122) == (2, False)
    assert Wheat.simulate_decay(4, mls, 128)[1] is True  # Weed


# ==========================================================================
# Crop & Animal Specs
# ==========================================================================

def test_crop_specs():
    assert Wheat.seed_cost == 10 and Wheat.base_market_price == 25
    assert Wheat.bonus_window == (2, 4)  # ceil(4/2) = 2, max_yield_day = 4
    assert Wheat.unfertilized_max_yield == 4 and Wheat.max_yield == 6

    assert Carrot.seed_cost == 20 and Carrot.bonus_window == (2, 3)
    assert Carrot.unfertilized_max_yield == 3

    assert Tomato.interval == 1 and Tomato.max_yield == 4
    assert Strawberry.interval == 2 and Strawberry.first_yield_day == 10

    assert Melon.bonus_window == (6, 12)
    assert Melon.seed_cost == 80 and Melon.base_market_price == 250

    # Wheat yield via accumulated_yield_units (standalone calculation)
    # Bonus window (2,4): watering days 2,3,4 = 3 bonus days → yield = 1 + 3 = 4
    assert Wheat.calculate_yield(0, 4, {2, 3, 4}) == 4
    # Fertilized: 2 per day → yield = 1 + 6 = 7 → capped at 6
    assert Wheat.calculate_yield(0, 4, {2, 3, 4}, fertilized_until_day=4) == 6

    # Melon peaks at age 10 with daily watering (base 1 + ages 6..10 = 5 days = 6)
    assert Melon.calculate_yield(0, 10, {6, 7, 8, 9, 10}) == 6


def test_animal_specs():
    assert Goose.cost == 300 and Goose.product == "EGG" and Goose.interval == 1
    assert Cow.cost == 400 and Cow.product == "MILK" and Cow.interval == 2
    assert Sheep.cost == 500 and Sheep.product == "WOOL" and Sheep.interval == 3
    assert Fertilizer.cost == 100 and Fertilizer.duration_days == 3


def test_lookups():
    assert get_crop("WHEAT") is Wheat
    assert get_crop(Plants.MELON) is Melon
    assert get_animal("GOOSE") is Goose
    with pytest.raises(KeyError): get_crop("UNKNOWN")
    with pytest.raises(KeyError): get_animal("UNKNOWN")
