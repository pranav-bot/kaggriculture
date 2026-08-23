import pytest
from kaggriculture.actions import (
    Actions,
    Wheat,
    Carrot,
    Tomato,
    Strawberry,
    Melon,
    Goose,
    Cow,
    Sheep,
    Fertilizer,
    CROPS,
    ANIMALS,
    get_crop,
    get_animal,
)
from kaggriculture.env.items import Plants, Animals, Products, Structures, YieldType


def test_actions_movement():
    assert Actions.north() == ["NORTH"]
    assert Actions.south() == ["SOUTH"]
    assert Actions.east() == ["EAST"]
    assert Actions.west() == ["WEST"]
    assert Actions.pass_action() == ["PASS"]
    assert Actions.move("north") == ["NORTH"]

    # Bounds checking
    assert Actions.can_move((0, 0), "NORTH", board_size=10) is False
    assert Actions.can_move((0, 0), "WEST", board_size=10) is False
    assert Actions.can_move((0, 0), "SOUTH", board_size=10) is True
    assert Actions.can_move((0, 0), "EAST", board_size=10) is True
    assert Actions.can_move((9, 9), "SOUTH", board_size=10) is False
    assert Actions.can_move((9, 9), "EAST", board_size=10) is False


def test_actions_tile_and_shed():
    assert Actions.water() == ["WATER"]
    assert Actions.harvest() == ["HARVEST"]
    assert Actions.fertilize() == ["FERTILIZE"]
    assert Actions.dig() == ["DIG"]
    assert Actions.drop() == ["DROP"]
    assert Actions.pickup("WHEAT", 5) == ["PICKUP", "WHEAT", 5]
    assert Actions.place("WHEAT", 2) == ["PLACE", "WHEAT", 2]

    # Shed adjacency
    assert Actions.is_shed_adjacent((4, 4)) is True
    assert Actions.is_shed_adjacent((5, 4)) is True
    assert Actions.is_shed_adjacent((4, 5)) is True
    assert Actions.is_shed_adjacent((5, 5)) is True
    assert Actions.is_shed_adjacent((0, 0)) is False

    assert Actions.can_drop((4, 4)) is True
    assert Actions.can_drop((0, 0)) is False

    shed = {"WHEAT": 10, "FERTILIZER": 0}
    assert Actions.can_pickup((4, 4), "WHEAT", shed) is True
    assert Actions.can_pickup((4, 4), "FERTILIZER", shed) is False
    assert Actions.can_pickup((0, 0), "WHEAT", shed) is False


def test_actions_feasibility_predicates():
    # Planting
    assert Actions.can_plant(None, "WHEAT", {"WHEAT": 2}) is True
    assert Actions.can_plant(None, "WHEAT", {"WHEAT": 0}) is False
    assert Actions.can_plant("LOCKED", "WHEAT", {"WHEAT": 2}) is False
    assert Actions.can_plant({"kind": "WEED"}, "WHEAT", {"WHEAT": 2}) is False

    # Watering
    plant_tile = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False}
    watered_tile = {"kind": "PLANT", "crop": "WHEAT", "watered_today": True}
    assert Actions.can_water(plant_tile) is True
    assert Actions.can_water(watered_tile) is False
    assert Actions.can_water(None) is False
    assert Actions.can_water("LOCKED") is False

    # Harvesting
    ripe_plant = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 4}
    immature_plant = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 3, "yield_units": 0}
    assert Actions.can_harvest(ripe_plant, current_day=4) is True
    assert Actions.can_harvest(immature_plant, current_day=3) is False

    # Fertilizing
    assert Actions.can_fertilize(plant_tile, {"FERTILIZER": 1}) is True
    assert Actions.can_fertilize(plant_tile, {"FERTILIZER": 0}) is False

    # Digging
    weed_tile = {"kind": "WEED"}
    empty_coop = {"kind": "COOP"}
    occupied_coop = {"kind": "COOP", "animal": "GOOSE"}
    assert Actions.can_dig(weed_tile) is True
    assert Actions.can_dig(empty_coop) is True
    assert Actions.can_dig(occupied_coop) is False
    assert Actions.can_dig(None) is False
    assert Actions.can_dig("LOCKED") is False

    # Animal operations
    assert Actions.can_place_animal(empty_coop, "GOOSE", {"GOOSE": 1}) is True
    assert Actions.can_place_animal(empty_coop, "GOOSE", {"GOOSE": 0}) is False
    assert Actions.can_place_animal(empty_coop, "COW", {"COW": 1}) is False  # Cow needs PASTURE

    assert Actions.can_feed(occupied_coop, {"WHEAT": 1}) is True
    assert Actions.can_feed(occupied_coop, {"WHEAT": 0}) is False
    assert Actions.can_care(occupied_coop) is True

    fertilizer_coop = {"kind": "COOP", "animal": "GOOSE", "fertilizer_available": True}
    no_fert_coop = {"kind": "COOP", "animal": "GOOSE", "fertilizer_available": False}
    assert Actions.can_collect_fertilizer(fertilizer_coop) is True
    assert Actions.can_collect_fertilizer(no_fert_coop) is False


def test_actions_animal_and_structures():
    assert Actions.build_coop() == ["BUILD_COOP"]
    assert Actions.build_pasture() == ["BUILD_PASTURE"]
    assert Actions.feed() == ["FEED"]
    assert Actions.collect_fertilizer() == ["COLLECT_FERTILIZER"]
    assert Actions.care() == ["CARE"]


def test_actions_market():
    assert Actions.buy_seed("WHEAT", 3) == ["BUY_SEED", "WHEAT", 3]
    assert Actions.buy_animal("GOOSE", 1) == ["BUY_ANIMAL", "GOOSE", 1]
    assert Actions.buy_product("WHEAT", 2) == ["BUY_PRODUCT", "WHEAT", 2]
    assert Actions.sell("MELON", 4) == ["SELL", "MELON", 4]
    assert Actions.hire() == ["HIRE"]
    assert Actions.buy_land() == ["BUY_LAND"]


def test_actions_plant_helpers():
    assert Actions.plant("WHEAT") == ["PLANT", "WHEAT"]
    assert Actions.plant_wheat() == ["PLANT", "WHEAT"]
    assert Actions.plant_carrot() == ["PLANT", "CARROT"]
    assert Actions.plant_tomato() == ["PLANT", "TOMATO"]
    assert Actions.plant_strawberry() == ["PLANT", "STRAWBERRY"]
    assert Actions.plant_melon() == ["PLANT", "MELON"]


def test_wheat_specifications():
    w = Wheat
    assert w.name == Plants.WHEAT
    assert w.yield_type == YieldType.ONE_TIME
    assert w.seed_cost == 10
    assert w.base_market_price == 25
    assert w.time_to_first_yield == 2
    assert w.time_to_max_yield == 4
    assert w.unfertilized_max_yield == 4
    assert w.fertilized_max_yield == 6
    assert w.max_yield == 6
    assert w.action_cost == 1
    assert w.yield_per_tile_per_day == 0.80
    assert w.subsequent_yields == "none"
    assert w.bonus_window == (2, 4)

    # Yield simulation
    # Unfertilized: base 1 + 3 bonus days = 4
    watered_days = {2, 3, 4}
    assert w.calculate_yield(planted_day=0, current_day=4, watered_days=watered_days) == 4
    # Fertilized: base 1 + 3 * 2 = 7 capped at 6
    assert w.calculate_yield(planted_day=0, current_day=4, watered_days=watered_days, fertilized_until_day=4) == 6


def test_carrot_specifications():
    c = Carrot
    assert c.name == Plants.CARROT
    assert c.yield_type == YieldType.ONE_TIME
    assert c.seed_cost == 20
    assert c.base_market_price == 35
    assert c.time_to_first_yield == 2
    assert c.time_to_max_yield == 3
    assert c.unfertilized_max_yield == 3
    assert c.fertilized_max_yield == 4
    assert c.max_yield == 4
    assert c.action_cost == 1
    assert c.yield_per_tile_per_day == 0.75
    assert c.bonus_window == (2, 3)

    # Yield simulation
    # Unfertilized: base 1 + 2 bonus days = 3
    watered_days = {2, 3}
    assert c.calculate_yield(planted_day=0, current_day=3, watered_days=watered_days) == 3
    # Fertilized: base 1 + 2 * 2 = 5 capped at 4
    assert c.calculate_yield(planted_day=0, current_day=3, watered_days=watered_days, fertilized_until_day=3) == 4


def test_tomato_specifications():
    t = Tomato
    assert t.name == Plants.TOMATO
    assert t.yield_type == YieldType.ONGOING
    assert t.seed_cost == 50
    assert t.base_market_price == 60
    assert t.time_to_first_yield == 8
    assert t.time_to_max_yield == 11
    assert t.interval == 1
    assert t.max_yield == 4
    assert t.action_cost == 1
    assert t.yield_per_tile_per_day == 0.33
    assert t.subsequent_yields == "every day ×4"


def test_strawberry_specifications():
    s = Strawberry
    assert s.name == Plants.STRAWBERRY
    assert s.yield_type == YieldType.ONGOING
    assert s.seed_cost == 100
    assert s.base_market_price == 120
    assert s.time_to_first_yield == 10
    assert s.time_to_max_yield == 16
    assert s.interval == 2
    assert s.max_yield == 4
    assert s.action_cost == 1
    assert s.yield_per_tile_per_day == 0.24
    assert s.subsequent_yields == "every other day ×4"


def test_melon_specifications():
    m = Melon
    assert m.name == Plants.MELON
    assert m.yield_type == YieldType.ONE_TIME
    assert m.seed_cost == 80
    assert m.base_market_price == 250
    assert m.time_to_first_yield == 10
    assert m.time_to_max_yield == 10
    assert m.max_yield == 6
    assert m.action_cost == 1
    assert m.yield_per_tile_per_day == 0.55
    assert m.bonus_window == (6, 12)

    # Accumulated units during growth:
    watered_days = {6, 7, 8, 9, 10, 11, 12}
    assert m.accumulated_yield_units(planted_day=0, current_day=10, watered_days=watered_days) == 6
    assert m.calculate_yield(planted_day=0, current_day=10, watered_days=watered_days) == 6
    assert m.is_optimal_harvest_age(planted_day=0, current_day=10, fertilized=False) is True

    # Fertilized reaches cap 6 at age 8 on tile:
    assert m.accumulated_yield_units(planted_day=0, current_day=8, watered_days=watered_days, fertilized_until_day=8) == 6
    assert m.calculate_yield(planted_day=0, current_day=10, watered_days=watered_days, fertilized_until_day=8) == 6


def test_animals_specifications():
    g = Goose
    assert g.name == Animals.GOOSE
    assert g.cost == 300
    assert g.base_market_price == 50
    assert g.structure == Structures.COOP
    assert g.product == Products.EGG
    assert g.first_yield_day == 4
    assert g.interval == 1
    assert g.max_held == 4
    assert g.action_cost == "1 + 1 (build coop)"
    assert g.yield_per_tile_per_day == 1.00
    assert g.feed_item == Products.WHEAT

    c = Cow
    assert c.name == Animals.COW
    assert c.cost == 400
    assert c.base_market_price == 160
    assert c.structure == Structures.PASTURE
    assert c.product == Products.MILK
    assert c.first_yield_day == 8
    assert c.interval == 2
    assert c.max_held == 6
    assert c.action_cost == "1 + 1 (build pasture)"
    assert c.yield_per_tile_per_day == 0.50

    sh = Sheep
    assert sh.name == Animals.SHEEP
    assert sh.cost == 500
    assert sh.base_market_price == 200
    assert sh.structure == Structures.PASTURE
    assert sh.product == Products.WOOL
    assert sh.first_yield_day == 6
    assert sh.interval == 3
    assert sh.max_held == 6
    assert sh.action_cost == "1 + 1 (build pasture)"
    assert sh.yield_per_tile_per_day == 0.33


def test_fertilizer_specifications():
    f = Fertilizer
    assert f.cost == 100
    assert f.action_cost == 1
    assert f.duration_days == 3
    assert f.base_market_price == 100


def test_get_crop_and_animal():
    assert get_crop("WHEAT") == Wheat
    assert get_crop(Plants.MELON) == Melon
    assert get_animal("GOOSE") == Goose
    assert get_animal(Animals.COW) == Cow

    with pytest.raises(KeyError):
        get_crop("UNKNOWN_CROP")

    with pytest.raises(KeyError):
        get_animal("UNKNOWN_ANIMAL")
