import random
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


def test_map_quadrants_and_geometry():
    assert Actions.quadrant_of(0, 0, 10) == "NW"
    assert Actions.quadrant_of(4, 4, 10) == "NW"
    assert Actions.quadrant_of(5, 0, 10) == "NE"
    assert Actions.quadrant_of(9, 4, 10) == "NE"
    assert Actions.quadrant_of(0, 5, 10) == "SW"
    assert Actions.quadrant_of(4, 9, 10) == "SW"
    assert Actions.quadrant_of(5, 5, 10) == "SE"
    assert Actions.quadrant_of(9, 9, 10) == "SE"

    assert Actions.get_quadrant_bounds("NW", 10) == (0, 5, 0, 5)
    assert Actions.get_quadrant_bounds("NE", 10) == (5, 10, 0, 5)
    assert Actions.get_quadrant_bounds("SW", 10) == (0, 5, 5, 10)
    assert Actions.get_quadrant_bounds("SE", 10) == (5, 10, 5, 10)

    unlocked = ["NW", "NE"]
    assert Actions.is_tile_unlocked(2, 2, unlocked, 10) is True   # in NW
    assert Actions.is_tile_unlocked(7, 2, unlocked, 10) is True   # in NE
    assert Actions.is_tile_unlocked(2, 7, unlocked, 10) is False  # in SW
    assert Actions.is_tile_unlocked(7, 7, unlocked, 10) is False  # in SE

    assert Actions.shed_access_tiles(10) == [(4, 4), (5, 4), (4, 5), (5, 5)]
    assert Actions.default_spawn(10) == (4, 4)


def test_shed_drop_and_capacity_discard():
    # Shed has 90 items, inventories have 20 items total. Cap is 100.
    shed = {"WHEAT": 90}
    inventories = [{"MELON": 15}, {"CARROT": 5}]

    new_shed, new_invs, discarded = Actions.simulate_shed_drop(shed, inventories, capacity=100)
    # Room is 10. Takes 10 MELON, discards 5 MELON + 5 CARROT = 10 discarded
    assert sum(new_shed.values()) == 100
    assert new_shed["WHEAT"] == 90
    assert new_shed["MELON"] == 10
    assert discarded == 10
    assert len(new_invs[0]) == 0
    assert len(new_invs[1]) == 0


def test_weed_spawning_simulation():
    tiles = [
        [None, None, "LOCKED"],
        [None, {"kind": "PLANT"}, "LOCKED"],
        ["LOCKED", "LOCKED", "LOCKED"],
    ]
    rng = random.Random(42)
    # With 100% chance, all None tiles should become WEED
    spawned = Actions.simulate_weed_spawns(tiles, weed_chance=1.0, rng=rng)
    assert spawned[0][0] == {"kind": "WEED"}
    assert spawned[0][1] == {"kind": "WEED"}
    assert spawned[1][0] == {"kind": "WEED"}
    assert spawned[1][1] == {"kind": "PLANT"}  # Plant not overwritten
    assert spawned[0][2] == "LOCKED"          # Locked not overwritten


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


def test_actions_market():
    assert Actions.buy_seed("WHEAT", 3) == ["BUY_SEED", "WHEAT", 3]
    assert Actions.buy_animal("GOOSE", 1) == ["BUY_ANIMAL", "GOOSE", 1]
    assert Actions.buy_product("WHEAT", 2) == ["BUY_PRODUCT", "WHEAT", 2]
    assert Actions.buy_product("FERTILIZER", 1) == ["BUY_PRODUCT", "FERTILIZER", 1]
    assert Actions.sell("MELON", 4) == ["SELL", "MELON", 4]
    assert Actions.hire() == ["HIRE"]
    assert Actions.buy_land() == ["BUY_LAND"]

    # Fibonacci Hire Costs: 1, 1, 2, 3, 5, 8, 13, 21...
    assert [Actions.hire_cost(i) for i in range(8)] == [1, 1, 2, 3, 5, 8, 13, 21]

    # Land Costs: 1000, 2000, 4000
    assert Actions.land_cost(["NW"]) == 1000
    assert Actions.land_cost(["NW", "NE"]) == 2000
    assert Actions.land_cost(["NW", "NE", "SW"]) == 4000
    assert Actions.land_cost(["NW", "NE", "SW", "SE"]) is None

    assert Actions.next_quadrant(["NW"]) == "NE"
    assert Actions.next_quadrant(["NW", "NE"]) == "SW"
    assert Actions.next_quadrant(["NW", "NE", "SW"]) == "SE"
    assert Actions.next_quadrant(["NW", "NE", "SW", "SE"]) is None

    # Market feasibility checks
    assert Actions.can_buy_land(1500, ["NW"]) is True
    assert Actions.can_buy_land(500, ["NW"]) is False
    assert Actions.can_buy_land(5000, ["NW", "NE", "SW", "SE"]) is False

    assert Actions.can_hire(10, hires_already_today=0) is True
    assert Actions.can_hire(0, hires_already_today=0) is False

    assert Actions.can_buy_seed(100, "WHEAT", quantity=5) is True
    assert Actions.can_buy_seed(30, "WHEAT", quantity=5) is False

    shed = {"WHEAT": 20, "MELON": 10}
    assert Actions.can_buy_animal(500, "GOOSE", shed=shed, shed_capacity=100, quantity=1) is True
    assert Actions.can_buy_animal(200, "GOOSE", shed=shed, shed_capacity=100, quantity=1) is False
    assert Actions.can_buy_animal(500, "GOOSE", shed={"WHEAT": 100}, shed_capacity=100, quantity=1) is False

    assert Actions.can_buy_product(50, "WHEAT", current_price=25, shed=shed, quantity=2) is True
    assert Actions.can_buy_product(40, "WHEAT", current_price=25, shed=shed, quantity=2) is False

    assert Actions.can_sell("WHEAT", shed=shed, quantity=15) is True
    assert Actions.can_sell("WHEAT", shed=shed, quantity=25) is False
    assert Actions.can_sell("STRAWBERRY", shed=shed, quantity=1) is False


def test_end_of_day_plant_and_weed_danger():
    new_plant = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 1,
    }

    t1 = Actions.simulate_end_of_day_plant(new_plant, was_watered=False, current_day=0)
    assert t1 == {"kind": "WEED"}

    t2 = Actions.simulate_end_of_day_plant(new_plant, was_watered=True, current_day=0)
    assert t2["kind"] == "PLANT"
    assert t2["consecutive_unwatered"] == 0


def test_end_of_day_animal_and_escape():
    new_animal = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }

    a1 = Actions.simulate_end_of_day_animal(new_animal, was_fed=False, was_cared=False, current_day=0)
    assert a1["consecutive_unfed"] == 1
    assert "animal" in a1

    a2 = Actions.simulate_end_of_day_animal(a1, was_fed=False, was_cared=False, current_day=1)
    assert a2 == {"kind": "COOP"}


def test_plant_decay_simulation():
    w = Wheat
    mls = (0 + 4 + 1) * 24
    res, is_weed = w.simulate_decay(current_yield=4, max_lifespan_step=mls, current_step=119)
    assert res == 4 and not is_weed

    res1, _ = w.simulate_decay(current_yield=4, max_lifespan_step=mls, current_step=120)
    assert res1 == 3
    res2, _ = w.simulate_decay(current_yield=4, max_lifespan_step=mls, current_step=122)
    assert res2 == 2
    res_final, is_weed_final = w.simulate_decay(current_yield=4, max_lifespan_step=mls, current_step=128)
    assert is_weed_final is True


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

    watered_days = {2, 3, 4}
    assert w.calculate_yield(planted_day=0, current_day=4, watered_days=watered_days) == 4
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

    watered_days = {2, 3}
    assert c.calculate_yield(planted_day=0, current_day=3, watered_days=watered_days) == 3
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

    watered_days = {6, 7, 8, 9, 10, 11, 12}
    assert m.accumulated_yield_units(planted_day=0, current_day=10, watered_days=watered_days) == 6
    assert m.calculate_yield(planted_day=0, current_day=10, watered_days=watered_days) == 6
    assert m.is_optimal_harvest_age(planted_day=0, current_day=10, fertilized=False) is True

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
