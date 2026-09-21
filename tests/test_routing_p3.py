from kaggriculture.actions.actions import Actions
from kaggriculture.actions.controller import ActionController
from kaggriculture.helpers.water_rescue import build_water_rescue_route


def test_direction_alternates_by_unit_index():
    assert ActionController.direction_towards((2, 2), (4, 3), 0) == Actions.EAST
    assert ActionController.direction_towards((2, 2), (4, 3), 1) == Actions.SOUTH


def test_loss_sort_prefers_high_value_water_rescue():
    controller = ActionController(target_crop="WHEAT")
    farm = {
        "tiles": [
            [
                {
                    "kind": "PLANT",
                    "crop": "WHEAT",
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 1,
                    "consecutive_unwatered": 1,
                },
                {
                    "kind": "PLANT",
                    "crop": "MELON",
                    "planted_day": 0,
                    "watered_today": False,
                    "yield_units": 6,
                    "consecutive_unwatered": 1,
                },
            ]
        ],
    }
    target = controller.find_best_tile_for_unit(
        (0, 0),
        {},
        farm,
        {},
        {},
        10,
        10,
        set(),
        {"WHEAT": 25, "MELON": 250},
    )
    assert target is not None
    assert target[2] == "water"
    assert farm["tiles"][0][1]["crop"] == "MELON"


def test_water_rescue_route_fits_seven_commands():
    route = build_water_rescue_route((4, 4), (5, 4))
    assert route is not None
    assert len(route) <= 7
    assert route[len(route) // 2] == ["WATER"]
