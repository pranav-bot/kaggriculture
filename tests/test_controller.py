import pytest
from kaggriculture.actions.controller import ActionController
from kaggriculture.actions.actions import Actions
from kaggriculture.env.items import Plants


def test_controller_initialization():
    controller = ActionController(target_crop=Plants.MELON)
    assert controller.target_crop == "MELON"
    assert controller.auto_water is True
    assert controller.auto_harvest is True
    assert controller.auto_sell is True


def test_controller_navigation():
    # Test directions
    assert ActionController.direction_towards((2, 2), (2, 0)) == Actions.NORTH
    assert ActionController.direction_towards((2, 2), (2, 4)) == Actions.SOUTH
    assert ActionController.direction_towards((2, 2), (4, 2)) == Actions.EAST
    assert ActionController.direction_towards((2, 2), (0, 2)) == Actions.WEST
    assert ActionController.direction_towards((2, 2), (2, 2)) == Actions.PASS

    # Test shed adjacency
    assert ActionController.is_shed_adjacent((4, 4)) is True
    assert ActionController.is_shed_adjacent((5, 4)) is True
    assert ActionController.is_shed_adjacent((4, 5)) is True
    assert ActionController.is_shed_adjacent((5, 5)) is True
    assert ActionController.is_shed_adjacent((0, 0)) is False


def test_controller_act_structure():
    controller = ActionController(target_crop=Plants.WHEAT)
    sample_obs = {
        "player": 0,
        "step": 1,
        "day": 0,
        "hour": 1,
        "farms": [
            {
                "money": 3000.0,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": 3000.0,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {"WHEAT": 0, "MELON": 0},
            "seeds": {"WHEAT": 2},
            "inventories": [{}],
        },
        "market": {
            "inventory": {"WHEAT": 10000, "MELON": 10000},
            "prices": {"WHEAT": 25, "MELON": 250},
        },
        "town": {"unlocked_shops": []},
    }

    action = controller.act(sample_obs)
    assert "farmer" in action
    assert "hands" in action
    assert "market" in action
    assert isinstance(action["farmer"], list)
    assert isinstance(action["hands"], list)
    assert isinstance(action["market"], list)
