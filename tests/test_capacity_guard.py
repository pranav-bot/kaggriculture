from kaggriculture.helpers.capacity_guard import (
    clamp_sells,
    planned_drop_inventory,
    projected_shed_from_action,
    room_guard_99,
)


def test_projected_shed_includes_drop_at_shed():
    shed = {"WHEAT": 10}
    actions = [["DROP"], ["PASS"]]
    positions = [[4, 4], [0, 0]]
    inventories = [{"WHEAT": 5}, {}]
    projected = projected_shed_from_action(shed, actions, positions, inventories, board_size=10)
    assert projected["WHEAT"] == 15


def test_clamp_sells_uses_projected_stock():
    projected = {"WHEAT": 3}
    orders = [["SELL", "WHEAT", 8], ["BUY_SEED", "MELON", 1]]
    assert clamp_sells(projected, orders) == [
        ["SELL", "WHEAT", 3],
        ["BUY_SEED", "MELON", 1],
    ]


def test_planned_drop_inventory_counts_shed_adjacent_drop():
    farm = {"farmer": [4, 4], "hands": [[5, 4]]}
    private = {"inventories": [{"MELON": 2}, {"CARROT": 3}]}
    planned = planned_drop_inventory(
        farm,
        private,
        [["DROP"], ["DROP"]],
        board_size=10,
    )
    assert planned == {"MELON": 2, "CARROT": 3}


def test_room_guard_99_adds_sell_at_day_close():
    obs = {
        "player": 0,
        "step": 23,
        "day": 0,
        "hour": 23,
        "farms": [
            {
                "farmer": [4, 4],
                "hands": [],
                "tiles": [[None] * 10 for _ in range(10)],
            }
        ],
        "private": {
            "shed": {"WHEAT": 60, "MELON": 45},
            "inventories": [{}],
        },
        "market": {"prices": {"WHEAT": 25, "MELON": 250}},
    }
    orders = room_guard_99(obs, [], [["PASS"]])
    assert any(order[0] == "SELL" for order in orders)
    total_sell = sum(int(o[2]) for o in orders if o[0] == "SELL")
    assert total_sell >= 6
