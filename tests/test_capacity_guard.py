"""Synthetic tests for helpers/capacity_guard (research/06 portables)."""

from kaggriculture.helpers.capacity_guard import (
    clamp_sells,
    dead_stock_sells,
    planned_drop_inventory,
    projected_shed_from_action,
    room_guard_99,
)


def _farm_obs(
    *,
    step: int = 1,
    shed: dict | None = None,
    inventories: list | None = None,
    prices: dict | None = None,
    tiles: list | None = None,
):
    day, hour = divmod(step, 24)
    return {
        "player": 0,
        "step": step,
        "day": day,
        "hour": hour,
        "farms": [
            {
                "farmer": [4, 4],
                "hands": [],
                "tiles": tiles or [[None] * 10 for _ in range(10)],
            }
        ],
        "private": {
            "shed": shed or {},
            "inventories": inventories or [{}],
        },
        "market": {"prices": prices or {}},
    }


def test_projected_shed_drop_respects_capacity():
    shed = {"WHEAT": 98}
    actions = [["DROP"]]
    positions = [[4, 4]]
    inventories = [{"WHEAT": 10}]
    projected = projected_shed_from_action(shed, actions, positions, inventories, board_size=10)
    assert sum(projected.values()) == 100
    assert projected["WHEAT"] == 100


def test_projected_shed_place_non_animal_at_shed():
    shed = {"WHEAT": 5}
    actions = [["PLACE", "WHEAT", 3]]
    positions = [[4, 4]]
    inventories = [{"WHEAT": 3}]
    projected = projected_shed_from_action(shed, actions, positions, inventories, board_size=10)
    assert projected["WHEAT"] == 8


def test_clamp_sells_drops_zero_and_shrinks():
    projected = {"WHEAT": 3, "MELON": 0}
    orders = [["SELL", "WHEAT", 8], ["SELL", "MELON", 5], ["HIRE"]]
    assert clamp_sells(projected, orders) == [["SELL", "WHEAT", 3], ["HIRE"]]


def test_planned_drop_inventory_counts_shed_adjacent_drop():
    farm = {"farmer": [4, 4], "hands": [[5, 4]]}
    private = {"inventories": [{"MELON": 2}, {"CARROT": 3}]}
    planned = planned_drop_inventory(farm, private, [["DROP"], ["DROP"]], board_size=10)
    assert planned == {"MELON": 2, "CARROT": 3}


def test_room_guard_99_noop_before_hour_23():
    obs = _farm_obs(step=10, shed={"WHEAT": 100})
    assert room_guard_99(obs, [["BUY_SEED", "WHEAT", 1]], [["PASS"]]) == [["BUY_SEED", "WHEAT", 1]]


def test_room_guard_99_adds_sell_when_over_capacity():
    obs = _farm_obs(
        step=23,
        shed={"WHEAT": 60, "MELON": 45},
        prices={"WHEAT": 25, "MELON": 250},
    )
    orders = room_guard_99(obs, [], [["PASS"]])
    assert any(order[0] == "SELL" for order in orders)
    assert sum(int(o[2]) for o in orders if o[0] == "SELL") >= 6


def test_room_guard_prefers_no_future_sell_items():
    obs = _farm_obs(
        step=23,
        shed={"WHEAT": 55, "MELON": 50},
        prices={"WHEAT": 25, "MELON": 250},
    )

    def future(item: str, _step: int) -> int:
        return 50 if item == "MELON" else 0

    orders = room_guard_99(obs, [], [["PASS"]], future_sells_fn=future)
    wheat_sells = [o for o in orders if o[0] == "SELL" and o[1] == "WHEAT"]
    melon_sells = [o for o in orders if o[0] == "SELL" and o[1] == "MELON"]
    assert wheat_sells and sum(int(o[2]) for o in wheat_sells) >= sum(int(o[2]) for o in melon_sells)


def test_dead_stock_sells_surplus_without_future_route():
    obs = _farm_obs(step=100, prices={"WHEAT": 25, "MELON": 250})
    projected = {"WHEAT": 10, "MELON": 4}
    market = [["SELL", "WHEAT", 3]]
    result = dead_stock_sells(projected, market, obs)
    assert sum(int(o[2]) for o in result if o[0] == "SELL" and o[1] == "WHEAT") == 10
    assert sum(int(o[2]) for o in result if o[0] == "SELL" and o[1] == "MELON") == 4


def test_dead_stock_respects_future_sells_fn():
    obs = _farm_obs(step=50, prices={"WHEAT": 25})
    projected = {"WHEAT": 10}
    market = [["SELL", "WHEAT", 2]]

    def future(item: str, _step: int) -> int:
        return 5 if item == "WHEAT" else 0

    result = dead_stock_sells(projected, market, obs, future_sells_fn=future)
    wheat_sells = [o for o in result if o[0] == "SELL" and o[1] == "WHEAT"]
    assert len(wheat_sells) == 2
    assert sum(int(o[2]) for o in wheat_sells) == 5


def test_dead_stock_terminal_day_liquidates_all_unplanned():
    obs = _farm_obs(step=29 * 24 + 5, prices={"CARROT": 35})
    obs["day"] = 29
    projected = {"CARROT": 8}
    result = dead_stock_sells(projected, [], obs, future_sells_fn=lambda _i, _s: 100)
    assert result == [["SELL", "CARROT", 8]]


def test_dead_stock_skips_price_floor():
    obs = _farm_obs(step=10, prices={"WHEAT": 1})
    result = dead_stock_sells({"WHEAT": 5}, [], obs)
    assert not any(o[0] == "SELL" for o in result)
