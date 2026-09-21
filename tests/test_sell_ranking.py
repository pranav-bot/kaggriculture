from kaggriculture.helpers.sell_ranking import (
    clamp_sells,
    projected_shed,
    rank_sell_slots,
    room_guard_99,
)


def test_rank_sell_slots_puts_higher_impact_order_first():
    orders = [["SELL", "WHEAT", 2], ["SELL", "MELON", 2]]
    ranked = rank_sell_slots(
        orders,
        {"prices": {"WHEAT": 25, "MELON": 250}},
    )
    assert ranked == [["SELL", "MELON", 2], ["SELL", "WHEAT", 2]]


def test_sell_repairs_are_conservative():
    orders = [["SELL", "WHEAT", 8], ["BUY_SEED", "MELON", 1]]
    assert clamp_sells(orders, {"WHEAT": 3}) == [
        ["SELL", "WHEAT", 3],
        ["BUY_SEED", "MELON", 1],
    ]
    assert projected_shed({"WHEAT": 8}, orders) == {"WHEAT": 0}
    assert room_guard_99({"WHEAT": 99}, orders) == [["SELL", "WHEAT", 8]]
