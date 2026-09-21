"""Tests for opponent sell-volume inference and premium sell duel."""

from kaggriculture.env.items import MARKET_I0
from kaggriculture.helpers.opponent import estimate_imminent_sell_volume
from kaggriculture.helpers.sell_duel import duel_sell_qty


def _melon_tile(planted_day: int = 0, yield_units: int = 4):
    return {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": planted_day,
        "yield_units": yield_units,
        "watered_today": True,
        "consecutive_unwatered": 0,
    }


def test_estimate_imminent_sell_volume_melons_near_shed():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[4][4] = _melon_tile(yield_units=6)
    tiles[4][5] = _melon_tile(yield_units=3)
    tiles[0][0] = _melon_tile(yield_units=99)

    farm = {"tiles": tiles, "unlocked_quadrants": ["NW"]}
    vol = estimate_imminent_sell_volume(farm, "MELON", current_day=10, shed_radius=4)
    assert vol == 9


def test_estimate_imminent_sell_volume_animal_product():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[2][2] = {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 5,
        "placed_day": 0,
    }
    farm = {"tiles": tiles}
    assert estimate_imminent_sell_volume(farm, "MILK") == 5
    assert estimate_imminent_sell_volume(farm, "WOOL") == 0


def test_duel_sell_qty_holds_when_opponent_dump_is_severe():
    have = 20
    q_low_opp = duel_sell_qty("MELON", have, MARKET_I0, opp_hat=0, shed_total=50)
    q_high_opp = duel_sell_qty("MELON", have, MARKET_I0, opp_hat=400, shed_total=50)
    assert q_low_opp >= q_high_opp


def test_duel_sell_qty_force_sell_under_pressure():
    q = duel_sell_qty("MELON", 12, MARKET_I0, opp_hat=500, shed_total=90)
    assert q == 12
