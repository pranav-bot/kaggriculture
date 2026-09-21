"""Tests for opponent regime classification and counter knobs."""

from kaggriculture.helpers.opponent import OpponentProfile, analyze_opponent_farm
from kaggriculture.helpers.regime_counter import (
    OpponentRegime,
    classify_regime,
    desired_mix_with_regime,
    knobs_for_regime,
    sell_delay_items,
)


def _melon_rush_farm():
    tiles = [[None] * 10 for _ in range(10)]
    for i in range(10):
        tiles[0][i] = {
            "kind": "PLANT",
            "crop": "MELON",
            "planted_day": 0,
            "yield_units": 0,
        }
    return {"tiles": tiles, "unlocked_quadrants": ["NW"], "money": 2000.0}


def test_classify_melon_rush():
    farm = _melon_rush_farm()
    profile = analyze_opponent_farm(farm, current_day=5)
    regime = classify_regime(profile, {"day": 5})
    assert regime == OpponentRegime.MELON_RUSH


def test_desired_mix_shifts_away_from_melon():
    knobs = knobs_for_regime(OpponentRegime.MELON_RUSH)
    mix = desired_mix_with_regime(5, [], knobs)
    base_melon = 14
    assert mix["MELON"] < base_melon
    assert mix["CARROT"] >= 14


def test_sell_delay_melon_rush():
    farm = _melon_rush_farm()
    profile = analyze_opponent_farm(farm, current_day=9)
    knobs = knobs_for_regime(OpponentRegime.MELON_RUSH)
    delayed = sell_delay_items(knobs, profile, clone_like_opponent=False)
    assert "MELON" in delayed


def test_mixed_inactive_collision_aggressive():
    profile = OpponentProfile(
        opponent_money=3000,
        unlocked_quadrants=["NW"],
        total_crops_planted=3,
        crop_counts={"WHEAT": 3},
        animal_counts={"COW": 0, "SHEEP": 0, "GOOSE": 0},
        weeds_count=0,
        vacant_count=90,
        crop_groups=[],
        sabotage_opportunities=[],
    )
    regime = classify_regime(profile, {"day": 10})
    assert regime == OpponentRegime.MIXED_INACTIVE
    assert knobs_for_regime(regime).collision_aggressive is True
