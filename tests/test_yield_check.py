import pytest

from kaggriculture.helpers import check_max_yield_met, compute_one_time_yield


# --- One-time crops ---


def test_wheat_unfertilized_peak_at_day_4():
    state = {
        "plant_type": "WHEAT",
        "age_in_days": 4,
        "watered_bonus_days_count": 3,  # bonus days 2, 3, 4
        "fertilized_bonus_days_count": 0,
    }
    assert compute_one_time_yield(3, 0, 6) == 4
    assert check_max_yield_met(state) is True


def test_wheat_underwatered_not_ready():
    state = {
        "plant_type": "WHEAT",
        "age_in_days": 4,
        "watered_bonus_days_count": 2,
        "fertilized_bonus_days_count": 0,
    }
    assert check_max_yield_met(state) is False


def test_wheat_immature_not_ready():
    state = {
        "plant_type": "WHEAT",
        "age_in_days": 3,
        "watered_bonus_days_count": 3,
        "fertilized_bonus_days_count": 0,
    }
    assert check_max_yield_met(state) is False


def test_wheat_fertilized_absolute_max():
    state = {
        "plant_type": "WHEAT",
        "age_in_days": 4,
        "watered_bonus_days_count": 0,
        "fertilized_bonus_days_count": 3,
    }
    assert compute_one_time_yield(0, 3, 6) == 6
    assert check_max_yield_met(state) is True


def test_carrot_unfertilized_cap():
    state = {
        "plant_type": "CARROT",
        "age_in_days": 3,
        "watered_bonus_days_count": 2,  # bonus days 2, 3
        "fertilized_bonus_days_count": 0,
    }
    assert check_max_yield_met(state) is True


def test_melon_reaches_max_without_fertilizer():
    state = {
        "plant_type": "MELON",
        "age_in_days": 10,
        "watered_bonus_days_count": 6,  # bonus days 5–10
        "fertilized_bonus_days_count": 0,
    }
    assert compute_one_time_yield(6, 0, 6) == 6
    assert check_max_yield_met(state) is True


# --- Ongoing crops ---


def test_tomato_first_harvest_day():
    state = {
        "plant_type": "TOMATO",
        "age_in_days": 8,
        "watered_bonus_days_count": 0,
        "fertilized_bonus_days_count": 0,
        "harvests_collected": 0,
    }
    assert check_max_yield_met(state) is True


def test_tomato_between_harvest_days():
    state = {
        "plant_type": "TOMATO",
        "age_in_days": 9,
        "watered_bonus_days_count": 0,
        "fertilized_bonus_days_count": 0,
        "harvests_collected": 0,
    }
    assert check_max_yield_met(state) is False


def test_tomato_after_harvest_collected():
    state = {
        "plant_type": "TOMATO",
        "age_in_days": 8,
        "watered_bonus_days_count": 0,
        "fertilized_bonus_days_count": 0,
        "harvests_collected": 1,
    }
    assert check_max_yield_met(state) is False


def test_strawberry_every_other_day():
    ready = {
        "plant_type": "STRAWBERRY",
        "age_in_days": 12,
        "watered_bonus_days_count": 0,
        "fertilized_bonus_days_count": 0,
        "harvests_collected": 1,
    }
    not_ready = {**ready, "age_in_days": 11}
    assert check_max_yield_met(ready) is True
    assert check_max_yield_met(not_ready) is False


def test_unknown_plant_raises():
    with pytest.raises(ValueError, match="Unknown plant_type"):
        check_max_yield_met({"plant_type": "POTATO", "age_in_days": 1})
