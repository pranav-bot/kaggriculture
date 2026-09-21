from kaggriculture.helpers.phase_brain import (
    adaptive_shift,
    alpha_sale_quantity,
    desired_mix,
    seed_deficits,
    terminal_return_active,
)


def test_desired_mix_switches_on_day_13():
    early = desired_mix(10, [])
    late = desired_mix(13, [])
    assert "MELON" in early and early["MELON"] == 14
    assert "MELON" not in late


def test_adaptive_shift_moves_wheat_to_carrot_for_pet_cafe():
    mix = {"CARROT": 20, "WHEAT": 20}
    shifted = adaptive_shift(mix, ["PET_CAFE"])
    assert shifted["CARROT"] > mix["CARROT"]
    assert shifted["WHEAT"] < mix["WHEAT"]


def test_alpha_sale_quantity_batches_premium():
    assert alpha_sale_quantity("MELON", 20, day=10, shed_total=50) == 8
    assert alpha_sale_quantity("MELON", 20, day=10, shed_total=82) == 20


def test_terminal_return_active():
    assert terminal_return_active({"day": 29, "hour": 13}) is True
    assert terminal_return_active({"day": 29, "hour": 12}) is False


def test_seed_deficits_counts_active_and_seeds():
    farm = {"tiles": [[{"kind": "PLANT", "crop": "WHEAT"}] + [None] * 9]}
    deficits = seed_deficits(farm, {"WHEAT": 2, "CARROT": 0, "MELON": 0}, day=5, unlocked_shops=[])
    assert deficits.get("WHEAT", 0) > 0
