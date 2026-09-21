"""Phase P1 production knobs (Architecture Alpha champion mix / labor / sells)."""

from __future__ import annotations

from typing import Any, Mapping

from kaggriculture.actions.actions import CROPS
from kaggriculture.env.items import SHOPS

EARLY_MIX: dict[str, int] = {"MELON": 14, "CARROT": 14, "WHEAT": 12}
LATE_MIX: dict[str, int] = {"CARROT": 20, "WHEAT": 20}
MIX_SWITCH_DAY = 13
MAX_ACTIVE = 40
CROP_ORDER = ("MELON", "CARROT", "WHEAT")

HANDS_BY_UNLOCKED: dict[int, int] = {1: 6, 2: 9}

PREMIUM_ITEMS = frozenset({"STRAWBERRY", "MELON", "MILK", "WOOL"})
PREMIUM_BATCH = 8

TERMINAL_DAY = 29
TERMINAL_RETURN_HOUR = 13
SEASON_DAYS = 30

SHED_PRESSURE_FULL_SELL = 82


def last_profitable_start_day(crop: str, season_days: int = SEASON_DAYS) -> int:
    cfg = CROPS[crop]
    if crop == "MELON":
        grow = 10
    elif cfg.ongoing:
        grow = cfg.first_yield_day
    else:
        grow = cfg.time_to_max_yield
    return season_days - 1 - grow


def adaptive_shift(mix: dict[str, int], unlocked_shops: list[str] | None) -> dict[str, int]:
    """Shift up to ±3 tiles between carrot and wheat from shop pull."""
    result = dict(mix)
    shops = unlocked_shops or []
    carrot_pull = sum(2 if shop == "PET_CAFE" else 1 for shop in shops if "CARROT" in SHOPS.get(shop, ()))
    wheat_pull = sum(1 for shop in shops if "WHEAT" in SHOPS.get(shop, ()))
    shift = min(3, abs(carrot_pull - wheat_pull))
    if carrot_pull > wheat_pull and result.get("WHEAT", 0) >= shift:
        result["CARROT"] = result.get("CARROT", 0) + shift
        result["WHEAT"] -= shift
    elif wheat_pull > carrot_pull and result.get("CARROT", 0) >= shift:
        result["WHEAT"] = result.get("WHEAT", 0) + shift
        result["CARROT"] -= shift
    return result


def desired_mix(day: int, unlocked_shops: list[str] | None = None) -> dict[str, int]:
    source = EARLY_MIX if day < MIX_SWITCH_DAY else LATE_MIX
    mix = {str(crop): int(count) for crop, count in source.items()}
    if day < MIX_SWITCH_DAY:
        return mix
    return adaptive_shift(mix, unlocked_shops)


def crop_counts(farm: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in farm.get("tiles", []) or []:
        for tile in row or []:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = str(tile.get("crop"))
                counts[crop] = counts.get(crop, 0) + 1
    return counts


def seed_deficits(
    farm: Mapping[str, Any],
    seeds: Mapping[str, int],
    day: int,
    unlocked_shops: list[str] | None = None,
) -> dict[str, int]:
    active = crop_counts(farm)
    targets = desired_mix(day, unlocked_shops)
    deficits: dict[str, int] = {}
    for crop, target in targets.items():
        if day > last_profitable_start_day(crop):
            continue
        have = active.get(crop, 0) + int(seeds.get(crop, 0))
        if target > have:
            deficits[crop] = target - have
    return deficits


def pick_plant_crop(
    available_seeds: Mapping[str, int],
    day: int,
    farm: Mapping[str, Any],
    unlocked_shops: list[str] | None = None,
) -> str | None:
    """Choose crop maximizing deficit ratio among seeds on hand."""
    targets = desired_mix(day, unlocked_shops)
    active = crop_counts(farm)
    deficits = {
        crop: max(0, targets.get(crop, 0) - active.get(crop, 0))
        for crop in CROP_ORDER
        if day <= last_profitable_start_day(crop)
    }
    choices = [crop for crop in CROP_ORDER if deficits.get(crop, 0) > 0 and int(available_seeds.get(crop, 0)) > 0]
    if not choices:
        return next((crop for crop, qty in available_seeds.items() if int(qty) > 0), None)
    return min(choices, key=lambda crop: (-deficits[crop] / max(1, targets.get(crop, 1)), CROP_ORDER.index(crop)))


def target_hired_hands(unlocked_quadrants: list[str] | None) -> int:
    unlocked = len(unlocked_quadrants or ["NW"])
    if unlocked in HANDS_BY_UNLOCKED:
        return HANDS_BY_UNLOCKED[unlocked]
    return HANDS_BY_UNLOCKED.get(max(HANDS_BY_UNLOCKED), 0)


def terminal_return_active(obs: Mapping[str, Any]) -> bool:
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", obs.get("step", 0) % 24) or 0)
    return day >= TERMINAL_DAY and hour >= TERMINAL_RETURN_HOUR


def alpha_sale_quantity(
    item: str,
    amount: int,
    day: int,
    shed_total: int,
    *,
    terminal_day: int = TERMINAL_DAY,
) -> int:
    if amount <= 0:
        return 0
    if day >= terminal_day or shed_total >= SHED_PRESSURE_FULL_SELL:
        return amount
    if item in PREMIUM_ITEMS:
        return min(amount, PREMIUM_BATCH)
    return amount
