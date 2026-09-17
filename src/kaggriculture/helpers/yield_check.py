"""
Yield readiness helpers for Kaggriculture crops.

Determines whether a plant has reached its maximum possible yield (one-time crops)
or is on a scheduled harvest day with produce ready (ongoing crops).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, TypedDict, Union


class PlantState(TypedDict, total=False):
    """Current status of a single plant tile."""

    plant_type: str
    age_in_days: int
    watered_bonus_days_count: int
    fertilized_bonus_days_count: int
    harvests_collected: int


@dataclass(frozen=True)
class _OneTimeCropSpec:
    time_to_max_yield: int
    absolute_max_yield: int
    unfertilized_max_yield: int | None = None  # None → same as absolute (e.g. melon)


@dataclass(frozen=True)
class _OngoingCropSpec:
    first_yield_day: int
    interval: int
    max_lifetime_harvests: int


_ONE_TIME_CROPS: dict[str, _OneTimeCropSpec] = {
    "WHEAT": _OneTimeCropSpec(4, 6, 4),
    "CARROT": _OneTimeCropSpec(3, 4, 3),
    "MELON": _OneTimeCropSpec(10, 6),
}

_ONGOING_CROPS: dict[str, _OngoingCropSpec] = {
    "TOMATO": _OngoingCropSpec(first_yield_day=8, interval=1, max_lifetime_harvests=4),
    "STRAWBERRY": _OngoingCropSpec(first_yield_day=10, interval=2, max_lifetime_harvests=4),
}


def _bonus_window_days(time_to_max_yield: int) -> tuple[int, int, int]:
    """Return (start_age, end_age, day_count) for the watering bonus window.

    The bonus window opens at ceil(time_to_max_yield / 2) and closes at
    time_to_max_yield. Each in-window watered day adds +1 yield; watered +
    fertilized days add +2 instead.
    """
    start = math.ceil(time_to_max_yield / 2)
    end = time_to_max_yield
    return start, end, end - start + 1


def compute_one_time_yield(
    watered_bonus_days_count: int,
    fertilized_bonus_days_count: int,
    absolute_max_yield: int,
) -> int:
    """Yield for a one-time crop: base 1 + bonus contributions, capped."""
    raw = 1 + watered_bonus_days_count + (2 * fertilized_bonus_days_count)
    return min(absolute_max_yield, raw)


def max_achievable_one_time_yield(
    spec: _OneTimeCropSpec,
    fertilized_bonus_days_count: int,
) -> int:
    """Highest yield this crop can reach given whether fertilizer was ever used."""
    _, _, bonus_days = _bonus_window_days(spec.time_to_max_yield)
    if fertilized_bonus_days_count > 0:
        return min(spec.absolute_max_yield, 1 + bonus_days * 2)
    cap = spec.unfertilized_max_yield or spec.absolute_max_yield
    return min(cap, 1 + bonus_days)


def _one_time_max_yield_met(state: Mapping[str, int | str], spec: _OneTimeCropSpec) -> bool:
    age = int(state["age_in_days"])
    watered = int(state["watered_bonus_days_count"])
    fertilized = int(state["fertilized_bonus_days_count"])

    # Yield only stops increasing once the plant reaches time_to_max_yield.
    if age < spec.time_to_max_yield:
        return False

    _, _, bonus_days = _bonus_window_days(spec.time_to_max_yield)
    if watered + fertilized > bonus_days:
        watered = max(0, bonus_days - fertilized)

    current_yield = compute_one_time_yield(watered, fertilized, spec.absolute_max_yield)
    achievable = max_achievable_one_time_yield(spec, fertilized)
    return current_yield >= achievable


def _ongoing_max_yield_met(state: Mapping[str, int | str], spec: _OngoingCropSpec) -> bool:
    age = int(state["age_in_days"])
    harvests_collected = int(state.get("harvests_collected", 0))

    if age < spec.first_yield_day:
        return False
    if harvests_collected >= spec.max_lifetime_harvests:
        return False

    days_since_first = age - spec.first_yield_day
    if days_since_first % spec.interval != 0:
        return False

    # harvest_index is 1-based; produce is ready when the next collection is due.
    harvest_index = days_since_first // spec.interval + 1
    return harvest_index == harvests_collected + 1


def check_max_yield_met(plant_state: Union[PlantState, Mapping[str, int | str]]) -> bool:
    """Return True when a plant is at peak yield and ready for immediate harvest.

    One-time crops (wheat, carrot, melon):
        - Must have reached ``time_to_max_yield``.
        - Current yield (from bonus-day counts) must equal the best yield
          achievable for that crop given whether fertilizer was used during
          the bonus window.

    Ongoing crops (tomato, strawberry):
        - Must be on a scheduled yield day with an uncollected harvest pending.
        - Watering/fertilizing does not change unit count (always 1 per tick).
    """
    plant_type = str(plant_state["plant_type"]).upper()

    if plant_type in _ONE_TIME_CROPS:
        return _one_time_max_yield_met(plant_state, _ONE_TIME_CROPS[plant_type])
    if plant_type in _ONGOING_CROPS:
        return _ongoing_max_yield_met(plant_state, _ONGOING_CROPS[plant_type])

    known = sorted(_ONE_TIME_CROPS) + sorted(_ONGOING_CROPS)
    raise ValueError(f"Unknown plant_type '{plant_type}'. Expected one of: {known}")
