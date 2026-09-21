"""Opponent regime classification and counter-mix knobs (research/05 innovation 4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from kaggriculture.helpers.opponent import OpponentProfile, analyze_opponent_farm
from kaggriculture.helpers.phase_brain import (
    CROP_ORDER,
    EARLY_MIX,
    LATE_MIX,
    MIX_SWITCH_DAY,
    PREMIUM_ITEMS,
    adaptive_shift,
    target_hired_hands,
)


class OpponentRegime(str, Enum):
    MELON_RUSH = "melon_rush"
    YARN_SHEEP = "yarn_sheep"
    AGGRESSIVE_EXPANDER = "aggressive_expander"
    MIXED_INACTIVE = "mixed_inactive"


@dataclass(frozen=True)
class RegimeKnobs:
    """Production and market overlays keyed off opponent regime."""

    mix_weights: dict[str, float]
    plant_priority: tuple[str, ...]
    delay_premium_sells: bool
    delay_melon_sells: bool
    delay_wool_sells: bool
    collision_aggressive: bool
    expander_match_pace: bool
    melon_sell_after_opponent_peak: bool


@dataclass
class RegimeState:
    regime: OpponentRegime
    knobs: RegimeKnobs
    profile: OpponentProfile


def _knobs_melon_rush() -> RegimeKnobs:
    return RegimeKnobs(
        mix_weights={"MELON": 0.55, "CARROT": 1.25, "WHEAT": 1.25},
        plant_priority=("CARROT", "WHEAT", "MELON"),
        delay_premium_sells=True,
        delay_melon_sells=True,
        delay_wool_sells=False,
        collision_aggressive=False,
        expander_match_pace=False,
        melon_sell_after_opponent_peak=True,
    )


def _knobs_yarn_sheep() -> RegimeKnobs:
    return RegimeKnobs(
        mix_weights={"WHEAT": 1.2, "CARROT": 1.1, "MELON": 0.85},
        plant_priority=("CARROT", "WHEAT", "MELON"),
        delay_premium_sells=False,
        delay_melon_sells=False,
        delay_wool_sells=True,
        collision_aggressive=False,
        expander_match_pace=False,
        melon_sell_after_opponent_peak=False,
    )


def _knobs_aggressive_expander() -> RegimeKnobs:
    return RegimeKnobs(
        mix_weights={"MELON": 1.0, "CARROT": 1.0, "WHEAT": 1.0},
        plant_priority=CROP_ORDER,
        delay_premium_sells=False,
        delay_melon_sells=False,
        delay_wool_sells=False,
        collision_aggressive=True,
        expander_match_pace=True,
        melon_sell_after_opponent_peak=False,
    )


def _knobs_mixed_inactive() -> RegimeKnobs:
    return RegimeKnobs(
        mix_weights={"MELON": 1.0, "CARROT": 1.0, "WHEAT": 1.0},
        plant_priority=CROP_ORDER,
        delay_premium_sells=False,
        delay_melon_sells=False,
        delay_wool_sells=False,
        collision_aggressive=True,
        expander_match_pace=False,
        melon_sell_after_opponent_peak=False,
    )


def classify_regime(profile: OpponentProfile, obs: Mapping[str, Any]) -> OpponentRegime:
    day = int(obs.get("day", 0) or 0)
    crops = profile.crop_counts
    animals = profile.animal_counts
    unlocked = len(profile.unlocked_quadrants or ["NW"])
    total_crop = max(1, profile.total_crops_planted)

    sheep = int(animals.get("SHEEP", 0) or 0)
    cows = int(animals.get("COW", 0) or 0)
    melon = int(crops.get("MELON", 0) or 0)

    if sheep >= 2 or (sheep >= 1 and cows >= 1):
        return OpponentRegime.YARN_SHEEP
    if melon >= 8 and melon / total_crop >= 0.35 and day <= 14:
        return OpponentRegime.MELON_RUSH
    if unlocked >= 3 and day <= 14:
        return OpponentRegime.AGGRESSIVE_EXPANDER
    if profile.total_crops_planted <= 8 and sum(int(v) for v in animals.values()) <= 1:
        return OpponentRegime.MIXED_INACTIVE
    return OpponentRegime.MIXED_INACTIVE


def knobs_for_regime(regime: OpponentRegime) -> RegimeKnobs:
    if regime == OpponentRegime.MELON_RUSH:
        return _knobs_melon_rush()
    if regime == OpponentRegime.YARN_SHEEP:
        return _knobs_yarn_sheep()
    if regime == OpponentRegime.AGGRESSIVE_EXPANDER:
        return _knobs_aggressive_expander()
    return _knobs_mixed_inactive()


def desired_mix_with_regime(
    day: int,
    unlocked_shops: list[str] | None,
    knobs: RegimeKnobs,
) -> dict[str, int]:
    source = EARLY_MIX if day < MIX_SWITCH_DAY else LATE_MIX
    mix = {str(crop): int(count) for crop, count in source.items()}
    if day >= MIX_SWITCH_DAY:
        mix = adaptive_shift(mix, unlocked_shops)
    adjusted: dict[str, int] = {}
    for crop, target in mix.items():
        weight = float(knobs.mix_weights.get(crop, 1.0))
        adjusted[crop] = max(0, int(round(target * weight)))
    return adjusted


def target_hired_hands_with_regime(
    unlocked_quadrants: list[str] | None,
    knobs: RegimeKnobs,
) -> int:
    base = target_hired_hands(unlocked_quadrants)
    if knobs.expander_match_pace:
        return base + 2
    return base


def opponent_melon_dump_imminent(profile: OpponentProfile) -> bool:
    for group in profile.crop_groups:
        if group.crop != "MELON":
            continue
        if group.days_until_optimal_harvest <= 1 and group.estimated_total_yield >= 10:
            return True
    return False


def sell_delay_items(
    knobs: RegimeKnobs,
    profile: OpponentProfile,
    *,
    clone_like_opponent: bool,
) -> set[str]:
    delayed: set[str] = set()
    if knobs.delay_melon_sells or (
        knobs.melon_sell_after_opponent_peak and opponent_melon_dump_imminent(profile)
    ):
        delayed.add("MELON")
    if knobs.delay_premium_sells:
        delayed |= set(PREMIUM_ITEMS)
    if knobs.delay_wool_sells and clone_like_opponent:
        delayed.add("WOOL")
    return delayed


def analyze_regime(obs: Mapping[str, Any]) -> RegimeState:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    opp_id = 1 - player
    opp_farm = farms[opp_id] if len(farms) > opp_id else {}
    market = obs.get("market") or {}
    profile = analyze_opponent_farm(
        opp_farm,
        current_day=int(obs.get("day", 0) or 0),
        current_hour=int(obs.get("hour", 0) or 0),
        market_prices=market.get("prices"),
    )
    regime = classify_regime(profile, obs)
    return RegimeState(regime=regime, knobs=knobs_for_regime(regime), profile=profile)
