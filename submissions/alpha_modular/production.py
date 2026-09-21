from __future__ import annotations
from typing import Any

from .mechanics import SHOPS, last_profitable_start_day, read
from .state import GameState


def desired_mix(state: GameState, config: dict[str, Any]) -> dict[str, int]:
    source = config["early_mix"] if state.day < int(config["mix_switch_day"]) else config["late_mix"]
    mix = {str(crop): int(count) for crop, count in source.items()}
    if not config.get("adaptive_mix") or state.day < int(config["mix_switch_day"]):
        return mix

    shops = read(state.town, "unlocked_shops", []) or []
    carrot_pull = sum(2 if shop == "PET_CAFE" else 1 for shop in shops if "CARROT" in SHOPS.get(shop, ()))
    wheat_pull = sum(1 for shop in shops if "WHEAT" in SHOPS.get(shop, ()))
    shift = min(3, abs(carrot_pull - wheat_pull))
    if carrot_pull > wheat_pull and mix.get("WHEAT", 0) >= shift:
        mix["CARROT"] = mix.get("CARROT", 0) + shift
        mix["WHEAT"] -= shift
    elif wheat_pull > carrot_pull and mix.get("CARROT", 0) >= shift:
        mix["WHEAT"] = mix.get("WHEAT", 0) + shift
        mix["CARROT"] -= shift
    return mix


def seed_deficits(state: GameState, config: dict[str, Any]) -> dict[str, int]:
    active = state.crop_counts()
    targets = desired_mix(state, config)
    deficits: dict[str, int] = {}
    for crop, target in targets.items():
        if state.day > last_profitable_start_day(crop):
            continue
        have = active.get(crop, 0) + int(state.seeds.get(crop, 0))
        if target > have:
            deficits[crop] = target - have
    return deficits


def plant_assignments(state: GameState, config: dict[str, Any]) -> list[tuple[tuple[int, int], str]]:
    active = state.crop_counts()
    targets = desired_mix(state, config)
    room = max(0, int(config["max_active"]) - sum(active.values()))
    if room <= 0:
        return []

    available = {crop: int(state.seeds.get(crop, 0)) for crop in targets}
    deficits = {
        crop: max(0, target - active.get(crop, 0))
        for crop, target in targets.items()
        if state.day <= last_profitable_start_day(crop)
    }
    order = [crop for crop in config["crop_order"] if crop in deficits]
    center = state.board_size // 2 - 1
    empties = sorted(
        state.empty_positions(),
        key=lambda p: (abs(p[0] - center) + abs(p[1] - center), p[1], p[0]),
    )
    assignments: list[tuple[tuple[int, int], str]] = []
    for position in empties:
        choices = [crop for crop in order if deficits.get(crop, 0) > 0 and available.get(crop, 0) > 0]
        if not choices or len(assignments) >= room:
            break
        crop = min(choices, key=lambda c: (-deficits[c] / max(1, targets[c]), order.index(c)))
        assignments.append((position, crop))
        deficits[crop] -= 1
        available[crop] -= 1
    return assignments
