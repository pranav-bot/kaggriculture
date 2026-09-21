"""Idle-hand water rescue overlay (research/08 Overlay D)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from kaggriculture.env.items import TURNS_PER_DAY

_OPPOSITE = {"EAST": "WEST", "WEST": "EAST", "NORTH": "SOUTH", "SOUTH": "NORTH"}

RESCUE_DAY_START = 6
RESCUE_DAY_END = 17
RESCUE_HOUR_START = 16
RESCUE_HOUR_END = 21
MAX_ROUTE_COMMANDS = 7
MAX_RESCUES_PER_DAY = 2


@dataclass
class WaterRescueTracker:
    """Per-hand rescue routes and daily cap."""

    routes: dict[int, list[list[Any]]] = field(default_factory=dict)
    cursors: dict[int, int] = field(default_factory=dict)
    rescues_today: int = 0
    tracked_day: int = -1

    def reset_day(self, day: int) -> None:
        if day != self.tracked_day:
            self.tracked_day = day
            self.rescues_today = 0
            self.routes.clear()
            self.cursors.clear()


def _path_to(src: tuple[int, int], dst: tuple[int, int]) -> list[list[str]]:
    x, y = src
    tx, ty = dst
    cmds: list[list[str]] = []
    while x != tx:
        cmds.append(["EAST" if tx > x else "WEST"])
        x += 1 if tx > x else -1
    while y != ty:
        cmds.append(["SOUTH" if ty > y else "NORTH"])
        y += 1 if ty > y else -1
    return cmds


def build_water_rescue_route(
    start: tuple[int, int],
    target_xy: tuple[int, int],
    max_commands: int = MAX_ROUTE_COMMANDS,
) -> list[list[Any]] | None:
    go = _path_to(start, target_xy)
    work = go + [["WATER"]]
    back = [[_OPPOSITE[step[0]]] for step in reversed(go)]
    route = work + back
    if len(route) > max_commands:
        return None
    return route


def _rescue_window_active(obs: Mapping[str, Any]) -> bool:
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", obs.get("step", 0) % TURNS_PER_DAY) or 0)
    step = int(obs.get("step", day * TURNS_PER_DAY + hour) or 0)
    if not (RESCUE_DAY_START <= day < RESCUE_DAY_END):
        return False
    if not (RESCUE_HOUR_START <= hour <= RESCUE_HOUR_END):
        return False
    return RESCUE_DAY_START * TURNS_PER_DAY <= step < RESCUE_DAY_END * TURNS_PER_DAY


def _find_rescue_targets(farm: Mapping[str, Any]) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row or []):
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            if tile.get("watered_today"):
                continue
            if int(tile.get("consecutive_unwatered", 0) or 0) < 1:
                continue
            targets.append((x, y))
    targets.sort(key=lambda p: (p[1], p[0]))
    return targets


def apply_idle_water_rescue(
    obs: Mapping[str, Any],
    farmer_act: list[Any],
    hands_act: list[list[Any]],
    tracker: WaterRescueTracker,
    farm: Mapping[str, Any],
    private: Mapping[str, Any],
) -> tuple[list[Any], list[list[Any]]]:
    """Replace qualifying hand PASS actions with one step of a water rescue route."""
    if not _rescue_window_active(obs):
        return farmer_act, hands_act

    day = int(obs.get("day", 0) or 0)
    tracker.reset_day(day)
    positions = [tuple(farm.get("farmer", [0, 0])), *(tuple(p) for p in farm.get("hands") or [])]
    inventories = private.get("inventories") or []
    targets = _find_rescue_targets(farm)
    updated_hands = [list(action) for action in hands_act]

    for hand_idx, action in enumerate(updated_hands):
        unit_idx = hand_idx + 1
        if hand_idx in tracker.routes:
            route = tracker.routes[hand_idx]
            cursor = tracker.cursors.get(hand_idx, 0)
            if cursor < len(route):
                updated_hands[hand_idx] = list(route[cursor])
                tracker.cursors[hand_idx] = cursor + 1
                if tracker.cursors[hand_idx] >= len(route):
                    tracker.routes.pop(hand_idx, None)
                    tracker.cursors.pop(hand_idx, None)
            continue

        if action and action[0] != "PASS":
            continue
        if unit_idx >= len(positions):
            continue
        inv = inventories[unit_idx] if unit_idx < len(inventories) else {}
        if sum(int(v or 0) for v in inv.values()) > 0:
            continue
        if tracker.rescues_today >= MAX_RESCUES_PER_DAY or not targets:
            continue

        start = (int(positions[unit_idx][0]), int(positions[unit_idx][1]))
        target = min(targets, key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]))
        route = build_water_rescue_route(start, target)
        if not route:
            continue
        tracker.routes[hand_idx] = route
        tracker.cursors[hand_idx] = 0
        tracker.rescues_today += 1
        updated_hands[hand_idx] = list(route[0])

    return farmer_act, updated_hands
