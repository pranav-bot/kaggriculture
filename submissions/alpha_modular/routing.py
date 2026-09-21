from __future__ import annotations
from typing import Any

from mechanics import is_shed_adjacent, shed_access_tiles
from state import GameState, Unit
from tasks import Task


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(source: tuple[int, int], target: tuple[int, int], unit_index: int) -> list[str]:
    dx, dy = target[0] - source[0], target[1] - source[1]
    horizontal_first = unit_index % 2 == 0
    if horizontal_first and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    return ["PASS"]


def _nearest_shed(position: tuple[int, int], board_size: int) -> tuple[int, int]:
    return min(shed_access_tiles(board_size), key=lambda tile: (distance(position, tile), tile[1], tile[0]))


def _return_action(unit: Unit, state: GameState) -> list[Any]:
    if is_shed_adjacent(unit.position, state.board_size):
        return ["DROP"]
    return _step_toward(unit.position, _nearest_shed(unit.position, state.board_size), unit.index)


def assign_actions(state: GameState, tasks: list[Task], config: dict[str, Any]) -> tuple[list[list[Any]], dict[str, int]]:
    units = state.units()
    actions: list[list[Any]] = [["PASS"] for _ in units]
    assigned_units: set[int] = set()
    assigned_tasks: set[str] = set()
    planned_drop: dict[str, int] = {}
    terminal_return = state.day >= int(config["terminal_day"]) and state.hour >= int(config["terminal_return_hour"])

    if terminal_return:
        for unit in units:
            if sum(int(n) for n in unit.inventory.values()) <= 0:
                continue
            actions[unit.index] = _return_action(unit, state)
            assigned_units.add(unit.index)
            if actions[unit.index][0] == "DROP":
                for item, amount in unit.inventory.items():
                    planned_drop[item] = planned_drop.get(item, 0) + int(amount)

    pairs = []
    for task in tasks:
        for unit in units:
            if unit.index in assigned_units:
                continue
            pairs.append((task.priority, distance(unit.position, task.target), task.deadline_step, task.task_id, unit.index, task))
    for _, _, _, _, unit_index, task in sorted(pairs):
        if unit_index in assigned_units or task.task_id in assigned_tasks:
            continue
        unit = units[unit_index]
        actions[unit_index] = list(task.action) if unit.position == task.target else _step_toward(unit.position, task.target, unit.index)
        assigned_units.add(unit_index)
        assigned_tasks.add(task.task_id)

    return actions, planned_drop
