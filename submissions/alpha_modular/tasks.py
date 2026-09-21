from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

from mechanics import ANIMALS, CROPS, expected_unfertilized_yield, harvest_age, read
from production import plant_assignments
from state import GameState


@dataclass(frozen=True)
class Task:
    task_id: str
    target: tuple[int, int]
    action: tuple[Any, ...]
    priority: int
    deadline_step: int
    loss_if_omitted: float
    reason: str


def _plant_task(state: GameState, x: int, y: int, tile: Mapping[str, Any]) -> Task | None:
    crop = str(read(tile, "crop"))
    data = CROPS[crop]
    age = state.day - int(read(tile, "planted_day", state.day))
    watered = bool(read(tile, "watered_today", False))
    missed = int(read(tile, "consecutive_unwatered", 0))
    held = int(read(tile, "yield_units", 0))
    terminal = state.day >= 29

    if data["ongoing"]:
        if held > 0 and (terminal or held >= int(data["max_yield"])):
            return Task(f"harvest:{x}:{y}", (x, y), ("HARVEST",), 1 if terminal else 3, state.step + 1, held * 50.0, "collect ongoing yield")
        if not watered and not terminal:
            priority = 0 if missed >= 1 else 3
            return Task(f"water:{x}:{y}", (x, y), ("WATER",), priority, state.step + (1 if missed >= 1 else 20), 200.0, "prevent crop death")
        return None

    target_age = harvest_age(crop)
    target_yield = expected_unfertilized_yield(crop)
    if terminal and held > 0:
        return Task(f"harvest:{x}:{y}", (x, y), ("HARVEST",), 0, state.step + 1, held * state.market_price(crop), "terminal liquidation harvest")
    if age >= target_age:
        can_add_yield = age <= int(data["max_yield_day"]) and held < target_yield
        if not watered and can_add_yield:
            return Task(f"water-peak:{x}:{y}", (x, y), ("WATER",), 1, state.step + 8, state.market_price(crop), "realize peak yield")
        return Task(f"harvest:{x}:{y}", (x, y), ("HARVEST",), 2, state.step + 12, held * state.market_price(crop), "mature harvest")
    if not watered:
        priority = 0 if missed >= 1 else 4
        return Task(
            f"water:{x}:{y}",
            (x, y),
            ("WATER",),
            priority,
            state.step + (1 if missed >= 1 else 20),
            float(CROPS[crop]["seed"] + target_yield * state.market_price(crop)),
            "prevent crop death" if missed >= 1 else "daily yield service",
        )
    return None


def materialize_tasks(state: GameState, config: dict[str, Any]) -> list[Task]:
    tasks: list[Task] = []
    for x, y, tile in state.iter_tiles():
        if tile is None or tile == "LOCKED":
            continue
        if isinstance(tile, Mapping) and read(tile, "kind") == "PLANT":
            task = _plant_task(state, x, y, tile)
            if task:
                tasks.append(task)
        elif isinstance(tile, Mapping) and "animal" in tile:
            animal = str(read(tile, "animal"))
            if not bool(read(tile, "fed_today", False)):
                tasks.append(Task(f"feed:{x}:{y}", (x, y), ("FEED",), 0, state.step + 1, float(ANIMALS[animal]["cost"]), "prevent animal escape"))
            elif int(read(tile, "yield_units", 0)) > 0:
                tasks.append(Task(f"animal-harvest:{x}:{y}", (x, y), ("HARVEST",), 3, state.step + 20, 100.0, "collect animal product"))
        elif isinstance(tile, Mapping) and read(tile, "kind") == "WEED" and state.day < 27:
            tasks.append(Task(f"dig:{x}:{y}", (x, y), ("DIG",), 8, state.step + 48, 10.0, "recover productive tile"))

    for (x, y), crop in plant_assignments(state, config):
        tasks.append(Task(f"plant:{crop}:{x}:{y}", (x, y), ("PLANT", crop), 7, state.step + 20, 50.0, "selected production pattern"))
    return sorted(tasks, key=lambda task: (task.priority, task.deadline_step, task.target[1], task.target[0], task.task_id))
