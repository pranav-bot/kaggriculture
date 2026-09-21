"""Grid routing helpers for terminal-week search."""

from __future__ import annotations

from typing import Any


def shed_access_tiles(board_size: int = 10) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def walk(start: tuple[int, int], end: tuple[int, int]) -> list[list[str]]:
    x, y = start
    tx, ty = end
    return (
        [["EAST"]] * max(0, tx - x)
        + [["WEST"]] * max(0, x - tx)
        + [["SOUTH"]] * max(0, ty - y)
        + [["NORTH"]] * max(0, y - ty)
    )


def return_to_shed_drop(pos: tuple[int, int], board_size: int = 10) -> list[list[Any]]:
    targets = shed_access_tiles(board_size)
    target = min(targets, key=lambda xy: (abs(pos[0] - xy[0]) + abs(pos[1] - xy[1]), targets.index(xy)))
    return walk(pos, target) + [["DROP"]]
