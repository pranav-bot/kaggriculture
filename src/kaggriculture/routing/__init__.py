"""High-density spatial packing for the Kuhn-Munkres routing layer.

Ladder autopsies: elites score $108k+ on three quadrants by segregating
livestock from cash crops. This module adds:

- Dynamic zoning of the unlocked 10x10 grid: NW = GRAZING (pastures), NE/SW
  = HIGH_YIELD (fertilized strawberries/melons), SE = FLEX overflow.
- Zone-violation penalties inside bipartite assignment: animal jobs on
  high-yield tiles (and crop jobs on grazing tiles) cost +100, which exceeds
  any Manhattan travel saving (max 18), so cows never pull work into the
  cash-crop zones and pastures are never sited there.
- Automated reclamation: weeds/exhausted tiles in HIGH_YIELD zones sort first
  and are KM-assigned to the nearest workers for instant DIG, so a tile is
  re-plantable the very next hour.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from kaggriculture.env.items import get_quadrant_bounds

Pos = Tuple[int, int]

GRAZING = "GRAZING"
HIGH_YIELD = "HIGH_YIELD"
FLEX = "FLEX"
LOCKED = "LOCKED"

# Hardcoded segregation: starter quadrant grazes; bought quadrants grow cash
# crops (Boey pattern: strawberries across NE/SW); late SE stays flexible.
ZONE_BY_QUADRANT = {"NW": GRAZING, "NE": HIGH_YIELD, "SW": HIGH_YIELD, "SE": FLEX}

# Must exceed the maximum 10x10 Manhattan distance (18) so zone discipline
# always beats travel convenience.
ZONE_VIOLATION_PENALTY = 100.0

CASH_CROPS = ("STRAWBERRY", "MELON")


def zone_map(unlocked_quadrants: Sequence[str]) -> Dict[Pos, str]:
    """Tile -> zone for every tile, locked quadrants map to LOCKED."""
    unlocked = {str(q).upper() for q in (unlocked_quadrants or [])}
    zones: Dict[Pos, str] = {}
    for quad, zone in ZONE_BY_QUADRANT.items():
        x0, x1, y0, y1 = get_quadrant_bounds(quad)
        for r in range(y0, y1):
            for c in range(x0, x1):
                zones[(c, r)] = zone if quad in unlocked else LOCKED
    return zones


def zone_of(pos: Sequence[int], zones: Mapping[Pos, str]) -> str:
    """Zone for an (x, y) position; off-grid positions are LOCKED."""
    return zones.get((int(pos[0]), int(pos[1])), LOCKED)


def _manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def packing_cost(
    unit_pos: Sequence[int],
    target: Pos,
    job_kind: str,
    zones: Mapping[Pos, str],
    base: Optional[Callable[[Sequence[int], Pos], float]] = None,
) -> float:
    """Travel cost plus hard zone-segregation penalties.

    job_kind "animal" (feed/care/collect/milk/wool/pasture-build) on a
    HIGH_YIELD tile, or job_kind "crop" (plant/water/harvest/fertilize) on a
    GRAZING tile, pays ZONE_VIOLATION_PENALTY.
    """
    cost = float(base(unit_pos, target) if base else _manhattan(unit_pos, target))
    zone = zone_of(target, zones)
    if job_kind == "animal" and zone == HIGH_YIELD:
        cost += ZONE_VIOLATION_PENALTY
    elif job_kind == "crop" and zone == GRAZING:
        cost += ZONE_VIOLATION_PENALTY
    return cost


def assign_with_penalties(
    unit_positions: Sequence[Sequence[int]],
    jobs: Sequence[Tuple[Pos, str, str]],
    zones: Mapping[Pos, str],
) -> List[Tuple[int, int]]:
    """Bipartite assign units to (target, command, job_kind) jobs, zone-aware.

    Same Hungarian/greedy structure as the scratch routing layer, with
    packing_cost as the edge weight. Returns (unit_idx, job_idx) pairs.
    """
    if not unit_positions or not jobs:
        return []
    costs = [[packing_cost(u, t, kind, zones) for (t, _, kind) in jobs]
             for u in unit_positions]
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(costs)
        return list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        pairs, used = [], set()
        for r in range(len(costs)):
            best, bj = None, -1
            for c in range(len(jobs)):
                if c not in used and (best is None or costs[r][c] < best):
                    best, bj = costs[r][c], c
            if bj >= 0:
                used.add(bj)
                pairs.append((r, bj))
        return pairs


def rank_pasture_sites(empties: Sequence[Pos], zones: Mapping[Pos, str]) -> List[Pos]:
    """Pasture construction sites: grazing first, flex ok, high-yield NEVER."""
    def key(p: Pos) -> tuple:
        z = zone_of(p, zones)
        return (0 if z == GRAZING else 1 if z == FLEX else 2, p[1], p[0])
    return sorted([p for p in empties if zone_of(p, zones) != HIGH_YIELD
                   and zone_of(p, zones) != LOCKED], key=key)


def _farm_tiles(obs: Mapping[str, Any], seat: int) -> tuple:
    farms = obs.get("farms") or []
    farm = farms[int(seat)] if 0 <= int(seat) < len(farms) else {}
    tiles = farm.get("tiles") or []
    return farm, tiles


def reclamation_jobs(obs: Mapping[str, Any], seat: int = 0,
                     day: int = 0) -> List[Tuple[Pos, str, str]]:
    """Ordered (target, command, job_kind) DIG/reclaim work list.

    HIGH_YIELD weeds first, then HIGH_YIELD exhausted plants, then everything
    else: every hour a high-yield tile sits unusable is lost terminal cash.
    Exhausted = zero-yield plant past first harvest age (HARVEST to clear,
    DIG attempt otherwise).
    """
    farm, tiles = _farm_tiles(obs, seat)
    quads = farm.get("unlocked_quadrants") or ["NW"]
    zones = zone_map(quads)
    hy_weeds, hy_spent, weeds, spent = [], [], [], []
    for r in range(10):
        for c in range(10):
            try:
                cell = tiles[r][c]
            except (IndexError, TypeError):
                continue
            if not isinstance(cell, Mapping):
                continue
            kind = cell.get("kind")
            pos = (c, r)
            hy = zone_of(pos, zones) == HIGH_YIELD
            if kind == "WEED":
                (hy_weeds if hy else weeds).append((pos, ["DIG"], "crop"))
            elif kind == "PLANT" and int(cell.get("yield_units", 1) or 0) <= 0:
                age = int(day) - int(cell.get("planted_day", day) or day)
                cmd = ["HARVEST"] if age >= 0 else ["DIG"]
                (hy_spent if hy else spent).append((pos, cmd, "crop"))
    return hy_weeds + hy_spent + weeds + spent


def reclamation_plan(obs: Mapping[str, Any], seat: int = 0,
                     day: int = 0) -> Dict[str, Any]:
    """Automated reclamation script: KM-assign workers to DIG/reclaim jobs.

    Returns {"farmer": cmd, "hands": [cmds], "jobs": [...]}. A unit standing
    on its job emits the reclaim command immediately, otherwise steps toward
    it -- so high-yield tiles are re-plantable the very next hour.
    """
    farm, _ = _farm_tiles(obs, seat)
    jobs = reclamation_jobs(obs, seat, day)
    farmer = tuple(farm.get("farmer", [4, 4]))
    hands = [tuple(p) for p in (farm.get("hands") or [])]
    units = [farmer, *hands]
    zones = zone_map(farm.get("unlocked_quadrants") or ["NW"])
    pairs = assign_with_penalties(units, [(t, c, k) for (t, c, k) in jobs], zones)
    cmds: List[list] = [["PASS"]] * (1 + len(hands))
    for ui, ji in pairs:
        (tx, ty), cmd, _ = jobs[ji]
        ux, uy = int(units[ui][0]), int(units[ui][1])
        if (ux, uy) == (tx, ty):
            cmds[ui] = cmd
        elif ux != tx:
            cmds[ui] = ["EAST" if tx > ux else "WEST"]
        else:
            cmds[ui] = ["SOUTH" if ty > uy else "NORTH"]
    return {"farmer": cmds[0], "hands": cmds[1:], "jobs": jobs,
            "assignments": pairs}
