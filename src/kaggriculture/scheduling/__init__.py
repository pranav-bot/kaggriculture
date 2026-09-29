"""Combinatorial strawberry scheduler: stagger planting under worker bandwidth.

Dual-engine constraint: 9 workers = 9 physical actions per hourly turn, shared
between the herd's strict dawn feeding block and the strawberry matrix. All 14
tiles planted at once would stack 14 simultaneous harvests onto feeding hours;
instead batches are day-staggered so harvest windows interleave, and every
operation is hour-slotted around the feeding block with a formal fit proof.

Hardcoded fertilized-strawberry lifecycle (env/actions configs):
  PLANT (day P) -> WATER daily -> first yield P+10 -> harvests P+10/+12/+14/+16
  (4 units each, fertilized cap) -> DIG to clear -> tile free P+17.
  Miss a watering (1+ unwatered days, unwatered today) -> WEED at end of day.
Herd block (18 head): 18 FEED + 18 CARE in dawn hours 0-3 (9/slot exactly).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# --- hardcoded lifecycle -------------------------------------------------------
STRAWBERRY_FIRST_YIELD = 10
STRAWBERRY_INTERVAL = 2
STRAWBERRY_HARVESTS = 4
STRAWBERRY_YIELD_UNITS = 4
FERTILIZE_EVERY_DAYS = 3

# --- bandwidth ------------------------------------------------------------------
WORKERS_PER_HOUR = 9
TURNS_PER_DAY = 24
FEED_BLOCK_HOURS = (0, 1, 2, 3)  # dawn herd block, 9 ops/slot
SEASON_DAYS = 30


def harvest_days(plant_day: int) -> List[int]:
    """The four harvest days of a batch planted on plant_day."""
    return [plant_day + STRAWBERRY_FIRST_YIELD + i * STRAWBERRY_INTERVAL
            for i in range(STRAWBERRY_HARVESTS)]


def batch_daily_ops(batch_size: int, age: int) -> Dict[str, int]:
    """Operations a batch of `batch_size` needs on a day of given age.

    age < 0 (unplanted) or age > 16 (cleared): nothing. Every growing day
    needs WATER per tile; harvest days add HARVEST per tile; every third day
    adds FERTILIZE per tile; day 17 needs DIG per tile (clear for replant).
    """
    if age < 0 or age > 17:
        return {}
    ops = {"WATER": batch_size}
    if age in (10, 12, 14, 16):
        ops["HARVEST"] = batch_size
    if age % FERTILIZE_EVERY_DAYS == 0:
        ops["FERTILIZE"] = batch_size
    if age == 17:
        ops = {"DIG": batch_size, "PLANT": batch_size}  # turnover day
    return ops


@dataclass
class TargetPlantingSchedule:
    """Staggered planting matrix + hour-slot proof for the routing layer."""
    batches: List[Tuple[int, int]]  # (plant_day, size), chronological
    herd_size: int = 18
    feed_block: Tuple[int, ...] = FEED_BLOCK_HOURS
    slots: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)
    peak_load: int = 0
    peak_harvest_day: int = 0

    def jobs_for_slot(self, day: int, hour: int) -> List[str]:
        """Op kinds due this hour (routing layer converts to tile jobs)."""
        return list(self.slots.get((day, hour), []))

    def to_dict(self) -> Dict[str, Any]:
        return {"batches": [{"plant_day": d, "size": s} for d, s in self.batches],
                "herd_size": self.herd_size, "peak_load": self.peak_load,
                "n_slots_used": len(self.slots)}

    def verify(self) -> Dict[str, Any]:
        """Prove: no slot exceeds bandwidth; no harvest inside feeding block."""
        worst = 0
        harvest_in_block = 0
        for (day, hour), ops in self.slots.items():
            worst = max(worst, len(ops))
            if hour in self.feed_block and "HARVEST" in ops:
                harvest_in_block += len([o for o in ops if o == "HARVEST"])
        feeding = 2 * self.herd_size  # FEED + CARE per head, inside block
        return {"max_ops_per_slot": worst,
                "within_bandwidth": worst <= WORKERS_PER_HOUR,
                "harvest_in_feed_block": harvest_in_block,
                "no_harvest_overlap": harvest_in_block == 0,
                "feeding_block_load": feeding,
                "feeding_block_capacity": len(self.feed_block) * WORKERS_PER_HOUR,
                "feeding_fits": feeding <= len(self.feed_block) * WORKERS_PER_HOUR}


def _day_demand(batches: List[Tuple[int, int]], day: int) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for plant_day, size in batches:
        for op, n in batch_daily_ops(size, day - plant_day).items():
            total[op] = total.get(op, 0) + n
    return total


def _slot_day(demand: Dict[str, int], day: int,
              feed_block: Tuple[int, ...]) -> Dict[Tuple[int, int], List[str]]:
    """Greedy hour assignment: HARVEST first (must finish same day), then
    DIG/PLANT turnover, FERTILIZE, WATER into earliest free non-block slots."""
    free = {h: WORKERS_PER_HOUR for h in range(TURNS_PER_DAY) if h not in feed_block}
    slots: Dict[Tuple[int, int], List[str]] = {}
    for op in ("HARVEST", "DIG", "PLANT", "FERTILIZE", "WATER"):
        remaining = demand.get(op, 0)
        for h in sorted(free):
            while remaining > 0 and len(slots.get((day, h), [])) < free[h]:
                slots.setdefault((day, h), []).append(op)
                remaining -= 1
            if remaining <= 0:
                break
        if remaining > 0:
            raise ValueError(f"day {day}: {remaining} {op} ops do not fit "
                             f"outside the feeding block")
    return slots


def build_schedule(batches: List[Tuple[int, int]],
                   herd_size: int = 18) -> TargetPlantingSchedule:
    """Slot a fixed batch plan; raises ValueError if bandwidth is violated."""
    sched = TargetPlantingSchedule(batches=sorted(batches), herd_size=herd_size)
    for day in range(SEASON_DAYS):
        demand = _day_demand(sched.batches, day)
        if demand:
            sched.slots.update(_slot_day(demand, day, sched.feed_block))
    sched.peak_load = max((len(v) for v in sched.slots.values()), default=0)
    sched.peak_harvest_day = 0
    for day in range(SEASON_DAYS):
        h = _day_demand(sched.batches, day).get("HARVEST", 0)
        sched.peak_harvest_day = max(sched.peak_harvest_day, h)
    proof = sched.verify()
    if not proof["within_bandwidth"] or not proof["no_harvest_overlap"] \
            or not proof["feeding_fits"]:
        raise ValueError(f"infeasible plan: {proof}")
    return sched


def optimize_stagger(n_tiles: int = 14, herd_size: int = 18,
                     first_day: int = 2, last_plant_day: int = 12,
                     stagger_days: Sequence[int] = (2, 3, 4)) -> TargetPlantingSchedule:
    """Search batch splits + stagger offsets; keep the lowest peak-load plan.

    Candidate splits cover 5-4-5 style partitions of n_tiles; each batch is
    planted stagger days after the previous, starting first_day. The winner
    The winner minimizes same-day harvest concurrency (the time-critical
    peak), then hourly peak, then finish day.
    """
    best: Optional[TargetPlantingSchedule] = None
    best_key: Optional[tuple] = None
    # Ordered 3-way integer partitions of n_tiles with each batch >= 3.
    splits = {(a, b, n_tiles - a - b)
              for a in range(3, n_tiles - 5) for b in range(3, n_tiles - a - 2)}
    for stagger in stagger_days:
        for split in sorted(splits):
            days = [first_day + i * stagger for i in range(len(split))]
            if days[-1] > last_plant_day:
                continue
            try:
                sched = build_schedule(list(zip(days, split)), herd_size)
            except ValueError:
                continue
            key = (sched.peak_harvest_day, sched.peak_load, days[-1])
            if best is None or (best_key is not None and key < best_key):
                best, best_key = sched, key
    if best is None:
        raise ValueError("no feasible stagger in the search grid")
    return best
