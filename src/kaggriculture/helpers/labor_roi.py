"""Continuous Marginal Labor Value calculator for Kaggriculture.

Replaces the heuristic ``target_hands`` / naive ``HIRE_WORKER`` (``HIRE_MAX``)
trigger with a rigorous ROI test: hire the Nth worker today **only if** the
amortized dollar value of the extra physical actions strictly exceeds the
geometric compounding value of keeping that cash for land expansion
(Section: Day 0-15 liquidity race for the $1,000 NE / $2,000 SW quadrants).

Model
-----
Marginal cost of the Nth worker hired at turn ``t`` (0-719):

    direct(t, N) = FARM_HAND_COST_MULT * fib(hires_today + N - 1)
        fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ...

    The hire is a same-day throughput purchase (hands reset at dawn), so the
    only inter-temporal cost is *opportunity*: the locked ``direct`` dollars
    cannot compound in Strawberries (the dominant cash engine) nor be saved
    toward the next land quadrant.  With a strawberry gross multiple ``M``
    per ``C``-day cycle, the continuous daily rate is::

        r = ln(M) / C

    and the geometric compounding factor to turn 720 is::

        G(t) = M ** (days_remaining(t) / C) = exp(r * days_remaining(t))

    Total marginal cost::

        MC(N, t) = direct(N) * G(t)

Marginal value of 1 extra physical action per hour:

    One worker = 1 action per hourly turn for the rest of today, i.e.
    ``actions_left = 24 - hour`` actions.  Each action can clear at most one
    queued op.  Only time-critical decaying ops count: ``DIG`` (a blocked
    tile cannot be replanted) and ``HARVEST`` (yield decays 1 unit every
    other step past ``max_lifespan_step``, then weeds).  The scheduler query
    (``kaggriculture.scheduling``, Subtask 29) provides the queued,
    unfulfilled ``DIG``/``HARVEST`` demand that exceeds current bandwidth.
    Sorted by value-at-risk descending, the rescuable value of the Nth
    worker is the sum of the top-``actions_left`` task values.

Threshold solver::

    should_hire_worker(cash, queue, t) = rescuable_value(t) > MC(N, t)
        AND cash >= direct + OPERATING_RESERVE
        AND (outside Days 0-15 OR land-delay guard passes)

    The strict ``>`` guarantees we save for tomorrow's land expansion unless
    clearing the queue today is mathematically superior.

Beam-search override
--------------------
During the Day 0-15 liquidity race any ``HIRE_WORKER`` / ``HIRE_MAX``
macro-intent (index 4, see ``features/macro_intents.py``) or ``["HIRE"]``
market order must be vetoed when ``should_hire_worker`` is False.  Use
:func:`veto_hire_worker_macro`, :func:`filter_beam_trajectory`, or
:func:`filter_hire_market_orders` to enforce this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Engine constants (with standalone fallbacks)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - prefer authoritative engine values
    from kaggriculture.env.items import (
        EPISODE_STEPS as _EPISODE_STEPS,
        TURNS_PER_DAY as _TURNS_PER_DAY,
        FARM_HAND_COST_MULT as _MULT,
        MARKET_PARAMS as _MARKET_PARAMS,
    )
    from kaggriculture.actions.actions import Actions as _Actions

    EPISODE_END_TURN: int = int(_EPISODE_STEPS)  # 720
    TURNS_PER_DAY: int = int(_TURNS_PER_DAY)  # 24
    FARM_HAND_COST_MULT: int = int(_MULT)  # 1
    _STRAWBERRY_BASE: float = float(_MARKET_PARAMS["STRAWBERRY"]["base"])  # 120
    _STRAWBERRY_SEED: float = float(100)  # CROPS_DATA seed; Actions canonical
    try:
        _STRAWBERRY_SEED = float(_Actions.CROPS["STRAWBERRY"].seed_cost)
    except Exception:
        pass
except Exception:  # standalone fallback (tests without installed package)
    EPISODE_END_TURN = 720
    TURNS_PER_DAY = 24
    FARM_HAND_COST_MULT = 1
    _STRAWBERRY_BASE = 120.0
    _STRAWBERRY_SEED = 100.0
    _Actions = None  # type: ignore[assignment]

SEASON_DAYS: int = 30
LIQUIDITY_RACE_END_DAY: int = 15  # critical Day 0-15 liquidity race
OPERATING_RESERVE: float = 60.0  # matches scratch_grandmaster.OPERATING_RESERVE

# Land expansion ladder: NE $1k -> SW $2k -> SE $4k.
LAND_ORDER: Tuple[str, ...] = ("NE", "SW", "SE")
LAND_PRICES: Tuple[int, ...] = (1000, 2000, 4000)
LAND_PRICE_BY_QUADRANT: Dict[str, int] = dict(zip(LAND_ORDER, LAND_PRICES))

# Beam-search macro intent indices (keep in sync with features/macro_intents.py
# and scratch_grandmaster.DISCRETE_MACRO_ACTIONS).
HIRE_WORKER_INDEX: int = 4
HIRE_MAX_INDEX: int = 4  # alias: beam's naive HIRE_MAX intent is the same slot
HIRE_MACRO_NAMES = frozenset({"HIRE_WORKER", "HIRE_MAX", "HIRE"})
PASS_MACRO_INDEX: int = 0

# Strawberry compounding engine (opportunity-cost benchmark).
# Conservative unfertilized tile: 4 production events x 1 unit = 4 units total.
STRAWBERRY_SEED_COST: float = float(_STRAWBERRY_SEED)
STRAWBERRY_BASE_PRICE: float = float(_STRAWBERRY_BASE)
STRAWBERRY_UNITS_PER_TILE: int = 4
STRAWBERRY_CYCLE_DAYS: int = 17  # PLANT -> 4 harvests (P+10..P+16) -> DIG P+17
STRAWBERRY_GROSS_MULTIPLE: float = (
    STRAWBERRY_UNITS_PER_TILE * STRAWBERRY_BASE_PRICE / STRAWBERRY_SEED_COST
)  # 4*120/100 = 4.8x per 17-day cycle
STRAWBERRY_NET_PER_TILE: float = (
    STRAWBERRY_UNITS_PER_TILE * STRAWBERRY_BASE_PRICE - STRAWBERRY_SEED_COST
)  # $380 conservative NPV per freed tile

# Default per-op time-critical values when the queue only carries counts.
# HARVEST of one strawberry production ~= 1 unit @ base price.
DEFAULT_HARVEST_VALUE: float = float(STRAWBERRY_BASE_PRICE)  # $120
# DIG frees a tile for replant; worth the single-cycle net if still viable.
# Computed dynamically by dig_tile_npv(); this is the early-game ceiling.
DEFAULT_DIG_VALUE_CEILING: float = float(STRAWBERRY_NET_PER_TILE)  # $380

# Ops that decay in value if left unfulfilled (all others are non-blocking).
DECAYING_OPS = frozenset({"HARVEST", "DIG"})

# Backwards-compat alias for price overrides.
PRICE_PER_UNIT_FALLBACK: float = float(_STRAWBERRY_BASE)


# ---------------------------------------------------------------------------
# 1. Marginal cost: Fibonacci wage + geometric strawberry opportunity cost
# ---------------------------------------------------------------------------

def fib(n: int) -> int:
    """fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ..."""
    a, b = 1, 1
    for _ in range(max(0, int(n))):
        a, b = b, a + b
    return a


def hire_cost(hires_today: int, mult: int = FARM_HAND_COST_MULT) -> int:
    """Direct cash cost of the next hire given ``hires_today`` already hired."""
    if _Actions is not None:
        try:
            return int(_Actions.hire_cost(int(hires_today), mult=int(mult)))
        except Exception:
            pass
    return int(mult) * fib(int(hires_today))


def turn_to_day_hour(current_turn: int) -> Tuple[int, int]:
    """Map global turn 0-719 to (day 0-29, hour 0-23)."""
    t = max(0, min(int(current_turn), EPISODE_END_TURN - 1))
    return (t // TURNS_PER_DAY, t % TURNS_PER_DAY)


def remaining_turns(current_turn: int) -> int:
    """Turns left in the season including the current turn."""
    return max(0, EPISODE_END_TURN - int(current_turn))


def remaining_days(current_turn: int) -> float:
    """Fractional days left in the season."""
    return remaining_turns(current_turn) / float(TURNS_PER_DAY)


def strawberry_daily_rate(
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
) -> float:
    """Continuous daily compounding rate ``r = ln(M) / C`` of strawberries."""
    m = max(1e-12, float(gross_multiple))
    c = max(1, int(cycle_days))
    return math.log(m) / float(c)


def strawberry_growth_factor(
    current_turn: int,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
) -> float:
    """Geometric compounding factor ``G(t) = M ** (days_left / C)``.

    $1 of liquidity held at turn ``t`` and rolled into strawberries every
    cycle is worth ``G(t)`` dollars by turn 720 (before price slippage).
    At ``t = 720`` the factor is exactly 1.0.
    """
    days_left = remaining_days(current_turn)
    if days_left <= 0:
        return 1.0
    return float(gross_multiple) ** (days_left / float(max(1, int(cycle_days))))


def compounded_opportunity_cost(
    direct_cost: float,
    current_turn: int,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
) -> float:
    """Future value at turn 720 of ``direct_cost`` if invested in strawberries."""
    return float(direct_cost) * strawberry_growth_factor(
        current_turn, gross_multiple=gross_multiple, cycle_days=cycle_days
    )


@dataclass(frozen=True)
class MarginalCostBreakdown:
    """Itemized marginal cost of the Nth worker hired at turn ``t``."""

    nth_worker: int  # 1-indexed hire order today (1 = first hire today)
    hires_today: int  # hires already made today before this hire
    current_turn: int
    current_day: int
    direct_cost: int
    days_remaining: float
    growth_factor: float
    compounded_cost: float  # direct * G(t): the threshold the queue must beat


def marginal_cost_of_nth_worker(
    n: int,
    current_turn: int,
    hires_today: int = 0,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
) -> MarginalCostBreakdown:
    """Marginal cost of the Nth worker (1-indexed) hired at ``current_turn``.

    When the caller tracks ``hires_today`` explicitly, ``n`` should equal
    ``hires_today + 1``; when it does not, the Fibonacci index is
    ``hires_today`` (the cost of the *next* hire) and ``n`` is informational.
    For the common case (``hires_today=0``) the Nth worker costs ``fib(N-1)``.
    """
    h = max(0, int(hires_today))
    nth = max(1, int(n))
    # Canonical: Nth hire today costs fib(N-1) when starting from zero hires.
    # With pre-existing hires, the next hire costs fib(hires_today).
    direct = hire_cost(h, mult=FARM_HAND_COST_MULT)
    day, _ = turn_to_day_hour(current_turn)
    g = strawberry_growth_factor(current_turn, gross_multiple, cycle_days)
    return MarginalCostBreakdown(
        nth_worker=nth,
        hires_today=h,
        current_turn=int(current_turn),
        current_day=day,
        direct_cost=int(direct),
        days_remaining=remaining_days(current_turn),
        growth_factor=float(g),
        compounded_cost=float(direct) * float(g),
    )


def marginal_cost_of_next_worker(
    hires_today: int,
    current_turn: int,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
) -> MarginalCostBreakdown:
    """Convenience wrapper: cost of the next (``hires_today + 1``-th) worker."""
    return marginal_cost_of_nth_worker(
        int(hires_today) + 1, current_turn,
        hires_today=hires_today,
        gross_multiple=gross_multiple, cycle_days=cycle_days,
    )


def marginal_cost_curve(
    current_turn: int,
    max_n: int = 8,
    hires_today: int = 0,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
) -> List[MarginalCostBreakdown]:
    """Continuous marginal-cost curve for the next ``max_n`` hires (N=1..max)."""
    out: List[MarginalCostBreakdown] = []
    for k in range(1, max(1, int(max_n)) + 1):
        out.append(
            marginal_cost_of_nth_worker(
                int(hires_today) + k, current_turn,
                hires_today=int(hires_today) + k - 1,
                gross_multiple=gross_multiple, cycle_days=cycle_days,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 2. Marginal value: one extra physical action per hour via CropScheduler
# ---------------------------------------------------------------------------

@dataclass
class PendingTask:
    """One queued physical op that an extra worker could clear."""

    op: str  # "HARVEST" | "DIG" (only decaying ops carry value)
    value: float  # dollars rescued by executing this op today
    quantity: int = 1  # repeated identical ops collapse into one entry
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        return float(self.value) * int(self.quantity)


def dig_tile_npv(
    current_day: int,
    net_per_tile: float = STRAWBERRY_NET_PER_TILE,
    maturation_days: int = 10,
    horizon_day: int = SEASON_DAYS,
) -> float:
    """Net present value of freeing one tile with DIG on ``current_day``.

    A freed tile can only pay off if a replanted strawberry still matures
    before day 30 (first yield P+10 < 30).  Conservative single-cycle NPV:
    ``4 units x $120 - $100 seed = $380`` when viable, else $0.  Late-game
    DIG that cannot mature is worthless (no future cycles fit).
    """
    if int(current_day) + int(maturation_days) < int(horizon_day):
        return float(net_per_tile)
    return 0.0


def decaying_harvest_value(
    base_value: float,
    turns_overdue: int = 0,
    price_per_unit: float = STRAWBERRY_BASE_PRICE,
) -> float:
    """Value of a HARVEST op decaying per the engine's yield-decay rule.

    Past ``max_lifespan_step`` the tile loses 1 yield unit every other step
    (``CropConfig.simulate_decay``).  Each lost unit destroys ``price``.
    Models the decay as ``base - ceil((overdue+1)/2) * price`` floored at 0.
    An on-time harvest (``turns_overdue <= 0``) keeps full ``base_value``.
    """
    overdue = int(turns_overdue)
    if overdue <= 0:
        return max(0.0, float(base_value))
    decay_ticks = (overdue // 2) + 1
    return max(0.0, float(base_value) - decay_ticks * float(price_per_unit))


def normalize_pending_queue(
    pending_task_queue: Any,
    current_day: int = 0,
    harvest_value: float = DEFAULT_HARVEST_VALUE,
    price_per_unit: float = STRAWBERRY_BASE_PRICE,
) -> List[PendingTask]:
    """Normalize free-form queue input into a list of :class:`PendingTask`.

    Accepted shapes (all optional-value forms get engine-grounded defaults):

    - ``None`` / ``[]`` -> ``[]``
    - ``{"HARVEST": 5, "DIG": 3}`` (counts per op)
    - ``{"HARVEST": {"count": 5, "value_per_op": 120.0}, ...}``
    - ``["HARVEST", "DIG", "harvest", ...]`` (1.0 value each default)
    - ``[{"op": "HARVEST", "value": 120.0, "quantity": 2}, ...]``
    - ``[{"op": "HARVEST", "yield_units": 4, "price": 120.0}, ...]``
    - Objects with ``.op`` / ``.value`` attributes (e.g. scheduler jobs).
    """
    if pending_task_queue is None:
        return []
    tasks: List[PendingTask] = []

    def _default_value(op: str) -> float:
        op_u = str(op).upper()
        if op_u == "HARVEST":
            return float(harvest_value)
        if op_u == "DIG":
            return float(dig_tile_npv(int(current_day)))
        return 0.0

    def _push(op: str, value: Optional[float], qty: int = 1, detail: Optional[Dict[str, Any]] = None) -> None:
        op_u = str(op).upper()
        v = float(value) if value is not None else _default_value(op_u)
        q = max(1, int(qty))
        # Non-decaying ops (WATER/PLANT/...) carry no time-critical value here.
        if op_u not in DECAYING_OPS:
            v = 0.0
        # Expand quantity into per-op entries capped at unit granularity so
        # the top-k selection below is exact; collapse only for huge counts.
        if q <= 256:
            for _ in range(q):
                tasks.append(PendingTask(op=op_u, value=max(0.0, v), quantity=1, detail=dict(detail or {})))
        else:
            tasks.append(PendingTask(op=op_u, value=max(0.0, v), quantity=q, detail=dict(detail or {})))

    # Mapping form: {"HARVEST": 5} or {"HARVEST": {"count":..,"value_per_op":..}}
    if isinstance(pending_task_queue, Mapping):
        for op, spec in pending_task_queue.items():
            if isinstance(spec, Mapping):
                qty = int(spec.get("quantity", spec.get("count", 1)))
                val = spec.get("value_per_op", spec.get("value", spec.get("price", None)))
                if val is None and "yield_units" in spec:
                    try:
                        val = float(spec.get("yield_units", 1)) * float(spec.get("price", price_per_unit))
                    except Exception:
                        val = None
                overdue = int(spec.get("turns_overdue", spec.get("overdue", 0)))
                if str(op).upper() == "HARVEST" and overdue > 0 and val is None:
                    val = decaying_harvest_value(harvest_value, overdue, price_per_unit)
                elif str(op).upper() == "HARVEST" and overdue > 0 and val is not None:
                    val = decaying_harvest_value(float(val), overdue, price_per_unit)
                _push(str(op), val, qty, {str(k): v for k, v in spec.items()})
            else:
                try:
                    _push(str(op), None, int(spec))
                except Exception:
                    _push(str(op), None, 1)
        return tasks

    # Sequence form.
    if isinstance(pending_task_queue, (list, tuple)):
        for item in pending_task_queue:
            if item is None:
                continue
            if isinstance(item, str):
                _push(item, None, 1)
            elif isinstance(item, Mapping):
                op = str(item.get("op", item.get("action", item.get("type", "HARVEST"))))
                if "value" in item and "yield_units" not in item:
                    val: Optional[float] = float(item["value"])
                elif "yield_units" in item:
                    try:
                        val = float(item.get("yield_units", 1)) * float(item.get("price", price_per_unit))
                    except Exception:
                        val = _default_value(op)
                elif "value_per_op" in item:
                    val = float(item["value_per_op"])
                else:
                    val = None
                qty = int(item.get("quantity", item.get("count", 1)))
                overdue = int(item.get("turns_overdue", item.get("overdue", 0)))
                if op.upper() == "HARVEST" and overdue > 0:
                    val = decaying_harvest_value(float(val) if val is not None else harvest_value, overdue, price_per_unit)
                _push(op, val, qty, {str(k): v for k, v in item.items()})
            else:  # attribute-style job objects
                op = str(getattr(item, "op", getattr(item, "action", "HARVEST")))
                val = getattr(item, "value", None)
                qty = int(getattr(item, "quantity", 1))
                _push(op, float(val) if val is not None else None, qty)
        return tasks

    return []


def scheduler_unfulfilled_tasks(
    day_demand: Mapping[str, int],
    n_workers_available: int,
    current_day: int = 0,
    harvest_value: float = DEFAULT_HARVEST_VALUE,
    feed_block_hours: int = 4,
) -> List[PendingTask]:
    """Build the decaying-task queue from a CropScheduler day-demand dict.

    Queries the Subtask-29 scheduler convention: ``demand`` maps op ->
    required count today (see ``scheduling.batch_daily_ops`` /
    ``_day_demand``).  Daily bandwidth outside the dawn herd feeding block is
    ``n_workers * (24 - feed_block_hours)`` physical actions.  Any
    ``DIG``/``HARVEST`` demand above that bandwidth is *unfulfilled* and
    decaying — exactly the queue the marginal worker would clear first.

    Falls back gracefully when ``n_workers_available < 1`` (whole demand is
    unfulfilled) and ignores non-decaying ops (WATER/FERTILIZE/PLANT).
    """
    demand = {str(k).upper(): max(0, int(v)) for k, v in dict(day_demand or {}).items()}
    n = max(0, int(n_workers_available))
    capacity = n * max(0, TURNS_PER_DAY - int(feed_block_hours))
    # Time-critical ops first: HARVEST (same-day expiry) before DIG.
    unfulfilled: List[PendingTask] = []
    # Fulfilled capacity is shared across all ops; decaying ops are assumed
    # to be scheduled first (scheduler priority: HARVEST > DIG > ...), so
    # overflow is computed on the decaying subset after non-decaying load.
    other_load = sum(v for k, v in demand.items() if k not in DECAYING_OPS)
    free_for_decaying = max(0, capacity - other_load)
    decaying_need = sum(v for k, v in demand.items() if k in DECAYING_OPS)
    overflow = max(0, decaying_need - free_for_decaying)
    # Attribute overflow to HARVEST first (it decays fastest), then DIG.
    for op in ("HARVEST", "DIG"):
        need = int(demand.get(op, 0))
        if need <= 0 or overflow <= 0:
            continue
        take = min(need, overflow)
        overflow -= take
        if op == "HARVEST":
            unfulfilled.extend(
                PendingTask(op="HARVEST", value=float(harvest_value), quantity=1)
                for _ in range(take)
            )
        else:
            unfulfilled.extend(
                PendingTask(op="DIG", value=float(dig_tile_npv(int(current_day))), quantity=1)
                for _ in range(take)
            )
    return [t for t in unfulfilled if t.value > 0]


def scheduler_demand_from_batches(
    batches: Sequence[Tuple[int, int]],
    day: int,
) -> Dict[str, int]:
    """Total op demand for ``day`` from scheduler ``(plant_day, size)`` batches.

    Thin wrapper over ``kaggriculture.scheduling.batch_daily_ops`` so this
    module queries the CropScheduler instead of re-implementing lifecycle
    math.  Falls back to an empty demand when the scheduler is unavailable.
    """
    try:
        from kaggriculture.scheduling import batch_daily_ops as _ops
    except Exception:
        return {}
    total: Dict[str, int] = {}
    for plant_day, size in list(batches or []):
        try:
            for op, cnt in dict(_ops(int(size), int(day) - int(plant_day))).items():
                total[str(op).upper()] = total.get(str(op).upper(), 0) + int(cnt)
        except Exception:
            continue
    return total


def extra_worker_capacity(current_turn: int) -> int:
    """Physical actions the Nth worker adds: 1/hour for the rest of today."""
    _, hour = turn_to_day_hour(current_turn)
    return max(0, TURNS_PER_DAY - int(hour))


@dataclass(frozen=True)
class MarginalValueBreakdown:
    """Itemized marginal value of the Nth worker hired at turn ``t``."""

    current_turn: int
    current_day: int
    current_hour: int
    extra_actions: int
    n_decaying_tasks: int
    n_clearable: int
    rescuable_value: float  # amortized $ value of clearing top-k tasks today
    value_per_action: float  # rescuable / extra_actions (0 when no capacity)
    top_task_value: float  # value of the single best (next-action) task
    growth_adjusted: bool = False


def marginal_value_of_extra_worker(
    pending_task_queue: Any,
    current_turn: int,
    harvest_value: float = DEFAULT_HARVEST_VALUE,
    price_per_unit: float = STRAWBERRY_BASE_PRICE,
) -> MarginalValueBreakdown:
    """Value of 1 extra physical action/hour applied to the decaying queue.

    Normalizes ``pending_task_queue`` (any
    :func:`normalize_pending_queue` shape), keeps only decaying
    ``DIG``/``HARVEST`` ops with positive value, sorts descending, and sums
    the top-``extra_worker_capacity`` entries.  The result is the
    mathematically amortized same-day rescue value of hiring now.
    """
    day, hour = turn_to_day_hour(current_turn)
    tasks = normalize_pending_queue(
        pending_task_queue, current_day=day,
        harvest_value=harvest_value, price_per_unit=price_per_unit,
    )
    decaying = [t for t in tasks if t.op in DECAYING_OPS and t.value > 0]
    # Expand quantity>1 entries for exact top-k (quantities are small here).
    flat: List[float] = []
    for t in decaying:
        flat.extend([float(t.value)] * max(1, int(t.quantity)))
    flat.sort(reverse=True)
    cap = extra_worker_capacity(current_turn)
    take = min(cap, len(flat))
    rescued = float(sum(flat[:take])) if take > 0 else 0.0
    top = float(flat[0]) if flat else 0.0
    per_action = (rescued / float(cap)) if cap > 0 else 0.0
    return MarginalValueBreakdown(
        current_turn=int(current_turn),
        current_day=int(day),
        current_hour=int(hour),
        extra_actions=int(cap),
        n_decaying_tasks=int(len(flat)),
        n_clearable=int(take),
        rescuable_value=rescued,
        value_per_action=per_action,
        top_task_value=top,
    )


# ---------------------------------------------------------------------------
# 3. Threshold solver: should_hire_worker
# ---------------------------------------------------------------------------

def next_land_target(unlocked_quadrants: Optional[Sequence[str]] = None) -> Optional[Tuple[str, int]]:
    """Next unlockable quadrant and its price, or None when fully expanded."""
    have = {str(q).upper() for q in (unlocked_quadrants or ["NW"])}
    for q, price in zip(LAND_ORDER, LAND_PRICES):
        if q not in have:
            return (q, int(price))
    return None


@dataclass(frozen=True)
class HireDecision:
    """Full audit trail for one threshold evaluation."""

    hire: bool
    current_turn: int
    current_day: int
    current_hour: int
    direct_cost: int
    growth_factor: float
    compounded_cost: float
    rescuable_value: float
    extra_actions: int
    n_clearable: int
    affordable: bool
    in_liquidity_race: bool
    land_guard_pass: bool
    reason: str


def should_hire_worker(
    current_cash: float,
    pending_task_queue: Any,
    current_turn: int,
    hires_today: int = 0,
    unlocked_quadrants: Optional[Sequence[str]] = None,
    harvest_value: float = DEFAULT_HARVEST_VALUE,
    price_per_unit: float = STRAWBERRY_BASE_PRICE,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
    operating_reserve: float = OPERATING_RESERVE,
    liquidity_race_end_day: int = LIQUIDITY_RACE_END_DAY,
) -> bool:
    """Return True only if hiring the next worker has strictly positive ROI.

    The amortized same-day rescue value of clearing the decaying
    ``DIG``/``HARVEST`` queue must *strictly exceed* the geometric
    compounding value of saving the hire cash for land expansion::

        sum(top-k decaying task values) > direct * M ** (days_left / C)

    plus two hard guards:

    - *Affordability*: ``cash >= direct + operating_reserve`` (never breach
      the dawn-wage reserve).
    - *Liquidity-race land guard* (Days 0-``liquidity_race_end_day``): when
      the next quadrant (NE $1k / SW $2k) is affordable today, a hire that
      would drop cash below its price is vetoed — the discrete land unlock
      strictly dominates one day of throughput.  A hire that leaves land
      affordable (or land already out of reach either way) still needs the
      strict ROI inequality above.

    Args:
        current_cash: liquid cash on hand right now.
        pending_task_queue: decaying-task queue in any
            :func:`normalize_pending_queue` shape (counts, op strings, or
            explicit ``{op, value}`` dicts).  Only ``DIG``/``HARVEST``
            entries with positive value count.
        current_turn: global turn 0-719 (``day = turn // 24``).
        hires_today: hires already made today (Fibonacci index of next hire).
        unlocked_quadrants: e.g. ``["NW"]`` or ``["NW", "NE"]``.
        harvest_value / price_per_unit: per-op valuation overrides.
        gross_multiple / cycle_days: strawberry compounding overrides.
        operating_reserve: minimum cash to preserve after hiring.
        liquidity_race_end_day: last day the land guard applies (inclusive).

    Returns:
        True iff the Nth worker's amortized queue-clearing value strictly
        exceeds its compounded opportunity cost.
    """
    decision = explain_hire_decision(
        current_cash, pending_task_queue, current_turn,
        hires_today=hires_today, unlocked_quadrants=unlocked_quadrants,
        harvest_value=harvest_value, price_per_unit=price_per_unit,
        gross_multiple=gross_multiple, cycle_days=cycle_days,
        operating_reserve=operating_reserve,
        liquidity_race_end_day=liquidity_race_end_day,
    )
    return bool(decision.hire)


def explain_hire_decision(
    current_cash: float,
    pending_task_queue: Any,
    current_turn: int,
    hires_today: int = 0,
    unlocked_quadrants: Optional[Sequence[str]] = None,
    harvest_value: float = DEFAULT_HARVEST_VALUE,
    price_per_unit: float = STRAWBERRY_BASE_PRICE,
    gross_multiple: float = STRAWBERRY_GROSS_MULTIPLE,
    cycle_days: int = STRAWBERRY_CYCLE_DAYS,
    operating_reserve: float = OPERATING_RESERVE,
    liquidity_race_end_day: int = LIQUIDITY_RACE_END_DAY,
) -> HireDecision:
    """Same as :func:`should_hire_worker` but returns the full audit trail."""
    cash = float(current_cash)
    t = int(current_turn)
    day, hour = turn_to_day_hour(t)

    cost = marginal_cost_of_next_worker(
        int(hires_today), t,
        gross_multiple=gross_multiple, cycle_days=cycle_days,
    )
    value = marginal_value_of_extra_worker(
        pending_task_queue, t,
        harvest_value=harvest_value, price_per_unit=price_per_unit,
    )

    affordable = cash >= float(cost.direct_cost) + float(operating_reserve)
    if not affordable:
        return HireDecision(
            hire=False, current_turn=t, current_day=day, current_hour=hour,
            direct_cost=cost.direct_cost, growth_factor=cost.growth_factor,
            compounded_cost=cost.compounded_cost,
            rescuable_value=value.rescuable_value,
            extra_actions=value.extra_actions, n_clearable=value.n_clearable,
            affordable=False,
            in_liquidity_race=day <= int(liquidity_race_end_day),
            land_guard_pass=False,
            reason=(f"unaffordable: cash ${cash:.0f} < hire ${cost.direct_cost} "
                    f"+ reserve ${float(operating_reserve):.0f}"),
        )

    if value.extra_actions <= 0 or value.n_clearable <= 0 or value.rescuable_value <= 0:
        return HireDecision(
            hire=False, current_turn=t, current_day=day, current_hour=hour,
            direct_cost=cost.direct_cost, growth_factor=cost.growth_factor,
            compounded_cost=cost.compounded_cost,
            rescuable_value=value.rescuable_value,
            extra_actions=value.extra_actions, n_clearable=value.n_clearable,
            affordable=True,
            in_liquidity_race=day <= int(liquidity_race_end_day),
            land_guard_pass=True,
            reason="no decaying DIG/HARVEST tasks for the extra worker to clear",
        )

    # Strict ROI inequality: amortized rescue must beat compounded savings.
    roi_pass = bool(value.rescuable_value > cost.compounded_cost)

    # Liquidity-race land-delay guard (Days 0-15): never let a marginal hire
    # knock an affordable land unlock out of reach.
    in_race = bool(day <= int(liquidity_race_end_day))
    land_guard_pass = True
    land_note = ""
    if in_race:
        target = next_land_target(unlocked_quadrants)
        if target is not None:
            quad, price = target
            cash_after = cash - float(cost.direct_cost)
            if cash >= float(price) and cash_after < float(price):
                land_guard_pass = False
                land_note = (f"liquidity-race veto: hiring drops cash ${cash:.0f} "
                             f"-> ${cash_after:.0f} below {quad} ${price}; ")
            elif cash < float(price):
                # Land out of reach either way: the hire does not change land
                # timing, so only the strict ROI inequality decides.  Record
                # the shortfall for the audit trail.
                land_note = (f"race saving toward {quad} ${price} "
                             f"(shortfall ${float(price) - cash:.0f}); ")

    hire = bool(roi_pass and land_guard_pass)
    if hire:
        reason = (f"HIRE: rescue ${value.rescuable_value:.0f} ({value.n_clearable} ops) "
                  f"> compounded ${cost.compounded_cost:.1f} "
                  f"(direct ${cost.direct_cost} x G={cost.growth_factor:.2f}). {land_note}".strip())
    elif not roi_pass:
        reason = (f"save: rescue ${value.rescuable_value:.0f} <= compounded "
                  f"${cost.compounded_cost:.1f} (direct ${cost.direct_cost} x "
                  f"G={cost.growth_factor:.2f}). {land_note}".strip())
    else:
        reason = (land_note + f"rescue ${value.rescuable_value:.0f} blocked by land guard "
                  f"(compounded ${cost.compounded_cost:.1f})").strip()

    return HireDecision(
        hire=hire, current_turn=t, current_day=day, current_hour=hour,
        direct_cost=cost.direct_cost, growth_factor=cost.growth_factor,
        compounded_cost=cost.compounded_cost,
        rescuable_value=value.rescuable_value,
        extra_actions=value.extra_actions, n_clearable=value.n_clearable,
        affordable=True,
        in_liquidity_race=in_race,
        land_guard_pass=land_guard_pass,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# 4. Beam-search override: veto naive HIRE_MAX / HIRE_WORKER in the race
# ---------------------------------------------------------------------------

def veto_hire_worker_macro(
    current_cash: float,
    pending_task_queue: Any,
    current_turn: int,
    **kwargs: Any,
) -> bool:
    """True when the beam's naive hire intent must be overridden (vetoed).

    Thin negation of :func:`should_hire_worker` for call sites that think in
    terms of vetoes: returns True iff hiring is *not* ROI-positive.  Extra
    ``kwargs`` (``hires_today``, ``unlocked_quadrants``, ...) forward to the
    threshold solver.
    """
    return not bool(should_hire_worker(current_cash, pending_task_queue, current_turn, **kwargs))


def filter_beam_trajectory(
    trajectory: Sequence[int],
    current_cash: float,
    pending_task_queue: Any,
    current_turn: int,
    **kwargs: Any,
) -> List[int]:
    """Replace vetoed ``HIRE_WORKER`` (4) / ``HIRE_MAX`` macros with ``PASS`` (0).

    Applies the liquidity-race override to a beam-search macro trajectory so
    the naive hire intent can never execute when the ROI test fails.  Only
    the hire slots are touched; every other macro passes through unchanged.
    When the queue is empty or cash is tight, every hire in the trajectory
    becomes ``PASS``.
    """
    traj = [int(a) for a in list(trajectory or [])]
    if HIRE_WORKER_INDEX not in traj:
        return traj
    if should_hire_worker(current_cash, pending_task_queue, current_turn, **kwargs):
        return traj
    return [PASS_MACRO_INDEX if a == HIRE_WORKER_INDEX else a for a in traj]


def filter_hire_market_orders(
    orders: Sequence[Sequence[Any]],
    current_cash: float,
    pending_task_queue: Any,
    current_turn: int,
    **kwargs: Any,
) -> List[List[Any]]:
    """Strip ``["HIRE"]`` market orders when the ROI test fails.

    Drop-in guard for the ``_market``-style heuristic hire loop
    (``while hires < target_hands: orders.append(["HIRE"])``): call it on the
    final order list, or check :func:`should_hire_worker` inside the loop
    before appending each ``["HIRE"]``.  Non-hire orders pass through
    untouched and order is preserved.

    Each approved hire debits both cash (Fibonacci rung) and the decaying
    queue (top-``extra_worker_capacity`` tasks cleared), so a batch of HIRE
    orders is evaluated marginally — the 2nd HIRE faces a higher price and
    a smaller rescue pool than the 1st.
    """
    kept: List[List[Any]] = []
    cash = float(current_cash)
    hires_today = int(kwargs.get("hires_today", 0))
    base_kwargs = {k: v for k, v in kwargs.items() if k != "hires_today"}
    day, _ = turn_to_day_hour(int(current_turn))
    tasks = normalize_pending_queue(
        pending_task_queue, current_day=day,
        harvest_value=float(kwargs.get("harvest_value", DEFAULT_HARVEST_VALUE)),
        price_per_unit=float(kwargs.get("price_per_unit", PRICE_PER_UNIT_FALLBACK)),
    )
    remaining: List[float] = []
    for t in tasks:
        if t.op in DECAYING_OPS and t.value > 0:
            remaining.extend([float(t.value)] * max(1, int(t.quantity)))
    remaining.sort(reverse=True)
    cap = extra_worker_capacity(int(current_turn))
    for order in list(orders or []):
        tokens = [str(x).upper() for x in (order or [])]
        is_hire = bool(tokens) and tokens[0] in ("HIRE", "HIRE_HAND", "HIRE_WORKER", "HIRE_MAX")
        if not is_hire:
            kept.append(list(order))
            continue
        # Evaluate each successive hire against its own Fibonacci rung and
        # the not-yet-cleared remainder of the queue.
        view = [{"op": "HARVEST", "value": v} for v in remaining]
        if should_hire_worker(cash, view, current_turn,
                              hires_today=hires_today, **base_kwargs):
            kept.append(list(order))
            try:
                cash -= float(hire_cost(hires_today))
            except Exception:
                pass
            hires_today += 1
            del remaining[:max(0, cap)]
        # else: vetoed — drop this HIRE order.
    return kept


def max_roi_positive_hires(
    current_cash: float,
    pending_task_queue: Any,
    current_turn: int,
    max_cap: int = 8,
    **kwargs: Any,
) -> int:
    """Largest N such that hires 1..N are *each* ROI-positive in sequence.

    Replaces the fixed ``target_hands = 6/7/8`` heuristic: walk the Fibonacci
    ladder, spending cash rung by rung, and stop at the first hire whose
    rescuable remainder no longer strictly beats its compounded cost.  The
    consumed queue is debited as hires are granted (each hire clears the top
    ``extra_worker_capacity`` decaying tasks), so later hires face a
    shrinking rescue pool against a rising Fibonacci price — the continuous
    marginal-value curve crossing the marginal-cost curve.
    """
    day, _ = turn_to_day_hour(int(current_turn))
    tasks = normalize_pending_queue(
        pending_task_queue, current_day=day,
        harvest_value=float(kwargs.get("harvest_value", DEFAULT_HARVEST_VALUE)),
        price_per_unit=float(kwargs.get("price_per_unit", PRICE_PER_UNIT_FALLBACK)),
    )
    # Expand quantity>1.
    expanded: List[float] = []
    for t in tasks:
        if t.op in DECAYING_OPS and t.value > 0:
            expanded.extend([float(t.value)] * max(1, int(t.quantity)))
    expanded.sort(reverse=True)

    cash = float(current_cash)
    hires_today = int(kwargs.get("hires_today", 0))
    cap = extra_worker_capacity(int(current_turn))
    count = 0
    remaining = list(expanded)
    for _ in range(max(0, int(max_cap))):
        sub_kwargs = dict(kwargs)
        sub_kwargs["hires_today"] = hires_today
        # Build the remaining-queue view for this rung.
        view = [{"op": "HARVEST", "value": v} for v in remaining]
        if not should_hire_worker(cash, view, int(current_turn), **sub_kwargs):
            break
        direct = hire_cost(hires_today)
        cash -= float(direct)
        # Debit the tasks this hire clears.
        del remaining[:max(0, cap)]
        hires_today += 1
        count += 1
    return int(count)


__all__ = [
    "EPISODE_END_TURN",
    "TURNS_PER_DAY",
    "SEASON_DAYS",
    "LIQUIDITY_RACE_END_DAY",
    "OPERATING_RESERVE",
    "LAND_ORDER",
    "LAND_PRICES",
    "LAND_PRICE_BY_QUADRANT",
    "HIRE_WORKER_INDEX",
    "HIRE_MAX_INDEX",
    "HIRE_MACRO_NAMES",
    "PASS_MACRO_INDEX",
    "STRAWBERRY_SEED_COST",
    "STRAWBERRY_BASE_PRICE",
    "STRAWBERRY_UNITS_PER_TILE",
    "STRAWBERRY_CYCLE_DAYS",
    "STRAWBERRY_GROSS_MULTIPLE",
    "STRAWBERRY_NET_PER_TILE",
    "DEFAULT_HARVEST_VALUE",
    "DEFAULT_DIG_VALUE_CEILING",
    "DECAYING_OPS",
    "PendingTask",
    "MarginalCostBreakdown",
    "MarginalValueBreakdown",
    "HireDecision",
    "fib",
    "hire_cost",
    "turn_to_day_hour",
    "remaining_turns",
    "remaining_days",
    "strawberry_daily_rate",
    "strawberry_growth_factor",
    "compounded_opportunity_cost",
    "marginal_cost_of_nth_worker",
    "marginal_cost_of_next_worker",
    "marginal_cost_curve",
    "dig_tile_npv",
    "decaying_harvest_value",
    "normalize_pending_queue",
    "scheduler_unfulfilled_tasks",
    "scheduler_demand_from_batches",
    "extra_worker_capacity",
    "marginal_value_of_extra_worker",
    "next_land_target",
    "should_hire_worker",
    "explain_hire_decision",
    "veto_hire_worker_macro",
    "filter_beam_trajectory",
    "filter_hire_market_orders",
    "max_roi_positive_hires",
]
