"""Resource Starvation subroutine for Kaggriculture (fertilizer denial).

Bankrupts an opponent's high-yield crop cycle on the shared wholesale market:
when the opponent-intent forecaster predicts a fertilizer-intensive rush
(e.g. ``PREPARING_MELON_RUSH``), buy just enough Fertilizer to spike its
unit price above the opponent's liquid cash, mathematically locking them out
of the required volume — but only when our expected holding loss is strictly
less than the terminal cash the opponent loses by missing the harvest window.

Model
-----
Shared-market pricing (``env/items.py::market_price``):

    P(inv) = base + amp * f(|inv - I0|),  floored at $1.

Fertilizer is linear both sides (base 100, T 200, target 0.40), so
``amp = 0.40 * 100 / 200 = 0.2 $/unit`` and every bought unit durably lowers
market inventory (no town shop consumes fertilizer, so there is no drain
recovery — denial is persistent until someone sells back).

BUY execution quotes at post-buy inventory
(``helpers/market_prediction.py::simulate_buy_slippage``), hence exactly:

    buy_cost(q, inv0)  = sum_{i=1..q} P(inv0 - i)
    opp_cost(q)        = buy_cost(V, inv0 - q)   # opponent buys V after us

``opp_cost(q)`` is monotone increasing in ``q`` (scarcity raises price), so
the optimal denial volume is the least ``q`` with:

    opp_cost(q) > opp_cash                      # strict lockout

subject to our feasibility (``buy_cost(q) <= our_cash - reserve`` and
``q <= shed_room``).  If ``opp_cost(0) > opp_cash`` the opponent is already
locked and the optimal volume is 0 (no-op).  If no feasible ``q`` locks them,
denial is infeasible.

Opponent requirement for a melon rush (``actions/actions.py``: melon bonus
window ages 6-12 = 7 days; ``FertilizerConfig`` duration 3 days):

    V = max(0, melon_tiles * ceil(7 / 3) - self_supply - held)

Melon economics (``CROPS_DATA``: seed 80, max yield 6 units, base price 250):

    L_opp = tiles * (6 * melon_price - 80)      # net margin of missed window
            (0 when day + 10 >= 30: replant cannot mature — nothing at risk)

Our holding loss (conservative; resale reverses the linear curve):

    L_us(q) = buy_cost(q) - own_value(usable) - sell_revenue(q - usable)

where ``usable = min(q, own_need)`` units are valued at replacement cost
(procurement we would buy anyway) and the surplus is resold sequentially
(``simulate_sell_slippage`` semantics, floor-aware).  Near I0 the round trip
is nearly lossless; the binding constraints are cash and shed room.

Risk gate (strict, per spec):

    execute  <=>  feasible q* > 0  AND  L_us(q*) < L_opp  AND  L_opp > 0
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
        MARKET_I0 as _I0,
        MARKET_PARAMS as _PARAMS,
        PRICE_FLOOR as _FLOOR,
        SHED_CAPACITY as _SHED_CAP,
        CROPS_DATA as _CROPS,
        FERTILIZER_DATA as _FERT,
        market_price as _engine_price,
    )

    MARKET_I0: int = int(_I0)
    PRICE_FLOOR: int = int(_FLOOR)
    SHED_CAPACITY: int = int(_SHED_CAP)
    FERT_BASE: int = int(_PARAMS["FERTILIZER"]["base"])
    MELON_BASE: int = int(_PARAMS["MELON"]["base"])
    MELON_SEED: int = int(_CROPS["MELON"]["seed"])
    MELON_MAX_YIELD: int = int(_CROPS["MELON"]["max_yield"])
    FERT_DURATION_DAYS: int = int(_FERT["duration_days"])
except Exception:  # standalone fallback (tests without installed package)
    MARKET_I0 = 10_000
    PRICE_FLOOR = 1
    SHED_CAPACITY = 100
    FERT_BASE = 100
    MELON_BASE = 250
    MELON_SEED = 80
    MELON_MAX_YIELD = 6
    FERT_DURATION_DAYS = 3
    _engine_price = None  # type: ignore[assignment]

TARGET_FERTILIZER = "FERTILIZER"

# Melon bonus window ages 6..12 inclusive = 7 days (CropConfig.bonus_window).
MELON_BONUS_WINDOW_DAYS: int = 7
# Full-coverage applications per tile: ceil(7 / 3) = 3.
FERT_UNITS_PER_MELON_TILE: int = int(
    math.ceil(MELON_BONUS_WINDOW_DAYS / max(1, FERT_DURATION_DAYS))
)
# Melon needs 10 days seed -> harvest; planting on day >= 20 never matures.
MELON_MATURATION_DAYS: int = 10
SEASON_DAYS: int = 30

# Rush-like intents that require market fertilizer (normalized, upper-case).
RUSH_INTENTS = frozenset({
    "PREPARING_MELON_RUSH",
    "MELON_RUSH",
    "HOARDING_FERTILIZER",
})
# Intents that never justify fertilizer denial (no rush requirement).
NON_RUSH_INTENTS = frozenset({
    "PREPARING_EXPANSION",
    "LIVESTOCK_RUSH",
    "CROP_ROTATION",
    "MIXED_INACTIVE",
})


# ---------------------------------------------------------------------------
# 1. Intent ingestion -> fertilizer requirement
# ---------------------------------------------------------------------------

def normalize_intent(intent: Any) -> str:
    """Normalize a predicted macro-intent label (alias-tolerant, upper-case)."""
    s = str(intent or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "MELONRUSH": "MELON_RUSH",
        "PREPARING_MELONRUSH": "PREPARING_MELON_RUSH",
        "MELON_RUSH_PREP": "PREPARING_MELON_RUSH",
        "FERTILIZER_HOARD": "HOARDING_FERTILIZER",
        "HOARD_FERTILIZER": "HOARDING_FERTILIZER",
    }
    return aliases.get(s, s)


def select_rush_intent(
    intent_dist: Any,
    threshold: float = 0.5,
) -> Optional[Tuple[str, float]]:
    """Pick the top intent from a forecaster distribution if it is rush-like.

    Accepts a ``{intent: prob}`` mapping (``models/opponent_intent.py``) or a
    bare intent string (confidence 1.0).  Returns ``(intent, prob)`` when the
    argmax is rush-like and meets ``threshold``; otherwise ``None`` (abstain —
    no denial without a confident rush signal).
    """
    if intent_dist is None:
        return None
    if isinstance(intent_dist, str):
        name = normalize_intent(intent_dist)
        return (name, 1.0) if name in RUSH_INTENTS else None
    if isinstance(intent_dist, Mapping):
        best: Optional[str] = None
        best_p = -1.0
        for k, v in intent_dist.items():
            try:
                p = float(v)
            except Exception:
                continue
            if p > best_p:
                best_p, best = p, normalize_intent(k)
        if best in RUSH_INTENTS and best_p >= float(threshold):
            return (best, float(best_p))
        return None
    return None


def is_rush_intent(intent: Any) -> bool:
    """True when ``intent`` (string or distribution) signals a fertilizer rush."""
    if isinstance(intent, Mapping):
        return select_rush_intent(intent, threshold=0.0) is not None
    return normalize_intent(intent) in RUSH_INTENTS


def required_fertilizer_volume(
    intent: Any,
    melon_tiles: int = 0,
    *,
    fertilizer_per_tile: int = FERT_UNITS_PER_MELON_TILE,
    opponent_self_supply: int = 0,
    opponent_held_fertilizer: int = 0,
) -> int:
    """Opponent's net market Fertilizer requirement ``V`` for ``intent``.

    ``V = max(0, tiles * per_tile - self_supply - held)`` for rush intents,
    else 0.  Self-supply covers animal-produced fertilizer the opponent can
    collect without the market (1/day/head); held covers shed stock (publicly
    unobservable — default 0, conservative toward denial).
    """
    if not is_rush_intent(intent) if isinstance(intent, str) else (
        select_rush_intent(intent, threshold=0.0) is None
    ):
        return 0
    need = max(0, int(melon_tiles)) * max(0, int(fertilizer_per_tile))
    net = need - max(0, int(opponent_self_supply)) - max(0, int(opponent_held_fertilizer))
    return max(0, int(net))


def estimate_melon_tiles(
    opponent_farm: Optional[Mapping[str, Any]] = None,
    *,
    melon_tiles: Optional[int] = None,
    default_tiles: int = 10,
) -> int:
    """Tile count behind the rush: explicit override, else vacant-tile proxy.

    The opponent's public farm exposes no planting plan, so absent an explicit
    count we proxy with their vacant unlocked tiles (rush size is bounded by
    plantable room), floored at 0.  ``default_tiles`` applies when no farm is
    given at all.
    """
    if melon_tiles is not None:
        return max(0, int(melon_tiles))
    if not isinstance(opponent_farm, Mapping):
        return max(0, int(default_tiles))
    try:
        from kaggriculture.helpers.opponent import analyze_opponent_farm

        profile = analyze_opponent_farm(dict(opponent_farm))
        return max(0, int(profile.vacant_count))
    except Exception:
        pass
    # Fallback: count None tiles directly.
    try:
        n = 0
        for row in opponent_farm.get("tiles", []) or []:
            for tile in row or []:
                if tile is None:
                    n += 1
        return max(0, int(n))
    except Exception:
        return max(0, int(default_tiles))


# ---------------------------------------------------------------------------
# 2. Shared-market reads + exact sequential execution math
# ---------------------------------------------------------------------------

def fertilizer_price(inventory: int) -> int:
    """Unit Fertilizer price at market inventory (engine curve, floored)."""
    if _engine_price is not None:
        try:
            return int(_engine_price(TARGET_FERTILIZER, int(inventory)))
        except Exception:
            pass
    # Standalone linear fallback: base 100, amp 0.2, I0 10000.
    inv = int(inventory)
    if inv < MARKET_I0:
        return max(PRICE_FLOOR, int(round(FERT_BASE + 0.2 * (MARKET_I0 - inv))))
    return max(PRICE_FLOOR, int(round(FERT_BASE - 0.2 * (inv - MARKET_I0))))


def buy_cost(quantity: int, start_inventory: int) -> int:
    """Exact cash to BUY ``quantity`` Fertilizer starting at ``start_inventory``.

    Post-buy quoting: unit k costs ``P(inv0 - k)``.  Mirrors
    ``simulate_buy_slippage`` with no simultaneous opponent buys.
    """
    q = max(0, int(quantity))
    inv = int(start_inventory)
    total = 0
    for k in range(1, q + 1):
        total += fertilizer_price(inv - k)
    return int(total)


def sell_revenue(quantity: int, start_inventory: int) -> int:
    """Exact cash from SELLing ``quantity`` Fertilizer starting at inventory.

    Pre-sell quoting with the engine floor rule: a unit quoted at the $1
    floor does not increment market inventory (mirrors
    ``simulate_sell_slippage``).
    """
    q = max(0, int(quantity))
    inv = int(start_inventory)
    total = 0
    for _ in range(q):
        p = fertilizer_price(inv)
        total += p
        if p > PRICE_FLOOR:
            inv += 1
    return int(total)


def opponent_fertilizer_cost(required_volume: int, inventory_after_denial: int) -> int:
    """Opponent's total cost for the full required volume ``V`` post-denial."""
    return buy_cost(int(required_volume), int(inventory_after_denial))


def read_fertilizer_market(market_or_obs: Any) -> Tuple[int, int]:
    """Return ``(inventory, price)`` for shared-market Fertilizer.

    Accepts an ``obs`` dict (reads ``obs["market"]``), a bare market dict
    (``inventory``/``stocks`` + ``prices``/``price``), or ``None`` (assumes
    pristine ``I0``).  Missing inventory is verified against the quoted price
    via the engine curve inverse when only prices are present.
    """
    if market_or_obs is None:
        return (MARKET_I0, fertilizer_price(MARKET_I0))
    m: Any = market_or_obs
    if isinstance(m, Mapping) and "market" in m and isinstance(m["market"], Mapping):
        m = m["market"]
    if not isinstance(m, Mapping):
        return (MARKET_I0, fertilizer_price(MARKET_I0))
    inv_map = m.get("inventory", m.get("stocks", None))
    price_map = m.get("prices", m.get("price", None))
    inv = None
    if isinstance(inv_map, Mapping) and "FERTILIZER" in inv_map:
        try:
            inv = int(inv_map["FERTILIZER"])
        except Exception:
            inv = None
    if inv is None and isinstance(price_map, Mapping) and "FERTILIZER" in price_map:
        try:
            p = float(price_map["FERTILIZER"])
            # Linear inverse below I0: inv = I0 - (p - base) / 0.2.
            inv = int(round(MARKET_I0 - (p - FERT_BASE) / 0.2))
        except Exception:
            inv = None
    if inv is None:
        inv = MARKET_I0
    return (int(inv), fertilizer_price(int(inv)))


# ---------------------------------------------------------------------------
# 3. Optimal Denial_Purchase_Volume (least q with strict lockout)
# ---------------------------------------------------------------------------

def max_affordable_denial(
    market_inventory: int,
    our_cash: float,
    *,
    operating_reserve: float = 100.0,
) -> int:
    """Largest Fertilizer quantity we can BUY without breaching reserve."""
    budget = float(our_cash) - float(operating_reserve)
    if budget < fertilizer_price(int(market_inventory) - 1):
        return 0
    lo, hi = 0, int(budget // max(1, PRICE_FLOOR)) + 1
    hi = min(hi, SHED_CAPACITY, 10_000)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if buy_cost(mid, int(market_inventory)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return max(0, int(lo))


def denial_purchase_volume(
    required_volume: int,
    market_inventory: int,
    opponent_cash: float,
    *,
    our_cash: float = 0.0,
    shed_room: int = SHED_CAPACITY,
    operating_reserve: float = 100.0,
) -> Optional[int]:
    """Least ``q`` in ``[0, q_max]`` with ``opp_cost(q) > opp_cash``.

    Returns 0 when the opponent is already locked at current inventory
    (no purchase needed), or ``None`` when no feasible ``q`` achieves strict
    lockout (opponent too rich, or we lack cash/shed room).  Feasibility cap
    ``q_max = min(shed_room, max_affordable_denial(...))``.
    """
    V = max(0, int(required_volume))
    if V <= 0:
        return None
    inv0 = int(market_inventory)
    # Already locked: strict inequality holds with zero spend.
    if opponent_fertilizer_cost(V, inv0) > float(opponent_cash):
        return 0
    q_max = min(max(0, int(shed_room)),
                max_affordable_denial(inv0, float(our_cash),
                                      operating_reserve=operating_reserve))
    if q_max <= 0:
        return None
    # Monotone lockout predicate -> binary search the least locking q.
    if opponent_fertilizer_cost(V, inv0 - q_max) <= float(opponent_cash):
        return None
    lo, hi = 1, q_max
    while lo < hi:
        mid = (lo + hi) // 2
        if opponent_fertilizer_cost(V, inv0 - mid) > float(opponent_cash):
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# ---------------------------------------------------------------------------
# 4. Risk check: our holding loss vs opponent's missed-window loss
# ---------------------------------------------------------------------------

def holding_loss(
    denial_volume: int,
    market_inventory: int,
    *,
    own_fertilizer_need: int = 0,
) -> int:
    """Expected loss holding ``q`` excess Fertilizer (resale-valued).

    ``L = buy_cost(q) - buy_cost(usable) - sell_revenue(surplus)`` with
    ``usable = min(q, own_need)`` valued at pre-denial replacement cost
    (procurement we would buy anyway, hence zero incremental loss) and the
    surplus resold sequentially from the post-buy inventory.  Exactly 0 when
    ``q <= own_need``; near-I0 round trips lose only integer rounding.
    """
    q = max(0, int(denial_volume))
    if q <= 0:
        return 0
    inv0 = int(market_inventory)
    usable = min(q, max(0, int(own_fertilizer_need)))
    # Replacement cost of the usable portion at the pre-denial market: units
    # we would have bought anyway, so they carry no incremental denial loss.
    own_value = buy_cost(usable, inv0) if usable > 0 else 0
    surplus = q - usable
    resale = sell_revenue(surplus, inv0 - q) if surplus > 0 else 0
    return int(buy_cost(q, inv0) - own_value - resale)


def opponent_melon_loss(
    melon_tiles: int,
    *,
    current_day: int = 0,
    melon_price: Optional[int] = None,
    melon_yield_units: int = MELON_MAX_YIELD,
    melon_seed_cost: int = MELON_SEED,
) -> int:
    """Terminal cash the opponent loses by missing the Melon window.

    Net margin of therush ``tiles * (yield * price - seed)`` when a replant
    could still mature (``day + 10 < 30``); 0 once melons are nonviable (no
    window left to destroy) or with no tiles.  Uses the live melon price when
    given, else the engine base (250).
    """
    tiles = max(0, int(melon_tiles))
    if tiles <= 0:
        return 0
    if int(current_day) + MELON_MATURATION_DAYS >= SEASON_DAYS:
        return 0
    px = int(melon_price) if melon_price is not None else MELON_BASE
    return max(0, int(tiles * (max(0, int(melon_yield_units)) * px - max(0, int(melon_seed_cost)))))


@dataclass
class DenialPlan:
    """Auditable resource-starvation decision (all dollars integer)."""

    execute: bool
    intent: str
    intent_confidence: float
    required_volume: int
    melon_tiles: int
    market_inventory: int
    market_price_now: int
    denial_volume: Optional[int]
    our_cost: int
    opponent_cost_after: Optional[int]
    opponent_cash: float
    locked_out: bool
    holding_loss: int
    opponent_loss: int
    risk_pass: bool
    reason: str
    _orders: List[List[Any]] = field(default_factory=list, repr=False)

    def to_market_orders(self) -> List[List[Any]]:
        """``[["BUY_PRODUCT", "FERTILIZER", q]]`` when executing, else ``[]``."""
        return [list(o) for o in (self._orders or [])]


def plan_denial(
    intent: Any,
    *,
    opponent_cash: float = 0.0,
    market_inventory: int = MARKET_I0,
    our_cash: float = 0.0,
    shed_room: int = SHED_CAPACITY,
    melon_tiles: int = 0,
    opponent_farm: Optional[Mapping[str, Any]] = None,
    intent_confidence: Optional[float] = None,
    confidence_threshold: float = 0.5,
    fertilizer_per_tile: int = FERT_UNITS_PER_MELON_TILE,
    opponent_self_supply: int = 0,
    opponent_held_fertilizer: int = 0,
    own_fertilizer_need: int = 0,
    current_day: int = 0,
    melon_price: Optional[int] = None,
    operating_reserve: float = 100.0,
) -> DenialPlan:
    """Full starvation pipeline: intent -> requirement -> denial -> risk gate.

    Execute (emit the BUY order) iff the intent is a confident rush, a
    feasible ``q* > 0`` achieves strict lockout, and
    ``holding_loss(q*) < opponent_melon_loss`` strictly.  ``q* == 0``
    (already locked) and infeasible/unguarded cases return ``execute=False``
    with an audit ``reason`` and empty orders.
    """
    # -- 1. Ingest intent (string or forecaster distribution). --
    if isinstance(intent, Mapping):
        sel = select_rush_intent(intent, threshold=confidence_threshold)
        if sel is None:
            name = str(max(intent, key=lambda k: float(intent[k]))) if intent else "UNKNOWN"
            return DenialPlan(
                execute=False, intent=normalize_intent(name), intent_confidence=0.0,
                required_volume=0, melon_tiles=0, market_inventory=int(market_inventory),
                market_price_now=fertilizer_price(int(market_inventory)),
                denial_volume=None, our_cost=0, opponent_cost_after=None,
                opponent_cash=float(opponent_cash), locked_out=False,
                holding_loss=0, opponent_loss=0, risk_pass=False,
                reason=f"abstain: no confident rush intent (need >= {confidence_threshold})",
            )
        name, conf = sel
    else:
        name = normalize_intent(intent)
        conf = float(intent_confidence) if intent_confidence is not None else 1.0
        if name not in RUSH_INTENTS or conf < float(confidence_threshold):
            return DenialPlan(
                execute=False, intent=name, intent_confidence=float(conf),
                required_volume=0, melon_tiles=0, market_inventory=int(market_inventory),
                market_price_now=fertilizer_price(int(market_inventory)),
                denial_volume=None, our_cost=0, opponent_cost_after=None,
                opponent_cash=float(opponent_cash), locked_out=False,
                holding_loss=0, opponent_loss=0, risk_pass=False,
                reason=f"abstain: intent {name} is not a confident fertilizer rush",
            )

    # -- 2. Requirement + market check. --
    tiles = estimate_melon_tiles(opponent_farm, melon_tiles=melon_tiles if melon_tiles else None,
                                 default_tiles=max(0, int(melon_tiles)))
    if melon_tiles:
        tiles = max(0, int(melon_tiles))
    V = required_fertilizer_volume(
        name, tiles, fertilizer_per_tile=fertilizer_per_tile,
        opponent_self_supply=opponent_self_supply,
        opponent_held_fertilizer=opponent_held_fertilizer,
    )
    inv0 = int(market_inventory)
    px0 = fertilizer_price(inv0)
    if V <= 0:
        return DenialPlan(
            execute=False, intent=name, intent_confidence=float(conf),
            required_volume=0, melon_tiles=int(tiles), market_inventory=inv0,
            market_price_now=px0, denial_volume=None, our_cost=0,
            opponent_cost_after=opponent_fertilizer_cost(0, inv0),
            opponent_cash=float(opponent_cash), locked_out=False,
            holding_loss=0, opponent_loss=0, risk_pass=False,
            reason="abstain: zero net market requirement (self-supply covers rush)",
        )

    # -- 3. Optimal denial volume. --
    q = denial_purchase_volume(
        V, inv0, float(opponent_cash), our_cash=float(our_cash),
        shed_room=int(shed_room), operating_reserve=operating_reserve,
    )
    if q is None:
        return DenialPlan(
            execute=False, intent=name, intent_confidence=float(conf),
            required_volume=int(V), melon_tiles=int(tiles), market_inventory=inv0,
            market_price_now=px0, denial_volume=None,
            our_cost=0,
            opponent_cost_after=opponent_fertilizer_cost(int(V), inv0),
            opponent_cash=float(opponent_cash), locked_out=False,
            holding_loss=0,
            opponent_loss=opponent_melon_loss(int(tiles), current_day=int(current_day),
                                              melon_price=melon_price),
            risk_pass=False,
            reason="abstain: no feasible purchase locks the opponent (too rich / we lack cash-room)",
        )
    if q == 0:
        return DenialPlan(
            execute=False, intent=name, intent_confidence=float(conf),
            required_volume=int(V), melon_tiles=int(tiles), market_inventory=inv0,
            market_price_now=px0, denial_volume=0, our_cost=0,
            opponent_cost_after=opponent_fertilizer_cost(int(V), inv0),
            opponent_cash=float(opponent_cash), locked_out=True,
            holding_loss=0,
            opponent_loss=opponent_melon_loss(int(tiles), current_day=int(current_day),
                                              melon_price=melon_price),
            risk_pass=False,
            reason="no-op: opponent already locked at current price (volume 0)",
        )

    # -- 4. Risk gate (strict). --
    our_c = buy_cost(int(q), inv0)
    opp_c = opponent_fertilizer_cost(int(V), inv0 - int(q))
    locked = bool(opp_c > float(opponent_cash))
    l_us = holding_loss(int(q), inv0, own_fertilizer_need=own_fertilizer_need)
    l_opp = opponent_melon_loss(int(tiles), current_day=int(current_day),
                                melon_price=melon_price)
    risk = bool(l_us < l_opp) and bool(l_opp > 0)
    if locked and risk:
        return DenialPlan(
            execute=True, intent=name, intent_confidence=float(conf),
            required_volume=int(V), melon_tiles=int(tiles), market_inventory=inv0,
            market_price_now=px0, denial_volume=int(q), our_cost=int(our_c),
            opponent_cost_after=int(opp_c), opponent_cash=float(opponent_cash),
            locked_out=True, holding_loss=int(l_us), opponent_loss=int(l_opp),
            risk_pass=True,
            reason=(f"execute: buy {q} @ ~${our_c} spikes V={V} cost to ${opp_c} "
                    f"> opp ${float(opponent_cash):.0f}; loss ${l_us} < opp loss ${l_opp}"),
            _orders=[["BUY_PRODUCT", TARGET_FERTILIZER, int(q)]],
        )
    return DenialPlan(
        execute=False, intent=name, intent_confidence=float(conf),
        required_volume=int(V), melon_tiles=int(tiles), market_inventory=inv0,
        market_price_now=px0, denial_volume=int(q), our_cost=int(our_c),
        opponent_cost_after=int(opp_c), opponent_cash=float(opponent_cash),
        locked_out=locked, holding_loss=int(l_us), opponent_loss=int(l_opp),
        risk_pass=risk,
        reason=(f"abstain: risk gate fails (loss ${l_us} vs opp loss ${l_opp})"
                if locked else
                f"abstain: q={q} does not lock (opp cost ${opp_c} vs cash ${float(opponent_cash):.0f})"),
    )


def should_execute_denial(*args: Any, **kwargs: Any) -> bool:
    """Boolean threshold: True iff :func:`plan_denial` says execute."""
    return bool(plan_denial(*args, **kwargs).execute)


def plan_denial_from_obs(
    obs: Mapping[str, Any],
    intent: Any,
    *,
    confidence_threshold: float = 0.5,
    melon_tiles: int = 0,
    opponent_self_supply: int = 0,
    own_fertilizer_need: int = 0,
    operating_reserve: float = 100.0,
    **kwargs: Any,
) -> DenialPlan:
    """Obs-level convenience wrapper (reads cash, shed room, market, day).

    ``obs`` supplies our cash (``farms[player].money``), shed room
    (``private.shed`` vs ``SHED_CAPACITY``), market inventory, melon price,
    and current day; the opponent's cash comes from ``farms[1-player]``.
    """
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    me = farms[player] if player < len(farms) else {}
    opp = farms[1 - player] if (1 - player) < len(farms) else {}
    our_cash = float(me.get("money", 0.0) or 0.0)
    opp_cash = float(opp.get("money", 0.0) or 0.0)
    private = obs.get("private") or {}
    shed = private.get("shed") or {}
    try:
        shed_total = sum(int(v or 0) for v in shed.values())
    except Exception:
        shed_total = 0
    market = obs.get("market") or {}
    inv, _ = read_fertilizer_market(market)
    prices = market.get("prices", market.get("price", {})) or {}
    try:
        melon_px: Optional[int] = int(prices.get("MELON", MELON_BASE))
    except Exception:
        melon_px = MELON_BASE
    day = int(obs.get("day", 0) or 0)
    tiles = melon_tiles or estimate_melon_tiles(opp, melon_tiles=None, default_tiles=0)
    return plan_denial(
        intent, opponent_cash=opp_cash, market_inventory=inv, our_cash=our_cash,
        shed_room=max(0, SHED_CAPACITY - int(shed_total)), melon_tiles=int(tiles),
        opponent_farm=opp, confidence_threshold=confidence_threshold,
        opponent_self_supply=opponent_self_supply, own_fertilizer_need=own_fertilizer_need,
        current_day=day, melon_price=melon_px, operating_reserve=operating_reserve,
        **kwargs,
    )


__all__ = [
    "TARGET_FERTILIZER",
    "MELON_BONUS_WINDOW_DAYS",
    "FERT_UNITS_PER_MELON_TILE",
    "MELON_MATURATION_DAYS",
    "SEASON_DAYS",
    "RUSH_INTENTS",
    "NON_RUSH_INTENTS",
    "DenialPlan",
    "normalize_intent",
    "select_rush_intent",
    "is_rush_intent",
    "required_fertilizer_volume",
    "estimate_melon_tiles",
    "fertilizer_price",
    "buy_cost",
    "sell_revenue",
    "opponent_fertilizer_cost",
    "read_fertilizer_market",
    "max_affordable_denial",
    "denial_purchase_volume",
    "holding_loss",
    "opponent_melon_loss",
    "plan_denial",
    "should_execute_denial",
    "plan_denial_from_obs",
]
