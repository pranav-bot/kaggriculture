"""PSRO meta-controller: hierarchical top level over MacroIntent profiles."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from kaggriculture.env.items import MARKET_PARAMS, get_quadrant_bounds
from kaggriculture.features.macro_intents import MACRO_INTENT_CLASSES
from kaggriculture.models.opponent_intent import OpponentIntentModel

INTENT_INDEX = {name: i for i, name in enumerate(MACRO_INTENT_CLASSES)}

# Latent opponent clusters (PSRO opponent model).
MELON_RUSH = "MELON_RUSH"
MILK_FLOODER = "MILK_FLOODER"
STRAWBERRY_CONTINGENCY = "STRAWBERRY_CONTINGENCY"
CONSERVATIVE_SAVER = "CONSERVATIVE_SAVER"
EXPANSION_RUSH = "EXPANSION_RUSH"
UNKNOWN = "UNKNOWN"

# Discrete public events emitted by the meta-controller.
OPPONENT_EXPANSION_DETECTED = "OPPONENT_EXPANSION_DETECTED"

# Our specialist sub-policies (PSRO response population).
POLICY_A = "Policy_A"  # Milk Dominance
POLICY_B = "Policy_B"  # Strawberry / Insulated Crop Defense
POLICY_C = "Policy_C"  # Deceptive Poisoned Well

# PSRO best-response matrix: Aggressive Milk beats Savers but loses to an
# opponent who ignores the milk book; hoard-then-dump milk flooders are met
# with insulated crops; a mass melon rush is met by poisoning the melon well
# (preemptive selling) before their harvest lands; a multi-quadrant land
# expansion (Rank-1 tell) is met with the Strawberry Contingency, racing the
# expander into insulated crops rather than contesting their scaling book.
RESPONSE_MATRIX = {
    EXPANSION_RUSH: POLICY_B,
    MILK_FLOODER: POLICY_B,
    MELON_RUSH: POLICY_C,
    STRAWBERRY_CONTINGENCY: POLICY_A,
    CONSERVATIVE_SAVER: POLICY_A,
    UNKNOWN: POLICY_A,
}

# Melon-rush tripwire (exact task rule).
MELON_RUSH_SW_EMPTY_THRESHOLD = 5
MELON_RUSH_CASH_THRESHOLD = 3000.0


def _prior(high: Mapping[str, float], low: float = 0.2) -> List[float]:
    """24-dim intent prior: `high` intents up-weighted, rest at `low`."""
    vec = [low] * len(MACRO_INTENT_CLASSES)
    for name, w in high.items():
        vec[INTENT_INDEX[name]] = float(w)
    total = sum(vec)
    return [v / total for v in vec]


def build_default_profiles() -> Dict[str, Dict[str, Any]]:
    """Three distinct MacroIntent priors (JSON-serializable, loadable)."""
    return {
        POLICY_A: {
            "id": POLICY_A,
            "name": "Milk Dominance",
            "doctrine": (
                "Out-produce and out-sell milk: fast pasture + cow ramp, "
                "continuous DUMP_MILK. Beats savers and uncontested books; "
                "loses to opponents who ignore milk (strawberry contingency)."
            ),
            "intent_prior": _prior({
                "BUY_COW": 5.0, "BUILD_PASTURE": 4.0, "FEED_ANIMAL": 3.0,
                "CARE_ANIMAL": 3.0, "COLLECT_FERTILIZER": 3.0,
                "DUMP_MILK": 5.0, "SELL_MARKET": 3.0, "HIRE_WORKER": 2.0,
                "EXPAND_NE": 2.0, "BUY_SEEDS": 1.5,
            }),
        },
        POLICY_B: {
            "id": POLICY_B,
            "name": "Strawberry / Insulated Crop Defense",
            "doctrine": (
                "Abandon the contested milk/wool book for insulated crops "
                "(strawberry/melon): plant, water, fertilize, harvest. "
                "Best response to an incoming milk/wool dump."
            ),
            "intent_prior": _prior({
                "PLANT_CROPS": 5.0, "WATER_CROPS": 4.0, "MAINTAIN_CROPS": 4.0,
                "HARVEST_CROPS": 4.0, "FERTILIZE_CROPS": 3.0,
                "BUY_SEEDS": 3.0, "SELL_CROPS": 4.0, "CLEAR_WEED": 2.0,
                "EXPAND_SW": 2.0, "SELL_MARKET": 2.0,
            }),
            "reclaim_pastures": True,
        },
        POLICY_C: {
            "id": POLICY_C,
            "name": "Deceptive Poisoned Well",
            "doctrine": (
                "Feint herd expansion, then spoil the shared book: preemptive "
                "DUMP/SELL_MARKET crashes the target price (melon/wool fall "
                "super-linearly) before the opponent's harvest lands, while "
                "our cash stays liquid for the post-crash recovery."
            ),
            "intent_prior": _prior({
                "SELL_MARKET": 5.0, "DUMP_MILK": 3.0, "DUMP_WOOL": 3.0,
                "SELL_CROPS": 4.0, "BUY_MARKET": 2.0, "MOVE_WORKERS": 2.0,
                "BUILD_PASTURE": 2.0, "BUY_COW": 2.0, "BUY_SHEEP": 2.0,
                "HIRE_WORKER": 1.5,
            }),
        },
    }


def opponent_spatial_footprint(obs: Mapping[str, Any], seat: int = 0) -> Dict[str, Any]:
    """Count opponent tiles by kind, per quadrant + globally.

    Tracks the opponent's spatial footprint: empty/development-ready tiles
    (melon-rush staging), pasture mass (herd threat), planted tiles.
    """
    opp = 1 - int(seat)
    farms = obs.get("farms") or []
    farm = farms[opp] if 0 <= opp < len(farms) else {}
    tiles = farm.get("tiles") or []

    def kind_at(r: int, c: int) -> str:
        try:
            cell = tiles[r][c]
        except (IndexError, TypeError):
            return "MISSING"
        if cell is None:
            return "EMPTY"
        if isinstance(cell, str):
            return cell  # "LOCKED" etc.
        if isinstance(cell, Mapping):
            return str(cell.get("kind", "UNKNOWN"))
        return "UNKNOWN"

    quads = ["NW", "NE", "SW", "SE"]
    footprint: Dict[str, Any] = {
        "cash": float(farm.get("money", 0.0) or 0.0),
        "unlocked_quadrants": list(farm.get("unlocked_quadrants") or []),
        "quadrants": {},
        "pastures": 0,
        "pasture_tiles": [],
        "plants": 0,
        "empty": 0,
        "empty_tiles": [],
    }
    for q in quads:
        x0, x1, y0, y1 = get_quadrant_bounds(q)
        counts: Dict[str, int] = {}
        for r in range(y0, y1):
            for c in range(x0, x1):
                k = kind_at(r, c)
                counts[k] = counts.get(k, 0) + 1
                if k == "EMPTY":
                    footprint["empty"] += 1
                    footprint["empty_tiles"].append((r, c))
                elif k == "PASTURE":
                    footprint["pastures"] += 1
                    footprint["pasture_tiles"].append((r, c))
                elif k == "PLANT":
                    footprint["plants"] += 1
        footprint["quadrants"][q] = counts
    footprint["sw_empty"] = sum(
        1 for (r, c) in footprint["empty_tiles"] if 5 <= r < 10 and 0 <= c < 5
    )
    return footprint


def own_reclaim_targets(obs: Mapping[str, Any], seat: int = 0) -> List[Dict[str, Any]]:
    """Own PASTURE/WEED tiles to reclaim into plantable soil via DIG.

    Only UNOCCUPIED pastures are targeted: tiles holding an animal (or with
    fertilizer ready to collect) are deferred, never destroyed. Each target
    becomes one queued DIG command for the transition layer.
    """
    farms = obs.get("farms") or []
    farm = farms[int(seat)] if 0 <= int(seat) < len(farms) else {}
    tiles = farm.get("tiles") or []
    targets = []
    for r in range(10):
        for c in range(10):
            try:
                cell = tiles[r][c]
            except (IndexError, TypeError):
                continue
            if not isinstance(cell, Mapping):
                continue
            kind = cell.get("kind")
            if kind == "WEED":
                targets.append({"tile": (r, c), "kind": "WEED", "occupied": False})
            elif kind == "PASTURE":
                occupied = bool(cell.get("animal")) or bool(cell.get("fertilizer_available"))
                targets.append({"tile": (r, c), "kind": "PASTURE", "occupied": occupied})
    return targets


def default_holding_penalty(inventory: Mapping[str, float]) -> float:
    """Milk/Wool holding penalty (2x base per unit) for the beam layer."""
    penalty = 0.0
    for item in ("MILK", "WOOL"):
        qty = float((inventory or {}).get(item, 0.0) or 0.0)
        base = float(MARKET_PARAMS.get(item, {}).get("base", 0.0) or 0.0)
        penalty += max(0.0, qty) * 2.0 * base
    return penalty


class MetaController:
    """PSRO meta-agent: top of the hierarchy above the Beam Search layer.

    Each turn: ingest obs + BOCD status -> infer latent opponent strategy ->
    gate (instantly switch) the active MacroIntent profile -> expose the
    profile + transition DIG queue + holding-penalty hook to Beam Search.
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        default_policy: str = POLICY_A,
        holding_penalty_fn: Optional[Callable[[Mapping[str, float]], float]] = None,
        top_k_intents: int = 12,
        opponent_intent_model: Optional[OpponentIntentModel] = None,
        expansion_dwell_turns: int = 72,
    ) -> None:
        self.profiles = profiles if profiles is not None else build_default_profiles()
        if default_policy not in self.profiles:
            raise ValueError(f"unknown default policy {default_policy!r}")
        self.default_policy = default_policy
        self.active_policy = default_policy
        self.holding_penalty_fn = holding_penalty_fn or default_holding_penalty
        self.top_k_intents = int(top_k_intents)
        self.latent_strategy: str = UNKNOWN
        self.footprint: Dict[str, Any] = {}
        self.bocd_status: Dict[str, Any] = {}
        self.transition_queue: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.opponent_intent_model = opponent_intent_model or OpponentIntentModel()
        self._prev_opp_quads: Optional[Tuple[str, ...]] = None
        self.expansion_event: Optional[Dict[str, Any]] = None
        # Structural-event hysteresis: a confirmed land expansion is
        # irreversible, so the Policy_B response is HELD for this many turns
        # past the trigger even if heuristic inference flaps afterwards.
        self.expansion_dwell_turns = int(expansion_dwell_turns)
        self._dwell_until_turn: Optional[int] = None

    # -- perception ------------------------------------------------------
    def get_opponent_intent(
        self, historical_obs_buffer: Any
    ) -> Dict[str, float]:
        """Return the public-information intent distribution for the next 24h."""
        return self.opponent_intent_model.predict_proba(historical_obs_buffer)

    def infer_latent_strategy(
        self,
        obs: Mapping[str, Any],
        seat: int = 0,
        bocd_status: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classify the opponent into a latent strategy cluster.

        MELON_RUSH rule (exact): opponent holds >5 empty SW tiles AND cash
        hoard > $3000 (land staged + capital for mass melon seeds).

        EXPANSION_RUSH rule (exact, top precedence): the opponent's public
        unlocked-quadrant set GREW since the previous tick (e.g. a Rank-1
        multi-quadrant land grab). Raises OPPONENT_EXPANSION_DETECTED on the
        exact turn the new quadrant becomes observable.
        """
        fp = opponent_spatial_footprint(obs, seat)
        self.footprint = fp
        status = dict(bocd_status or {})
        self.bocd_status = status
        hoarding = bool(status.get("event_latched")) or status.get("event") is not None

        quads = tuple(fp.get("unlocked_quadrants") or [])
        turn = obs.get("step", 0)
        try:
            turn = int(turn or 0)
        except (TypeError, ValueError):
            turn = 0
        if self._prev_opp_quads is not None and set(quads) - set(self._prev_opp_quads):
            self.expansion_event = {
                "event": OPPONENT_EXPANSION_DETECTED,
                "turn": turn,
                "prev_quadrants": list(self._prev_opp_quads),
                "new_quadrants": sorted(set(quads) - set(self._prev_opp_quads)),
                "all_quadrants": list(quads),
            }
            self._prev_opp_quads = quads
            return EXPANSION_RUSH
        self._prev_opp_quads = quads

        if (
            fp["sw_empty"] > MELON_RUSH_SW_EMPTY_THRESHOLD
            and fp["cash"] > MELON_RUSH_CASH_THRESHOLD
        ):
            return MELON_RUSH
        if hoarding or fp["pastures"] >= 8:
            return MILK_FLOODER
        if fp["plants"] >= 15:
            return STRAWBERRY_CONTINGENCY
        farms = obs.get("farms") or []
        opp = farms[1 - int(seat)] if len(farms) > 1 else {}
        unlocked = len(opp.get("unlocked_quadrants") or [])
        if fp["cash"] > MELON_RUSH_CASH_THRESHOLD and fp["pastures"] <= 2 and unlocked <= 1:
            return CONSERVATIVE_SAVER
        return UNKNOWN

    # -- gating ----------------------------------------------------------
    def gate(
        self,
        latent: str,
        obs: Optional[Mapping[str, Any]] = None,
        seat: int = 0,
    ) -> str:
        """PSRO response oracle: instantly switch to the best-response profile.

        Switching INTO Policy_B auto-queues DIG commands reclaiming our
        unoccupied pastures (+ weeds) into plantable soil so the transition
        is smooth -- no stranded pasture tiles, no destroyed livestock.
        """
        if latent not in RESPONSE_MATRIX:
            latent = UNKNOWN
        new_policy = RESPONSE_MATRIX[latent]
        if new_policy != self.active_policy:
            self.active_policy = new_policy
            if new_policy == POLICY_B and obs is not None:
                self.plan_transition(obs, seat)
            else:
                self.transition_queue = []
        return self.active_policy

    def update(
        self,
        obs: Mapping[str, Any],
        seat: int = 0,
        bocd_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """One meta-tick: perceive -> infer -> gate. Returns the decision."""
        latent = self.infer_latent_strategy(obs, seat, bocd_status)
        self.latent_strategy = latent
        try:
            turn = int(obs.get("step", 0) or 0)
        except (TypeError, ValueError):
            turn = 0
        if latent == EXPANSION_RUSH:
            self._dwell_until_turn = turn + self.expansion_dwell_turns
        # Dwell override: hold the expansion response while the confirmed
        # structural event is being digested, regardless of heuristic flap.
        gated_latent = latent
        dwell_remaining = 0
        if self._dwell_until_turn is not None and turn <= self._dwell_until_turn:
            dwell_remaining = self._dwell_until_turn - turn
            if RESPONSE_MATRIX.get(latent, POLICY_A) != POLICY_B:
                gated_latent = EXPANSION_RUSH
        active = self.gate(gated_latent, obs, seat)
        bocd = bocd_status or {}
        event = None
        if latent == EXPANSION_RUSH and self.expansion_event is not None:
            event = self.expansion_event.get("event")
        elif bocd.get("event_latched"):
            event = bocd.get("event") or "OPPONENT_HOARDING_DETECTED"
        decision = {
            "latent_strategy": latent,
            "gated_latent": gated_latent,
            "dwell_remaining": dwell_remaining,
            "active_policy": active,
            "event": event,
            "footprint": {
                "sw_empty": self.footprint.get("sw_empty", 0),
                "pastures": self.footprint.get("pastures", 0),
                "plants": self.footprint.get("plants", 0),
                "cash": self.footprint.get("cash", 0.0),
                "quadrants": list(self.footprint.get("unlocked_quadrants") or []),
            },
            "hoarding": bool(bocd.get("event_latched")),
            "queued_digs": len(self.transition_queue),
        }
        self.history.append(decision)
        return decision

    # -- smooth transition ------------------------------------------------
    def plan_transition(
        self, obs: Mapping[str, Any], seat: int = 0
    ) -> List[Dict[str, Any]]:
        """(Re)build the DIG queue from live obs; called on Policy_B entry."""
        targets = own_reclaim_targets(obs, seat)
        self._last_reclaim_targets = targets
        self.transition_queue = [
            {"unit": None, "target": t["tile"], "command": ["DIG"],
             "kind": t["kind"], "status": "queued"}
            for t in targets
            if not t["occupied"]
        ]
        return list(self.transition_queue)

    def pop_transition_commands(
        self, unit_positions: Mapping[str, List[int]]
    ) -> Dict[str, Any]:
        """Render queued DIGs into one turn of mechanical farmer/hands actions.

        Greedy nearest-unit assignment: a unit standing ON its target emits
        DIG, otherwise steps toward it. Consumed orders leave the queue, so
        the Beam Search layer below keeps working its own plan concurrently.
        """
        farmer_pos = tuple(unit_positions.get("farmer", [4, 4]))
        hands_pos = [tuple(p) for p in unit_positions.get("hands", [])]
        free = [("farmer", farmer_pos)] + [
            (f"hand_{i}", p) for i, p in enumerate(hands_pos)
        ]

        actions: Dict[str, Any] = {"farmer": ["PASS"], "hands": [["PASS"]] * len(hands_pos)}
        remaining = []
        for order in self.transition_queue:
            tr, tc = order["target"]
            best, best_d = None, None
            for name, (ur, uc) in free:
                d = abs(ur - tr) + abs(uc - tc)
                if best_d is None or d < best_d:
                    best, best_d = (name, (ur, uc)), d
            if best is None:
                remaining.append(order)
                continue
            name, (ur, uc) = best
            free = [f for f in free if f[0] != name]
            if (ur, uc) == (tr, tc):
                cmd = ["DIG"]
            elif ur != tr:
                cmd = ["SOUTH" if tr > ur else "NORTH"]
            else:
                cmd = ["EAST" if tc > uc else "WEST"]
            if name == "farmer":
                actions["farmer"] = cmd
            else:
                actions["hands"][int(name.split("_")[1])] = cmd
            if (ur, uc) != (tr, tc):
                remaining.append(order)  # still en route
        self.transition_queue = remaining
        return actions

    # -- beam-search interface --------------------------------------------
    def profile_for_beam(
        self, inventory: Optional[Mapping[str, float]] = None
    ) -> Dict[str, Any]:
        """Package the active profile for the Beam Search layer."""
        profile = self.profiles[self.active_policy]
        prior = list(profile["intent_prior"])
        ranked = sorted(range(len(prior)), key=lambda i: -prior[i])
        hoarding = bool(self.bocd_status.get("event_latched"))
        penalty_fn = self.holding_penalty_fn if hoarding else None
        return {
            "policy_id": self.active_policy,
            "latent_strategy": self.latent_strategy,
            "intent_prior": prior,
            "action_space": ranked[: self.top_k_intents],
            "holding_penalty_fn": penalty_fn,
            "transition_queue": list(self.transition_queue),
            "hoarding_active": hoarding,
        }

    def meta_distribution(self) -> Dict[str, float]:
        """Current PSRO meta-distribution (pure best response)."""
        return {pid: 1.0 if pid == self.active_policy else 0.0 for pid in self.profiles}
