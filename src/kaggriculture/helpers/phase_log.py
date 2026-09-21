"""Episode phase timeline (research/01 §6). PolicyBrain calls log_transition when wired."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from kaggriculture.helpers.market_overlays import PREMIUM_BASE_PRICE, PREMIUM_ITEMS
from kaggriculture.helpers.phase_brain import MIX_SWITCH_DAY, SHED_PRESSURE_FULL_SELL

_TERMINAL_DAY = 27
_CASH_CRITICAL = 1500
_COLLISION_INV_DELTA = 3
_COLLISION_PRICE_DELTA = -3
_COLLISION_MAX_PRICE_RATIO = 1.10


class Phase(str, Enum):
    OPENING = "OPENING"
    EXPANSION = "EXPANSION"
    ADAPTIVE_MID = "ADAPTIVE_MID"
    PRESSURE = "PRESSURE"
    COLLISION = "COLLISION"
    TERMINAL = "TERMINAL"


@dataclass
class PhaseTransition:
    step: int
    day: int
    hour: int
    phase: Phase


@dataclass
class PhaseLog:
    """Collects phase changes for one episode."""

    transitions: List[PhaseTransition] = field(default_factory=list)
    _last_phase: Optional[Phase] = None

    def record(self, obs: Mapping[str, Any], phase: Phase) -> None:
        if phase == self._last_phase:
            return
        self._last_phase = phase
        self.transitions.append(
            PhaseTransition(
                step=int(obs.get("step", 0) or 0),
                day=int(obs.get("day", 0) or 0),
                hour=int(obs.get("hour", 0) or 0),
                phase=phase,
            )
        )

    def to_timeline(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": t.step,
                "day": t.day,
                "hour": t.hour,
                "phase": t.phase.value,
            }
            for t in self.transitions
        ]


_active_log: Optional[PhaseLog] = None


def start_phase_log() -> PhaseLog:
    """Begin collecting transitions for the current episode."""
    global _active_log
    _active_log = PhaseLog()
    return _active_log


def get_active_phase_log() -> Optional[PhaseLog]:
    return _active_log


def clear_phase_log() -> None:
    global _active_log
    _active_log = None


def log_transition(obs: Mapping[str, Any], phase: Phase) -> None:
    """Record a phase change. PolicyBrain.update() should call this when implemented."""
    if _active_log is not None:
        _active_log.record(obs, phase)


def _player_farm(obs: Mapping[str, Any]) -> Mapping[str, Any]:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    if len(farms) > player:
        return farms[player]
    return farms[0] if farms else {}


def _shed_total(obs: Mapping[str, Any]) -> int:
    private = obs.get("private") or {}
    shed = private.get("shed") or {}
    return int(sum(max(0, int(v or 0)) for v in shed.values()))


def _cash_critical(obs: Mapping[str, Any]) -> bool:
    money = float(_player_farm(obs).get("money", 0) or 0)
    return money < _CASH_CRITICAL


def _pressure_active(obs: Mapping[str, Any]) -> bool:
    return _shed_total(obs) >= SHED_PRESSURE_FULL_SELL or _cash_critical(obs)


def _collision_active(
    obs: Mapping[str, Any],
    prev_inventory: Optional[Mapping[str, Any]],
    prev_prices: Optional[Mapping[str, Any]],
) -> bool:
    day = int(obs.get("day", 0) or 0)
    if day >= _TERMINAL_DAY or _pressure_active(obs):
        return False
    if prev_inventory is None or prev_prices is None:
        return False
    market = obs.get("market") or {}
    inv = market.get("inventory") or {}
    prices = market.get("prices") or {}
    for item in PREMIUM_ITEMS:
        base = float(PREMIUM_BASE_PRICE.get(item, 0) or 0)
        if base <= 0:
            continue
        price_ratio = float(prices.get(item, 0) or 0) / base
        if price_ratio >= _COLLISION_MAX_PRICE_RATIO:
            continue
        inv_delta = float(inv.get(item, 0) or 0) - float(prev_inventory.get(item, 0) or 0)
        price_delta = float(prices.get(item, 0) or 0) - float(prev_prices.get(item, 0) or 0)
        if inv_delta >= _COLLISION_INV_DELTA or price_delta <= _COLLISION_PRICE_DELTA:
            return True
    return False


def infer_phase(
    obs: Mapping[str, Any],
    prev_inventory: Optional[Mapping[str, Any]] = None,
    prev_prices: Optional[Mapping[str, Any]] = None,
) -> Phase:
    """Heuristic phase from public obs (used until PolicyBrain owns transitions)."""
    day = int(obs.get("day", 0) or 0)
    if day >= _TERMINAL_DAY:
        return Phase.TERMINAL
    if _collision_active(obs, prev_inventory, prev_prices):
        return Phase.COLLISION
    if _pressure_active(obs):
        return Phase.PRESSURE
    if day >= MIX_SWITCH_DAY:
        return Phase.ADAPTIVE_MID
    if day >= 3:
        return Phase.EXPANSION
    return Phase.OPENING


def infer_and_log(
    obs: Mapping[str, Any],
    prev_inventory: Optional[Mapping[str, Any]] = None,
    prev_prices: Optional[Mapping[str, Any]] = None,
) -> Phase:
    """Infer phase from obs and append to the active PhaseLog if any."""
    phase = infer_phase(obs, prev_inventory, prev_prices)
    log_transition(obs, phase)
    return phase


class PhaseLogWrapper:
    """Wrap an agent callable; logs inferred phases each turn (pre-PolicyBrain)."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._prev_inv: Optional[Dict[str, Any]] = None
        self._prev_prices: Optional[Dict[str, Any]] = None

    def _advance_market_snapshot(self, obs: Mapping[str, Any]) -> None:
        market = obs.get("market") or {}
        self._prev_inv = dict(market.get("inventory") or {})
        self._prev_prices = dict(market.get("prices") or {})

    def __call__(self, obs: Mapping[str, Any], configuration: Any = None) -> Dict[str, Any]:
        infer_and_log(obs, self._prev_inv, self._prev_prices)
        self._advance_market_snapshot(obs)
        if hasattr(self._agent, "act") and callable(getattr(self._agent, "act")):
            return self._agent.act(obs)
        if configuration is not None:
            return self._agent(obs, configuration)
        return self._agent(obs)
