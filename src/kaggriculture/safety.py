"""Pure-Python SafeFallbackController & @impenetrable_agent Watchdog Decorator.

Bulletproofs agents against:
1. Python exceptions (uninitialized variables, KeyError, NameError, FFI crashes,
   tensor shape mismatches, memory errors).
2. Kaggle 1.0-second soft turn limits and 60-second overage bank exhaustion.
3. Automatically triggers SafeFallbackController (<1ms execution) and completely
   disables heavy neural networks/Beam Search when remaining overage bank < 5.0s.
4. Logs detailed diagnostics to sys.stderr for post-match autopsy.
"""

from __future__ import annotations

import functools
import logging
import math
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

Pos = Tuple[int, int]

# Kaggle Environments timing constraints
DEFAULT_SOFT_LIMIT_S = 1.0       # Per-turn soft compute limit
DEFAULT_OVERAGE_BANK_S = 60.0    # Match-level overage time bank
DEFAULT_SAFETY_THRESHOLD_S = 5.0 # Circuit breaker trip threshold (< 5.0s remaining)


# =============================================================================
# 1. Pure-Python, Zero-Dependency SafeFallbackController (<1ms)
# =============================================================================

class SafeFallbackController:
    """Pure-Python, zero-dependency fallback controller executing in <1 millisecond.

    Implements the robust 'Care Mill' strategy:
    - Feeds existing cows (fetches wheat from shed if needed).
    - Cares for cows daily to maintain compounding productivity.
    - Collects produced fertilizer from animal pastures.
    - Harvests available milk.
    - Drops harvested goods into the shed.
    - Sells 100% of stored fertilizer if Day < 8; otherwise holds fertilizer.
    - Flushes residual shed inventory on Day 29 to maximize terminal score.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.total_act_duration_s: float = 0.0

    def _dist(self, a: Sequence[int], b: Pos) -> int:
        """Manhattan distance between two points."""
        return abs(int(a[0]) - b[0]) + abs(int(a[1]) - b[1])

    def _shed_tiles(self, board_size: int = 10) -> List[Pos]:
        """Central shed access tiles."""
        h = board_size // 2
        return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]

    def _move(self, src: Sequence[int], target: Pos) -> List[str]:
        """Fast orthogonal movement toward target."""
        sx, sy = int(src[0]), int(src[1])
        tx, ty = int(target[0]), int(target[1])
        dx, dy = tx - sx, ty - sy

        if abs(dx) >= abs(dy) and dx != 0:
            return ["EAST" if dx > 0 else "WEST"]
        if dy != 0:
            return ["SOUTH" if dy > 0 else "NORTH"]
        if dx != 0:
            return ["EAST" if dx > 0 else "WEST"]
        return ["PASS"]

    def act(self, obs: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Generates valid Kaggle environment action dictionary in <1 millisecond."""
        t_start = time.perf_counter()
        self.call_count += 1

        # Safe parsing with defaults
        player = int(obs.get("player", 0))
        farms = obs.get("farms", [{}])
        farm = farms[player] if player < len(farms) else (farms[0] if farms else {})
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        money = float(farm.get("money", 0.0))

        tiles = farm.get("tiles", [])
        board_size = len(tiles) if tiles else 10
        shed_access = self._shed_tiles(board_size)

        farmer_pos = farm.get("farmer", [4, 4])
        hands_pos = farm.get("hands", [])
        all_workers = [farmer_pos] + list(hands_pos)

        private = obs.get("private", farm.get("private", {}))
        shed = dict(private.get("shed", farm.get("shed", {})))
        inventories = list(private.get("inventories", farm.get("inventories", [])))

        # Scan cows on pastures
        cows: List[Tuple[Pos, Dict[str, Any]]] = []
        for r, row in enumerate(tiles):
            for c, tile in enumerate(row):
                if isinstance(tile, dict):
                    anim = str(tile.get("animal", "")).upper()
                    # Primarily cows, but handles sheep/geese if cows aren't present
                    if anim in ("COW", "SHEEP", "GOOSE") or tile.get("kind") in ("PASTURE", "COOP"):
                        cows.append(((c, r), tile))

        # Sort cows prioritizing cows
        cows.sort(key=lambda item: 0 if str(item[1].get("animal", "")).upper() == "COW" else 1)

        # Worker action planning
        claimed_targets: set[Pos] = set()
        worker_actions: List[List[str]] = []

        unfed_cows = [pos for pos, t in cows if not bool(t.get("fed_today", False))]
        uncared_cows = [pos for pos, t in cows if not bool(t.get("cared_today", False))]
        fert_cows = [pos for pos, t in cows if bool(t.get("fertilizer_available", False))]
        harvest_cows = [pos for pos, t in cows if int(t.get("yield_units", 0)) > 0]

        cow_at_pos = {pos: t for pos, t in cows}

        def nearest_shed(pos: Pos) -> Pos:
            return min(shed_access, key=lambda s: (self._dist(pos, s), s))

        def take_nearest(pos: Pos, candidates: List[Pos]) -> Optional[Pos]:
            available = [c for c in candidates if c not in claimed_targets]
            if not available:
                return None
            best = min(available, key=lambda c: (self._dist(pos, c), c))
            claimed_targets.add(best)
            return best

        for idx, worker_pos in enumerate(all_workers):
            pos: Pos = (int(worker_pos[0]), int(worker_pos[1]))
            inv = inventories[idx] if idx < len(inventories) and isinstance(inventories[idx], dict) else {}
            wheat_count = int(inv.get("WHEAT", 0))
            carried_goods = sum(int(v) for k, v in inv.items() if k != "WHEAT")

            here_tile = cow_at_pos.get(pos)

            # 1. Underfoot cow actions
            if here_tile is not None:
                if wheat_count > 0 and not bool(here_tile.get("fed_today", False)):
                    here_tile["fed_today"] = True
                    if pos in unfed_cows:
                        unfed_cows.remove(pos)
                    worker_actions.append(["FEED"])
                    continue
                if not bool(here_tile.get("cared_today", False)):
                    here_tile["cared_today"] = True
                    if pos in uncared_cows:
                        uncared_cows.remove(pos)
                    worker_actions.append(["CARE"])
                    continue
                if int(here_tile.get("yield_units", 0)) > 0:
                    here_tile["yield_units"] = 0
                    if pos in harvest_cows:
                        harvest_cows.remove(pos)
                    worker_actions.append(["HARVEST"])
                    continue
                if bool(here_tile.get("fertilizer_available", False)):
                    here_tile["fertilizer_available"] = False
                    if pos in fert_cows:
                        fert_cows.remove(pos)
                    worker_actions.append(["COLLECT_FERTILIZER"])
                    continue

            # 2. Feed hungry cows
            if unfed_cows:
                if wheat_count > 0:
                    target = take_nearest(pos, unfed_cows)
                    if target is not None:
                        worker_actions.append(["FEED"] if pos == target else self._move(pos, target))
                        continue
                elif int(shed.get("WHEAT", 0)) > 0:
                    shed_target = nearest_shed(pos)
                    if pos in shed_access:
                        take_qty = min(4, int(shed.get("WHEAT", 0)))
                        shed["WHEAT"] = max(0, int(shed.get("WHEAT", 0)) - take_qty)
                        worker_actions.append(["PICKUP", "WHEAT", str(take_qty)])
                    else:
                        worker_actions.append(self._move(pos, shed_target))
                    continue

            # 3. Care for cows
            if uncared_cows:
                target = take_nearest(pos, uncared_cows)
                if target is not None:
                    worker_actions.append(["CARE"] if pos == target else self._move(pos, target))
                    continue

            # 4. Harvest cow yields
            if harvest_cows:
                target = take_nearest(pos, harvest_cows)
                if target is not None:
                    worker_actions.append(["HARVEST"] if pos == target else self._move(pos, target))
                    continue

            # 5. Collect fertilizer
            if fert_cows:
                target = take_nearest(pos, fert_cows)
                if target is not None:
                    worker_actions.append(["COLLECT_FERTILIZER"] if pos == target else self._move(pos, target))
                    continue

            # 6. Drop goods at shed
            if carried_goods > 0:
                shed_target = nearest_shed(pos)
                if pos in shed_access:
                    worker_actions.append(["DROP"])
                else:
                    worker_actions.append(self._move(pos, shed_target))
                continue

            # Default: PASS
            worker_actions.append(["PASS"])

        # Market Orders
        market_orders: List[List[str]] = []

        # Fertilizer logic: Sell all fertilizer if Day < 8, otherwise hold
        fert_in_shed = int(shed.get("FERTILIZER", 0))
        if day < 8 and fert_in_shed > 0:
            market_orders.append(["SELL", "FERTILIZER", str(fert_in_shed)])

        # Terminal Day 29 flush (hours 20..23)
        if day == 29 and hour >= 20:
            for item, qty in shed.items():
                if int(qty) > 0 and len(market_orders) < 10:
                    market_orders.append(["SELL", str(item).upper(), str(qty)])
        else:
            # Regularly liquidate milk and wool if buffered
            milk_qty = int(shed.get("MILK", 0))
            if milk_qty >= 3 and len(market_orders) < 10:
                market_orders.append(["SELL", "MILK", str(min(6, milk_qty))])
            wool_qty = int(shed.get("WOOL", 0))
            if wool_qty >= 2 and len(market_orders) < 10:
                market_orders.append(["SELL", "WOOL", str(min(4, wool_qty))])

        # Emergency wheat replenishment if cows are starving
        wheat_in_shed = int(shed.get("WHEAT", 0))
        if cows and wheat_in_shed < 4 and money >= 25.0 and len(market_orders) < 10:
            buy_wheat = min(4, int(money // 25.0))
            if buy_wheat > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", str(buy_wheat)])

        farmer_cmd = worker_actions[0] if worker_actions else ["PASS"]
        hands_cmds = worker_actions[1:] if len(worker_actions) > 1 else []

        dur = time.perf_counter() - t_start
        self.total_act_duration_s += dur

        return {
            "farmer": farmer_cmd,
            "hands": hands_cmds,
            "market": market_orders[:10],
            "_fallback_active": True,
            "_fallback_latency_ms": dur * 1000.0,
        }


# =============================================================================
# 2. @impenetrable_agent Decorator & Watchdog
# =============================================================================

class ImpenetrableAgentWrapper:
    """Watchdog and circuit-breaker wrapper around an agent(obs, config) callable."""

    def __init__(
        self,
        agent_fn: Callable[..., Any],
        fallback_controller: Optional[SafeFallbackController] = None,
        overage_bank_s: float = DEFAULT_OVERAGE_BANK_S,
        soft_limit_s: float = DEFAULT_SOFT_LIMIT_S,
        safety_threshold_s: float = DEFAULT_SAFETY_THRESHOLD_S,
    ) -> None:
        self.agent_fn = agent_fn
        self.fallback_controller = fallback_controller or SafeFallbackController()
        self.overage_bank_s: float = float(overage_bank_s)
        self.soft_limit_s: float = float(soft_limit_s)
        self.safety_threshold_s: float = float(safety_threshold_s)

        self.remaining_overage: float = self.overage_bank_s
        self.cumulative_overage_used: float = 0.0
        self.circuit_broken: bool = False
        self.fallback_trigger_count: int = 0
        self.last_error: Optional[str] = None
        self.turn_history: List[float] = []

        functools.update_wrapper(self, agent_fn)

    def reset(self) -> None:
        """Reset watchdog state for new matches or benchmark seeds."""
        self.remaining_overage = self.overage_bank_s
        self.cumulative_overage_used = 0.0
        self.circuit_broken = False
        self.fallback_trigger_count = 0
        self.last_error = None
        self.turn_history.clear()

    def __call__(self, obs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Executes agent turn protected by try/except and time.perf_counter() watchdog."""
        step = int(obs.get("step", 0)) if isinstance(obs, Mapping) else 0
        day = int(obs.get("day", 0)) if isinstance(obs, Mapping) else 0
        hour = int(obs.get("hour", 0)) if isinstance(obs, Mapping) else 0

        # Check environment-reported overage if present
        if isinstance(obs, Mapping):
            env_overage = obs.get("remainingOverageTime", obs.get("remaining_overage_time"))
            if env_overage is not None:
                self.remaining_overage = min(self.remaining_overage, float(env_overage))

        # 1. Circuit Breaker Active: Completely bypass primary agent & Beam Search
        if self.circuit_broken:
            return self.fallback_controller.act(obs, *args, **kwargs)

        # 2. Execute Primary Agent with Strict Exception Shield
        t_start = time.perf_counter()
        action: Optional[Dict[str, Any]] = None

        try:
            action = self.agent_fn(obs, *args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.fallback_trigger_count += 1
            self.last_error = f"{type(e).__name__}: {e}"

            # Log to sys.stderr for post-match diagnosis
            sys.stderr.write(
                f"[IMPENETRABLE_AGENT] CRITICAL: Caught {type(e).__name__} at Step {step} "
                f"(Day {day}, Hour {hour}):\n{e}\n{tb_str}\n"
                f"Activating SafeFallbackController for this turn.\n"
            )
            sys.stderr.flush()

            # Execute SafeFallbackController
            action = self.fallback_controller.act(obs, *args, **kwargs)
            action["_exception_shield_triggered"] = True
            action["_last_error"] = self.last_error

        t_end = time.perf_counter()
        turn_duration = t_end - t_start
        self.turn_history.append(turn_duration)

        # 3. Watchdog Accounting
        if turn_duration > self.soft_limit_s:
            overage_consumed = turn_duration - self.soft_limit_s
            self.cumulative_overage_used += overage_consumed
            self.remaining_overage = max(0.0, self.overage_bank_s - self.cumulative_overage_used)

        # 4. Check Safety Threshold (< 5.0s remaining)
        if self.remaining_overage < self.safety_threshold_s and not self.circuit_broken:
            self.circuit_broken = True
            sys.stderr.write(
                f"[IMPENETRABLE_AGENT_WATCHDOG] CRITICAL: Remaining overage bank {self.remaining_overage:.3f}s "
                f"dropped below safety threshold ({self.safety_threshold_s:.1f}s) at Step {step} (Day {day})! "
                f"Permanently disabling primary neural network & Beam Search; "
                f"SafeFallbackController taking over for remainder of match.\n"
            )
            sys.stderr.flush()

        # Ensure return format is valid dictionary
        if not isinstance(action, dict):
            sys.stderr.write(f"[IMPENETRABLE_AGENT] Non-dict action returned: {type(action)}. Falling back.\n")
            sys.stderr.flush()
            action = self.fallback_controller.act(obs, *args, **kwargs)

        return action

    def stats(self) -> Dict[str, Any]:
        """Diagnostic state for monitoring and post-match reports."""
        return {
            "circuit_broken": self.circuit_broken,
            "remaining_overage_s": self.remaining_overage,
            "cumulative_overage_used_s": self.cumulative_overage_used,
            "fallback_trigger_count": self.fallback_trigger_count,
            "last_error": self.last_error,
            "total_turns": len(self.turn_history),
            "max_turn_duration_ms": max(self.turn_history, default=0.0) * 1000.0,
            "avg_turn_duration_ms": (sum(self.turn_history) / max(1, len(self.turn_history))) * 1000.0,
            "fallback_call_count": self.fallback_controller.call_count,
        }


def impenetrable_agent(
    fn: Optional[Callable[..., Any]] = None,
    *,
    fallback_controller: Optional[SafeFallbackController] = None,
    overage_bank_s: float = DEFAULT_OVERAGE_BANK_S,
    soft_limit_s: float = DEFAULT_SOFT_LIMIT_S,
    safety_threshold_s: float = DEFAULT_SAFETY_THRESHOLD_S,
) -> Union[ImpenetrableAgentWrapper, Callable[[Callable[..., Any]], ImpenetrableAgentWrapper]]:
    """Decorator wrapping agent(obs, config) with exception shielding & overage watchdog.

    Can be used with or without arguments:
        @impenetrable_agent
        def agent(obs, config=None):
            ...

        @impenetrable_agent(safety_threshold_s=3.0)
        def agent(obs, config=None):
            ...
    """
    def decorator(target_fn: Callable[..., Any]) -> ImpenetrableAgentWrapper:
        return ImpenetrableAgentWrapper(
            agent_fn=target_fn,
            fallback_controller=fallback_controller,
            overage_bank_s=overage_bank_s,
            soft_limit_s=soft_limit_s,
            safety_threshold_s=safety_threshold_s,
        )

    if fn is not None:
        return decorator(fn)
    return decorator
