"""Replay-aware, failure-safe surrogate agent for Kaggriculture.

The class in this module deliberately has no simulator dependency.  It accepts
the observation dictionaries produced by Kaggriculture and returns the normal
``{"farmer": ..., "hands": ..., "market": ...}`` action object.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

try:
    from kaggriculture.actions.actions import Actions
except ImportError:  # Useful when this file is loaded directly by a submission.
    class Actions:  # type: ignore[no-redef]
        PASS = "PASS"


SAFE_NOOP = {"farmer": ["PASS"], "hands": [], "market": []}
_UNIT_OPS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLANT", "WATER",
    "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "DIG", "PLACE",
    "FEED", "COLLECT_FERTILIZER", "CARE", "DROP",
}
_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}
_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


def _pass() -> list[str]:
    return [getattr(Actions, "PASS", "PASS")]


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


class SurrogateAgent:
    """Three-tier replay agent with deterministic counters and safe fallbacks.

    ``recorded_actions`` may be a list indexed by replay turn, a mapping from
    turn to action, or a replay JSON object containing ``steps``.  Kaggriculture
    replays are post-transition indexed, so the default lookup uses
    ``observation.step + 1``.
    """

    def __init__(
        self,
        recorded_actions: Any = None,
        *,
        replay_player: int = 1,
        recorded_offset: int = 1,
        replay_seed: Any = None,
        target_crop: str = "WHEAT",
    ) -> None:
        self.replay_player = replay_player
        self.recorded_offset = recorded_offset
        self.replay_seed = replay_seed
        self.target_crop = str(target_crop).upper()
        self._recorded = self._extract_actions(recorded_actions)
        self.counters: Counter[str] = Counter()
        self.fallback_reasons: Counter[str] = Counter()
        self.last_tier = "none"
        self.last_reason = ""

    @staticmethod
    def _extract_actions(source: Any) -> Any:
        if isinstance(source, (str, Path)):
            with Path(source).open() as handle:
                source = json.load(handle)
        if isinstance(source, Mapping) and "steps" in source:
            source = source["steps"]
        if isinstance(source, Mapping):
            return source
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            result = {}
            for index, frame in enumerate(source):
                if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
                    result[index] = frame
                else:
                    result[index] = frame
            return result
        return {}

    @property
    def exact_replay_turns(self) -> int:
        return self.counters["exact_replay"]

    @property
    def surrogate_turns(self) -> int:
        return self.counters["surrogate"]

    @property
    def safe_noop_turns(self) -> int:
        return self.counters["safe_noop"]

    @property
    def divergence_count(self) -> int:
        return self.counters["divergence"]

    @property
    def fallback_count(self) -> int:
        return self.counters["surrogate"] + self.counters["safe_noop"]

    def metrics(self) -> dict[str, Any]:
        return {
            "exact_replay_turns": self.exact_replay_turns,
            "surrogate_turns": self.surrogate_turns,
            "safe_noop_turns": self.safe_noop_turns,
            "divergence_count": self.divergence_count,
            "fallback_count": self.fallback_count,
            "fallback_reasons": dict(self.fallback_reasons),
        }

    def _lookup_recorded(self, obs: Mapping[str, Any]) -> Any:
        if self.replay_seed is not None and obs.get("seed") is not None and obs.get("seed") != self.replay_seed:
            self.fallback_reasons["seed_mismatch"] += 1
            return None
        step = _positive_int(obs.get("step"), 0) + self.recorded_offset
        raw = self._recorded.get(step) if isinstance(self._recorded, Mapping) else None
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            player = _positive_int(obs.get("player"), self.replay_player)
            raw = raw[player] if player < len(raw) else None
        if isinstance(raw, Mapping) and "action" in raw:
            raw = raw["action"]
        return raw

    @staticmethod
    def _valid_action(action: Any) -> bool:
        if not isinstance(action, Mapping) or not isinstance(action.get("farmer"), list):
            return False
        if not isinstance(action.get("hands", []), list) or not isinstance(action.get("market", []), list):
            return False
        if not action["farmer"] or str(action["farmer"][0]).upper() not in _UNIT_OPS:
            return False
        for unit in action["hands"]:
            if not isinstance(unit, list) or not unit or str(unit[0]).upper() not in _UNIT_OPS:
                return False
        for order in action["market"]:
            if not isinstance(order, list) or not order or str(order[0]).upper() not in _MARKET_OPS:
                return False
        return True

    def _scale(self, action: Mapping[str, Any], obs: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        out = {"farmer": list(action["farmer"]), "hands": [list(x) for x in action.get("hands", [])], "market": []}
        private = obs.get("private") or {}
        shed = private.get("shed") or {}
        seeds = private.get("seeds") or {}
        money = _positive_int((obs.get("farms") or [{}])[int(obs.get("player", 0))].get("money", 0))
        prices = (obs.get("market") or {}).get("prices") or {}
        changed = False
        for key in ("farmer", "hands"):
            units = [out[key]] if key == "farmer" else out[key]
            for unit in units:
                if unit and str(unit[0]).upper() == "PLANT" and len(unit) > 1:
                    crop = str(unit[1]).upper()
                    if _positive_int(seeds.get(crop)) <= 0:
                        replacement = next((c for c, n in seeds.items() if _positive_int(n) > 0), None)
                        if replacement:
                            unit[1], changed = str(replacement).upper(), True
                        else:
                            unit[:], changed = _pass(), True
        for order in action.get("market", []):
            if not isinstance(order, list) or not order:
                continue
            item = str(order[1]).upper() if len(order) > 1 else ""
            op = str(order[0]).upper()
            if op == "SELL":
                available = _positive_int(shed.get(item))
                quantity = min(_positive_int(order[2] if len(order) > 2 else 0), available)
            elif op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
                unit_price = _positive_int(prices.get(item), 20)
                quantity = min(_positive_int(order[2] if len(order) > 2 else 0), money // max(1, unit_price))
                money -= quantity * unit_price
            else:
                out["market"].append(list(order))
                continue
            if quantity:
                normalized = [op, item, quantity]
                out["market"].append(normalized)
                changed |= normalized != order
            else:
                changed = True
        return out, changed

    @staticmethod
    def _tile_at(farm: Mapping[str, Any], pos: Any) -> Any:
        try:
            x, y = int(pos[0]), int(pos[1])
            tiles = farm.get("tiles") or []
            return tiles[y][x]
        except (IndexError, KeyError, TypeError, ValueError):
            return None

    def _fallback(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        player = _positive_int(obs.get("player"), 0)
        farms = obs.get("farms") or []
        if not isinstance(farms, list) or player >= len(farms) or not isinstance(farms[player], Mapping):
            raise ValueError("observation has no usable local farm")
        farm = farms[player]
        private = obs.get("private") or {}
        if not isinstance(private, Mapping):
            raise ValueError("observation has no usable private state")
        positions = [farm.get("farmer", [0, 0]), *(farm.get("hands") or [])]
        inventories = private.get("inventories") or []
        units: list[list[Any]] = []
        for index, position in enumerate(positions):
            tile = self._tile_at(farm, position)
            action = _pass()
            if isinstance(tile, Mapping):
                if tile.get("kind") == "PLANT" and not tile.get("watered_today", False):
                    action = ["WATER"]
                elif tile.get("kind") == "PLANT" and _positive_int(tile.get("yield_units")) > 0:
                    action = ["HARVEST"]
                elif tile.get("kind") == "WEED":
                    action = ["DIG"]
                elif tile.get("animal") and not tile.get("fed_today", False):
                    action = ["FEED"]
                elif tile.get("animal") and not tile.get("cared_today", False):
                    action = ["CARE"]
            units.append(action)
        if not units:
            units = [_pass()]
        market: list[list[Any]] = []
        for item, quantity in sorted((private.get("shed") or {}).items()):
            quantity = _positive_int(quantity)
            if quantity:
                market.append(["SELL", str(item).upper(), quantity])
        market = market[:10]
        return {"farmer": units[0], "hands": units[1:], "market": market}

    def step(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return one action without raising on malformed/impossible observations."""
        try:
            recorded = self._lookup_recorded(observation)
            if self._valid_action(recorded):
                action, changed = self._scale(recorded, observation)
                self.counters["exact_replay"] += 1
                self.last_tier = "recorded"
                self.last_reason = "scaled" if changed else "exact"
                if changed:
                    self.counters["divergence"] += 1
                return action
            if recorded is not None:
                self.fallback_reasons["invalid_recorded_action"] += 1
            action = self._fallback(observation)
            if self._valid_action(action):
                self.counters["surrogate"] += 1
                self.last_tier, self.last_reason = "surrogate", "behavioral_archetype"
                return action
            raise ValueError("surrogate produced invalid action")
        except Exception as exc:
            self.counters["safe_noop"] += 1
            reason = type(exc).__name__.lower()
            self.fallback_reasons[reason] += 1
            self.last_tier, self.last_reason = "safe_noop", reason
            return deepcopy(SAFE_NOOP)


# Public name used by the replay-engine specification. Keep the shorter
# implementation name for existing callers.
ShadowReplayAgent = SurrogateAgent


def make_agent(recorded_actions: Any = None, **kwargs: Any) -> SurrogateAgent:
    """Factory for loaders that expect an ``agent``-like callable object."""
    return SurrogateAgent(recorded_actions, **kwargs)
