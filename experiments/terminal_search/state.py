"""Mutable match state for terminal-search simulation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


SHED_CAPACITY = 100


@dataclass
class TerminalState:
    farm: dict[str, Any]
    private: dict[str, Any]
    player: int = 0
    day: int = 27
    overflow_units: int = 0

    @classmethod
    def from_obs(cls, obs: dict[str, Any]) -> TerminalState:
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms") or [{}]
        farm = deepcopy(farms[player] if len(farms) > player else farms[0])
        private = deepcopy(obs.get("private") or {})
        if "inventories" not in private:
            private["inventories"] = [{}]
        if "shed" not in private:
            private["shed"] = {}
        return cls(
            farm=farm,
            private=private,
            player=player,
            day=int(obs.get("day", 27) or 27),
        )

    def shed_total(self) -> int:
        return int(sum(max(0, int(v or 0)) for v in (self.private.get("shed") or {}).values()))

    def record_overflow(self) -> None:
        excess = self.shed_total() - SHED_CAPACITY
        if excess > 0:
            self.overflow_units += excess

    def actor_positions(self) -> list[tuple[int, int]]:
        farmer = tuple(self.farm.get("farmer", [0, 0]))
        hands = [tuple(pos) for pos in (self.farm.get("hands") or [])]
        return [farmer, *hands]

    def inventories(self) -> list[dict[str, int]]:
        invs = self.private.get("inventories") or [{}]
        return [dict(inv) if isinstance(inv, dict) else {} for inv in invs]
