"""Per-episode telemetry (research/01 §5, research/04 §5)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from kaggriculture.helpers.sell_ranking import impact_score

SHED_CAPACITY = 100


def shed_total(private: Mapping[str, Any] | None) -> int:
    shed = (private or {}).get("shed") or {}
    return int(sum(max(0, int(v or 0)) for v in shed.values()))


def carried_total(private: Mapping[str, Any] | None) -> int:
    total = 0
    for inv in (private or {}).get("inventories") or []:
        if isinstance(inv, dict):
            total += int(sum(max(0, int(v or 0)) for v in inv.values()))
    return total


def _sell_slots(orders: Sequence[Sequence[Any]]) -> List[tuple[str, int]]:
    slots: List[tuple[str, int]] = []
    for order in orders:
        if len(order) < 3 or str(order[0]).upper() != "SELL":
            continue
        slots.append((str(order[1]).upper(), int(order[2])))
    return slots


def count_unfillable_sell_slots(
    planned: Sequence[Sequence[Any]],
    after_repair: Sequence[Sequence[Any]],
) -> int:
    """SELL order slots present before capacity repair but not fully honored after."""
    before = Counter(_sell_slots(planned))
    after = Counter(_sell_slots(after_repair))
    burned = 0
    for slot, count in before.items():
        missing = count - after.get(slot, 0)
        if missing > 0:
            burned += missing
    return burned


def _premium_collision_diff(
    before: Sequence[Sequence[Any]],
    after: Sequence[Sequence[Any]],
) -> tuple[int, int]:
    """Approximate collision holds (premium sells dropped) and releases (added)."""
    premium = {"MELON", "CARROT", "WHEAT", "PUMPKIN", "CABBAGE", "ONION", "TOMATO"}

    def premium_qty(orders: Sequence[Sequence[Any]]) -> Counter[str]:
        c: Counter[str] = Counter()
        for order in orders:
            if len(order) < 3 or str(order[0]).upper() != "SELL":
                continue
            item = str(order[1]).upper()
            if item in premium:
                c[item] += int(order[2])
        return c

    b, a = premium_qty(before), premium_qty(after)
    holds = 0
    releases = 0
    for item in set(b) | set(a):
        if b[item] > a[item]:
            holds += b[item] - a[item]
        if a[item] > b[item]:
            releases += a[item] - b[item]
    return holds, releases


def percentile_ms(samples: List[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


@dataclass
class EpisodeMetrics:
    """Aggregated episode telemetry."""

    steps: int = 0
    final_cash: float = 0.0
    shed_overflow_units_lost: int = 0
    unfillable_sell_slots_burned: int = 0
    hires_by_day: Dict[int, int] = field(default_factory=dict)
    sell_impact_scores: List[float] = field(default_factory=list)
    collision_holds: int = 0
    collision_releases: int = 0
    terminal_carried_goods_step_718: int = 0
    decide_ms_samples: List[float] = field(default_factory=list)
    decide_ms_p50: float = 0.0
    decide_ms_p95: float = 0.0

    @property
    def mean_impact_score_of_sells(self) -> float:
        if not self.sell_impact_scores:
            return 0.0
        return float(sum(self.sell_impact_scores) / len(self.sell_impact_scores))

    def finalize_latencies(self) -> None:
        self.decide_ms_p50 = percentile_ms(self.decide_ms_samples, 50.0)
        self.decide_ms_p95 = percentile_ms(self.decide_ms_samples, 95.0)

    def to_summary_dict(self) -> Dict[str, Any]:
        self.finalize_latencies()
        payload = asdict(self)
        payload["mean_impact_score_of_sells"] = self.mean_impact_score_of_sells
        payload.pop("decide_ms_samples", None)
        payload["hires_by_day"] = {str(k): v for k, v in sorted(self.hires_by_day.items())}
        return payload


class EpisodeMetricsRecorder:
    """Mutable accumulator; attach to ActionController.metrics_recorder."""

    def __init__(self) -> None:
        self.metrics = EpisodeMetrics()
        self._last_shed_total = 0

    def begin_step(self, obs: Mapping[str, Any]) -> None:
        private = obs.get("private") or {}
        current = shed_total(private)
        if current > SHED_CAPACITY and current > self._last_shed_total:
            self.metrics.shed_overflow_units_lost += current - max(
                SHED_CAPACITY, self._last_shed_total
            )
        self._last_shed_total = current

        step = int(obs.get("step", -1))
        if step == 718:
            self.metrics.terminal_carried_goods_step_718 = carried_total(private)

    def record_market_repair(
        self,
        planned: Sequence[Sequence[Any]],
        after_repair: Sequence[Sequence[Any]],
    ) -> None:
        self.metrics.unfillable_sell_slots_burned += count_unfillable_sell_slots(
            planned, after_repair
        )

    def record_hires(self, day: int, market_act: Sequence[Sequence[Any]]) -> None:
        hires = sum(
            1
            for order in market_act
            if order and str(order[0]).upper() == "HIRE"
        )
        if hires:
            self.metrics.hires_by_day[int(day)] = (
                self.metrics.hires_by_day.get(int(day), 0) + hires
            )

    def record_postprocess(
        self,
        before: Sequence[Sequence[Any]],
        after: Sequence[Sequence[Any]],
    ) -> None:
        holds, releases = _premium_collision_diff(before, after)
        self.metrics.collision_holds += holds
        self.metrics.collision_releases += releases

    def record_final_market(
        self,
        market_act: Sequence[Sequence[Any]],
        market_info: Mapping[str, Any] | None,
        town_shops: Sequence[str] | None,
    ) -> None:
        for order in market_act:
            score = impact_score(order, market_info, town_shops)
            if score > float("-inf"):
                self.metrics.sell_impact_scores.append(score)

    def finish_step(self, decide_ms: float) -> None:
        self.metrics.steps += 1
        self.metrics.decide_ms_samples.append(float(decide_ms))

    def finalize(self, obs: Mapping[str, Any], steps: Optional[int] = None) -> Dict[str, Any]:
        if steps is not None:
            self.metrics.steps = int(steps)
        farm = (obs.get("farms") or [{}])[0] if isinstance(obs, dict) else {}
        if isinstance(obs, dict) and "farms" in obs:
            player = int(obs.get("player", 0))
            farms = obs.get("farms") or []
            if len(farms) > player:
                farm = farms[player]
        self.metrics.final_cash = float(farm.get("money", 0.0) or 0.0)
        private = obs.get("private") if isinstance(obs, dict) else {}
        if int(obs.get("step", -1)) == 718 and isinstance(private, dict):
            self.metrics.terminal_carried_goods_step_718 = carried_total(private)
        return self.metrics.to_summary_dict()


class MetricsWrapper:
    """Wraps a callable or ActionController; wires metrics_recorder when supported."""

    def __init__(
        self,
        agent: Any,
        recorder: Optional[EpisodeMetricsRecorder] = None,
    ) -> None:
        self._agent = agent
        self.recorder = recorder or EpisodeMetricsRecorder()
        if hasattr(agent, "metrics_recorder"):
            agent.metrics_recorder = self.recorder

    def __call__(self, obs: Mapping[str, Any], configuration: Any = None) -> Dict[str, Any]:
        if hasattr(self._agent, "act") and callable(getattr(self._agent, "act")):
            return self._agent.act(obs)
        if configuration is not None:
            return self._agent(obs, configuration)
        return self._agent(obs)
