"""Online Counterfactual Regret (CFR) matching for meta-policy selection.

Replaces the MetaController's hardcoded Bayesian confidence thresholds —
deducible by Rank-1 opponents and exploitable via deceptive signaling — with
a no-regret mixed strategy over the 3 core macro-policies that converges
toward Nash equilibrium during the live match.

Protocol (one CFR iteration per in-game day, at Hour 23)
-------------------------------------------------------
1. Observe the end-of-day state ``s`` and the policy ``a`` played today.
2. Counterfactual reward: for each policy ``p``, estimate the terminal cash
   we would hold right now had ``p`` been played today instead of ``a``::

       u_p = V(s_p)          # s_p = counterfactual leaf state for policy p

   ``V`` is the IQL Value Network (any ``predict``/``evaluate``/``value``/
   ``__call__`` API, as in ``scratch_grandmaster._call_value_net``), with a
   material fallback (liquid cash + fertilizer x $100 + cows x $400, as in
   ``scratch_grandmaster._fallback_leaf_value``) when inference is
   unavailable.
3. Instantaneous regret per policy::

       r_p = u_p - u_a       # "how much more terminal cash if B, not A, today?"

4. Accumulate into the running Regret Table:: ``R_p += r_p``.
5. Tomorrow's mixed strategy via standard regret matching::

       sigma_p = R_p^+ / sum_q R_q^+   (uniform 1/3 when all R <= 0)

The current strategy is the no-regret iterate (unpredictable under deceptive
signaling); the time-average strategy is what converges toward Nash
equilibrium.  Both are exposed (``strategy_for_tomorrow`` /
``average_strategy``).

Cost: O(3) arithmetic per day, stdlib only, no torch/numpy/scipy — safe to
call inline at Hour 23 inside the live turn budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Policy set (kept as literals so this module has zero package imports and
# therefore cannot create an import cycle with kaggriculture.meta).
# ---------------------------------------------------------------------------
POLICY_A = "Policy_A"  # Milk Dominance  (task name: "Milk Flooder")
POLICY_B = "Policy_B"  # Strawberry / Insulated Crop Defense ("Strawberry Contingency")
POLICY_C = "Policy_C"  # Deceptive Poisoned Well ("Poisoned Well")

POLICIES: Tuple[str, str, str] = (POLICY_A, POLICY_B, POLICY_C)

# Human-readable task names accepted anywhere a policy id is expected.
POLICY_ALIASES = {
    "MILK_FLOODER": POLICY_A,
    "MILK DOMINANCE": POLICY_A,
    "MILK_DOMINANCE": POLICY_A,
    "STRAWBERRY_CONTINGENCY": POLICY_B,
    "STRAWBERRY CONTINGENCY": POLICY_B,
    "STRAWBERRY_CONTINGENT": POLICY_B,
    "POISONED_WELL": POLICY_C,
    "POISONED WELL": POLICY_C,
    "DECEPTIVE_POISONED_WELL": POLICY_C,
}

END_OF_DAY_HOUR = 23
TURNS_PER_DAY = 24


def normalize_policy(policy: Any) -> str:
    """Canonicalize a policy id or task-name alias to Policy_A/B/C."""
    s = str(policy or "").strip().upper().replace("-", "_")
    s = "_".join(s.split())
    canonical = {"POLICY_A": POLICY_A, "POLICY_B": POLICY_B, "POLICY_C": POLICY_C}
    if s in canonical:
        return canonical[s]
    if s in POLICY_ALIASES:
        return POLICY_ALIASES[s]
    raise ValueError(f"unknown macro-policy {policy!r}; expected one of {list(POLICIES)}")


def normalize_policies(policies: Optional[Sequence[Any]] = None) -> Tuple[str, ...]:
    """Canonicalize an explicit policy set (default: the 3 core policies)."""
    if policies is None:
        return POLICIES
    out = tuple(normalize_policy(p) for p in policies)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate policies in {policies!r}")
    if not out:
        raise ValueError("policy set must be non-empty")
    return out


# ---------------------------------------------------------------------------
# Leaf-state evaluation: IQL Value Network first, material fallback second.
# Mirrors scratch_grandmaster._call_value_net / _fallback_leaf_value.
# ---------------------------------------------------------------------------

_MATERIAL_KEYS = (
    ("Liquid Cash", "liquid_cash", "money", "cash"),
    ("Fertilizer_Stock", "fertilizer_stock", "FERTILIZER", "fertilizer"),
    ("Cows", "cows", "COW"),
)


def _find_material(leaf: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        if name in leaf:
            try:
                return float(leaf[name])  # type: ignore[index]
            except (TypeError, ValueError):
                continue
    for value in leaf.values():
        if isinstance(value, Mapping):
            found = _find_material(value, names)
            if found:
                return found
    return 0.0


def fallback_leaf_value(leaf_state: Any) -> float:
    """Material terminal value: cash + fertilizer x $100 + cows x $400."""
    if isinstance(leaf_state, Mapping):
        cash = _find_material(leaf_state, _MATERIAL_KEYS[0])
        fertilizer = _find_material(leaf_state, _MATERIAL_KEYS[1])
        cows = _find_material(leaf_state, _MATERIAL_KEYS[2])
        return cash + fertilizer * 100.0 + cows * 400.0
    try:
        return float(leaf_state)
    except (TypeError, ValueError):
        return 0.0


def call_value_net(value_net: Any, leaf_state: Any) -> float:
    """Call common IQL value-network APIs; raise honestly when unavailable."""
    if value_net is None:
        raise LookupError("IQL_Value_Net is unavailable")
    for name in ("predict", "evaluate", "value", "__call__"):
        method = getattr(value_net, name, None)
        if callable(method):
            value = method(leaf_state)
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (list, tuple)):
                value = value[0]
            return float(value)
    raise TypeError("IQL_Value_Net is not callable")


def evaluate_leaf_value(leaf_state: Any, value_net: Any = None) -> float:
    """Dual leaf evaluator: IQL first, then the material fallback."""
    try:
        return call_value_net(value_net, leaf_state)
    except (LookupError, OSError, TypeError, ValueError, RuntimeError):
        return fallback_leaf_value(leaf_state)


# ---------------------------------------------------------------------------
# Counterfactual leaf construction.
# ---------------------------------------------------------------------------

def project_counterfactual_leaves(
    base_leaf: Mapping[str, Any],
    policy_deltas: Mapping[str, Mapping[str, float]],
    policies: Optional[Sequence[Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build one leaf state per policy from a shared end-of-day base leaf.

    ``policy_deltas[p]`` maps material field -> additive shift expressing what
    today's books would look like had ``p`` been played instead (e.g. extra
    milk stock under Policy_A, extra crop yield under Policy_B).  Missing
    policies default to zero shift (leaf == base).  Callers that fork real
    simulator states (e.g. via the counterfactual rollout engine) should pass
    those leaves directly to ``counterfactual_values`` instead.
    """
    order = normalize_policies(policies)
    leaves: Dict[str, Dict[str, Any]] = {}
    for policy in order:
        leaf = dict(base_leaf)
        for key, delta in dict(policy_deltas.get(policy, {}) or {}).items():
            try:
                leaf[key] = float(leaf.get(key, 0.0)) + float(delta)
            except (TypeError, ValueError):
                leaf[key] = float(delta)
        leaves[policy] = leaf
    return leaves


def counterfactual_values(
    leaf_states: Mapping[str, Any],
    value_net: Any = None,
    policies: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Estimate terminal cash per policy via the IQL Value Network.

    ``leaf_states`` maps policy -> leaf state (arbitrary object the value net
    accepts, or a material dict for the fallback).  Every policy in
    ``policies`` must be present; evaluation failures fall back per-leaf, so
    one bad leaf never poisons the other two estimates.
    """
    order = normalize_policies(policies)
    values: Dict[str, float] = {}
    for policy in order:
        if policy not in leaf_states:
            raise KeyError(f"missing counterfactual leaf for policy {policy!r}")
        values[policy] = evaluate_leaf_value(leaf_states[policy], value_net)
    return values


# ---------------------------------------------------------------------------
# Regret table + regret matching.
# ---------------------------------------------------------------------------

def regret_matching_strategy(
    cumulative_regret: Mapping[str, float],
    policies: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Standard regret matching: sigma_p propto max(R_p, 0), else uniform."""
    order = normalize_policies(policies)
    positive = {p: max(0.0, float(cumulative_regret.get(p, 0.0))) for p in order}
    total = sum(positive.values())
    if total <= 0.0:
        return {p: 1.0 / len(order) for p in order}
    return {p: positive[p] / total for p in order}


@dataclass
class DayUpdate:
    """Audit trail of one Hour-23 CFR iteration."""

    day: int
    played_policy: str
    values: Dict[str, float]
    regrets: Dict[str, float]
    cumulative_regret: Dict[str, float]
    strategy_for_tomorrow: Dict[str, float]


class RegretTable:
    """Running cumulative regret over the 3 core macro-policies."""

    def __init__(self, policies: Optional[Sequence[Any]] = None) -> None:
        self.policies = normalize_policies(policies)
        self.cumulative: Dict[str, float] = {p: 0.0 for p in self.policies}
        self.history: List[DayUpdate] = []

    def add(
        self,
        day: int,
        played_policy: Any,
        values: Mapping[str, float],
    ) -> DayUpdate:
        """Apply one day's counterfactual rewards; return the audit record."""
        played = normalize_policy(played_policy)
        if played not in self.cumulative:
            raise ValueError(f"played policy {played_policy!r} not in {self.policies!r}")
        played_value = float(values[played])
        regrets = {p: float(values[p]) - played_value for p in self.policies
                   if p in values}
        missing = [p for p in self.policies if p not in values]
        if missing:
            raise KeyError(f"missing counterfactual values for {missing!r}")
        for p, r in regrets.items():
            self.cumulative[p] += r
        update = DayUpdate(
            day=int(day),
            played_policy=played,
            values={p: float(values[p]) for p in self.policies},
            regrets=dict(regrets),
            cumulative_regret=dict(self.cumulative),
            strategy_for_tomorrow=regret_matching_strategy(self.cumulative, self.policies),
        )
        self.history.append(update)
        return update

    def strategy(self) -> Dict[str, float]:
        """Current regret-matching distribution (tomorrow's mix)."""
        return regret_matching_strategy(self.cumulative, self.policies)

    def reset(self) -> None:
        """Clear cumulative regret and history (e.g. episode restart)."""
        self.cumulative = {p: 0.0 for p in self.policies}
        self.history = []


# ---------------------------------------------------------------------------
# Online selector: the live-game CFR loop.
# ---------------------------------------------------------------------------

def obs_day(obs: Mapping[str, Any]) -> int:
    """Day index from obs (explicit day, else step // 24)."""
    try:
        return int(obs.get("day", int(obs.get("step", 0) or 0) // TURNS_PER_DAY))
    except (TypeError, ValueError):
        return 0


def obs_hour(obs: Mapping[str, Any]) -> int:
    """Hour index from obs (explicit hour, else step % 24)."""
    try:
        return int(obs.get("hour", int(obs.get("step", 0) or 0) % TURNS_PER_DAY))
    except (TypeError, ValueError):
        return 0


def is_end_of_day(obs: Mapping[str, Any], hour: int = END_OF_DAY_HOUR) -> bool:
    """True only at the Hour-23 CFR tick (the day's books are final)."""
    return obs_hour(obs) == int(hour)


class CFRPolicySelector:
    """Lightweight online CFR matcher driving tomorrow's policy mix.

    Wraps a :class:`RegretTable` with the live-game plumbing: Hour-23 gating,
    IQL-valued counterfactual leaves, current + average strategies, and
    sampling.  ``to_meta_distribution()`` is a drop-in replacement for
    ``MetaController.meta_distribution()``'s pure best response.
    """

    def __init__(
        self,
        policies: Optional[Sequence[Any]] = None,
        value_net: Any = None,
        seed: Optional[int] = None,
    ) -> None:
        self.table = RegretTable(policies)
        self.value_net = value_net
        self._rng = random.Random(seed)
        self._strategy_sum: Dict[str, float] = {p: 0.0 for p in self.table.policies}
        self._iterations = 0

    @property
    def policies(self) -> Tuple[str, ...]:
        return self.table.policies

    @property
    def cumulative_regret(self) -> Dict[str, float]:
        return dict(self.table.cumulative)

    def end_of_day_update(
        self,
        day: int,
        played_policy: Any,
        leaf_states_or_values: Mapping[str, Any],
        value_net: Any = None,
    ) -> DayUpdate:
        """Run one CFR iteration for the finished day.

        ``leaf_states_or_values`` is either ``{policy: terminal-cash float}``
        (already estimated, e.g. from rollout forks) or ``{policy: leaf}``
        evaluated here through ``value_net or self.value_net`` with the
        material fallback.  A leaf mapping is treated as a leaf state, never
        as a scalar value.
        """
        first = next(iter(leaf_states_or_values.values()))
        if isinstance(first, Mapping):
            values = counterfactual_values(
                leaf_states_or_values,
                value_net if value_net is not None else self.value_net,
                self.table.policies,
            )
        else:
            values = {p: float(leaf_states_or_values[p]) for p in self.table.policies
                      if p in leaf_states_or_values}
            missing = [p for p in self.table.policies if p not in leaf_states_or_values]
            if missing:
                raise KeyError(f"missing counterfactual values for {missing!r}")
        update = self.table.add(day, played_policy, values)
        strategy = update.strategy_for_tomorrow
        for p in self.table.policies:
            self._strategy_sum[p] += strategy[p]
        self._iterations += 1
        return update

    def maybe_update_from_obs(
        self,
        obs: Mapping[str, Any],
        played_policy: Any,
        leaf_states_or_values: Mapping[str, Any],
        value_net: Any = None,
    ) -> Optional[DayUpdate]:
        """Hour-23-gated wrapper: no-op (None) on non-update turns."""
        if not is_end_of_day(obs):
            return None
        return self.end_of_day_update(
            obs_day(obs), played_policy, leaf_states_or_values, value_net)

    def strategy_for_tomorrow(self) -> Dict[str, float]:
        """Current regret-matching mix (the no-regret iterate for tomorrow)."""
        return self.table.strategy()

    def average_strategy(self) -> Dict[str, float]:
        """Time-average mix — the iterate that converges toward Nash."""
        if self._iterations <= 0:
            return {p: 1.0 / len(self.table.policies) for p in self.table.policies}
        return {p: self._strategy_sum[p] / self._iterations for p in self.table.policies}

    def select_for_tomorrow(self, rng: Optional[random.Random] = None) -> str:
        """Sample tomorrow's policy from the mixed strategy (unpredictable)."""
        strategy = self.strategy_for_tomorrow()
        draw = (rng or self._rng).random()
        cumulative = 0.0
        for p in self.table.policies:
            cumulative += strategy[p]
            if draw < cumulative:
                return p
        return self.table.policies[-1]

    def to_meta_distribution(self) -> Dict[str, float]:
        """Drop-in for ``MetaController.meta_distribution()`` (mixed, not pure)."""
        return self.strategy_for_tomorrow()

    def max_average_regret(self) -> float:
        """max_p R_p / T: vanishes for no-regret play; diagnostic for tests."""
        if self._iterations <= 0:
            return 0.0
        return max(self.table.cumulative.values()) / self._iterations

    def reset(self) -> None:
        """Clear regret, history, and the average-strategy accumulator."""
        self.table.reset()
        self._strategy_sum = {p: 0.0 for p in self.table.policies}
        self._iterations = 0


__all__ = [
    "POLICY_A",
    "POLICY_B",
    "POLICY_C",
    "POLICIES",
    "POLICY_ALIASES",
    "END_OF_DAY_HOUR",
    "TURNS_PER_DAY",
    "DayUpdate",
    "RegretTable",
    "CFRPolicySelector",
    "normalize_policy",
    "normalize_policies",
    "fallback_leaf_value",
    "call_value_net",
    "evaluate_leaf_value",
    "project_counterfactual_leaves",
    "counterfactual_values",
    "regret_matching_strategy",
    "obs_day",
    "obs_hour",
    "is_end_of_day",
]
