"""ShadowInjector: offline telemetry for Kaggle-stripped agents.
The production submission stays under 100MB / <1s per turn by shipping zero
debugging logs.  This offline tool replays a downloaded ``.json`` match,
rebuilds the exact ``obs`` dict each turn, and re-runs a local agent version
while *intercepting* subsystem internals via ``unittest.mock.patch`` — the
agent's source code is never modified.

Example:
    python tools/shadow_telemetry.py \\
        --replay replays/loss_123.json \\
        --agent src.kaggriculture.agent_final

Per turn the tool captures (when the agent actually uses the subsystem;
otherwise the block reports ``present: false``):

- CFR Regret Table values + policy probabilities (CFR Meta-Controller).
- Top 3 Beam Search macro-actions and their IQL expected values.  The leaf
  evaluator is not candidate-tagged, so only ``winner_score`` is a verified
  macro->value mapping; ``macro_iql_pairs`` is ``None`` with
  ``iql_values_attributed: false`` whenever attribution is not certain.
- Bayesian Opponent Intent probabilities (e.g. ``MELON_RUSH = 0.82``).
- Raw Claude DP Endgame solver output (safe sale volumes).
- Kuhn-Munkres routing assignment + cost-matrix summary.

Rows are appended to ``telemetry_<EPISODE_ID>.jsonl`` (one JSON object per
turn) for offline visualization.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from unittest import mock

MAX_LIST_ITEMS = 64
MAX_REPR_CHARS = 200
MAX_DEPTH = 6


# ---------------------------------------------------------------------------
# JSON sanitizer (bounded, never raises).
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return str(v)
        return round(v, 4)
    return v


def sanitize(value: Any, depth: int = 0) -> Any:
    """Convert arbitrary internals to bounded JSON-safe structures."""
    try:
        if depth > MAX_DEPTH:
            return "<max-depth>"
        if value is None or isinstance(value, (bool, int, str)):
            return _safe_float(value)
        if isinstance(value, float):
            return _safe_float(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return sanitize(dataclasses.asdict(value), depth + 1)
        if isinstance(value, Mapping):
            out = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= MAX_LIST_ITEMS:
                    out["<truncated>"] = f"{len(value) - i} more keys"
                    break
                out[str(k)] = sanitize(v, depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            items = [sanitize(v, depth + 1) for v in list(value)[:MAX_LIST_ITEMS]]
            if len(value) > MAX_LIST_ITEMS:
                items.append({"<truncated>": f"{len(value) - MAX_LIST_ITEMS} more"})
            return items
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return sanitize(value.to_dict(), depth + 1)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            return sanitize(
                {k: v for k, v in vars(value).items() if not k.startswith("_")},
                depth + 1,
            )
        return repr(value)[:MAX_REPR_CHARS]
    except Exception as exc:  # never break telemetry on one bad value
        return f"<unsanitizable: {type(exc).__name__}>"


# ---------------------------------------------------------------------------
# Agent + replay loading.
# ---------------------------------------------------------------------------

def load_agent(spec: str) -> Tuple[Callable[[Dict[str, Any]], Dict[str, Any]], Any]:
    """Import an agent module and resolve its ``agent(obs)`` entry point.

    ``spec`` accepts a dotted path (a leading ``src.`` prefix is tolerated:
    ``src.kaggriculture.agent_final`` -> ``kaggriculture.agent_final``), a
    ``.py`` file path, or a bare repo-root module name (``agent_final``).
    """
    if spec.endswith(".py") or os.sep in spec or "/" in spec:
        path = os.path.abspath(spec)
        name = os.path.splitext(os.path.basename(path))[0]
        pkg_dir = os.path.dirname(path)
        sys.path.insert(0, pkg_dir)
        sys.path.insert(0, os.path.join(pkg_dir, "src"))
        module = importlib.import_module(name)
    else:
        # Candidate dotted names, most specific first: the spec as given, then
        # with a leading ``src.`` stripped, then progressively shorter
        # suffixes so ``src.kaggriculture.agent_final`` still resolves when the
        # agent module actually lives at the repo root.
        candidates = [spec]
        if spec.startswith("src."):
            candidates.append(spec[4:])
        parts = spec.split(".")
        candidates.extend(".".join(parts[i:]) for i in range(1, len(parts)))
        for root in ("", os.getcwd()):
            for path_root in [p for p in (root, os.path.join(root, "src")) if p]:
                for cand in candidates:
                    if path_root not in sys.path:
                        sys.path.insert(0, path_root)
        module = None
        errors: List[str] = []
        seen: set = set()
        for path_root in ("", os.getcwd()):
            for cand in candidates:
                for base in (path_root, os.path.join(path_root, "src")):
                    if not base:
                        continue
                    tag = (base, cand)
                    if tag in seen:
                        continue
                    seen.add(tag)
                    try:
                        module = importlib.import_module(cand)
                        break
                    except ImportError as exc:
                        errors.append(f"{cand} (from {base or '.'}): {exc}")
                if module is not None:
                    break
            if module is not None:
                break
        if module is None:
            raise ImportError(
                f"could not import agent {spec!r}; tried: " + "; ".join(errors[:4]))
    entry = getattr(module, "agent", None)
    if callable(entry):
        return entry, module
    entry = getattr(module, "act", None)
    if callable(entry):
        return entry, module
    for attr in ("controller", "agent_", "policy"):
        obj = getattr(module, attr, None)
        if obj is not None:
            for meth in ("act", "agent", "__call__"):
                fn = getattr(obj, meth, None)
                if callable(fn):
                    return fn, module
    raise AttributeError(
        f"agent module {spec!r} exposes no agent(obs)/act entry point")


def _steps_from_replay(replay: Any) -> List[Any]:
    if isinstance(replay, Mapping) and "steps" in replay:
        return list(replay["steps"])
    if isinstance(replay, list):
        return list(replay)
    raise ValueError("replay JSON has no 'steps' list (expected Kaggle replay)")


def load_replay(path: str) -> Tuple[List[Any], str]:
    """Load a Kaggle replay file; return (steps, episode_id).

    Kaggle replay ``id`` is a UUID, while the human-facing episode number lives
    in ``info.EpisodeId`` (and in the ``episode-<id>-replay.json`` filename).
    ``info.EpisodeId`` is preferred so the output is ``telemetry_<EPISODE_ID>``
    rather than ``telemetry_<uuid>``.
    """
    with open(path) as fh:
        replay = json.load(fh)
    steps = _steps_from_replay(replay)
    episode_id = None
    if isinstance(replay, Mapping):
        info = replay.get("info")
        if isinstance(info, Mapping) and info.get("EpisodeId") is not None:
            episode_id = str(info["EpisodeId"])
        if episode_id is None:
            for field in ("episode_id", "episodeId", "id"):
                val = replay.get(field)
                if val is not None and str(val) not in ("", "None", "unknown"):
                    episode_id = str(val)
                    break
    if episode_id is None:
        stem = os.path.basename(path)
        match = re.search(r"episode[-_]?(\d+)", stem)
        episode_id = match.group(1) if match else os.path.splitext(stem)[0]
    return steps, episode_id


def obs_for_turn(steps: List[Any], turn: int, seat: int) -> Dict[str, Any]:
    """Rebuild the exact obs dict our seat saw at ``turn`` (deep-copied)."""
    row = steps[turn]
    if seat >= len(row):
        raise IndexError(
            f"turn {turn} has {len(row)} player entr(ies); seat {seat} requested")
    entry = row[seat]
    if isinstance(entry, Mapping) and "observation" in entry:
        entry = entry["observation"]
    if not isinstance(entry, Mapping):
        raise ValueError(f"turn {turn} seat {seat}: no observation dict")
    return copy.deepcopy(dict(entry))


# ---------------------------------------------------------------------------
# ShadowInjector: mock.patch interception without touching agent source.
# ---------------------------------------------------------------------------

def _manhattan(a: Any, b: Any) -> int:
    try:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
    except Exception:
        return 0


class ShadowInjector:
    """Enter/exit patch scope around one agent call; harvest per-turn logs."""

    def __init__(self, agent_module: Any) -> None:
        self.module = agent_module
        self._patches: List[Any] = []
        self.mocks: Dict[str, Any] = {}
        self.macro_names: Optional[List[str]] = None
        names = getattr(agent_module, "DISCRETE_MACRO_ACTIONS", None)
        if isinstance(names, (list, tuple)) and names:
            self.macro_names = [str(n) for n in names]

    # -- patch plumbing ----------------------------------------------------
    def _wrap(self, namespace: Any, attr: str, key: str) -> bool:
        try:
            original = getattr(namespace, attr, None)
        except Exception:
            return False
        if not callable(original):
            return False
        # NOTE: wraps-mocks neither retain returns nor bind `self` on methods
        # (instance.method loses the instance).  A real-function wrapper
        # installed via mock.patch keeps descriptor binding AND archives every
        # call/return for the harvesters — behavior is unchanged, interception
        # is complete.  Agent source is never edited.
        calls: List[Tuple[Any, Any]] = []
        returns: List[Any] = []
        try:
            sig = inspect.signature(original)
        except (TypeError, ValueError):
            sig = None

        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            calls.append((args, kwargs))
            returns.append(result)
            return result

        try:
            functools.update_wrapper(_wrapper, original)
        except Exception:
            pass
        patcher = mock.patch.object(namespace, attr, new=_wrapper)
        patcher.start()
        self._patches.append(patcher)
        self.mocks[key] = types.SimpleNamespace(
            calls=calls, returns=returns, original=original, sig=sig)
        return True

    def _arg_named(self, key: str, param: str, index: int = 0) -> Any:
        """Read a call argument by NAME.

        Patching a class method intercepts *unbound* calls, so positional
        index 1 is not the same parameter as on a module function (arg 0 is
        ``self``).  Binding the original signature is the only reliable way to
        pick out e.g. ``played_policy``.
        """
        calls = self._calls(key)
        if not calls:
            return None
        args, kwargs = calls[index]
        sig = getattr(self.mocks[key], "sig", None)
        if sig is not None:
            try:
                bound = sig.bind_partial(*args, **kwargs)
                if param in bound.arguments:
                    return bound.arguments[param]
            except TypeError:
                pass
        if param in kwargs:
            return kwargs[param]
        return None

    def _returns(self, key_prefix: str) -> List[Any]:
        """Archived returns for hooks under ``key_prefix`` (call order)."""
        out: List[Any] = []
        for key, entry in self.mocks.items():
            if key.startswith(key_prefix):
                try:
                    out.extend(getattr(entry, "returns", []) or [])
                except Exception:
                    continue
        return out

    def _returns_of(self, key: str) -> List[Any]:
        """Archived returns for one exact hook key (no cross-hook bleed)."""
        try:
            return list(getattr(self.mocks[key], "returns", []) or [])
        except Exception:
            return []

    def _keys(self, prefix: str) -> List[str]:
        return [k for k in self.mocks if k.startswith(prefix)]

    def _calls(self, key: str) -> List[Tuple[Any, Any]]:
        try:
            return list(getattr(self.mocks[key], "calls", []) or [])
        except Exception:
            return []

    def _reset_turn(self) -> None:
        for entry in self.mocks.values():
            try:
                del getattr(entry, "calls", [])[:]
            except Exception:
                pass
            try:
                del getattr(entry, "returns", [])[:]
            except Exception:
                pass

    def _wrap_all(self, namespace: Any, names: Tuple[str, ...],
                  prefix: str) -> bool:
        """Wrap EVERY name that exists — never short-circuit.

        ``any(genexpr)`` would stop at the first hit, silently leaving the
        entry point the agent actually calls unhooked (e.g. hooking
        ``step_level_beam_search`` while the agent runs
        ``beam_search_operations``), which shows up as a false
        ``present: false``.  Full coverage is required for correct
        telemetry.
        """
        hooked = False
        for name in names:
            if self._wrap(namespace, name, f"{prefix}:{name}"):
                hooked = True
        return hooked

    def install(self) -> Dict[str, bool]:
        """Patch every discoverable subsystem hook; report what was found."""
        mod = self.module
        found: Dict[str, bool] = {}

        # IQL Value Net: module-level leaf evaluators.
        found["iql"] = self._wrap_all(
            mod, ("evaluate_leaf", "_call_value_net", "IQL_Value_Net"), "iql")

        # CFR Meta-Controller: class methods (shared class object is fine —
        # patching is process-local to this offline run).
        cfr_cls = self._find_cfr_class()
        found["cfr"] = cfr_cls is not None and self._wrap_all(
            cfr_cls, ("end_of_day_update", "strategy_for_tomorrow",
                      "select_for_tomorrow", "cumulative_regret", "add"), "cfr")
        self._cfr_cls = cfr_cls

        # Beam Search: winner-returning entry points.
        found["beam"] = self._wrap_all(
            mod, ("step_level_beam_search", "beam_search",
                  "beam_search_operations"), "beam")

        # Kuhn-Munkres routing + cost matrix.
        found["km"] = self._wrap_all(
            mod, ("kuhn_munkres_route", "linear_sum_assignment"), "km")

        # Bayesian Opponent Intent: model methods + module functions.
        found["intent"] = (self._wrap_intent_class() or False) or self._wrap_all(
            mod, ("get_opponent_intent", "top_opponent_macros",
                  "predict_proba"), "intent")

        # Claude DP Endgame solver: quota-producing methods on used classes.
        hooked = False
        for cls in self._find_solver_classes():
            hooked = self._wrap_all(
                cls, ("plan_liquidation", "solve", "solve_commodity",
                      "generate_market_orders", "get_intent_for_day"),
                f"endgame:{cls.__name__}") or hooked
        found["endgame"] = hooked
        return found

    def _find_cfr_class(self) -> Optional[Any]:
        # Only classes the agent module actually references (defined or
        # imported there).  No global fallback: patching the shared class for
        # an agent that never touches CFR would be pure global side effects.
        for attr in ("CFRPolicySelector", "cfr_selector", "selector", "cfr",
                     "meta_controller", "MetaController"):
            obj = getattr(self.module, attr, None)
            if isinstance(obj, type) and hasattr(obj, "end_of_day_update"):
                return obj
            if obj is not None and hasattr(obj, "end_of_day_update") \
                    and hasattr(obj, "strategy_for_tomorrow"):
                return type(obj)
        return None

    def _wrap_intent_class(self) -> bool:
        for attr in ("OpponentIntentModel", "BayesianMarketPredictor",
                     "opponent_intent_model", "intent_model", "predictor"):
            obj = getattr(self.module, attr, None)
            cls = obj if isinstance(obj, type) else (
                type(obj) if obj is not None and hasattr(obj, "predict_proba") else None)
            if cls is not None and hasattr(cls, "predict_proba"):
                return self._wrap(cls, "predict_proba", f"intent:{cls.__name__}")
        try:
            from kaggriculture.models.opponent_intent import OpponentIntentModel
            if getattr(self.module, "OpponentIntentModel", None) is OpponentIntentModel:
                return self._wrap(OpponentIntentModel, "predict_proba", "intent:model")
        except Exception:
            pass
        return False

    def _find_solver_classes(self) -> List[Any]:
        classes = []
        for attr in ("TerminalLiquidationSolver", "RetrogradeDPSolver",
                     "MILPLiquidationSolver", "LiquidationController",
                     "liquidation_solver", "endgame_solver", "solver"):
            obj = getattr(self.module, attr, None)
            if isinstance(obj, type):
                classes.append(obj)
            elif obj is not None and type(obj) not in classes \
                    and hasattr(type(obj), "solve"):
                classes.append(type(obj))
        return classes

    def stop(self) -> None:
        for m in self._patches:
            try:
                m.stop()
            except Exception:
                pass
        self._patches = []

    # -- harvesters (per-turn telemetry blocks) -----------------------------
    def cfr_block(self) -> Dict[str, Any]:
        n_updates = 0
        last_update = None
        played: List[str] = []
        for key in self._keys("cfr:"):
            if not key.endswith("end_of_day_update"):
                continue
            try:
                n = len(self._calls(key))
                n_updates += n
                for ret in self._returns_of(key):
                    last_update = sanitize(ret)
                for i in range(n):
                    val = self._arg_named(key, "played_policy", i)
                    if val is not None:
                        played.append(str(val))
            except Exception:
                continue
        # Read the live table directly when the agent holds a selector.
        table, live_strategy = None, None
        obj = self._find_cfr_object()
        if obj is not None:
            # Prefer the selector's own accessors; fall back to its table so a
            # custom/duck-typed controller still reports.
            table = self._cfr_table(obj)
            if table is not None:
                # Ask the controller for tomorrow's distribution rather than
                # re-deriving regret matching here — the tool must never
                # disagree with the policy the agent actually applied.
                live_strategy = self._cfr_strategy(obj, table)
        if table is None and isinstance(last_update, Mapping):
            # Fall back to the update payload's own regret snapshot.
            for field in ("cumulative_regret", "regrets"):
                cand = last_update.get(field)
                if isinstance(cand, Mapping) and cand:
                    table = {str(k): float(v) for k, v in cand.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    break
            if live_strategy is None:
                strat = last_update.get("strategy_for_tomorrow")
                if isinstance(strat, Mapping) and strat:
                    live_strategy = {str(k): round(float(v), 4)
                                     for k, v in strat.items()}
        # Evidence rule: a patch alone is not presence — require an observed
        # update call or a table with real history (nonzero regret).
        used = n_updates > 0 or (table is not None and any(
            float(v) != 0.0 for v in table.values()))
        return {"present": bool(used), "cumulative_regret": table,
                "policy_probabilities": live_strategy,
                "n_day_updates": n_updates,
                "played_policy": played[-1] if played else None,
                "last_day_update": last_update}

    def _find_cfr_object(self) -> Optional[Any]:
        """Find the CFR selector *instance* the agent holds, if any."""
        seen: List[int] = []
        for name, obj in list(vars(self.module).items()):
            if name.startswith("__") or id(obj) in seen:
                continue
            seen.append(id(obj))
            if self._cfr_table(obj) is not None:
                return obj
        return None

    def _cfr_table(self, obj: Any) -> Optional[Dict[str, float]]:
        """Extract cumulative regret from a CFR controller or its table."""
        if obj is None or isinstance(obj, type):
            return None
        for meth in ("cumulative_regret",):
            fn = getattr(obj, meth, None)
            if callable(fn):
                try:
                    val = fn()
                except Exception:
                    val = None
                if isinstance(val, Mapping) and val:
                    return {str(k): float(v) for k, v in val.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}
        tab = getattr(obj, "table", None)
        if isinstance(tab, Mapping) and tab:
            return {str(k): float(v) for k, v in tab.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if tab is not None and not isinstance(tab, type):
            cum = getattr(tab, "cumulative", None)
            if isinstance(cum, Mapping) and cum:
                return {str(k): float(v) for k, v in cum.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return None

    def _cfr_strategy(self, obj: Any,
                      table: Mapping[str, float]) -> Optional[Dict[str, float]]:
        """Tomorrow's distribution, straight from the controller if possible."""
        for meth in ("strategy_for_tomorrow", "strategy"):
            fn = getattr(obj, meth, None)
            if not callable(fn):
                continue
            try:
                val = fn()
            except Exception:
                continue
            if isinstance(val, Mapping) and val:
                return {str(k): round(float(v), 4) for k, v in val.items()}
        # Last resort: standard regret matching over the observed table.
        total = sum(max(0.0, float(v)) for v in table.values())
        if total > 0:
            return {str(k): round(max(0.0, float(v)) / total, 4)
                    for k, v in table.items()}
        n = len(table)
        return {str(k): round(1.0 / n, 4) for k in table} if n else None

    def beam_block(self, iql_values: List[float]) -> Dict[str, Any]:
        winner_actions, winner_score, calls = None, None, 0
        for ret in self._returns("beam:"):
            calls += 1
            try:
                acts = getattr(ret, "actions", None)
                if acts is not None:
                    winner_actions = [int(a) for a in acts]
                    winner_score = float(getattr(ret, "score", float("nan")))
            except Exception:
                continue
        for key in self.mocks:
            if key.startswith("beam:"):
                try:
                    calls = max(calls, len(self._calls(key)))
                except Exception:
                    continue
        if winner_actions is None:
            return {"present": False, "n_searches": calls}
        names = [self.macro_names[a] if self.macro_names and 0 <= a < len(self.macro_names)
                 else f"MACRO_{a}" for a in winner_actions]
        top = names[:3]
        # A leaf evaluator is called once per candidate but its return value is
        # NOT tagged with the candidate it scored, so a macro -> value mapping
        # cannot be recovered from the leaf stream alone; zipping a ranked
        # value list onto the trajectory would invent (and can invert) the
        # correspondence.  The only verified mapping is the beam's own score
        # for the candidate it returned.  Report that, expose the raw ranked
        # leaf values, and mark the pairing unavailable rather than guess.
        pairs = None
        attributed = False
        if len(iql_values) == 1 and top:
            pairs = [{"macro": top[0], "iql_value": round(iql_values[0], 4),
                      "source": "sole_leaf_eval"}]
            attributed = True
        return {"present": True, "n_searches": calls,
                "winner_trajectory": names,
                "winner_score": winner_score,
                "top_macros": top,
                "winner_macro_value": winner_score,
                "macro_iql_pairs": pairs,
                "iql_values_attributed": attributed,
                "iql_expected_values": sorted(iql_values, reverse=True)[:3],
                "n_leaf_evals": len(iql_values)}

    def intent_block(self) -> Dict[str, Any]:
        probs = None
        for ret in self._returns("intent:"):
            try:
                if isinstance(ret, Mapping):
                    probs = {str(k): float(v) for k, v in ret.items()}
                elif isinstance(ret, (list, tuple)):
                    # top_opponent_macros-style [(macro_index, prob), ...]
                    seq = [(k, v) for k, v in ret
                           if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    if seq:
                        probs = {str(k): float(v) for k, v in seq}
            except Exception:
                continue
        return {"present": probs is not None, "probabilities": probs}

    def endgame_block(self) -> Dict[str, Any]:
        calls = []
        outputs: List[Any] = []
        # Pair each hook's own returns with that hook's name — positional
        # zipping across hooks desynchronises as soon as one is called twice.
        for key in self._keys("endgame:"):
            label = key.split(":", 1)[1]
            for ret in self._returns_of(key):
                out = sanitize(ret)
                outputs.append(out)
                calls.append({"method": label, "output": out})
        safe_volumes: Dict[str, Any] = {}
        for out in outputs:
            if isinstance(out, Mapping):
                for k, v in out.items():
                    if isinstance(v, Mapping) and "sell_quotas" in v:
                        safe_volumes[str(k)] = v["sell_quotas"]
        if not safe_volumes and outputs:
            safe_volumes = {"raw": outputs[-1]}
        return {"present": bool(outputs), "safe_sale_volumes": safe_volumes or None,
                "calls": calls}

    def km_block(self) -> Dict[str, Any]:
        assignments, units, targets = None, None, None
        for key in self._keys("km:"):
            # Hook keys are `km:<attr>`; match on the attribute name so the
            # block survives renames of the key prefix.
            attr = key.split(":", 1)[1]
            calls = self._calls(key)
            rets = self._returns_of(key)
            if not calls:
                continue
            if attr == "kuhn_munkres_route":
                args, _kwargs = calls[-1]
                if len(args) >= 2 and rets:
                    units, targets = list(args[0]), list(args[1])
                    assignments = [[int(a), int(b)] for a, b in rets[-1]]
                    break
            elif attr == "linear_sum_assignment":
                args, _kwargs = calls[-1]
                mat = args[0] if args else None
                if mat is not None and rets:
                    assignments = [[int(a), int(b)] for a, b in rets[-1]]
                    units = list(range(len(mat)))
                    targets = list(range(len(mat[0]))) if len(mat) else []
                    break
        if assignments is None:
            return {"present": False}
        total = 0
        if units is not None and targets is not None:
            for a, b in assignments:
                try:
                    total += _manhattan(units[a], targets[b])
                except Exception:
                    continue
        return {"present": True, "n_units": len(units or []),
                "n_targets": len(targets or []),
                "cost_matrix_shape": [len(units or []), len(targets or [])],
                "assignment": assignments, "total_manhattan_cost": total}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_telemetry(
    agent_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    injector: ShadowInjector,
    steps: List[Any],
    seat: int,
    max_turns: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Step the replay; call the agent under interception; harvest rows."""
    rows: List[Dict[str, Any]] = []
    n_turns = len(steps) if max_turns is None else min(len(steps), max_turns)
    for turn in range(n_turns):
        obs = obs_for_turn(steps, turn, seat)
        injector._reset_turn()
        try:
            action = agent_fn(obs)
            error = None
        except Exception as exc:  # telemetry must survive agent crashes
            action = None
            error = f"{type(exc).__name__}: {exc}"[:MAX_REPR_CHARS]
        series: List[float] = []
        for ret in injector._returns("iql:"):
            try:
                series.append(float(ret))
            except Exception:
                continue
        iql_block = {
            "present": bool(series),
            "n_evals": len(series),
            "last_value": round(series[-1], 4) if series else None,
            "values_sample": [round(v, 4) for v in series[-MAX_LIST_ITEMS:]],
        }
        rows.append({
            "turn": turn,
            "day": int(obs.get("day", turn // 24)),
            "hour": int(obs.get("hour", turn % 24)),
            "step": int(obs.get("step", turn)),
            "player": int(obs.get("player", seat)),
            "agent_action": sanitize(action),
            "error": error,
            "cfr": injector.cfr_block(),
            "beam": injector.beam_block(list(series)),
            "iql": iql_block,
            "opponent_intent": injector.intent_block(),
            "endgame": injector.endgame_block(),
            "km": injector.km_block(),
        })
    return rows


def main(argv: Optional[List[str]] = None) -> str:
    """CLI: parse args, install shadow hooks, stream JSONL telemetry."""
    parser = argparse.ArgumentParser(description="ShadowInjector offline telemetry")
    parser.add_argument("--replay", required=True, help="downloaded .json replay")
    parser.add_argument("--agent", required=True, help="agent module (dotted path or .py file)")
    parser.add_argument("--seat", type=int, default=0, help="player seat to shadow (default 0)")
    parser.add_argument("--out", default=".", help="output directory (default .)")
    parser.add_argument("--max-turns", type=int, default=None, help="cap turns (default: all)")
    args = parser.parse_args(argv)

    steps, episode_id = load_replay(args.replay)
    try:
        agent_fn, agent_module = load_agent(args.agent)
    except (ImportError, AttributeError) as exc:
        parser.error(f"cannot load agent {args.agent!r}: {exc}")
    injector = ShadowInjector(agent_module)
    found = injector.install()
    try:
        rows = run_telemetry(agent_fn, injector, steps, args.seat, args.max_turns)
    finally:
        injector.stop()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"telemetry_{episode_id}.jsonl")
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    hooked = sorted(k for k, v in found.items() if v)
    print(f"shadow telemetry: {len(rows)} turns -> {out_path} "
          f"(hooked: {', '.join(hooked) if hooked else 'none — graceful action-only mode'})")
    return out_path


if __name__ == "__main__":
    main()
