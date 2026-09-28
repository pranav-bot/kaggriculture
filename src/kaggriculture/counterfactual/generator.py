"""Counterfactual Market Data Generator.

Uses the Rust simulation engine (kaggsim.env) to retroactively evaluate
alternative market decisions (HOLD instead of SELL >10 Milk/Wool) in historical
replays, generating counterfactual_value labels for market forecasting and offline RL.
"""

from __future__ import annotations

import copy
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class CounterfactualUnavailableError(RuntimeError):
    """Raised when the Rust ``kaggsim`` bindings are not available."""


# Ensure scratch_grandmaster can be imported
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scratch_grandmaster import agent as default_heuristic_agent
except ImportError:
    default_heuristic_agent = None


class CounterfactualRecord(dict):
    """Dictionary representing a counterfactual transition record with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'CounterfactualRecord' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def detect_large_sells(action: Any) -> List[Tuple[int, str, int]]:
    """Detect whenever an action contains a SELL order for >10 units of MILK or WOOL.

    Args:
        action: Player action (dict with 'market', list of orders, or order string).

    Returns:
        List of tuples: (order_index, commodity, quantity).
    """
    if action is None:
        return []

    orders: List[Any] = []
    if isinstance(action, Mapping):
        orders = action.get("market") or []
    elif isinstance(action, (list, tuple)):
        # Could be list of orders or single order
        if action and isinstance(action[0], (str, int, float)) and len(action) >= 3 and str(action[0]).upper() == "SELL":
            orders = [action]
        else:
            orders = list(action)
    elif isinstance(action, str):
        # Tape string format: check market section after tabs
        parts = action.split("\t")
        market_str = parts[-1] if len(parts) >= 3 else action
        for cmd in market_str.split(";"):
            tokens = cmd.strip().split()
            if len(tokens) >= 3 and tokens[0].upper() == "SELL":
                orders.append(tokens)

    results: List[Tuple[int, str, int]] = []
    for idx, order in enumerate(orders):
        item = ""
        qty = 0

        if isinstance(order, (list, tuple)):
            if len(order) >= 3 and str(order[0]).upper() == "SELL":
                item = str(order[1]).upper()
                try:
                    qty = int(order[2])
                except (ValueError, TypeError):
                    continue
        elif isinstance(order, Mapping):
            act_type = str(order.get("action", order.get("type", ""))).upper()
            if act_type == "SELL":
                item = str(order.get("item", order.get("product", ""))).upper()
                try:
                    qty = int(order.get("quantity", order.get("amount", 0)))
                except (ValueError, TypeError):
                    continue
        elif isinstance(order, str):
            tokens = order.strip().split()
            if len(tokens) >= 3 and tokens[0].upper() == "SELL":
                item = tokens[1].upper()
                try:
                    qty = int(tokens[2])
                except (ValueError, TypeError):
                    continue

        if item in ("MILK", "WOOL") and qty > 10:
            results.append((idx, item, qty))

    return results


def create_hold_action(action: Any, target_commodity: Optional[str] = None) -> Any:
    """Create a counterfactual HOLD action by suppressing SELL of target commodity.

    Args:
        action: Original historical action.
        target_commodity: 'MILK' or 'WOOL' to suppress, or None to suppress all >10 sells.

    Returns:
        Counterfactual action dict or structure with the SELL order omitted.
    """
    if action is None:
        return {"farmer": ["PASS"], "hands": [], "market": []}

    target_comm = target_commodity.upper() if target_commodity else None

    if isinstance(action, Mapping):
        cf_action = copy.deepcopy(dict(action))
        raw_market = cf_action.get("market") or []
        new_market: List[Any] = []
        for order in raw_market:
            suppress = False
            if isinstance(order, (list, tuple)) and len(order) >= 3:
                if str(order[0]).upper() == "SELL":
                    item = str(order[1]).upper()
                    try:
                        q = int(order[2])
                    except (ValueError, TypeError):
                        q = 0
                    if (target_comm is None or item == target_comm) and item in ("MILK", "WOOL") and q > 10:
                        suppress = True
            elif isinstance(order, Mapping):
                act_type = str(order.get("action", order.get("type", ""))).upper()
                if act_type == "SELL":
                    item = str(order.get("item", order.get("product", ""))).upper()
                    try:
                        q = int(order.get("quantity", order.get("amount", 0)))
                    except (ValueError, TypeError):
                        q = 0
                    if (target_comm is None or item == target_comm) and item in ("MILK", "WOOL") and q > 10:
                        suppress = True
            if not suppress:
                new_market.append(order)
        cf_action["market"] = new_market
        return cf_action

    elif isinstance(action, (list, tuple)):
        # List of market orders
        new_orders = []
        for order in action:
            if isinstance(order, (list, tuple)) and len(order) >= 3 and str(order[0]).upper() == "SELL":
                item = str(order[1]).upper()
                try:
                    q = int(order[2])
                except (ValueError, TypeError):
                    q = 0
                if (target_comm is None or item == target_comm) and item in ("MILK", "WOOL") and q > 10:
                    continue
            new_orders.append(order)
        return new_orders

    elif isinstance(action, str):
        # Tape string format
        parts = action.split("\t")
        if len(parts) >= 3:
            market_cmds = parts[2].split(";")
            kept_cmds = []
            for cmd in market_cmds:
                toks = cmd.strip().split()
                if len(toks) >= 3 and toks[0].upper() == "SELL":
                    item = toks[1].upper()
                    try:
                        q = int(toks[2])
                    except (ValueError, TypeError):
                        q = 0
                    if (target_comm is None or item == target_comm) and item in ("MILK", "WOOL") and q > 10:
                        continue
                kept_cmds.append(cmd.strip())
            parts[2] = ";".join(c for c in kept_cmds if c)
            return "\t".join(parts)
        return "PASS\t\t"

    return action


def _default_policy_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback rollout step function using scratch_grandmaster or PASS."""
    if default_heuristic_agent is not None:
        try:
            return default_heuristic_agent(obs)
        except Exception:
            pass
    return {"farmer": ["PASS"], "hands": [], "market": []}


def generate_counterfactuals(
    state_dict: Mapping[str, Any],
    actual_action: Any,
    *,
    historical_terminal_cash: Optional[Union[float, Mapping[int, float], Sequence[float]]] = None,
    rollout_policy: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    kagg_binary: Optional[str] = None,
    seed: int = 42,
) -> List[CounterfactualRecord]:
    """Identify whenever a historical player sold >10 units of Milk or Wool,
    initialize the Rust environment (kaggsim.env) from state_dict, inject HOLD,
    rollout to turn 720, and calculate counterfactual_value.

    Args:
        state_dict: Engine state or Kaggle observation dictionary at timestep t.
        actual_action: Action(s) taken at step t. Can be:
            - A dict for one seat: {"farmer": [...], "market": [...]}
            - A list of two dicts: [action_seat0, action_seat1]
            - A list of market orders directly
            - A tape string
        historical_terminal_cash: Terminal cash from historical trajectory (optional).
            Can be scalar, or mapping/sequence {0: cash0, 1: cash1}.
            If omitted, a factual rollout baseline is computed.
        rollout_policy: Callable(obs) -> action for rollout to step 720.
            Defaults to scratch_grandmaster apex heuristic agent.
        kagg_binary: Optional path to Rust kagg executable.
        seed: Simulation seed.

    Returns:
        List of CounterfactualRecord objects (one for each detected large sell).
    """
    try:
        from kaggsim.env import Env
    except ImportError as exc:
        raise CounterfactualUnavailableError(
            "Counterfactual generation requires the Rust-backed kaggsim Python "
            "package. Install/build kaggriculture-simulation before running this "
            "pipeline."
        ) from exc
    from kaggriculture.counterfactual.serializer import extract_engine_state

    policy = rollout_policy or _default_policy_step

    # Inspect actual_action to find all seat candidates and large sells
    seat_actions: Dict[int, Any] = {}
    if isinstance(actual_action, (list, tuple)) and len(actual_action) == 2 and isinstance(actual_action[0], (Mapping, str)):
        seat_actions[0] = actual_action[0]
        seat_actions[1] = actual_action[1]
    elif isinstance(actual_action, Mapping) and 0 in actual_action and 1 in actual_action:
        seat_actions[0] = actual_action[0]
        seat_actions[1] = actual_action[1]
    else:
        # Single action dict or market order list
        primary_seat = int(state_dict.get("player", 0))
        seat_actions[primary_seat] = actual_action
        seat_actions[1 - primary_seat] = {"farmer": ["PASS"], "hands": [], "market": []}

    # Find which seats executed large sells of MILK or WOOL > 10
    detected_candidates: List[Tuple[int, str, int, Any]] = []
    for s, act in seat_actions.items():
        sells = detect_large_sells(act)
        for _, comm, qty in sells:
            detected_candidates.append((s, comm, qty, act))

    if not detected_candidates:
        return []

    # Ensure state_dict is in engine format
    full_state = extract_engine_state(state_dict, seed=seed)
    step = int(full_state.get("step", 0))

    # Helper to resolve historical terminal cash
    def get_hist_cash(seat_idx: int) -> Optional[float]:
        if historical_terminal_cash is not None:
            if isinstance(historical_terminal_cash, Mapping):
                return float(historical_terminal_cash.get(seat_idx, 0.0))
            if isinstance(historical_terminal_cash, (list, tuple)):
                if seat_idx < len(historical_terminal_cash):
                    return float(historical_terminal_cash[seat_idx])
            try:
                return float(historical_terminal_cash)
            except (ValueError, TypeError):
                pass
        if "historical_terminal_cash" in state_dict:
            htc = state_dict["historical_terminal_cash"]
            if isinstance(htc, Mapping):
                return float(htc.get(seat_idx, 0.0))
            return float(htc)
        return None

    results: List[CounterfactualRecord] = []

    for seat_idx, commodity, quantity, orig_act in detected_candidates:
        cf_action = create_hold_action(orig_act, target_commodity=commodity)
        opp_seat = 1 - seat_idx
        opp_act = seat_actions.get(opp_seat) or {"farmer": ["PASS"], "hands": [], "market": []}

        # Step t action pair with counterfactual injected for seat_idx
        if seat_idx == 0:
            a0_t, a1_t = cf_action, opp_act
        else:
            a0_t, a1_t = opp_act, cf_action

        # Initialize Rust environment from exact state_dict
        with Env(kagg=kagg_binary, state=full_state, seed=seed) as env:
            # 1. Inject counterfactual action at step t
            (obs0, obs1), rewards, done, info = env.step(a0_t, a1_t)

            # 2. Fast heuristic rollout to terminal step 719
            while not done:
                p0 = policy(obs0)
                p1 = policy(obs1)
                (obs0, obs1), rewards, done, info = env.step(p0, p1)

            synthetic_terminal_cash = float(info["banks"][seat_idx])

        # 3. Determine historical terminal cash
        hist_cash = get_hist_cash(seat_idx)
        if hist_cash is None:
            # Run baseline factual rollout to compute ground truth comparison
            with Env(kagg=kagg_binary, state=full_state, seed=seed) as env_f:
                if seat_idx == 0:
                    f_a0, f_a1 = orig_act, opp_act
                else:
                    f_a0, f_a1 = opp_act, orig_act
                (f_obs0, f_obs1), _, f_done, f_info = env_f.step(f_a0, f_a1)
                while not f_done:
                    p0 = policy(f_obs0)
                    p1 = policy(f_obs1)
                    (f_obs0, f_obs1), _, f_done, f_info = env_f.step(p0, p1)
                hist_cash = float(f_info["banks"][seat_idx])

        # 4. Calculate counterfactual_value
        cf_value = synthetic_terminal_cash - hist_cash

        # Extract market and private snapshots at decision point
        m_info = full_state.get("market") or {}
        prices = m_info.get("prices") or {}
        inv = m_info.get("inventory") or m_info.get("inventories") or {}

        priv_list = full_state.get("private")
        priv_seat = priv_list[seat_idx] if isinstance(priv_list, list) and seat_idx < len(priv_list) else {}
        shed = priv_seat.get("shed") if isinstance(priv_seat, dict) else {}

        record = CounterfactualRecord({
            "step": step,
            "day": step // 24,
            "hour": step % 24,
            "seat": seat_idx,
            "commodity": commodity,
            "historical_sell_quantity": quantity,
            "historical_action": copy.deepcopy(orig_act),
            "counterfactual_action": "HOLD",
            "counterfactual_action_dict": cf_action,
            "historical_terminal_cash": hist_cash,
            "synthetic_terminal_cash": synthetic_terminal_cash,
            "counterfactual_value": cf_value,
            "market_prices": copy.deepcopy(prices),
            "market_inventory": copy.deepcopy(inv),
            "private_shed": copy.deepcopy(shed),
        })
        results.append(record)

    return results


def _process_single_episode_worker(task: Tuple[str, Optional[str]]) -> List[Dict[str, Any]]:
    """Worker task executed in a separate process for multiprocessing.

    Args:
        task: Tuple of (episode_file_path, kagg_binary_path).

    Returns:
        List of generated counterfactual record dictionaries for this episode.
    """
    from kaggriculture.eda.parser import parse_episode, extract_frame

    ep_path, kagg_binary = task
    records: List[Dict[str, Any]] = []

    try:
        data = parse_episode(ep_path)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", ep_path, exc)
        return []

    steps = data.get("steps") or []
    if len(steps) < 2:
        return []

    # Historical terminal cash from the final step of the replay
    final_frame = steps[-1]
    final_obs0 = final_frame[0].get("observation") or {} if final_frame else {}
    final_obs1 = final_frame[1].get("observation") or {} if len(final_frame) > 1 else {}

    farms0 = final_obs0.get("farms") or []
    farms1 = final_obs1.get("farms") or []
    term_cash0 = float(farms0[0].get("money", 0.0)) if len(farms0) > 0 else 0.0
    term_cash1 = float(farms1[1].get("money", 0.0)) if len(farms1) > 1 else 0.0

    if term_cash0 == 0.0 and len(final_frame) > 0:
        term_cash0 = float(final_frame[0].get("reward", 0.0) or 0.0)
    if term_cash1 == 0.0 and len(final_frame) > 1:
        term_cash1 = float(final_frame[1].get("reward", 0.0) or 0.0)

    terminal_cash_map = {0: term_cash0, 1: term_cash1}
    num_steps = len(steps)

    for t in range(num_steps - 1):
        obs0, act0, _ = extract_frame(steps, t=t, seat=0)
        obs1, act1, _ = extract_frame(steps, t=t, seat=1)

        sells0 = detect_large_sells(act0)
        sells1 = detect_large_sells(act1)

        if not sells0 and not sells1:
            continue

        # Build full engine state mapping for step t
        st = dict(obs0)
        st["step"] = t
        st["day"] = t // 24
        st["hour"] = t % 24
        st["private"] = [obs0.get("private", {}), obs1.get("private", {})]

        cfs = generate_counterfactuals(
            st,
            [act0, act1],
            historical_terminal_cash=terminal_cash_map,
            kagg_binary=kagg_binary,
            seed=42 + t,
        )

        for cf in cfs:
            cf["episode_id"] = Path(ep_path).name
            records.append(dict(cf))

    return records


class CounterfactualPipeline:
    """Multiprocessing pipeline that processes replay episodes in parallel and streams
    counterfactual evaluation tuples to a JSONL file."""

    def __init__(
        self,
        output_path: Union[Path, str],
        num_workers: Optional[int] = None,
        kagg_binary: Optional[str] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.num_workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
        self.kagg_binary = kagg_binary

    def run(
        self,
        episode_paths: Sequence[Union[Path, str]],
        max_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the counterfactual generation pipeline over replay episodes.

        Args:
            episode_paths: Sequence of paths to .json or .json.gz replay files.
            max_episodes: Maximum number of episodes to process.

        Returns:
            Dictionary with execution summary statistics.
        """
        targets = list(episode_paths)
        if max_episodes is not None and max_episodes > 0:
            targets = targets[:max_episodes]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        tasks = [(str(p), self.kagg_binary) for p in targets]
        total_episodes = len(tasks)
        total_counterfactuals = 0
        episodes_with_counterfactuals = 0

        logger.info(
            "Starting CounterfactualPipeline on %d episodes using %d worker processes",
            total_episodes,
            self.num_workers,
        )

        # Open JSONL file for streaming writes
        with open(self.output_path, "w", encoding="utf-8") as out_f:
            if self.num_workers <= 1 or total_episodes <= 1:
                # Single-process mode
                for task in tasks:
                    ep_records = _process_single_episode_worker(task)
                    if ep_records:
                        episodes_with_counterfactuals += 1
                        for r in ep_records:
                            out_f.write(json.dumps(r) + "\n")
                            total_counterfactuals += 1
                        out_f.flush()
            else:
                # Multiprocessing pool mode
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=self.num_workers) as pool:
                    for ep_records in pool.imap_unordered(_process_single_episode_worker, tasks):
                        if ep_records:
                            episodes_with_counterfactuals += 1
                            for r in ep_records:
                                out_f.write(json.dumps(r) + "\n")
                                total_counterfactuals += 1
                            out_f.flush()

        stats = {
            "output_path": str(self.output_path),
            "episodes_processed": total_episodes,
            "episodes_with_large_sells": episodes_with_counterfactuals,
            "counterfactual_records_generated": total_counterfactuals,
            "num_workers": self.num_workers,
        }
        logger.info("Pipeline complete: %s", stats)
        return stats


def run_counterfactual_pipeline(
    episodes: Sequence[Union[Path, str]],
    output_file: Union[Path, str],
    num_workers: Optional[int] = None,
    max_episodes: Optional[int] = None,
    kagg_binary: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience runner for CounterfactualPipeline."""
    pipeline = CounterfactualPipeline(
        output_path=output_file,
        num_workers=num_workers,
        kagg_binary=kagg_binary,
    )
    return pipeline.run(episodes, max_episodes=max_episodes)
