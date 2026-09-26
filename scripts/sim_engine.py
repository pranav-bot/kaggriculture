"""sim_engine.py — Unified Simulation Engine Adapter for Kaggriculture.

Seamlessly bridges between:
1. High-Performance Rust Simulator (`kagg` from `kaggriculture-simulation`):
   - ~15-20x faster than official Python runner (~0.3-0.5s per 720-step episode).
   - 100% byte-identical state parity verified by differential testing.
   - Direct support for single-seat tapes, opponent replay feeds, and parallel execution.
2. Official Python Engine (`kaggle_environments`):
   - Used as reference fallback or when running under environments without the compiled binary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
KAGGSIM_DIR = ROOT / "kaggriculture-simulation"
sys.path.insert(0, str(KAGGSIM_DIR / "src-python"))

try:
    from kaggsim.binary import find_kagg
    from kaggsim.constants import FINAL_STEP
    from kaggsim.serve import Serve, Struct, call_agent, load_agent, obs_for, seat_view
    from kaggsim.tape import action_to_line, load_replay, replay_actions, replay_seed
    _HAS_KAGGSIM = True
except ImportError:
    _HAS_KAGGSIM = False

try:
    from kaggle_environments import make as kaggle_make
    _HAS_KAGGLE_ENV = True
except ImportError:
    _HAS_KAGGLE_ENV = False


def is_kagg_available() -> bool:
    """Check if the high-performance Rust engine is compiled and available."""
    if not _HAS_KAGGSIM:
        return False
    try:
        find_kagg()
        return True
    except FileNotFoundError:
        return False


def _pass_bot(obs: Any) -> Dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _resolve_agent(agent: Any, module_suffix: str = "") -> Callable:
    """Resolve an agent parameter (callable, submission name, path, or 'pass')."""
    if callable(agent):
        return agent
    if agent == "pass":
        return _pass_bot
    if agent == "random":
        if _HAS_KAGGSIM:
            from kaggsim.policies import make_policy
            return make_policy("random", 0)
        return _pass_bot

    path = Path(agent)
    if not path.is_file():
        # Check submissions directory
        cand = ROOT / "submissions" / agent / "main.py"
        if cand.is_file():
            path = cand

    if not path.is_file():
        raise FileNotFoundError(f"Cannot resolve agent: {agent}")

    if _HAS_KAGGSIM:
        return load_agent(str(path))
    else:
        from agent_utils import load_agent as py_load
        return py_load(str(path), module_suffix=module_suffix)


class FastSimulation:
    """Context-managed persistent Rust simulation server."""

    def __init__(self, kagg_bin: Optional[str] = None):
        if not is_kagg_available():
            raise RuntimeError(
                "kagg binary not available. Build it with: "
                "cd kaggriculture-simulation && cargo build --manifest-path src-rust/Cargo.toml --release"
            )
        self.srv = Serve(kagg_bin)

    def __enter__(self) -> "FastSimulation":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.srv:
            self.srv.close()

    def run_match(
        self,
        agent0: Any,
        agent1: Any,
        seed: int,
        record: bool = False,
        on_step: Optional[Callable[[Dict[str, Any], Tuple[Any, Any], Dict[str, Any]], None]] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Run a head-to-head match between two agents on a specific seed.
        
        Returns:
            (money_player0, money_player1, final_state_dict)
        """
        fn0 = _resolve_agent(agent0, f"m_{seed}_0")
        fn1 = _resolve_agent(agent1, f"m_{seed}_1")

        js = self.srv.reset(seed)
        trace: List[Dict[str, Any]] = []

        while js["step"] < FINAL_STEP:
            a0 = call_agent(fn0, obs_for(js, 0))
            a1 = call_agent(fn1, obs_for(js, 1))
            pre = js
            js = self.srv.step2(a0, a1)

            if on_step:
                on_step(pre, (a0, a1), js)

            if record:
                trace.append({
                    "step": js["step"],
                    "actions": [a0, a1],
                    "state": js,
                })

        b0 = float(js["farms"][0]["money"])
        b1 = float(js["farms"][1]["money"])
        if record:
            js["_trace"] = trace
        return b0, b1, js

    def run_replay(
        self,
        agent: Any,
        replay_data_or_path: Union[str, Path, Dict[str, Any]],
        our_player: int = 0,
        on_step: Optional[Callable[[int, Dict[str, Any], Dict[str, Any]], None]] = None,
    ) -> Tuple[float, float, bool, Dict[str, Any]]:
        """Run an agent against an opponent's recorded actions from a replay.
        
        Returns:
            (our_money, opp_money, won, final_state)
        """
        if isinstance(replay_data_or_path, (str, Path)):
            rep = load_replay(str(replay_data_or_path))
        else:
            rep = replay_data_or_path

        seed = replay_seed(rep)
        opp_player = 1 - our_player
        opp_actions = replay_actions(rep, opp_player)

        agent_fn = _resolve_agent(agent, f"rep_{seed}_{our_player}")

        def opp_bot(obs: Any) -> Dict[str, Any]:
            step = obs.step
            if step < len(opp_actions):
                act = opp_actions[step]
                if act:
                    return act
            return {"farmer": ["PASS"], "hands": [], "market": []}

        js = self.srv.reset(seed)
        while js["step"] < FINAL_STEP:
            obs_our = obs_for(js, our_player)
            obs_opp = obs_for(js, opp_player)

            a_our = call_agent(agent_fn, obs_our)
            a_opp = opp_bot(obs_opp)

            js = self.srv.step2(a_our, a_opp) if our_player == 0 else self.srv.step2(a_opp, a_our)

            if on_step:
                on_step(js["step"], js, obs_our)

        our_money = float(js["farms"][our_player]["money"])
        opp_money = float(js["farms"][opp_player]["money"])
        won = our_money > opp_money
        return our_money, opp_money, won, js


def run_official_match(
    agent0: Any,
    agent1: Any,
    seed: int,
    steps: int = 720,
) -> Tuple[float, float, Any]:
    """Fallback runner using official kaggle_environments engine."""
    if not _HAS_KAGGLE_ENV:
        raise RuntimeError("kaggle_environments is not installed")

    env = kaggle_make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    fn0 = _resolve_agent(agent0, f"off_{seed}_0")
    fn1 = _resolve_agent(agent1, f"off_{seed}_1")

    res = env.run([fn0, fn1])
    b0 = float(res[-1][0].observation.farms[0]["money"])
    b1 = float(res[-1][1].observation.farms[1]["money"])
    return b0, b1, res
