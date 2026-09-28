"""Tier 2 Boundary and Corner Case Tests: R3 Counterfactual Rollout Engine.

Validates:
- Counterfactual fork at terminal step t = 719 (0-horizon rollout)
- Counterfactual fork at opening step t = 0 (full 720-step horizon)
- Malformed state JSON rejection (engine error responses without Python crash)
- Process termination / broken pipe recovery
- Extreme order quantities (0, negative, 1,000,000 units)
- High-frequency pipe communication stress (no deadlocks or buffer desync)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    from kaggriculture.counterfactual.ipc import KaggServeProcess
except ImportError:
    KaggServeProcess = None

try:
    from kaggriculture.counterfactual.engine import CounterfactualRolloutEngine
except ImportError:
    CounterfactualRolloutEngine = None


def test_r3_terminal_step_719_rollout_boundary(mock_rust_engine_state: dict[str, Any], kagg_binary_path: Path | None):
    """Test 1 (R3 Boundary): Forking at step 719 with horizon 0 returns terminal state immediately."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    st = copy.deepcopy(mock_rust_engine_state)
    st["step"] = 719
    st["day"] = 29
    st["hour"] = 23
    st["done"] = True

    from kaggsim.serve import Serve
    with Serve() as srv:
        res = srv.rollout(st, horizon=0, lines0=[], lines1=[])
        assert "farms" in res
        assert res["step"] == 719
        assert res.get("done") is True


def test_r3_opening_step_0_full_game_rollout_boundary(mock_rust_engine_state: dict[str, Any], kagg_binary_path: Path | None):
    """Test 2 (R3 Boundary): Full 720-step rollout from step 0 terminates at step 719."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    st = copy.deepcopy(mock_rust_engine_state)
    st["step"] = 0

    from kaggsim.serve import Serve
    with Serve() as srv:
        res = srv.rollout(st, horizon=720, lines0=[], lines1=[])
        assert res["step"] == 719
        assert res.get("done") is True


def test_r3_malformed_state_rejection_boundary(kagg_binary_path: Path | None):
    """Test 3 (R3 Boundary): Invalid grid size or missing farms is rejected with error JSON / ServeError."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    bad_state = {
        "step": 0,
        "farms": [{"money": 1000}],  # missing second farm and 10x10 tiles
    }

    from kaggsim.serve import Serve, ServeError
    with Serve() as srv:
        with pytest.raises(ServeError) as exc_info:
            srv.load_state(bad_state)
        assert "farms" in str(exc_info.value) or "state" in str(exc_info.value)


def test_r3_broken_pipe_and_process_death_handling(kagg_binary_path: Path | None):
    """Test 4 (R3 Boundary): Killing the underlying process triggers automatic respawn without hanging."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    from kaggsim.serve import Serve
    srv = Serve()
    try:
        srv.reset(1)
        old_pid = srv.proc.pid if srv.proc else None
        # Kill the child process directly to simulate abrupt crash / SIGKILL
        if srv.proc:
            srv.proc.kill()
            srv.proc.wait()

        # Subsequent command should auto-respawn and succeed
        res = srv.reset(2)
        assert res["step"] == 0
        assert srv.proc is not None
        assert srv.proc.pid != old_pid, "Process manager failed to auto-respawn dead child process"
    finally:
        srv.close()


def test_r3_extreme_order_quantities_boundary(mock_rust_engine_state: dict[str, Any], kagg_binary_path: Path | None):
    """Test 5 (R3 Boundary): Zero, negative, and extreme order amounts do not crash simulator."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    from kaggsim.serve import Serve
    with Serve() as srv:
        # Step with zero/negative orders
        orders = [
            "\t\tBUY WHEAT 0 25",
            "\t\tBUY WHEAT -50 25",
            "\t\tBUY WHEAT 10000000 25",
        ]
        res = srv.rollout(mock_rust_engine_state, horizon=3, lines0=orders, lines1=[])
        assert "farms" in res
        assert not res.get("error")


def test_r3_rapid_sequential_queries_stress_boundary(kagg_binary_path: Path | None):
    """Test 6 (R3 Boundary): 50 rapid reset/step requests without pipe desynchronization."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    from kaggsim.serve import Serve
    with Serve() as srv:
        for seed in range(50):
            res = srv.reset(seed)
            assert res["step"] == 0
            step_res = srv.step("")
            assert step_res["step"] == 1
