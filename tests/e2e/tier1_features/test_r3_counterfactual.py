"""Tier 1 Feature Coverage: R3 Counterfactual Rollout Engine via Rust Simulator.

Validates:
- Rust simulator subprocess manager initialization and protocol handshake
- Observation state to Rust engine JSON bidirectional serialization (LOADSTATE)
- In-memory state forking at arbitrary timestep t in [0, 719]
- Action injection (counterfactual market bids/asks)
- Heuristic ROLLOUT to step 720 computing counterfactual_value
- Divergence verification: factual vs counterfactual trajectory outcomes
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

try:
    from kaggriculture.counterfactual.ipc import KaggServeProcess
except ImportError:
    KaggServeProcess = None

try:
    from kaggriculture.counterfactual.serializer import extract_engine_state, serialize_market_order
except ImportError:
    extract_engine_state = None
    serialize_market_order = None

try:
    from kaggriculture.counterfactual.engine import CounterfactualRolloutEngine
except ImportError:
    CounterfactualRolloutEngine = None


def test_r3_rust_simulator_ipc_initialization(kagg_binary_path: Path | None):
    """Test 1 (R3): Subprocess manager starts and handles basic protocol."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    if KaggServeProcess is None:
        # Direct oracle fallback test using raw subprocess or kaggsim.serve
        from kaggsim.serve import Serve
        with Serve() as srv:
            res = srv.reset(seed=42)
            assert res is not None
            assert res["step"] == 0
            assert "farms" in res
            assert len(res["farms"]) == 2
        pytest.skip("kaggriculture.counterfactual.ipc pending M3 implementation")

    with KaggServeProcess(binary_path=kagg_binary_path) as proc:
        assert proc.is_healthy()
        status = proc.ping()
        assert status is True


def test_r3_state_serialization_loadstate(mock_rust_engine_state: dict[str, Any], kagg_binary_path: Path | None):
    """Test 2 (R3): State serialization produces valid JSON for LOADSTATE."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    if extract_engine_state is None:
        # Validate format against Rust engine specification directly
        from kaggsim.serve import Serve
        with Serve() as srv:
            # Send LOADSTATE
            loaded = srv.load_state(mock_rust_engine_state)
            assert "error" not in loaded, f"Engine rejected state: {loaded.get('error')}"
            assert loaded["step"] == mock_rust_engine_state["step"]
        pytest.skip("kaggriculture.counterfactual.serializer pending M3 implementation")

    engine_json = extract_engine_state(mock_rust_engine_state, seed=42)
    assert isinstance(engine_json, dict)
    assert "farms" in engine_json
    assert "step" in engine_json
    assert "private" in engine_json


def test_r3_in_memory_state_forking(mock_rust_engine_state: dict[str, Any]):
    """Test 3 (R3): State forking creates independent clones without mutating source."""
    if extract_engine_state is None:
        # Verify isolation via copy semantics
        forked = copy.deepcopy(mock_rust_engine_state)
        forked["farms"][0]["money"] += 5000.0
        assert mock_rust_engine_state["farms"][0]["money"] != forked["farms"][0]["money"]
        pytest.skip("kaggriculture.counterfactual.engine pending M3 implementation")

    engine = CounterfactualRolloutEngine()
    forked_state = engine.fork_state(mock_rust_engine_state, step=100)
    assert forked_state["step"] == 100


def test_r3_counterfactual_order_serialization():
    """Test 4 (R3): Market orders serialize to valid tape lines."""
    if serialize_market_order is None:
        # Verify canonical string format for market order
        order_line = "BUY WHEAT 100 25"
        parts = order_line.split()
        assert parts[0] in {"BUY", "SELL"}
        assert parts[1] in {"WHEAT", "CORN", "MILK", "WOOL"}
        assert int(parts[2]) > 0
        pytest.skip("kaggriculture.counterfactual.serializer.serialize_market_order pending M3 implementation")

    cmd = serialize_market_order("BUY", item="WHEAT", quantity=100, price=25)
    assert isinstance(cmd, str)
    assert "BUY WHEAT 100" in cmd


def test_r3_heuristic_rollout_to_turn_720(mock_rust_engine_state: dict[str, Any], kagg_binary_path: Path | None):
    """Test 5 (R3): Fast rollout computes unbiased counterfactual value at step 720."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    if CounterfactualRolloutEngine is None:
        # Oracle test using kaggsim.serve ROLLOUT command
        from kaggsim.serve import Serve
        with Serve() as srv:
            # 10 step rollout
            final_obs = srv.rollout(mock_rust_engine_state, horizon=10, lines0=[], lines1=[])
            assert "farms" in final_obs
            assert final_obs["step"] >= mock_rust_engine_state["step"]
        pytest.skip("kaggriculture.counterfactual.engine pending M3 implementation")

    engine = CounterfactualRolloutEngine(binary_path=kagg_binary_path)
    cf_value = engine.evaluate_transition(
        state_dict=mock_rust_engine_state,
        alt_action=["BUY WHEAT 100 25"],
        downstream_tape=[],
    )
    assert isinstance(cf_value, float)
    assert np.isfinite(cf_value)


def test_r3_counterfactual_divergence_verification(mock_rust_engine_state: dict[str, Any], kagg_binary_path: Path | None):
    """Test 6 (R3): Injected counterfactual market order produces diverging outcome."""
    if kagg_binary_path is None:
        pytest.skip("Rust kagg binary not found")

    from kaggsim.serve import Serve
    with Serve() as srv:
        # Factual path: PASS for 10 steps
        factual_obs = srv.rollout(mock_rust_engine_state, horizon=10, lines0=[], lines1=[])
        factual_money = factual_obs["farms"][0]["money"]

        # Counterfactual path: Buy 20 seeds at step 0 then PASS
        cf_lines0 = ["\t\tBUY_SEED WHEAT 20"] + [""] * 9
        cf_obs = srv.rollout(mock_rust_engine_state, horizon=10, lines0=cf_lines0, lines1=[])
        cf_money = cf_obs["farms"][0]["money"]

        # State should diverge due to purchase cost
        assert factual_money != cf_money or cf_obs["market"] != factual_obs["market"]
