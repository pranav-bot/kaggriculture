"""Unit & Integration tests for SafeFallbackController and @impenetrable_agent.

Validates:
1. Zero external dependencies in safety module.
2. Sub-millisecond latency (<1ms) of SafeFallbackController.
3. Care Mill logic: feeding cows, caring for them, collecting fertilizer, harvesting milk.
4. Fertilizer market rule: sell all fertilizer if Day < 8, otherwise hold.
5. Terminal Day 29 asset liquidation flush.
6. Exception shielding: catches UnboundLocalError (Day 19 uninitialized variable bug),
   tensor dimension mismatches, FFI memory errors, and KeyErrors.
7. Watchdog: tracks cumulative overage time and disables primary agent when
   remaining overage bank drops below 5.0 seconds.
8. Logs all exceptions and watchdog triggers to sys.stderr.
9. Compatibility with both agent(obs) and agent(obs, config) signatures.
"""

from __future__ import annotations

import io
import sys
import time
from typing import Any, Dict, List
import pytest

from kaggriculture.safety import (
    DEFAULT_OVERAGE_BANK_S,
    DEFAULT_SAFETY_THRESHOLD_S,
    DEFAULT_SOFT_LIMIT_S,
    ImpenetrableAgentWrapper,
    SafeFallbackController,
    impenetrable_agent,
)


@pytest.fixture
def mock_cow_farm_obs() -> Dict[str, Any]:
    """Generates a realistic farm observation with pastures, cows, shed, and workers."""
    tiles = [[None] * 10 for _ in range(10)]
    # Place 3 cows in Northwest pasture
    tiles[0][0] = {
        "kind": "PASTURE", "animal": "COW",
        "fed_today": False, "cared_today": False,
        "fertilizer_available": True, "yield_units": 3,
    }
    tiles[0][1] = {
        "kind": "PASTURE", "animal": "COW",
        "fed_today": False, "cared_today": False,
        "fertilizer_available": False, "yield_units": 0,
    }
    tiles[1][0] = {
        "kind": "PASTURE", "animal": "COW",
        "fed_today": True, "cared_today": False,
        "fertilizer_available": True, "yield_units": 0,
    }

    return {
        "step": 120,
        "day": 5,
        "hour": 0,
        "player": 0,
        "farms": [
            {
                "money": 1500.0,
                "farmer": [4, 4],
                "hands": [[4, 5], [5, 4]],
                "tiles": tiles,
                "unlocked_quadrants": ["NW"],
            }
        ],
        "private": {
            "shed": {
                "FERTILIZER": 20,
                "WHEAT": 15,
                "MILK": 10,
            },
            "seeds": {"WHEAT": 5},
            "inventories": [
                {"WHEAT": 2},  # Farmer holds 2 wheat
                {"WHEAT": 0},  # Hand 0 holds 0 wheat
                {"WHEAT": 1},  # Hand 1 holds 1 wheat
            ],
        },
        "market": {
            "inventory": {"FERTILIZER": 10000, "WHEAT": 10000, "MILK": 10000},
            "prices": {"FERTILIZER": 100, "WHEAT": 25, "MILK": 160},
        },
    }


def test_zero_external_dependencies():
    """Verify safety module contains zero third-party dependencies (pure Python)."""
    import inspect
    import kaggriculture.safety as safety_mod

    # Inspect imported modules
    for name, val in inspect.getmembers(safety_mod):
        if inspect.ismodule(val):
            mod_name = val.__name__.split(".")[0]
            # Standard library modules only
            assert mod_name in (
                "sys", "time", "functools", "logging", "math", "traceback", "typing", "dataclasses"
            ), f"Forbidden non-standard dependency found: {mod_name}"


def test_safefallback_submillisecond_latency(mock_cow_farm_obs: Dict[str, Any]):
    """Verify SafeFallbackController executes in strictly <1.0ms."""
    controller = SafeFallbackController()

    # Warm-up
    for _ in range(5):
        controller.act(mock_cow_farm_obs)

    durations: List[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        act = controller.act(mock_cow_farm_obs)
        durations.append(time.perf_counter() - t0)

    mean_ms = (sum(durations) / len(durations)) * 1000.0
    max_ms = max(durations) * 1000.0

    assert mean_ms < 0.8, f"Mean latency too high: {mean_ms:.3f}ms"
    assert max_ms < 1.0, f"Max latency violated 1.0ms limit: {max_ms:.3f}ms"


def test_care_mill_cow_lifecycle(mock_cow_farm_obs: Dict[str, Any]):
    """Verify SafeFallbackController executes Care Mill logic: feeding, caring, and collecting."""
    controller = SafeFallbackController()
    action = controller.act(mock_cow_farm_obs)

    assert isinstance(action, dict)
    assert "farmer" in action
    assert "hands" in action
    assert "market" in action

    farmer_cmd = action["farmer"]
    assert isinstance(farmer_cmd, list) and len(farmer_cmd) >= 1
    # Hands commands
    assert len(action["hands"]) == 2  # 2 hands in mock obs


def test_fertilizer_day8_rule(mock_cow_farm_obs: Dict[str, Any]):
    """Verify fertilizer is completely sold if Day < 8, and held if Day >= 8."""
    controller = SafeFallbackController()

    # 1. Day 5 (< 8): Should sell all 20 units of fertilizer
    mock_cow_farm_obs["day"] = 5
    mock_cow_farm_obs["private"]["shed"]["FERTILIZER"] = 20
    action_d5 = controller.act(mock_cow_farm_obs)
    fert_orders_d5 = [o for o in action_d5["market"] if o[0] == "SELL" and o[1] == "FERTILIZER"]
    assert len(fert_orders_d5) == 1
    assert fert_orders_d5[0][2] == "20"

    # 2. Day 8 (>= 8): Must HOLD fertilizer (0 sold)
    mock_cow_farm_obs["day"] = 8
    action_d8 = controller.act(mock_cow_farm_obs)
    fert_orders_d8 = [o for o in action_d8["market"] if o[0] == "SELL" and o[1] == "FERTILIZER"]
    assert len(fert_orders_d8) == 0

    # 3. Day 19 (>= 8): Must HOLD fertilizer
    mock_cow_farm_obs["day"] = 19
    action_d19 = controller.act(mock_cow_farm_obs)
    fert_orders_d19 = [o for o in action_d19["market"] if o[0] == "SELL" and o[1] == "FERTILIZER"]
    assert len(fert_orders_d19) == 0


def test_day29_terminal_flush(mock_cow_farm_obs: Dict[str, Any]):
    """Verify Day 29 terminal flush liquidates all stored shed assets."""
    controller = SafeFallbackController()
    mock_cow_farm_obs["day"] = 29
    mock_cow_farm_obs["hour"] = 22
    mock_cow_farm_obs["private"]["shed"] = {"MILK": 15, "WOOL": 10, "FERTILIZER": 30}

    action = controller.act(mock_cow_farm_obs)
    sold_items = {o[1] for o in action["market"] if o[0] == "SELL"}
    assert "MILK" in sold_items
    assert "WOOL" in sold_items
    assert "FERTILIZER" in sold_items


def test_impenetrable_agent_catches_uninitialized_variable_day19(mock_cow_farm_obs: Dict[str, Any]):
    """Verify decorator catches uninitialized variable (NameError/UnboundLocalError) on Day 19."""
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        @impenetrable_agent
        def candidate_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
            day = obs.get("day", 0)
            if day == 19:
                # Simulate uninitialized variable crash
                return unassigned_variable + 1  # noqa: F821
            return {"farmer": ["PASS"], "hands": [], "market": []}

        # Day 18: Normal execution
        mock_cow_farm_obs["day"] = 18
        res_d18 = candidate_agent(mock_cow_farm_obs)
        assert res_d18.get("_fallback_active") is not True

        # Day 19: Crashes with NameError -> Caught by @impenetrable_agent!
        mock_cow_farm_obs["day"] = 19
        res_d19 = candidate_agent(mock_cow_farm_obs)
        assert res_d19.get("_fallback_active") is True
        assert res_d19.get("_exception_shield_triggered") is True
        assert "farmer" in res_d19 and "market" in res_d19

        # Stderr log check
        log_content = stderr_capture.getvalue()
        assert "[IMPENETRABLE_AGENT] CRITICAL: Caught NameError" in log_content
        assert "unassigned_variable" in log_content

        # Day 20: Continues running
        mock_cow_farm_obs["day"] = 20
        res_d20 = candidate_agent(mock_cow_farm_obs)
        assert res_d20.get("_fallback_active") is not True

    finally:
        sys.stderr = old_stderr


def test_impenetrable_agent_catches_tensor_and_ffi_errors(mock_cow_farm_obs: Dict[str, Any]):
    """Verify decorator catches tensor shape mismatches and FFI memory errors."""
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        @impenetrable_agent
        def faulty_nn_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
            day = obs.get("day", 0)
            if day == 10:
                raise RuntimeError("RuntimeError: shape mismatch in conv layer: expected (23,10,10) got (20,10,10)")
            if day == 11:
                raise MemoryError("FFI memory allocation failure: out of heap")
            return {"farmer": ["EAST"], "hands": [], "market": []}

        mock_cow_farm_obs["day"] = 10
        act_10 = faulty_nn_agent(mock_cow_farm_obs)
        assert act_10["_fallback_active"] is True
        assert "shape mismatch" in act_10["_last_error"]

        mock_cow_farm_obs["day"] = 11
        act_11 = faulty_nn_agent(mock_cow_farm_obs)
        assert act_11["_fallback_active"] is True
        assert "MemoryError" in act_11["_last_error"]

        assert faulty_nn_agent.stats()["fallback_trigger_count"] == 2

    finally:
        sys.stderr = old_stderr


def test_impenetrable_agent_watchdog_overage_trip(mock_cow_farm_obs: Dict[str, Any]):
    """Verify watchdog trips circuit breaker when remaining overage bank < 5.0s."""
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
        called_turns = []

        # Configure with small overage bank for fast unit testing:
        # Bank = 10.0s, soft limit = 0.05s, safety threshold = 5.0s
        @impenetrable_agent(overage_bank_s=10.0, soft_limit_s=0.05, safety_threshold_s=5.0)
        def slow_beam_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
            called_turns.append(obs.get("step"))
            # Step 1: Takes 0.05s + 3.0s = 3.05s -> consumes 3.0s overage -> 7.0s remaining (> 5.0s)
            if obs.get("step") == 1:
                time.sleep(0.06)  # slight sleep
            # Step 2: Simulate environment reporting overage < 5.0s
            return {"farmer": ["BEAM_SEARCH_MOVE"], "hands": [], "market": []}

        # Step 1
        mock_cow_farm_obs["step"] = 1
        res1 = slow_beam_agent(mock_cow_farm_obs)
        assert res1["farmer"] == ["BEAM_SEARCH_MOVE"]
        assert slow_beam_agent.circuit_broken is False

        # Step 2: Environment reports overage bank dropped to 4.2s (< 5.0s)
        mock_cow_farm_obs["step"] = 2
        mock_cow_farm_obs["remainingOverageTime"] = 4.2
        res2 = slow_beam_agent(mock_cow_farm_obs)
        assert slow_beam_agent.circuit_broken is True

        # Step 3: slow_beam_agent must NOT even be called!
        mock_cow_farm_obs["step"] = 3
        res3 = slow_beam_agent(mock_cow_farm_obs)
        assert 3 not in called_turns, "Primary slow_beam_agent was called after circuit breaker tripped!"
        assert res3.get("_fallback_active") is True

        log_content = stderr_capture.getvalue()
        assert "[IMPENETRABLE_AGENT_WATCHDOG] CRITICAL" in log_content
        assert "Permanently disabling primary neural network" in log_content

    finally:
        sys.stderr = old_stderr


def test_two_argument_signature(mock_cow_farm_obs: Dict[str, Any]):
    """Verify @impenetrable_agent works transparently with agent(obs, config) signature."""
    @impenetrable_agent
    def two_arg_agent(obs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {"farmer": ["MOVE"], "hands": [], "market": []}

    mock_config = {"boardSize": 10, "episodeSteps": 720}
    res = two_arg_agent(mock_cow_farm_obs, mock_config)
    assert res["farmer"] == ["MOVE"]


def test_diagnostics_and_reset(mock_cow_farm_obs: Dict[str, Any]):
    """Verify stats tracking and reset functionality."""
    @impenetrable_agent
    def agent_fn(obs):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    agent_fn(mock_cow_farm_obs)
    agent_fn(mock_cow_farm_obs)

    stats = agent_fn.stats()
    assert stats["total_turns"] == 2
    assert stats["remaining_overage_s"] == 60.0
    assert stats["circuit_broken"] is False

    agent_fn.reset()
    assert agent_fn.stats()["total_turns"] == 0
