"""Unit and Integration Tests for Subtask 6: Counterfactual Market Data Generator.

Validates:
1. detect_large_sells identifies whenever a player sold >10 units of Milk or Wool
2. create_hold_action injects HOLD by suppressing target commodity sells
3. generate_counterfactuals initializes Rust env (kaggsim.env), executes rollout to 720, and computes counterfactual_value
4. Multiprocessing pipeline generates valid JSONL records across replay episodes
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from kaggriculture.counterfactual import (
    CounterfactualPipeline,
    CounterfactualRecord,
    create_hold_action,
    detect_large_sells,
    generate_counterfactuals,
    run_counterfactual_pipeline,
)
from kaggsim.env import Env


def test_detect_large_sells():
    """Requirement 1: Identify whenever player sold >10 Milk or Wool."""
    # Sells MILK > 10
    act1 = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 16]]}
    assert detect_large_sells(act1) == [(0, "MILK", 16)]

    # Sells WOOL > 10
    act2 = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WOOL", 12]]}
    assert detect_large_sells(act2) == [(0, "WOOL", 12)]

    # Sells MILK <= 10 -> Should NOT trigger
    act3 = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 10]]}
    assert detect_large_sells(act3) == []

    # Sells other commodity (e.g. WHEAT > 10) -> Should NOT trigger
    act4 = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 50]]}
    assert detect_large_sells(act4) == []

    # BUY orders -> Should NOT trigger
    act5 = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 20]]}
    assert detect_large_sells(act5) == []

    # Multiple orders with one qualifying sell
    act6 = {
        "farmer": ["EAST"],
        "hands": [["WATER"]],
        "market": [["BUY_PRODUCT", "WHEAT", 2], ["SELL", "MILK", 18], ["HIRE"]],
    }
    assert detect_large_sells(act6) == [(1, "MILK", 18)]


def test_create_hold_action():
    """Requirement 3: Inject counterfactual action HOLD (do not sell)."""
    orig_act = {
        "farmer": ["EAST"],
        "hands": [["WATER"]],
        "market": [["SELL", "MILK", 16], ["BUY_SEED", "WHEAT", 4]],
    }
    hold_act = create_hold_action(orig_act, target_commodity="MILK")

    assert hold_act["farmer"] == ["EAST"]
    assert hold_act["hands"] == [["WATER"]]
    # The SELL MILK 16 order must be gone; BUY_SEED must remain
    assert hold_act["market"] == [["BUY_SEED", "WHEAT", 4]]


def test_generate_counterfactuals_execution(mock_rust_engine_state: Dict[str, Any]):
    """Requirements 1-5: generate_counterfactuals end-to-end execution."""
    st = copy.deepcopy(mock_rust_engine_state)
    st["step"] = 700  # near terminal for fast test execution
    st["day"] = 29
    st["hour"] = 4
    st["market"]["prices"]["MILK"] = 160.0
    st["private"][0]["shed"]["MILK"] = 30

    action_sell = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "MILK", 20]],
    }

    # When no large sell occurs, returns empty list
    no_sell_act = {"farmer": ["PASS"], "hands": [], "market": []}
    empty_cfs = generate_counterfactuals(st, no_sell_act)
    assert empty_cfs == []

    # When large sell occurs, evaluates counterfactual with kaggsim.env
    cfs = generate_counterfactuals(
        st,
        action_sell,
        historical_terminal_cash=50000.0,
        seed=123,
    )

    assert len(cfs) == 1
    cf = cfs[0]
    assert isinstance(cf, CounterfactualRecord)
    assert cf["commodity"] == "MILK"
    assert cf["historical_sell_quantity"] == 20
    assert cf["counterfactual_action"] == "HOLD"
    assert "synthetic_terminal_cash" in cf
    assert "historical_terminal_cash" in cf
    assert cf["historical_terminal_cash"] == 50000.0
    assert cf["counterfactual_value"] == cf["synthetic_terminal_cash"] - 50000.0
    # Check attribute access
    assert cf.counterfactual_value == cf["counterfactual_value"]


def test_counterfactual_pipeline_multiprocessing(tmp_path: Path):
    """Requirement 6: Multiprocessing pipeline generating JSONL dataset."""
    import glob

    episodes = sorted(glob.glob("datasets/il/episodes/*/*.json.gz"))
    if not episodes:
        pytest.skip("No replay episodes found in datasets/il/episodes/")

    # Select 2 episodes to process
    sample_episodes = episodes[:2]
    out_jsonl = tmp_path / "test_cfs.jsonl"

    stats = run_counterfactual_pipeline(
        episodes=sample_episodes,
        output_file=out_jsonl,
        num_workers=2,
    )

    assert stats["episodes_processed"] == 2
    assert out_jsonl.exists()

    # Read output JSONL lines
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    if lines:
        first_tuple = json.loads(lines[0])
        assert "step" in first_tuple
        assert "commodity" in first_tuple
        assert first_tuple["commodity"] in ("MILK", "WOOL")
        assert "counterfactual_value" in first_tuple
        assert "synthetic_terminal_cash" in first_tuple
        assert "historical_terminal_cash" in first_tuple
        assert first_tuple["counterfactual_action"] == "HOLD"
