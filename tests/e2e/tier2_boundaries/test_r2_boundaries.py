"""Tier 2 Boundary and Corner Case Tests: R2 Feature Transformation Pipeline.

Validates:
- Farm bankruptcy boundary (zero or negative cash; no NaN/Inf in continuous features)
- Extreme market inventory oversupply (100k units) and drought (0 units) without division-by-zero
- Initial timestep boundary t = 0
- Terminal timestep boundary t = 719 / 720 (return-to-go convergence)
- Unmapped or malformed mechanical actions mapping safely to PASS/default intent
- Degenerate / zero-variance opponent trajectory clustering
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pytest

try:
    from kaggriculture.features.global_state import extract_global_vector
except ImportError:
    extract_global_vector = None

try:
    from kaggriculture.features.market import extract_market_vector
except ImportError:
    extract_market_vector = None

try:
    from kaggriculture.features.macro_intents import classify_macro_intent
except ImportError:
    classify_macro_intent = None

try:
    from kaggriculture.features.returns import compute_return_to_go
except ImportError:
    compute_return_to_go = None

try:
    from kaggriculture.features.clustering import cluster_opponent_trajectory
except ImportError:
    cluster_opponent_trajectory = None


def test_r2_bankruptcy_zero_cash_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 1 (R2 Boundary): Zero or negative farm cash produces finite features (no NaN/Inf)."""
    obs = synthetic_edge_case_observations["bankrupt"]
    obs["farms"][0]["money"] = 0.0

    if extract_global_vector is None:
        pytest.skip("kaggriculture.features.global_state pending M2 implementation")

    vec = extract_global_vector(obs, seat=0)
    assert not np.isnan(vec).any(), "NaN found in global vector with 0 cash"
    assert not np.isinf(vec).any(), "Inf found in global vector with 0 cash (log(0) unprotected)"


def test_r2_extreme_market_supply_and_drought_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 2 (R2 Boundary): 100k oversupply and 0-unit drought produce finite vectors."""
    obs_extreme = synthetic_edge_case_observations["extreme_market"]
    obs_drought = synthetic_edge_case_observations["drought_market"]

    if extract_market_vector is None:
        pytest.skip("kaggriculture.features.market pending M2 implementation")

    vec_ext = extract_market_vector(obs_extreme["market"])
    vec_drg = extract_market_vector(obs_drought["market"])

    assert np.all(np.isfinite(vec_ext)), "Extreme supply produced non-finite values"
    assert np.all(np.isfinite(vec_drg)), "Market drought produced non-finite values"


def test_r2_timestep_t0_initial_boundary(sample_raw_observation: dict[str, Any]):
    """Test 3 (R2 Boundary): Step 0 produces valid initial state features."""
    obs0 = copy.deepcopy(sample_raw_observation)
    obs0["step"] = 0
    obs0["day"] = 0
    obs0["hour"] = 0

    if extract_global_vector is None:
        pytest.skip("kaggriculture.features.global_state pending M2 implementation")

    vec = extract_global_vector(obs0, seat=0)
    assert len(vec) == 30
    assert np.all(np.isfinite(vec))


def test_r2_terminal_timestep_t719_return_to_go_boundary():
    """Test 4 (R2 Boundary): Return-to-go at t=719 strictly equals the terminal reward."""
    if compute_return_to_go is None:
        pytest.skip("kaggriculture.features.returns.compute_return_to_go pending M2 implementation")

    # Final step only
    single_reward = np.array([42.0], dtype=np.float32)
    rtg = compute_return_to_go(single_reward)
    assert abs(rtg[0] - 42.0) < 1e-4

    # 720 steps with non-zero terminal reward
    rewards = np.zeros(720, dtype=np.float32)
    rewards[-1] = 1000.0  # Big terminal score
    full_rtg = compute_return_to_go(rewards)
    assert abs(full_rtg[-1] - 1000.0) < 1e-4
    assert abs(full_rtg[0] - 1000.0) < 1e-4


def test_r2_unmapped_malformed_action_intent_boundary():
    """Test 5 (R2 Boundary): Invalid/malformed action dictionaries map cleanly to fallback without crash."""
    if classify_macro_intent is None:
        pytest.skip("kaggriculture.features.macro_intents pending M2 implementation")

    malformed_actions = [
        {},
        {"invalid_key": 123},
        {"farmer": ["UNKNOWN_VERB_123"], "hands": [], "market": []},
        {"farmer": [None], "hands": [[None]], "market": ["INVALID ORDER FORMAT"]},
    ]

    for action in malformed_actions:
        intent = classify_macro_intent(action)
        assert isinstance(intent, int)
        assert 0 <= intent < 24, f"Intent {intent} for action {action} out of bounds"


def test_r2_degenerate_opponent_trajectory_clustering_boundary():
    """Test 6 (R2 Boundary): Zero-variance / identical opponent moves cluster without numerical error."""
    if cluster_opponent_trajectory is None:
        pytest.skip("kaggriculture.features.clustering pending M2 implementation")

    all_zero_traj = np.zeros((100, 25), dtype=np.float32)
    cluster_id = cluster_opponent_trajectory(all_zero_traj)
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0
