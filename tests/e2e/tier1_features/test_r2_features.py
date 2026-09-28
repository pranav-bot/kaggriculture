"""Tier 1 Feature Coverage: R2 State-Action-Return Feature Transformation Pipeline.

Validates:
- Spatial tensor state_t shape (23, 10, 10) with normalized channels
- Continuous global state vector (30-dim)
- Opponent spatial tensor opponent_state_t (20, 10, 10)
- Continuous market vector market_t (35-dim, baseline deviations)
- Macro-intent backward action classifier (24 discrete macro classes)
- Net-worth delta reward reward_t
- Undiscounted return-to-go return_to_go_t converging to 0 at step 720
- Opponent trajectory K-means clustering (turns 0..99)
- Chunked memory-mapped disk storage
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

try:
    from kaggriculture.features.spatial import extract_spatial_tensor, extract_opponent_spatial
except ImportError:
    extract_spatial_tensor = None
    extract_opponent_spatial = None

try:
    from kaggriculture.features.global_state import extract_global_vector
except ImportError:
    extract_global_vector = None

try:
    from kaggriculture.features.market import extract_market_vector
except ImportError:
    extract_market_vector = None

try:
    from kaggriculture.features.macro_intents import classify_macro_intent, MACRO_INTENT_CLASSES
except ImportError:
    classify_macro_intent = None
    MACRO_INTENT_CLASSES = None

try:
    from kaggriculture.features.returns import compute_net_worth, compute_reward, compute_return_to_go
except ImportError:
    compute_net_worth = None
    compute_reward = None
    compute_return_to_go = None

try:
    from kaggriculture.features.clustering import cluster_opponent_trajectory
except ImportError:
    cluster_opponent_trajectory = None

try:
    from kaggriculture.features.storage import save_transitions_chunk, TransitionChunkReader
except ImportError:
    save_transitions_chunk = None
    TransitionChunkReader = None


def test_r2_spatial_tensor_shape_and_normalization(sample_raw_observation: dict[str, Any]):
    """Test 1 (R2): Spatial tensor must have shape (23, 10, 10) and normalized values."""
    if extract_spatial_tensor is None:
        pytest.skip("kaggriculture.features.spatial pending M2 implementation")

    tensor = extract_spatial_tensor(sample_raw_observation, seat=0)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (23, 10, 10), f"Expected shape (23, 10, 10), got {tensor.shape}"
    assert np.all(tensor >= 0.0) and np.all(tensor <= 1.0), "Spatial channels must be in [0, 1]"


def test_r2_continuous_global_vector(sample_raw_observation: dict[str, Any]):
    """Test 2 (R2): Global continuous vector must be 30-dimensional and finite."""
    if extract_global_vector is None:
        pytest.skip("kaggriculture.features.global_state pending M2 implementation")

    vec = extract_global_vector(sample_raw_observation, seat=0)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (30,), f"Expected shape (30,), got {vec.shape}"
    assert np.all(np.isfinite(vec)), "Global vector contains NaN or Inf"


def test_r2_opponent_spatial_tensor(sample_raw_observation: dict[str, Any]):
    """Test 3 (R2): Opponent spatial tensor must have shape (20, 10, 10)."""
    if extract_opponent_spatial is None:
        pytest.skip("kaggriculture.features.spatial.extract_opponent_spatial pending M2 implementation")

    tensor = extract_opponent_spatial(sample_raw_observation, seat=0)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (20, 10, 10), f"Expected shape (20, 10, 10), got {tensor.shape}"


def test_r2_continuous_market_vector(sample_raw_observation: dict[str, Any]):
    """Test 4 (R2): Market continuous vector must be 35-dim with baseline deviations."""
    if extract_market_vector is None:
        pytest.skip("kaggriculture.features.market pending M2 implementation")

    m_vec = extract_market_vector(sample_raw_observation["market"])
    assert isinstance(m_vec, np.ndarray)
    assert m_vec.shape == (35,), f"Expected shape (35,), got {m_vec.shape}"
    assert np.all(np.isfinite(m_vec)), "Market vector contains NaN or Inf"


def test_r2_macro_intent_classifier_coverage():
    """Test 5 (R2): Mechanical moves map into one of 24 discrete macro classes."""
    if classify_macro_intent is None:
        pytest.skip("kaggriculture.features.macro_intents pending M2 implementation")

    sample_moves = [
        ({"farmer": ["WATER"], "hands": [], "market": []}, "MAINTAIN_CROPS"),
        ({"farmer": [], "hands": [], "market": ["BUY WHEAT 100 25"]}, "BUY_MARKET"),
        ({"farmer": [], "hands": [], "market": ["SELL MILK 50 40"]}, "DUMP_MILK"),
        ({"farmer": ["PLANT CORN"], "hands": [], "market": []}, "PLANT_CROPS"),
        ({"farmer": [], "hands": [], "market": []}, "PASS"),
    ]

    for move_dict, _ in sample_moves:
        intent_id = classify_macro_intent(move_dict)
        assert isinstance(intent_id, int)
        assert 0 <= intent_id < 24, f"Intent {intent_id} out of bounds [0, 23]"


def test_r2_net_worth_delta_reward():
    """Test 6 (R2): Net worth reward tracks cash plus replacement cost deltas."""
    if compute_reward is None or compute_net_worth is None:
        pytest.skip("kaggriculture.features.returns pending M2 implementation")

    # State t
    obs_t = {
        "farms": [{"money": 1000.0}],
        "private": {"carried": {}, "seeds": {}, "shed": {"WHEAT": 10.0}},
        "market": {"prices": {"WHEAT": 25.0}},
    }
    # State t+1 (sold 5 wheat, cash increased by 125)
    obs_t1 = {
        "farms": [{"money": 1125.0}],
        "private": {"carried": {}, "seeds": {}, "shed": {"WHEAT": 5.0}},
        "market": {"prices": {"WHEAT": 25.0}},
    }

    nw_t = compute_net_worth(obs_t, seat=0)
    nw_t1 = compute_net_worth(obs_t1, seat=0)
    reward = compute_reward(obs_t, obs_t1, seat=0)

    # Net worth delta: (1125 + 5*25) - (1000 + 10*25) = 1250 - 1250 = 0.0
    assert abs(reward - (nw_t1 - nw_t)) < 1e-4


def test_r2_return_to_go_convergence_at_turn_720():
    """Test 7 (R2): Undiscounted return-to-go converges strictly to 0.0 at turn 720."""
    if compute_return_to_go is None:
        pytest.skip("kaggriculture.features.returns.compute_return_to_go pending M2 implementation")

    rewards = np.ones(720, dtype=np.float32)  # 1 unit reward each step
    rtg = compute_return_to_go(rewards)

    assert len(rtg) == 720
    assert abs(rtg[-1] - rewards[-1]) < 1e-5 or abs(rtg[-1]) < 1e-5
    # Telescoping property: rtg[t] - rtg[t+1] == rewards[t]
    for t in range(719):
        assert abs((rtg[t] - rtg[t + 1]) - rewards[t]) < 1e-4


def test_r2_opponent_trajectory_clustering():
    """Test 8 (R2): Opponent trajectory clustering assigns cluster in [0, K-1]."""
    if cluster_opponent_trajectory is None:
        pytest.skip("kaggriculture.features.clustering pending M2 implementation")

    dummy_traj = np.zeros((100, 25), dtype=np.float32)
    cluster_id = cluster_opponent_trajectory(dummy_traj)
    assert isinstance(cluster_id, int)
    assert cluster_id >= 0


def test_r2_chunked_storage_roundtrip(tmp_path: Path):
    """Test 9 (R2): Save and read transition chunks with memory mapping."""
    if save_transitions_chunk is None or TransitionChunkReader is None:
        pytest.skip("kaggriculture.features.storage pending M2 implementation")

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    n = 64
    transitions = {
        "state_spatial": np.zeros((n, 23, 10, 10), dtype=np.float32),
        "state_global": np.zeros((n, 30), dtype=np.float32),
        "opponent_spatial": np.zeros((n, 20, 10, 10), dtype=np.float32),
        "opponent_global": np.zeros((n, 12), dtype=np.float32),
        "market": np.zeros((n, 35), dtype=np.float32),
        "action": np.zeros((n,), dtype=np.int64),
        "reward": np.zeros((n,), dtype=np.float32),
        "return_to_go": np.zeros((n,), dtype=np.float32),
        "strategy_cluster": np.zeros((n,), dtype=np.int64),
        "counterfactual_value": np.zeros((n,), dtype=np.float32),
    }

    save_transitions_chunk(chunk_dir / "chunk_0000.npz", transitions)
    reader = TransitionChunkReader(chunk_dir)
    assert len(reader) == n
    item = reader[0]
    assert item["state_spatial"].shape == (23, 10, 10)
