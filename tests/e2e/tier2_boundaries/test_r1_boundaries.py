"""Tier 2 Boundary and Corner Case Tests: R1 EDA & Schema Mapping.

Validates:
- All-null tiles board parsing (None for all 10x10 tiles)
- All-locked quadrants board parsing ("LOCKED" string in all cells)
- Completely empty shed and private inventory structures
- Shed overflow (>100 capacity) boundary conditions
- Zero and negative overage time tolerances
- Corrupted / truncated gzip archive handling with clean error propagation
- Variable farmhand counts (0 to N hands)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    from kaggriculture.eda.parser import parse_episode
except ImportError:
    parse_episode = None

try:
    from kaggriculture.eda.schema import validate_observation, validate_tile
except ImportError:
    validate_observation = None
    validate_tile = None


def test_r1_all_null_tiles_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 1 (R1 Boundary): Observation with all None tiles parses without exception."""
    obs = synthetic_edge_case_observations["all_null_tiles"]
    if validate_observation is None:
        # Direct assertion: all 10x10 cells are None and grid shape is preserved
        farm = obs["farms"][0]
        assert len(farm["tiles"]) == 10
        assert all(cell is None for row in farm["tiles"] for cell in row)
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    assert validate_observation(obs) is True


def test_r1_all_locked_tiles_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 2 (R1 Boundary): Observation with all 'LOCKED' tiles parses without exception."""
    obs = synthetic_edge_case_observations["all_locked_tiles"]
    if validate_observation is None:
        farm = obs["farms"][0]
        assert len(farm["tiles"]) == 10
        assert all(cell == "LOCKED" for row in farm["tiles"] for cell in row)
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    assert validate_observation(obs) is True


def test_r1_empty_private_inventory_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 3 (R1 Boundary): Completely empty shed and carried inventories parse safely."""
    obs = synthetic_edge_case_observations["empty_inventory"]
    if validate_observation is None:
        priv = obs["private"]
        assert priv["shed"] == {}
        assert priv["carried"] == {} or priv["inventories"] == []
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    assert validate_observation(obs) is True


def test_r1_shed_overflow_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 4 (R1 Boundary): Shed holding >100 units is flagged or handled gracefully."""
    obs = synthetic_edge_case_observations["shed_overflow"]
    shed = obs["private"]["shed"]
    total_shed = sum(shed.values())
    assert total_shed > 100.0, "Expected shed units to exceed 100 in boundary test"

    if validate_observation is None:
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    # The schema validator should either accept the state dictionary while flagging overflow,
    # or handle it without unhandled TypeError/IndexError
    try:
        res = validate_observation(obs)
        assert isinstance(res, bool)
    except ValueError:
        pass  # Raising specific ValueError for invalid capacity is acceptable


def test_r1_zero_overage_time_boundary(synthetic_edge_case_observations: dict[str, dict[str, Any]]):
    """Test 5 (R1 Boundary): remainingOverageTime = 0.0 handled without divide-by-zero or crash."""
    obs = synthetic_edge_case_observations["zero_overage"]
    assert obs["remainingOverageTime"] == 0.0

    if validate_observation is None:
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    assert validate_observation(obs) is True


def test_r1_corrupted_gzip_file_handling(tmp_path: Path):
    """Test 6 (R1 Boundary): Truncated or corrupt gzip file raises clean error, no crash."""
    corrupt_file = tmp_path / "corrupted_episode.json.gz"
    # Write corrupt gzip magic and partial bytes
    with open(corrupt_file, "wb") as f:
        f.write(b"\x1f\x8b\x08\x00truncateddatahere")

    if parse_episode is None:
        import zlib
        with pytest.raises((gzip.BadGzipFile, OSError, EOFError, zlib.error)):
            with gzip.open(corrupt_file, "rt") as f:
                f.read()
        pytest.skip("kaggriculture.eda.parser pending M1 implementation")

    import zlib
    with pytest.raises((gzip.BadGzipFile, OSError, ValueError, zlib.error)):
        parse_episode(corrupt_file)


def test_r1_variable_farmhands_count_boundary(sample_raw_observation: dict[str, Any]):
    """Test 7 (R1 Boundary): Farm with 0 hands vs multiple hands parses cleanly."""
    obs_0_hands = json.loads(json.dumps(sample_raw_observation))
    obs_0_hands["farms"][0]["hands"] = []

    obs_5_hands = json.loads(json.dumps(sample_raw_observation))
    obs_5_hands["farms"][0]["hands"] = [[4, 4], [4, 5], [5, 4], [5, 5], [6, 6]]

    if validate_observation is None:
        assert len(obs_0_hands["farms"][0]["hands"]) == 0
        assert len(obs_5_hands["farms"][0]["hands"]) == 5
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    assert validate_observation(obs_0_hands) is True
    assert validate_observation(obs_5_hands) is True
