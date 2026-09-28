"""Tier 1 Feature Coverage: R1 Exploratory Data Analysis (EDA) & Schema Mapping.

Validates:
- Raw JSON gzip episode streaming & step frame extraction
- Official Kaggle environment 1.32.7 observation schema mapping
- Tile state representations (None, LOCKED, WEED, PASTURE, PLANT)
- Private inventory sections and shed capacity constraints
- Automated EDA report metrics computation
- Temporal action alignment: action applied at state t is recorded in steps[t+1]
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any, Dict

import pytest

# Attempt imports from target implementation module (M1)
try:
    from kaggriculture.eda.parser import extract_frame, parse_episode
except ImportError:
    parse_episode = None
    extract_frame = None

try:
    from kaggriculture.eda.schema import validate_observation, validate_tile
except ImportError:
    validate_observation = None
    validate_tile = None

try:
    from kaggriculture.eda.report import generate_eda_report
except ImportError:
    generate_eda_report = None


# Reference schema definition for kaggle_environments 1.32.7
REQUIRED_OBS_KEYS = {
    "day",
    "hour",
    "step",
    "player",
    "remainingOverageTime",
    "farms",
    "market",
    "town",
    "private",
}

REQUIRED_FARM_KEYS = {
    "farmer",
    "hands",
    "hires_today",
    "money",
    "tiles",
    "unlocked_quadrants",
}


def test_r1_raw_json_step_frame_parsing(sample_episode_path: Path):
    """Test 1 (R1): Verify parsing of gzip episode replay files."""
    if parse_episode is None:
        # Fallback validation to verify oracle file integrity while M1 is under construction
        with gzip.open(sample_episode_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        assert "steps" in data, "Episode data missing 'steps' key"
        assert len(data["steps"]) >= 2, "Episode must have at least 2 steps"
        assert data["module_version"] == "1.32.7", "Expected engine version 1.32.7"
        pytest.skip("kaggriculture.eda.parser pending M1 implementation")

    parsed = parse_episode(sample_episode_path)
    assert isinstance(parsed, dict)
    assert "steps" in parsed
    assert "configuration" in parsed
    assert "rewards" in parsed
    assert len(parsed["steps"]) == 720 or len(parsed["steps"]) >= 2


def test_r1_observation_schema_validation(sample_raw_observation: dict[str, Any]):
    """Test 2 (R1): Validate observation structure conforms to 1.32.7 schema."""
    if validate_observation is None:
        # Direct authoritative assertion against Kaggle schema specification
        missing = REQUIRED_OBS_KEYS - set(sample_raw_observation.keys())
        assert not missing, f"Observation missing required keys: {missing}"
        farms = sample_raw_observation["farms"]
        assert isinstance(farms, list) and len(farms) == 2, "Farms must be list of 2 players"
        for farm in farms:
            assert REQUIRED_FARM_KEYS.issubset(farm.keys())
            assert len(farm["tiles"]) == 10 and len(farm["tiles"][0]) == 10
        pytest.skip("kaggriculture.eda.schema pending M1 implementation")

    is_valid = validate_observation(sample_raw_observation)
    assert is_valid is True


def test_r1_tile_state_kinds_mapping(sample_raw_observation: dict[str, Any]):
    """Test 3 (R1): Verify all standard tile states parse accurately."""
    sample_tiles = [
        None,
        "LOCKED",
        {"kind": "WEED"},
        {
            "kind": "PASTURE",
            "animal": "COW",
            "cared_today": True,
            "fed_today": True,
            "consecutive_unfed": 0,
            "fertilizer_available": True,
            "placed_day": 0,
            "pending_care_bonus": 0,
            "yield_units": 0,
        },
        {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "consecutive_unwatered": 0,
            "watered_today": True,
            "fertilized_until_day": -1,
            "max_lifespan_step": 120,
            "yield_units": 1,
        },
    ]

    if validate_tile is None:
        for t in sample_tiles:
            if t is None:
                assert t is None
            elif isinstance(t, str):
                assert t == "LOCKED"
            elif isinstance(t, dict):
                assert t.get("kind") in {"WEED", "PASTURE", "PLANT"}
        pytest.skip("kaggriculture.eda.schema.validate_tile pending M1 implementation")

    for t in sample_tiles:
        assert validate_tile(t) is True


def test_r1_private_inventory_and_shed_tracking(sample_raw_observation: dict[str, Any]):
    """Test 4 (R1): Validate private inventory sections and shed capacity constraints."""
    private = sample_raw_observation.get("private", {})
    assert "shed" in private
    assert "seeds" in private

    shed = private.get("shed", {})
    shed_units = sum(shed.values()) if isinstance(shed, dict) else 0.0
    # Baseline test replay should respect 100 max capacity in normal turns
    assert shed_units <= 100.0, f"Shed units exceeded 100 limit: {shed_units}"


def test_r1_automated_eda_report_generation(sample_episode_path: Path, tmp_path: Path):
    """Test 5 (R1): Verify automated EDA report produces required metrics."""
    if generate_eda_report is None:
        pytest.skip("kaggriculture.eda.report pending M1 implementation")

    output_file = tmp_path / "eda_report.json"
    report = generate_eda_report([sample_episode_path], output_path=output_file)

    assert isinstance(report, dict)
    assert "episode_count" in report
    assert "mean_episode_length" in report
    assert "tile_frequencies" in report
    assert "reward_summary" in report
    assert output_file.exists()


def test_r1_temporal_alignment_steps_t_plus_1(sample_episode_data: dict[str, Any]):
    """Test 6 (R1): Verify action at turn t is sourced from steps[t+1]."""
    steps = sample_episode_data["steps"]
    if len(steps) < 2:
        pytest.skip("Episode too short to test t+1 alignment")

    if extract_frame is None:
        pytest.skip("kaggriculture.eda.parser.extract_frame pending M1 implementation")

    obs_t, action_t, done = extract_frame(steps, t=0, seat=0)
    assert obs_t["step"] == 0
    assert action_t is not None
    assert not done
