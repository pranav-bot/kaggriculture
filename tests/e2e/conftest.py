"""Shared fixtures, mock states, and helper utilities for E2E tests."""

from __future__ import annotations

import copy
import gzip
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def find_sample_episode_path() -> Path | None:
    """Find a real episode .json.gz file in datasets/il/episodes/."""
    episodes_dir = PROJECT_ROOT / "datasets" / "il" / "episodes"
    if episodes_dir.exists():
        for gz_file in episodes_dir.glob("*/*.json.gz"):
            if gz_file.stat().st_size > 1000:
                return gz_file
    return None


def create_minimal_synthetic_episode() -> dict[str, Any]:
    """Create a minimal 10x10 compliant 2-step episode dict."""
    tiles_seat0 = [[None for _ in range(10)] for _ in range(10)]
    # Place sample plant, weed, pasture
    tiles_seat0[0][0] = {
        "consecutive_unwatered": 0,
        "crop": "WHEAT",
        "fertilized_until_day": -1,
        "kind": "PLANT",
        "max_lifespan_step": 120,
        "planted_day": 0,
        "watered_today": True,
        "yield_units": 1,
    }
    tiles_seat0[0][1] = {"kind": "WEED"}
    tiles_seat0[0][2] = {
        "animal": "COW",
        "cared_today": True,
        "consecutive_unfed": 0,
        "fed_today": True,
        "fertilizer_available": False,
        "kind": "PASTURE",
        "pending_care_bonus": 1,
        "placed_day": 0,
        "yield_units": 0,
    }
    for r in range(5, 10):
        for c in range(5, 10):
            tiles_seat0[r][c] = "LOCKED"

    tiles_seat1 = copy.deepcopy(tiles_seat0)

    farms = [
        {
            "farmer": [4, 4],
            "hands": [[4, 5]],
            "hires_today": 0,
            "money": 3000.0,
            "tiles": tiles_seat0,
            "unlocked_quadrants": ["NW"],
        },
        {
            "farmer": [4, 4],
            "hands": [],
            "hires_today": 0,
            "money": 3000.0,
            "tiles": tiles_seat1,
            "unlocked_quadrants": ["NW"],
        },
    ]

    market = {
        "inventory": {"WHEAT": 10000, "CORN": 10000, "MILK": 10000, "WOOL": 10000},
        "prices": {"WHEAT": 25, "CORN": 30, "MILK": 40, "WOOL": 50},
    }
    town = {"unlocked_shops": []}

    obs0 = {
        "day": 0,
        "hour": 0,
        "step": 0,
        "player": 0,
        "remainingOverageTime": 60.0,
        "farms": farms,
        "market": market,
        "town": town,
        "private": {
            "carried": {"WHEAT": 5.0},
            "inventories": [{"WHEAT": 5.0}],
            "seeds": {"WHEAT": 10.0},
            "shed": {"WHEAT": 20.0},
        },
    }

    obs1 = copy.deepcopy(obs0)
    obs1["player"] = 1

    action0 = {"farmer": ["EAST"], "hands": [["NORTH"]], "market": []}
    action1 = {"farmer": ["WEST"], "hands": [], "market": []}

    step0 = [
        {"action": None, "info": {}, "observation": obs0, "reward": 0, "status": "ACTIVE"},
        {"action": None, "info": {}, "observation": obs1, "reward": 0, "status": "ACTIVE"},
    ]

    obs0_t1 = copy.deepcopy(obs0)
    obs0_t1["step"] = 1
    obs0_t1["hour"] = 1
    obs0_t1["farms"][0]["money"] = 3010.0
    obs1_t1 = copy.deepcopy(obs1)
    obs1_t1["step"] = 1
    obs1_t1["hour"] = 1

    step1 = [
        {"action": action0, "info": {}, "observation": obs0_t1, "reward": 10.0, "status": "ACTIVE"},
        {"action": action1, "info": {}, "observation": obs1_t1, "reward": 0.0, "status": "ACTIVE"},
    ]

    return {
        "configuration": {"episodeSteps": 720, "actTimeout": 1.0},
        "description": "Synthetic minimal test episode",
        "id": "synthetic_001",
        "info": {"seed": 42},
        "module_version": "1.32.7",
        "name": "kaggriculture",
        "rewards": [10.0, 0.0],
        "schema_version": 1,
        "specification": {},
        "statuses": ["ACTIVE", "ACTIVE"],
        "steps": [step0, step1],
        "title": "Synthetic Episode",
        "version": "1.32.7",
    }


@pytest.fixture(scope="session")
def sample_episode_path(tmp_path_factory) -> Path:
    """Fixture providing path to an episode .json.gz file."""
    real_path = find_sample_episode_path()
    if real_path is not None:
        return real_path

    # Fallback: create temporary synthetic episode .json.gz
    temp_dir = tmp_path_factory.mktemp("synthetic_episodes")
    ep_file = temp_dir / "synthetic_episode.json.gz"
    data = create_minimal_synthetic_episode()
    with gzip.open(ep_file, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return ep_file


@pytest.fixture(scope="session")
def sample_episode_data(sample_episode_path) -> dict[str, Any]:
    """Fixture providing loaded episode dictionary."""
    with gzip.open(sample_episode_path, "rt", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_raw_observation(sample_episode_data) -> dict[str, Any]:
    """Fixture providing a valid player-0 step-0 observation dict."""
    step0 = sample_episode_data["steps"][0]
    p0_frame = step0[0]
    return copy.deepcopy(p0_frame["observation"])


@pytest.fixture
def synthetic_edge_case_observations(sample_raw_observation) -> dict[str, dict[str, Any]]:
    """Fixture providing boundary and corner case observations."""
    obs_base = copy.deepcopy(sample_raw_observation)

    # 1. All null tiles
    obs_null = copy.deepcopy(obs_base)
    obs_null["farms"][0]["tiles"] = [[None for _ in range(10)] for _ in range(10)]

    # 2. All locked tiles
    obs_locked = copy.deepcopy(obs_base)
    obs_locked["farms"][0]["tiles"] = [["LOCKED" for _ in range(10)] for _ in range(10)]

    # 3. Empty shed and carried inventories
    obs_empty_inv = copy.deepcopy(obs_base)
    obs_empty_inv["private"] = {
        "carried": {},
        "inventories": [],
        "seeds": {},
        "shed": {},
    }

    # 4. Shed overflow (>100 capacity)
    obs_shed_overflow = copy.deepcopy(obs_base)
    obs_shed_overflow["private"] = {
        "carried": {},
        "inventories": [],
        "seeds": {},
        "shed": {"WHEAT": 80.0, "CORN": 50.0},  # total 130 > 100
    }

    # 5. Bankrupt farm (0 cash or negative cash)
    obs_bankrupt = copy.deepcopy(obs_base)
    obs_bankrupt["farms"][0]["money"] = 0.0

    # 6. Exhausted overage time
    obs_zero_overage = copy.deepcopy(obs_base)
    obs_zero_overage["remainingOverageTime"] = 0.0

    # 7. Extreme market inventory (oversupply)
    obs_extreme_market = copy.deepcopy(obs_base)
    obs_extreme_market["market"]["inventory"] = {
        "WHEAT": 100000,
        "CORN": 100000,
        "MILK": 100000,
        "WOOL": 100000,
    }
    obs_extreme_market["market"]["prices"] = {
        "WHEAT": 1,
        "CORN": 1,
        "MILK": 1,
        "WOOL": 1,
    }

    # 8. Zero market inventory (market drought)
    obs_drought_market = copy.deepcopy(obs_base)
    obs_drought_market["market"]["inventory"] = {
        "WHEAT": 0,
        "CORN": 0,
        "MILK": 0,
        "WOOL": 0,
    }

    # 9. Terminal step 719
    obs_terminal = copy.deepcopy(obs_base)
    obs_terminal["step"] = 719
    obs_terminal["day"] = 29
    obs_terminal["hour"] = 23

    return {
        "all_null_tiles": obs_null,
        "all_locked_tiles": obs_locked,
        "empty_inventory": obs_empty_inv,
        "shed_overflow": obs_shed_overflow,
        "bankrupt": obs_bankrupt,
        "zero_overage": obs_zero_overage,
        "extreme_market": obs_extreme_market,
        "drought_market": obs_drought_market,
        "terminal_step": obs_terminal,
    }


@pytest.fixture(scope="session")
def kagg_binary_path() -> Path | None:
    """Fixture locating compiled Rust `kagg` executable."""
    candidates = [
        PROJECT_ROOT / "kaggriculture-simulation" / "src-rust" / "target" / "release" / "kagg",
        PROJECT_ROOT / "target" / "release" / "kagg",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    # Try system PATH
    sys_kagg = shutil.which("kagg")
    if sys_kagg:
        return Path(sys_kagg)
    return None


@pytest.fixture
def mock_rust_engine_state(sample_raw_observation) -> dict[str, Any]:
    """Fixture returning a full Rust-engine compatible state dict."""
    obs = copy.deepcopy(sample_raw_observation)
    return {
        "step": obs.get("step", 0),
        "day": obs.get("day", 0),
        "hour": obs.get("hour", 0),
        "done": False,
        "farms": obs.get("farms", []),
        "market": obs.get("market", {}),
        "town": obs.get("town", {}),
        "private": [
            obs.get("private", {}),
            obs.get("private", {}),
        ],
    }


@pytest.fixture
def temp_cache_dir(tmp_path) -> Path:
    """Fixture providing an isolated temporary directory for chunked data caching."""
    cache_path = tmp_path / "kagg_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
