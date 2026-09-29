"""Comprehensive verification suite for Kaggriculture Streamlit Dashboard."""

import os
import sys
import time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dashboard.data_loader import (
    MatchDataset,
    list_available_replays,
    load_dataset,
)
from dashboard.grid_view import render_farm_grid_html


@pytest.fixture(scope="module")
def default_replay_path():
    replays = list_available_replays()
    assert len(replays) > 0, "No replay files found in workspace"
    # Find replay with 114793445 if possible
    for r in replays:
        if "114793445" in r:
            return r
    return replays[0]


@pytest.fixture(scope="module")
def dataset(default_replay_path):
    return load_dataset(default_replay_path)


def test_loading_latency_under_3_seconds(default_replay_path):
    """Assert dashboard engine loads and indexes in under 3.0 seconds."""
    t0 = time.perf_counter()
    ds = load_dataset(default_replay_path)
    load_time = time.perf_counter() - t0
    print(f"Dataset load latency: {load_time:.3f}s")
    assert load_time < 3.0, f"Dashboard load time {load_time:.3f}s exceeded 3.0s limit"
    assert len(ds.turns) == 720, f"Expected 720 turns, got {len(ds.turns)}"


def test_grid_state_parsing(dataset):
    """Verify 10x10 farm grid states, coordinates, and worker positions."""
    turn0 = dataset.turns[0]
    assert turn0.turn == 0
    assert turn0.day == 0
    assert turn0.hour == 0
    assert len(turn0.tiles) == 10
    assert len(turn0.tiles[0]) == 10
    assert turn0.farmer_pos == (4, 4)
    assert turn0.unlocked_quadrants == ["NW"]

    # Check later turn
    turn100 = dataset.turns[100]
    assert len(turn100.crops) > 0 or len(turn100.animals) > 0
    assert len(turn100.hands_pos) > 0


def test_iql_value_and_divergence_detection(dataset):
    """Verify IQL Value Net predicted RTG and automatic divergence turn detection."""
    assert dataset.divergence_turn > 0
    assert dataset.divergence_turn < 720
    assert dataset.divergence_reason != ""
    assert dataset.final_cash > 0

    df = dataset.df_values
    assert len(df) == 720
    assert "pred_rtg" in df.columns
    assert "actual_rtg" in df.columns
    assert "divergence_error" in df.columns

    # Predicted RTG should be positive and bounded
    assert (df["pred_rtg"] >= 0).all()


def test_beam_search_candidates(dataset):
    """Verify 24-step Beam Search candidates, scores, and trajectories."""
    for test_turn in [0, 100, 384, 500, 700]:
        candidates = dataset.get_beam_search_candidates(test_turn)
        assert len(candidates) == 3, f"Expected 3 beam candidates, got {len(candidates)}"

        winner = candidates[0]
        assert winner.is_winner is True
        assert len(winner.actions) == 24, f"Expected 24-step trajectory, got {len(winner.actions)}"
        assert winner.score >= candidates[1].score
        assert candidates[1].score >= candidates[2].score


def test_kuhn_munkres_routing_and_debugger(dataset):
    """Verify Kuhn-Munkres cost matrix computation and Strawberry vs Weed debugger."""
    for test_turn in [50, 100, 385, 500]:
        labor = dataset.get_labor_cost_matrix(test_turn)
        assert len(labor.workers) >= 1
        assert len(labor.targets) >= 1
        assert labor.cost_matrix.shape == (len(labor.workers), len(labor.targets))
        assert len(labor.assignments) == min(len(labor.workers), len(labor.targets))

        # Heatmap grid must be 10x10
        assert labor.heatmap_grid.shape == (10, 10)

        # Check strawberry vs weed debugger if applicable
        if labor.strawberry_weed_debug:
            dbg = labor.strawberry_weed_debug
            assert "worker_name" in dbg
            assert "dist_weed" in dbg
            assert "dist_strawberry" in dbg
            assert "delta_cost" in dbg
            assert "reason" in dbg


def test_html_grid_rendering(dataset):
    """Verify HTML/CSS generation for the farm grid."""
    turn_data = dataset.turns[100]
    labor = dataset.get_labor_cost_matrix(100)
    html_normal = render_farm_grid_html(turn_data, labor, mode="normal")
    assert "farm-grid" in html_normal
    assert "farm-tile" in html_normal
    assert "🧑‍🌾" in html_normal

    html_heatmap = render_farm_grid_html(turn_data, labor, mode="heatmap")
    assert "farm-grid" in html_heatmap
    assert "background-color" in html_heatmap
