"""Unit tests for replay safety fallback auditor."""

import os
import pytest
from scripts.audit_replay_fallbacks import audit_single_replay, identify_player_index


def test_identify_player_index():
    mock_data = {
        "info": {
            "TeamNames": ["Pranav Advani", "OpponentBot"]
        }
    }
    assert identify_player_index(mock_data, "Pranav") == 0
    assert identify_player_index(mock_data, "Opponent") == 1
    assert identify_player_index(mock_data, "NonExistent") == 0


def test_audit_single_replay_on_actual_replay():
    replay_path = "replays/my_agents/psro_leauge_pick/114793445.json"
    if not os.path.exists(replay_path):
        pytest.skip(f"Replay {replay_path} not found")

    res = audit_single_replay(replay_path, player_query="Pranav")
    assert res["status"] == "DONE"
    assert res["total_steps"] == 720
    assert res["exact_fallback_matches"] == 0
    assert res["exact_fallback_rate"] == 0.0
    assert res["all_pass_rate"] < 0.05
    assert res["is_fallback_active"] is False
    assert res["crops_planted"]["STRAWBERRY"] > 0
    assert res["quadrants_bought"] == 2
