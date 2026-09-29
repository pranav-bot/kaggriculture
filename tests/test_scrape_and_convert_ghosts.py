"""Unit and integration tests for scrape_top10_replays.py and convert_replays_to_ghosts.py."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.scrape_top10_replays import (
    find_kaggle_cli,
    query_kaggle_json,
    copy_cached_fallback_replays,
)
from scripts.convert_replays_to_ghosts import (
    bridge_autopsy_to_build_order,
    extract_macro_build_order,
    resolve_target_seat,
    convert_single_replay,
    verify_ghost,
)
from scripts.ladder_ghost import parse_build_order


def test_find_kaggle_cli():
    cli = find_kaggle_cli()
    assert Path(cli).is_file()
    assert "kaggle" in cli


def test_query_kaggle_json_slicing(monkeypatch):
    import subprocess

    # Test array response with Next Page Token prefix
    fake_output_prefix = (
        "Next Page Token = CfDJ8LXGgraiXMNClPzcySXY5U4rAVJkbafAsWnvQY6w8hDY4sAZMvnm1f5WSIND5r7c_51ZikikcEoxlQnDVVY8eJ8\n"
        '[{"teamId": 12345, "teamName": "Test Team", "score": "3000.0"}]'
    )

    class MockCompletedProcess:
        returncode = 0
        stdout = fake_output_prefix
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockCompletedProcess())

    res = query_kaggle_json("fake_cli", ["competitions", "leaderboard"])
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]["teamId"] == 12345
    assert res[0]["teamName"] == "Test Team"

    # Test array response with instructional footer
    fake_output_footer = (
        '[{"id": 999, "state": "EpisodeState.COMPLETED"}]\n\n'
        'Use "kaggle competitions replay 999" to download a replay, or "kaggle competitions logs 999 0" for agent logs.\n'
    )
    MockCompletedProcess.stdout = fake_output_footer
    res_footer = query_kaggle_json("fake_cli", ["competitions", "episodes"])
    assert isinstance(res_footer, list)
    assert len(res_footer) == 1
    assert res_footer[0]["id"] == 999


def test_bridge_autopsy_to_build_order():
    autopsy_sample = {
        "episode_id": 112619304,
        "opponent_name": "BenPalmer59",
        "opponent_build_order": [
            {"day": 0, "event": "BUY_COW", "details": "Purchased 3 COW(s)"},
            {"day": 0, "event": "BUY_SHEEP", "details": "Purchased 2 SHEEP(s)"},
            {"day": 0, "event": "BUY_SEED_WHEAT", "details": "Bought 4 WHEAT seeds"},
            {"day": 0, "event": "HIRE_FARMHAND", "details": "Hired farmhand on Day 0"},
            {"day": 6, "event": "EXPAND_QUADRANT_NE", "details": "Unlocked NE quadrant"},
            {"day": 8, "event": "EXPAND_QUADRANT_SW", "details": "Unlocked SW quadrant"},
        ],
    }

    ep_id, opp_name, order = bridge_autopsy_to_build_order(autopsy_sample)
    assert ep_id == "112619304"
    assert opp_name == "BenPalmer59"
    assert len(order) >= 6

    # Validate against ladder_ghost schema parser
    ep_parsed, validated = parse_build_order({"episode_id": ep_id, "build_order": order})
    assert ep_parsed == "112619304"
    actions = [d["action"] for d in validated]
    assert "BUY_ANIMAL" in actions
    assert "EXPAND_NE" in actions
    assert "EXPAND_SW" in actions
    assert "DUMP" in actions


def test_extract_macro_build_order_from_live_replay():
    replay_path = Path("replays/live_top10/episode-114845092-replay.json")
    if not replay_path.is_file():
        pytest.skip("Live replay not found")

    with open(replay_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    seat, opp_name = resolve_target_seat(doc, preferred_team="Boey")
    assert seat == 0
    assert opp_name == "Boey"

    order = extract_macro_build_order(doc, seat)
    assert len(order) >= 6

    # Ensure Day 0 buys, expansions, and liquidation exist
    days = [d["day"] for d in order]
    actions = [d["action"] for d in order]
    assert 0 in days
    assert "BUY_ANIMAL" in actions
    assert "EXPAND_NE" in actions
    assert "EXPAND_SW" in actions
    assert "DUMP" in actions

    # Validate schema
    ep_parsed, validated = parse_build_order({"episode_id": "114845092", "build_order": order})
    assert ep_parsed == "114845092"
    assert len(validated) == len(order)


def test_convert_single_replay_and_verify(tmp_path):
    replay_path = Path("replays/live_top10/episode-114845092-replay.json")
    if not replay_path.is_file():
        pytest.skip("Live replay not found")

    dest = convert_single_replay(replay_path, submissions_dir=tmp_path)
    assert dest.is_dir()
    assert (dest / "main.py").is_file()
    assert (dest / "manifest.json").is_file()

    # Verify execution
    res = verify_ghost(dest, steps=720)
    assert res["clean"] is True
    assert res["completed_steps"] == 720
    assert res["ms_per_turn"] < 1.0
