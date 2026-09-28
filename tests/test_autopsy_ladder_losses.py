"""Unit tests for ladder loss autopsy and reverse-engineering pipeline."""

import json
import sqlite3
from pathlib import Path
import pytest

from scripts.autopsy_ladder_losses import (
    analyze_loss_match,
    ensure_database_schema,
    query_loss_episodes,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ensure_database_schema_and_query_loss_episodes(tmp_path: Path):
    db_file = tmp_path / "test_ratings.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row

    with conn:
        conn.execute("""
            CREATE TABLE episodes (
                submission_id TEXT,
                episode_id TEXT PRIMARY KEY,
                create_time TEXT
            )
        """)
        conn.execute("INSERT INTO episodes VALUES ('56647370', '101', '2026-09-28 12:00:00')")
        conn.execute("INSERT INTO episodes VALUES ('56647370', '102', '2026-09-28 13:00:00')")

    # Add columns
    ensure_database_schema(conn)

    # Set 101 as win, 102 as loss
    with conn:
        conn.execute("UPDATE episodes SET my_reward = 80000, opp_reward = 30000, is_loss = 0 WHERE episode_id = '101'")
        conn.execute("UPDATE episodes SET my_reward = 25000, opp_reward = 65000, is_loss = 1 WHERE episode_id = '102'")

    losses = query_loss_episodes(conn)
    assert len(losses) == 1
    assert losses[0]["episode_id"] == "102"
    assert losses[0]["my_reward"] == 25000.0
    assert losses[0]["opp_reward"] == 65000.0

    conn.close()


def test_analyze_loss_match_episode_114793445():
    replay_file = ROOT / "replays" / "live_losses" / "episode-114793445-replay.json"
    if not replay_file.is_file():
        pytest.skip(f"Replay {replay_file} not found")

    analysis = analyze_loss_match(replay_file, player_query="Pranav")

    assert analysis["episode_id"] == "114793445"
    assert analysis["match_metadata"]["match_result"] == "LOSS"
    assert analysis["match_metadata"]["our_terminal_score"] == 21926.0
    assert analysis["match_metadata"]["opponent_terminal_score"] == 63866.0
    assert analysis["match_metadata"]["score_deficit"] == 41940.0

    # Macro-milestones
    miles = analysis["opponent_macro_milestones"]
    assert miles["ne_quadrant_unlock"] is not None
    assert miles["ne_quadrant_unlock"]["day"] == 6
    assert miles["sw_quadrant_unlock"] is not None
    assert miles["sw_quadrant_unlock"]["day"] == 10
    assert miles["peak_herd_size"] >= 10
    assert miles["wool_market_crash"]["triggered"] is False

    # Crossover turn
    cross = analysis["crossover_analysis"]
    assert cross["crossover_turn"] == 165
    assert cross["crossover_day"] == 6
    assert cross["our_net_worth_at_crossover"] == 4951.0
    assert cross["opp_net_worth_at_crossover"] == 5233.0
    assert "Day 6" in cross["crossover_catalyst"]


def test_analyze_loss_match_wool_crash_episode_114797890():
    replay_file = ROOT / "replays" / "live_losses" / "episode-114797890-replay.json"
    if not replay_file.is_file():
        pytest.skip(f"Replay {replay_file} not found")

    analysis = analyze_loss_match(replay_file, player_query="Pranav")

    miles = analysis["opponent_macro_milestones"]
    wool = miles["wool_market_crash"]
    assert wool["triggered"] is True
    assert wool["min_wool_price"] == 1.0
    assert wool["opponent_wool_sold_total"] > 0
    assert analysis["crossover_analysis"]["crossover_turn"] > 0
