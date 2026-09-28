"""Unit tests for ladder performance tracker and rating trajectory visualizer."""

import os
import sqlite3
from pathlib import Path
import pytest

from scripts.track_ladder_ratings import (
    compute_rolling_deltas,
    evaluate_plateau_alerts,
    export_database_to_csv,
    generate_rating_trajectory_plot,
    init_database,
    load_submission_trajectory,
    parse_episodes_output,
    save_episodes_batch,
    save_submission_metadata,
)


def test_sqlite_persistence_roundtrip(tmp_path: Path):
    db_file = tmp_path / "test_ratings.db"
    conn = init_database(db_file)

    save_submission_metadata(
        conn,
        submission_id="12345",
        description="Test Agent",
        date="2026-09-28",
        status="COMPLETE",
        score=510.5,
    )

    dummy_episodes = [
        {"id": "1001", "createTime": "2026-09-28 10:00:00", "agent_score": 500.0},
        {"id": "1002", "createTime": "2026-09-28 11:00:00", "agent_score": 505.0},
        {"id": "1003", "createTime": "2026-09-28 12:00:00", "agent_score": 510.5},
    ]
    added = save_episodes_batch(conn, "12345", dummy_episodes)
    assert added == 3

    # Idempotence check
    added_again = save_episodes_batch(conn, "12345", dummy_episodes)
    assert added_again == 0

    history = load_submission_trajectory(conn, "12345")
    assert len(history) == 3
    assert history[0]["episode_id"] == "1001"
    assert history[0]["agent_score"] == 500.0
    assert history[2]["agent_score"] == 510.5

    # Export to CSV check
    csv_file = tmp_path / "export.csv"
    export_database_to_csv(conn, csv_file)
    assert csv_file.is_file()
    content = csv_file.read_text()
    assert "12345" in content
    assert "1001" in content

    conn.close()


def test_parse_episodes_output():
    raw_csv = (
        "id,createTime,endTime,state,type,agent_score\n"
        "2001,2026-09-28 10:00:00,2026-09-28 10:05:00,EpisodeState.COMPLETED,EpisodeType.EPISODE_TYPE_PUBLIC,520.0\n"
        "2002,2026-09-28 10:10:00,2026-09-28 10:15:00,EpisodeState.COMPLETED,EpisodeType.EPISODE_TYPE_PUBLIC,524.5\n"
        'Use "kaggle competitions logs <id> <index>" for agent logs.\n'
    )
    parsed = parse_episodes_output(raw_csv)
    assert len(parsed) == 2
    assert parsed[0]["id"] == "2001"
    assert parsed[0]["agent_score"] == 520.0
    assert parsed[1]["id"] == "2002"
    assert parsed[1]["agent_score"] == 524.5


def test_rolling_deltas_and_plateau_detection(tmp_path: Path):
    db_file = tmp_path / "alerts.db"
    conn = init_database(db_file)

    # 15 matches where agent climbs, peaks at match 10, then drops across matches 11-15
    scores = [
        400.0, 410.0, 420.0, 430.0, 440.0,
        450.0, 460.0, 470.0, 480.0, 500.0,  # Match 10 peak at 500
        495.0, 490.0, 485.0, 480.0, 475.0   # Match 15 drops to 475
    ]
    episodes = [
        {"id": str(1000 + i), "createTime": f"2026-09-28 {10 + (i//6):02d}:{(i%6)*10:02d}:00", "agent_score": s}
        for i, s in enumerate(scores)
    ]

    enriched = compute_rolling_deltas(episodes, window=10)
    assert len(enriched) == 15
    # Lookback at index 14 is index 4 (scores: 475 vs 440 -> delta = +35)
    # But from peak at index 9 (500), let's test a case where rolling 10 delta is negative
    peak_then_crash = [
        500.0, 500.0, 500.0, 500.0, 500.0,
        500.0, 500.0, 500.0, 500.0, 500.0,  # 10 matches at 500
        480.0  # Match 11 drops to 480 (delta vs index 0 is -20)
    ]
    crash_episodes = [
        {"id": str(2000 + i), "createTime": f"2026-09-28 {10 + (i//6):02d}:{(i%6)*10:02d}:00", "agent_score": s}
        for i, s in enumerate(peak_then_crash)
    ]

    alerts = evaluate_plateau_alerts(conn, "56647370", crash_episodes, window=10)
    assert len(alerts) == 1
    assert "SUBMISSION 56647370 PLATEAU DETECTED. ELO: 480.0." in alerts[0]

    # Verify persisted alert in database
    cur = conn.execute("SELECT * FROM plateau_alerts WHERE submission_id = '56647370'")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["current_elo"] == 480.0
    assert rows[0]["rolling_10_delta"] == -20.0

    conn.close()


def test_generate_rating_trajectory_plot(tmp_path: Path):
    plot_file = tmp_path / "test_trajectory.png"
    trajectories = {
        "56647370": {
            "description": "PSRO league pick",
            "episodes": [
                {"agent_score": 480.0, "rolling_delta": 0.0},
                {"agent_score": 490.0, "rolling_delta": 10.0},
                {"agent_score": 505.0, "rolling_delta": 25.0},
                {"agent_score": 498.0, "rolling_delta": -7.0},
            ]
        },
        "56612208": {
            "description": "two_team_grandmaster",
            "episodes": [
                {"agent_score": 450.0, "rolling_delta": 0.0},
                {"agent_score": 470.0, "rolling_delta": 20.0},
                {"agent_score": 510.0, "rolling_delta": 60.0},
                {"agent_score": 521.7, "rolling_delta": 71.7},
            ]
        }
    }

    res_path = generate_rating_trajectory_plot(trajectories, output_path=plot_file)
    assert res_path.is_file()
    assert res_path.stat().st_size > 5000  # valid image file > 5 KB
