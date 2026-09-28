"""Unit and Integration Tests for scripts/run_elo_tournament.py.

Validates:
1. McNemar exact two-sided p-value calculation
2. Iterative Elo rating update mechanics
3. Submission discovery and agent filtering
4. Markdown stdout table parsing
5. Latency profiling gate enforcement
6. End-to-end tournament execution via kagg tournament
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.run_elo_tournament import (
    compute_paired_mcnemar,
    discover_submissions,
    find_kagg_binary,
    mcnemar_exact,
    parse_stdout_standings,
    run_elo_tournament,
    update_elo_ratings,
)


def test_mcnemar_exact():
    """Verify exact two-sided McNemar binomial calculations."""
    # Symmetrical outcomes -> p = 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0

    # 6 vs 0 discordant pairs: (0.5^6) * 2 = 0.03125
    assert abs(mcnemar_exact(6, 0) - 0.03125) < 1e-6
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-6

    # 10 vs 0 discordant pairs: (0.5^10) * 2 = 0.001953125
    assert abs(mcnemar_exact(10, 0) - 0.001953125) < 1e-6


def test_update_elo_ratings():
    """Verify Elo rating adjustments based on match outcomes."""
    games = [
        {"agents": ["agent_a", "agent_b"], "scores": [1.0, 0.0]},
        {"agents": ["agent_a", "agent_b"], "scores": [1.0, 0.0]},
    ]
    ratings = update_elo_ratings(games, initial_ratings={"agent_a": 1500.0, "agent_b": 1500.0}, k_factor=32.0, iterations=1)

    assert ratings["agent_a"] > 1500.0
    assert ratings["agent_b"] < 1500.0
    # Zero-sum property for initial equal ratings
    assert abs((ratings["agent_a"] - 1500.0) + (ratings["agent_b"] - 1500.0)) < 1e-6


def test_discover_submissions():
    """Verify discovery and filtering of agent submissions."""
    subs = discover_submissions("submissions", filter_agents=["agent_final", "care_mill"])
    assert "agent_final" in subs
    assert "care_mill" in subs
    assert subs["agent_final"].endswith("main.py")


def test_parse_stdout_standings():
    """Verify parsing of Rust tournament Markdown standings table."""
    mock_stdout = """
# Tournament: test

4 game(s), 0 with errors, 0.01 s.

## Standings

| # | agent | games | W/D/L | score | 95% CI | mean margin | errors | ms/turn mean / max |
|---:|---|---:|---|---:|---|---:|---:|---|
| 1 | agent_alpha | 10 | 8/1/1 | 0.850 | 0.650 .. 1.000 | 12500 | 0 | 0.12 / 0.45 |
| 2 | agent_beta | 10 | 1/1/8 | 0.150 | 0.000 .. 0.350 | -12500 | 0 | 6.20 / 12.50 |
"""
    standings = parse_stdout_standings(mock_stdout)
    assert "agent_alpha" in standings
    assert "agent_beta" in standings

    alpha = standings["agent_alpha"]
    assert alpha["games"] == 10
    assert alpha["wins"] == 8
    assert alpha["draws"] == 1
    assert alpha["losses"] == 1
    assert alpha["score"] == 0.85
    assert alpha["mean_act_ms"] == 0.12
    assert alpha["max_act_ms"] == 0.45

    beta = standings["agent_beta"]
    assert beta["mean_act_ms"] == 6.20


def test_paired_mcnemar_computation():
    """Verify paired McNemar statistics and confidence interval calculation."""
    games = [
        {"agents": ["top", "bot"], "scores": [1.0, 0.0], "banks": [50000, 20000]},
        {"agents": ["top", "bot"], "scores": [1.0, 0.0], "banks": [45000, 15000]},
        {"agents": ["top", "bot"], "scores": [1.0, 0.0], "banks": [60000, 25000]},
        {"agents": ["top", "bot"], "scores": [0.0, 1.0], "banks": [20000, 30000]},
    ]
    res = compute_paired_mcnemar(games, "top", "bot")
    assert res["games"] == 4
    assert res["better_a"] == 3
    assert res["better_b"] == 1
    assert res["mean_diff"] == (30000 + 30000 + 35000 - 10000) / 4
    assert res["p_value"] > 0.0


def test_run_elo_tournament_e2e(tmp_path: Path):
    """Verify end-to-end tournament execution with kagg tournament."""
    lb_file = tmp_path / "test_lb.json"

    result = run_elo_tournament(
        submissions_dir="submissions",
        filter_agents=["agent_final", "melon_rusher"],
        num_seeds=2,
        workers=2,
        leaderboard_file=lb_file,
        latency_threshold_ms=5.0,
    )

    assert lb_file.exists()
    assert result["total_games"] == 4
    assert result["top_agent"] == "agent_final"
    assert len(result["leaderboard"]) == 2

    # Check top agent metrics
    top = result["leaderboard"][0]
    assert top["name"] == "agent_final"
    assert top["elo"] > 1500.0
    assert top["wins"] >= 1
    assert top["latency_violation"] is False
