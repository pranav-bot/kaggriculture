"""Smoke tests for panel analysis script."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_one_solo_episode_keys():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_core import aggregate_solo_rows, run_one_solo_episode

    row = run_one_solo_episode(
        "wheat_loop",
        "wheat_loop",
        seed=0,
        steps=24,
        opponent="pass",
        load_suffix="_test0",
    )
    assert row["agent"] == "wheat_loop"
    assert row["status"] == "ok"
    assert "terminal_cash" in row
    agg = aggregate_solo_rows([row])
    assert "mean_cash" in agg
    assert agg["episodes_ok"] == 1


def test_run_agent_analysis_subprocess_smoke(tmp_path):
    out = tmp_path / "panel_smoke"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_agent_analysis.py"),
        "--agents",
        "wheat_loop",
        "--seeds",
        "1",
        "--steps",
        "24",
        "--opponent",
        "pass",
        "--output-dir",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    summary = json.loads((out / "panel_summary.json").read_text())
    assert "leaderboard" in summary
    assert len(summary["leaderboard"]) >= 1
    assert (out / "panel_leaderboard.csv").is_file()
    assert (out / "wheat_loop_seeds.csv").is_file()
