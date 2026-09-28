"""Engine-free tests for scripts/find_best_agent.py."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from find_best_agent import (
    aggregate,
    build_specs,
    leaderboard_markdown,
    wilson_ci,
)
from run_elo_tournament import update_elo_ratings


def _row(a, b, sa, sb, ca, cb, ms=(0.5, 0.5)):
    return {"agents": [a, b], "scores": [sa, sb], "banks": [ca, cb],
            "act_ms_mean": list(ms), "act_ms_max": [m * 2 for m in ms],
            "error": None, "forfeit": False}


def test_wilson_ci_contains_rate_and_shrinks():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    lo2, hi2 = wilson_ci(500, 1000)
    assert (hi2 - lo2) < (hi - lo)
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_aggregate_cash_latency_and_records():
    rows = [_row("a", "b", 1, 0, 10000, 5000),
            _row("a", "b", 0.5, 0.5, 3000, 3000),
            _row("a", "b", 0, 1, 1000, 9000, ms=(7.0, 0.4))]
    stats = aggregate(rows, [])
    a = stats["a"]
    assert a["games"] == 3 and (a["w"], a["d"], a["l"]) == (1, 1, 1)
    assert a["avg_cash"] == pytest.approx((10000 + 3000 + 1000) / 3)
    assert a["worst_cash"] == pytest.approx(1000)  # robustness floor
    assert a["latency_flag"] == "OK"  # mean (0.5+0.5+7)/3 = 2.67ms
    assert a["opp"]["b"]["games"] == 3


def test_latency_flag_threshold():
    rows = [_row("slow", "b", 1, 0, 5000, 4000, ms=(6.0, 0.5))]
    stats = aggregate(rows, [])
    assert stats["slow"]["latency_flag"] == "DANGEROUS: High Latency."
    assert stats["slow"]["mean_ms"] == pytest.approx(6.0)


def test_leaderboard_orders_by_elo_and_recommends():
    rows = ([_row("champ", "mid", 1, 0, 9000, 5000) for _ in range(6)]
            + [_row("mid", "slow", 1, 0, 8000, 4000, ms=(0.5, 9.0)) for _ in range(6)]
            + [_row("champ", "slow", 1, 0, 9500, 3000, ms=(0.5, 9.0)) for _ in range(6)])
    stats = aggregate(rows, [])
    elos = update_elo_ratings(rows)
    md, pick = leaderboard_markdown(stats, elos, rows, list(range(6)), top_k=3)
    assert pick == "champ"  # best Elo among latency-compliant
    assert md.index("`champ`") < md.index("`mid`") < md.index("`slow`")
    assert "DANGEROUS: High Latency." in md
    assert "Worst cash" in md and "McNemar" in md and "submission.tar.gz" in md


def test_build_specs_skips_missing_tapes(tmp_path):
    agents = {"x": str(tmp_path / "main.py")}
    specs = build_specs(agents, [("t1", "replays/tapes/112542379_opp_seat1.tape"),
                                 ("t2", "nope/missing.tape")])
    kinds = {s["name"]: s["type"] for s in specs}
    assert kinds["x"] == "python" and kinds["t1"] == "tape"
    assert "t2" not in kinds
