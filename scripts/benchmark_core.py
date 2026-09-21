"""Shared solo-episode benchmarking for compare_agents and panel analysis."""

from __future__ import annotations

import statistics
import traceback
from time import perf_counter
from typing import Any, Dict, List, Optional

from agent_utils import load_agent
from kaggriculture.env import Environment
from kaggriculture.helpers.episode_metrics import EpisodeMetricsRecorder, MetricsWrapper
from kaggriculture.helpers.market_overlays import reset_market_overlay_state

SOLO_CSV_FIELDS = [
    "seed",
    "agent",
    "status",
    "terminal_cash",
    "overflow",
    "slots_burned",
    "mean_impact",
    "mean_revenue_per_sold_unit",
    "collision_holds",
    "collision_releases",
    "terminal_carried_goods_step_718",
    "decide_ms_p50",
    "decide_ms_p95",
    "elapsed_s",
]

COMPARE_CSV_FIELDS = [
    "seed",
    "agent",
    "terminal_cash",
    "overflow",
    "slots_burned",
    "mean_impact",
    "decide_ms_p50",
    "decide_ms_p95",
]


def run_solo(
    agent_spec: str,
    seed: int,
    steps: int,
    opponent: str,
    load_suffix: str,
) -> Dict[str, Any]:
    """Run one solo episode; return finalized episode metrics summary."""
    reset_market_overlay_state()
    agent = load_agent(agent_spec, module_suffix=load_suffix)
    recorder = EpisodeMetricsRecorder()
    wrapped = MetricsWrapper(agent, recorder)
    env = Environment(
        configuration={"episodeSteps": steps, "seed": seed},
        debug=False,
    )
    final = env.run_env(wrapped, opponent)
    obs = env.get_current_state(final, agent1=True)
    if not isinstance(obs, dict):
        obs = dict(obs) if hasattr(obs, "items") else {"farms": [{}]}
    return recorder.finalize(obs, steps=steps)


def run_one_solo_episode(
    agent_spec: str,
    agent_label: str,
    seed: int,
    steps: int,
    opponent: str,
    load_suffix: str,
) -> Dict[str, Any]:
    """Run solo episode with timing and error handling; returns a CSV-ready row."""
    started = perf_counter()
    try:
        summary = run_solo(agent_spec, seed, steps, opponent, load_suffix)
        elapsed = perf_counter() - started
        row = row_from_summary(seed, agent_label, summary, status="ok", elapsed_s=elapsed)
        row["_summary"] = summary
        return row
    except Exception as exc:
        elapsed = perf_counter() - started
        return {
            "seed": seed,
            "agent": agent_label,
            "status": "failed",
            "terminal_cash": "",
            "overflow": "",
            "slots_burned": "",
            "mean_impact": "",
            "mean_revenue_per_sold_unit": "",
            "collision_holds": "",
            "collision_releases": "",
            "terminal_carried_goods_step_718": "",
            "decide_ms_p50": "",
            "decide_ms_p95": "",
            "elapsed_s": round(elapsed, 3),
            "_error": f"{type(exc).__name__}: {exc}",
            "_traceback": traceback.format_exc(),
        }


def row_from_summary(
    seed: int,
    agent_label: str,
    summary: Dict[str, Any],
    *,
    status: str = "ok",
    elapsed_s: Optional[float] = None,
) -> Dict[str, Any]:
    scores = summary.get("sell_impact_scores") or []
    mean_impact = summary.get("mean_impact_score_of_sells")
    if mean_impact is None and scores:
        mean_impact = float(sum(scores) / len(scores))
    if mean_impact is None:
        mean_impact = ""
    row = {
        "seed": seed,
        "agent": agent_label,
        "status": status,
        "terminal_cash": summary.get("final_cash", ""),
        "overflow": summary.get("shed_overflow_units_lost", ""),
        "slots_burned": summary.get("unfillable_sell_slots_burned", ""),
        "mean_impact": mean_impact,
        "mean_revenue_per_sold_unit": summary.get("mean_revenue_per_sold_unit", ""),
        "collision_holds": summary.get("collision_holds", ""),
        "collision_releases": summary.get("collision_releases", ""),
        "terminal_carried_goods_step_718": summary.get(
            "terminal_carried_goods_step_718", ""
        ),
        "decide_ms_p50": summary.get("decide_ms_p50", ""),
        "decide_ms_p95": summary.get("decide_ms_p95", ""),
        "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else "",
    }
    return row


def compare_row_from_summary(seed: int, agent_label: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Narrow row for compare_agents.py backward compatibility."""
    full = row_from_summary(seed, agent_label, summary)
    return {k: full[k] for k in COMPARE_CSV_FIELDS}


def aggregate_solo_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok" and r.get("terminal_cash") != ""]
    cash = [float(r["terminal_cash"]) for r in ok]
    if not cash:
        return {
            "episodes": len(rows),
            "episodes_ok": 0,
            "mean_cash": 0.0,
            "median_cash": 0.0,
            "std_cash": 0.0,
            "overflow_mean": 0.0,
            "slots_mean": 0.0,
            "p95_ms_mean": 0.0,
        }

    def mean_field(key: str) -> float:
        vals = [float(r[key]) for r in ok if r.get(key) not in ("", None)]
        return statistics.mean(vals) if vals else 0.0

    return {
        "episodes": len(rows),
        "episodes_ok": len(ok),
        "mean_cash": statistics.mean(cash),
        "median_cash": statistics.median(cash),
        "std_cash": statistics.pstdev(cash) if len(cash) > 1 else 0.0,
        "overflow_mean": mean_field("overflow"),
        "slots_mean": mean_field("slots_burned"),
        "p95_ms_mean": mean_field("decide_ms_p95"),
        "overflow_sum": sum(int(r.get("overflow") or 0) for r in ok),
        "slots_sum": sum(int(r.get("slots_burned") or 0) for r in ok),
    }


def win_rate_vs_baseline(
    agent_rows: List[Dict[str, Any]],
    baseline_by_seed: Dict[int, float],
) -> Optional[float]:
    wins = 0
    paired = 0
    for row in agent_rows:
        if row.get("status") != "ok":
            continue
        seed = int(row["seed"])
        if seed not in baseline_by_seed:
            continue
        paired += 1
        if float(row["terminal_cash"]) > baseline_by_seed[seed]:
            wins += 1
    if paired == 0:
        return None
    return wins / paired
