#!/usr/bin/env python3
"""Compare two agents over N seeded solo episodes; emit CSV metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from kaggriculture.env import Environment
from kaggriculture.helpers.episode_metrics import EpisodeMetricsRecorder, MetricsWrapper
from kaggriculture.helpers.market_overlays import reset_market_overlay_state

CSV_FIELDS = [
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


def row_from_summary(seed: int, agent_label: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    scores = summary.get("sell_impact_scores") or []
    mean_impact = summary.get("mean_impact_score_of_sells")
    if mean_impact is None and scores:
        mean_impact = float(sum(scores) / len(scores))
    if mean_impact is None:
        mean_impact = ""
    return {
        "seed": seed,
        "agent": agent_label,
        "terminal_cash": summary.get("final_cash", ""),
        "overflow": summary.get("shed_overflow_units_lost", ""),
        "slots_burned": summary.get("unfillable_sell_slots_burned", ""),
        "mean_impact": mean_impact,
        "decide_ms_p50": summary.get("decide_ms_p50", ""),
        "decide_ms_p95": summary.get("decide_ms_p95", ""),
    }


def print_summary(agent_a: str, agent_b: str, rows: List[Dict[str, Any]]) -> str:
    def cash_for(label: str) -> List[float]:
        return [float(r["terminal_cash"]) for r in rows if r["agent"] == label]

    a_cash = cash_for(agent_a)
    b_cash = cash_for(agent_b)
    lines = [
        f"Compare: {agent_a} vs {agent_b} ({len(a_cash)} seeds)",
        f"  {agent_a} cash: mean={statistics.mean(a_cash):,.0f} "
        f"median={statistics.median(a_cash):,.0f}",
        f"  {agent_b} cash: mean={statistics.mean(b_cash):,.0f} "
        f"median={statistics.median(b_cash):,.0f}",
        f"  Δ mean cash ({agent_a} − {agent_b}): "
        f"{statistics.mean(a_cash) - statistics.mean(b_cash):+,.0f}",
    ]
    wins = sum(1 for i in range(len(a_cash)) if a_cash[i] > b_cash[i])
    lines.append(f"  {agent_a} higher cash on {wins}/{len(a_cash)} seeds")
    text = "\n".join(lines)
    print(text, file=sys.stderr)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two agents with episode metrics.")
    parser.add_argument("agent_a", help="First agent (submission name or path to main.py)")
    parser.add_argument("agent_b", help="Second agent")
    parser.add_argument(
        "--seeds",
        "-n",
        type=int,
        default=10,
        help="Number of consecutive seeds (default: 10)",
    )
    parser.add_argument("--start-seed", type=int, default=0, help="First seed value")
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--opponent", default="random")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="CSV path (default: stdout)",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        suffix = f"_s{seed}"
        summary_a = run_solo(args.agent_a, seed, args.steps, args.opponent, suffix + "a")
        rows.append(row_from_summary(seed, args.agent_a, summary_a))
        summary_b = run_solo(args.agent_b, seed, args.steps, args.opponent, suffix + "b")
        rows.append(row_from_summary(seed, args.agent_b, summary_b))

    print_summary(args.agent_a, args.agent_b, rows)

    out = args.output
    if out is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
