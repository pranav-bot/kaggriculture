#!/usr/bin/env python3
"""Compare two agents over N seeded solo episodes; emit CSV metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_core import COMPARE_CSV_FIELDS, compare_row_from_summary, run_solo


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
        rows.append(compare_row_from_summary(seed, args.agent_a, summary_a))
        summary_b = run_solo(args.agent_b, seed, args.steps, args.opponent, suffix + "b")
        rows.append(compare_row_from_summary(seed, args.agent_b, summary_b))

    print_summary(args.agent_a, args.agent_b, rows)

    out = args.output
    if out is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=COMPARE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COMPARE_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
