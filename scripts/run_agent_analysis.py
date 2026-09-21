#!/usr/bin/env python3
"""Panel benchmark for multiple Kaggriculture submissions with rich stderr progress.

Example (quick smoke):
  python scripts/run_agent_analysis.py --agents all --seeds 5 --steps 720 \\
    --output-dir standoff/smoke

Example (serious panel):
  python scripts/run_agent_analysis.py \\
    --agents market_velocity,alpha_velocity_p1,alpha_shop_hybrid,demand_mpc,shop_opportunist \\
    --seeds 30 --baseline shop_opportunist --head-to-head shop_opportunist \\
    --output-dir standoff/panel_main
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import SUBMISSIONS, load_agent
from benchmark_core import (
    SOLO_CSV_FIELDS,
    aggregate_solo_rows,
    run_one_solo_episode,
    win_rate_vs_baseline,
)
from kaggriculture.helpers.market_overlays import reset_market_overlay_state

try:
    from kaggle_environments import make
except ImportError:  # pragma: no cover
    make = None


def discover_agents() -> List[str]:
    return sorted(p.parent.name for p in SUBMISSIONS.glob("*/main.py"))


def parse_agents(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return discover_agents()
    return [a.strip() for a in spec.split(",") if a.strip()]


def _fmt_cash(value: Any) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):,.0f}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def log_progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def format_seed_line(agent: str, row: Dict[str, Any], verbose: bool) -> str:
    if row.get("status") != "ok":
        err = row.get("_error", "failed")
        return f"[{agent}] seed={row['seed']} FAILED {err} elapsed={row.get('elapsed_s', '?')}s"
    impact = _fmt_num(row.get("mean_impact"), 2)
    p95 = _fmt_num(row.get("decide_ms_p95"), 1)
    line = (
        f"[{agent}] seed={row['seed']} cash={_fmt_cash(row['terminal_cash'])} "
        f"overflow={row.get('overflow', 0)} slots={row.get('slots_burned', 0)} "
        f"impact={impact} p95={p95}ms elapsed={row.get('elapsed_s')}s"
    )
    if verbose:
        line += (
            f" rev/unit={_fmt_num(row.get('mean_revenue_per_sold_unit'), 2)} "
            f"coll={row.get('collision_holds', 0)}/{row.get('collision_releases', 0)} "
            f"carried718={row.get('terminal_carried_goods_step_718', 0)}"
        )
    return line


def format_agent_done(agent: str, agg: Dict[str, Any]) -> str:
    return (
        f"[{agent}] DONE mean={_fmt_cash(agg['mean_cash'])} "
        f"median={_fmt_cash(agg['median_cash'])} "
        f"std={_fmt_cash(agg['std_cash'])} "
        f"overflow_sum={int(agg.get('overflow_sum', 0))} "
        f"slots_sum={int(agg.get('slots_sum', 0))}"
    )


def _load_h2h_callable(agent_name: str, suffix: str):
    agent = load_agent(agent_name, module_suffix=suffix)
    if hasattr(agent, "act") and callable(getattr(agent, "act")):
        return lambda obs, configuration=None: agent.act(obs)
    if callable(agent):
        return agent
    raise TypeError(f"Agent {agent_name} is not callable")


def run_head_to_head_match(
    candidate: str,
    opponent: str,
    candidate_seat: int,
    steps: int,
    suffix: str,
) -> Dict[str, Any]:
    if make is None:
        raise RuntimeError("kaggle_environments is not installed")
    reset_market_overlay_state()
    cand = _load_h2h_callable(candidate, suffix + "c")
    opp = _load_h2h_callable(opponent, suffix + "o")
    order = [cand, opp] if candidate_seat == 0 else [opp, cand]
    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=False)
    match_steps = env.run(order)
    if not match_steps:
        return {"status": "failed", "error": "no_steps"}
    final = match_steps[-1]
    cand_idx = 0 if candidate_seat == 0 else 1
    opp_idx = 1 - cand_idx
    cand_obs = final[cand_idx].observation
    opp_obs = final[opp_idx].observation
    cand_cash = float(cand_obs["farms"][cand_idx]["money"])
    opp_cash = float(opp_obs["farms"][opp_idx]["money"])
    if cand_cash > opp_cash:
        winner = "candidate"
    elif opp_cash > cand_cash:
        winner = "opponent"
    else:
        winner = "tie"
    return {
        "status": "ok",
        "candidate_cash": cand_cash,
        "opponent_cash": opp_cash,
        "winner": winner,
        "candidate_seat": candidate_seat,
    }


def build_leaderboard(
    aggregates: Dict[str, Dict[str, Any]],
    baseline_name: Optional[str],
    baseline_by_seed: Dict[int, float],
    all_rows: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    board: List[Dict[str, Any]] = []
    for agent, agg in aggregates.items():
        agent_rows = all_rows.get(agent, [])
        win_vs = win_rate_vs_baseline(agent_rows, baseline_by_seed) if baseline_name else None
        board.append(
            {
                "agent": agent,
                "mean_cash": agg["mean_cash"],
                "median_cash": agg["median_cash"],
                "std_cash": agg["std_cash"],
                "win_vs_baseline": win_vs,
                "overflow_mean": agg["overflow_mean"],
                "slots_mean": agg["slots_mean"],
                "p95_ms_mean": agg["p95_ms_mean"],
                "episodes": agg["episodes_ok"],
                "is_baseline": baseline_name == agent,
            }
        )
    board.sort(key=lambda r: (-r["mean_cash"], r["agent"]))
    for i, row in enumerate(board, start=1):
        row["rank"] = i
    return board


def print_leaderboard_table(board: List[Dict[str, Any]], baseline_name: Optional[str]) -> None:
    headers = [
        "rank",
        "agent",
        "mean_cash",
        "median_cash",
        "std_cash",
        "win_vs_baseline",
        "overflow_mean",
        "slots_mean",
        "p95_ms_mean",
        "episodes",
    ]
    rows = []
    for row in board:
        name = row["agent"]
        if baseline_name and row.get("is_baseline"):
            name = f"{name}*"
        win = row.get("win_vs_baseline")
        win_s = f"{win * 100:.0f}%" if win is not None else "—"
        rows.append(
            [
                str(row["rank"]),
                name,
                _fmt_cash(row["mean_cash"]),
                _fmt_cash(row["median_cash"]),
                _fmt_cash(row["std_cash"]),
                win_s,
                _fmt_num(row["overflow_mean"], 1),
                _fmt_num(row["slots_mean"], 1),
                _fmt_num(row["p95_ms_mean"], 1),
                str(row["episodes"]),
            ]
        )
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))
    if baseline_name:
        print(f"\n* baseline = {baseline_name}")


def print_h2h_table(h2h: Dict[str, Dict[str, Any]]) -> None:
    headers = ["agent", "wins", "losses", "ties", "mean_candidate_cash", "mean_opp_cash"]
    rows = []
    for agent, stats in sorted(h2h.items()):
        rows.append(
            [
                agent,
                str(stats["wins"]),
                str(stats["losses"]),
                str(stats["ties"]),
                _fmt_cash(stats["mean_candidate_cash"]),
                _fmt_cash(stats["mean_opp_cash"]),
            ]
        )
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print("\nHead-to-head")
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def write_agent_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOLO_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SOLO_CSV_FIELDS})


def run_panel(args: argparse.Namespace) -> Dict[str, Any]:
    agents = parse_agents(args.agents)
    seeds = [args.start_seed + i for i in range(args.seeds)]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    aggregates: Dict[str, Dict[str, Any]] = {}
    load_failures: List[str] = []

    wall_start = perf_counter()
    total_agents = len(agents)

    for a_idx, agent in enumerate(agents, start=1):
        log_progress(f"Agent {a_idx}/{total_agents}: {agent}")
        try:
            load_agent(agent, module_suffix="_probe")
        except Exception as exc:
            log_progress(f"  SKIP load failed: {type(exc).__name__}: {exc}")
            load_failures.append(agent)
            continue

        rows: List[Dict[str, Any]] = []
        for s_idx, seed in enumerate(seeds, start=1):
            log_progress(f"  seed {s_idx}/{len(seeds)}")
            suffix = f"_s{seed}a"
            row = run_one_solo_episode(
                agent,
                agent,
                seed,
                args.steps,
                args.opponent,
                suffix,
            )
            if row.get("status") != "ok" and row.get("_traceback"):
                log_progress(f"  ERROR seed={seed}: {row.get('_error')}")
                if args.verbose:
                    log_progress(row["_traceback"].splitlines()[-1])
            log_progress(format_seed_line(agent, row, args.verbose))
            rows.append({k: v for k, v in row.items() if not k.startswith("_")})
        agg = aggregate_solo_rows(rows)
        all_rows[agent] = rows
        aggregates[agent] = agg
        log_progress(format_agent_done(agent, agg))
        write_agent_csv(output_dir / f"{agent}_seeds.csv", rows)

    baseline_name = args.baseline
    baseline_by_seed: Dict[int, float] = {}
    if baseline_name:
        if baseline_name not in all_rows:
            log_progress(f"Running baseline solo: {baseline_name}")
            base_rows = []
            for s_idx, seed in enumerate(seeds, start=1):
                log_progress(f"  baseline seed {s_idx}/{len(seeds)}")
                row = run_one_solo_episode(
                    baseline_name,
                    baseline_name,
                    seed,
                    args.steps,
                    args.opponent,
                    f"_s{seed}b",
                )
                base_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
            all_rows[baseline_name] = base_rows
            aggregates[baseline_name] = aggregate_solo_rows(base_rows)
            write_agent_csv(output_dir / f"{baseline_name}_seeds.csv", base_rows)
        for row in all_rows.get(baseline_name, []):
            if row.get("status") == "ok":
                baseline_by_seed[int(row["seed"])] = float(row["terminal_cash"])

    leaderboard = build_leaderboard(aggregates, baseline_name, baseline_by_seed, all_rows)

    h2h_results: Dict[str, Dict[str, Any]] = {}
    if args.head_to_head:
        opp = args.head_to_head
        candidates = [a for a in agents if a != opp and a not in load_failures]
        log_progress(f"Head-to-head vs {opp} ({len(candidates)} candidates)")
        for c_idx, candidate in enumerate(candidates, start=1):
            log_progress(f"  H2H {c_idx}/{len(candidates)}: {candidate}")
            stats = {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "candidate_cash": [],
                "opponent_cash": [],
            }
            for seat in (0, 1):
                log_progress(f"  H2H vs {opp} seat={seat}")
                try:
                    result = run_head_to_head_match(
                        candidate,
                        opp,
                        seat,
                        args.steps,
                        f"_h2h_{candidate}_{seat}",
                    )
                except Exception as exc:
                    log_progress(f"  H2H FAILED seat={seat}: {type(exc).__name__}: {exc}")
                    continue
                if result.get("status") != "ok":
                    continue
                stats["candidate_cash"].append(result["candidate_cash"])
                stats["opponent_cash"].append(result["opponent_cash"])
                w = result["winner"]
                if w == "candidate":
                    stats["wins"] += 1
                elif w == "opponent":
                    stats["losses"] += 1
                else:
                    stats["ties"] += 1
            if stats["candidate_cash"]:
                stats["mean_candidate_cash"] = sum(stats["candidate_cash"]) / len(
                    stats["candidate_cash"]
                )
                stats["mean_opp_cash"] = sum(stats["opponent_cash"]) / len(
                    stats["opponent_cash"]
                )
            else:
                stats["mean_candidate_cash"] = 0.0
                stats["mean_opp_cash"] = 0.0
            h2h_results[candidate] = stats

    payload = {
        "configuration": {
            "agents": agents,
            "seeds": seeds,
            "steps": args.steps,
            "opponent": args.opponent,
            "baseline": baseline_name,
            "head_to_head": args.head_to_head,
            "wall_seconds": round(perf_counter() - wall_start, 3),
        },
        "load_failures": load_failures,
        "leaderboard": leaderboard,
        "aggregates": aggregates,
        "head_to_head": h2h_results,
    }

    (output_dir / "panel_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output_dir / "panel_leaderboard.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "agent",
                "mean_cash",
                "median_cash",
                "std_cash",
                "win_vs_baseline",
                "overflow_mean",
                "slots_mean",
                "p95_ms_mean",
                "episodes",
                "is_baseline",
            ],
        )
        writer.writeheader()
        for row in leaderboard:
            writer.writerow(row)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent panel benchmark.")
    parser.add_argument(
        "--agents",
        default="all",
        help="Comma-separated submission names or 'all'",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--opponent", default="random")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "standoff" / f"panel_{date.today().strftime('%Y%m%d')}",
    )
    parser.add_argument("--baseline", default=None, help="Highlight Δ vs this agent")
    parser.add_argument(
        "--head-to-head",
        default=None,
        metavar="OPPONENT",
        help="Run two seat-swapped matches vs this opponent per candidate",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    payload = run_panel(args)
    print_leaderboard_table(payload["leaderboard"], args.baseline)
    if payload.get("head_to_head"):
        print_h2h_table(payload["head_to_head"])
    log_progress(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
