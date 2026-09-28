#!/usr/bin/env python3
"""Master Evaluation Script & Local Elo Rating System for Kaggriculture Agents.

Runs a fast parallel round-robin tournament across submission versions using the Rust
simulation engine (`kagg tournament`), parses standard outputs to extract per-agent win rates,
average terminal cash, and paired McNemar A/B test confidence intervals, computes updated
Elo ratings, and enforces a strict 5.0ms/turn latency profiling gate.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("elo_tournament")

# ANSI Color formatting
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class AgentResult:
    """Summary metrics for one agent in the tournament."""
    name: str
    path: str
    elo: float = 1500.0
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_rate: float = 0.0
    win_rate_ci95: Tuple[float, float] = (0.0, 0.0)
    avg_terminal_cash: float = 0.0
    mean_margin: float = 0.0
    mean_act_ms: float = 0.0
    max_act_ms: float = 0.0
    errors: int = 0
    forfeits: int = 0
    latency_violation: bool = False


def find_kagg_binary(override_path: Optional[str] = None) -> str:
    """Locate the native Rust kagg binary."""
    candidates = [
        override_path,
        os.environ.get("KAGG_PATH"),
        str(ROOT / "kaggriculture-simulation" / "src-rust" / "target" / "release" / "kagg"),
        str(ROOT / "kagg"),
        "kagg",
    ]
    for c in candidates:
        if c and (Path(c).is_file() or shutil_which(c)):
            return str(c)
    raise FileNotFoundError("Could not find native Rust 'kagg' executable. Run 'cargo build --release' in kaggriculture-simulation/src-rust.")


def shutil_which(pgm: str) -> bool:
    import shutil
    return shutil.which(pgm) is not None


def discover_submissions(
    submissions_dir: Path | str,
    filter_agents: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """Scan submissions directory and return mapping of {agent_name: main_py_path}."""
    sub_dir = Path(submissions_dir)
    pattern = str(sub_dir / "*" / "main.py")
    files = sorted(glob.glob(pattern))

    found: Dict[str, str] = {}
    for f in files:
        name = Path(f).parent.name
        found[name] = str(Path(f).resolve())

    if filter_agents:
        selected_names = set(filter_agents)
        found = {k: v for k, v in found.items() if k in selected_names or any(re.search(pat, k) for pat in selected_names)}

    if limit and len(found) > limit:
        # If no explicit filter was given, prioritize prominent agents
        priority_keys = ["agent_final", "care_mill", "shop_opportunist", "two_team_grandmaster", "compound_expansion", "melon_rusher"]
        selected = {}
        for pk in priority_keys:
            if pk in found:
                selected[pk] = found[pk]
        for k, v in found.items():
            if len(selected) >= limit:
                break
            selected[k] = v
        found = selected

    return found


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum binomial tail P(X <= k) with p = 0.5
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def compute_paired_mcnemar(
    games: List[Dict[str, Any]],
    agent_a: str,
    agent_b: str,
) -> Dict[str, Any]:
    """Compute paired McNemar test and margin CI between agent_a and agent_b."""
    relevant_games = []
    for g in games:
        ags = g.get("agents") or []
        if agent_a in ags and agent_b in ags and len(ags) == 2:
            relevant_games.append(g)

    if not relevant_games:
        return {"games": 0, "better_a": 0, "better_b": 0, "p_value": 1.0, "ci95": (0.0, 0.0), "mean_diff": 0.0}

    wa = 0
    wb = 0
    diffs = []

    for g in relevant_games:
        idx_a = g["agents"].index(agent_a)
        idx_b = g["agents"].index(agent_b)
        banks = g.get("banks") or [0, 0]
        cash_a = float(banks[idx_a])
        cash_b = float(banks[idx_b])
        scores = g.get("scores") or [0, 0]
        score_a = float(scores[idx_a])
        score_b = float(scores[idx_b])

        if score_a > score_b:
            wa += 1
        elif score_b > score_a:
            wb += 1

        diffs.append(cash_a - cash_b)

    n = len(diffs)
    p_val = mcnemar_exact(wa, wb)
    mean_d = sum(diffs) / n if n > 0 else 0.0
    if n > 1:
        variance = sum((x - mean_d) ** 2 for x in diffs) / (n - 1)
        stderr = math.sqrt(variance / n)
        ci95 = (mean_d - 1.96 * stderr, mean_d + 1.96 * stderr)
    else:
        ci95 = (mean_d, mean_d)

    return {
        "games": n,
        "better_a": wa,
        "better_b": wb,
        "p_value": p_val,
        "mean_diff": mean_d,
        "ci95": ci95,
    }


def update_elo_ratings(
    games: List[Dict[str, Any]],
    initial_ratings: Optional[Dict[str, float]] = None,
    k_factor: float = 32.0,
    iterations: int = 3,
) -> Dict[str, float]:
    """Calculate Elo ratings from game results using iterative logistic updates."""
    ratings = dict(initial_ratings) if initial_ratings else {}
    for g in games:
        for a in g.get("agents", []):
            if a not in ratings:
                ratings[a] = 1500.0

    # Iterative passes over tournament match history for rating stabilization
    for _ in range(iterations):
        for g in games:
            ags = g.get("agents") or []
            scores = g.get("scores") or [0.5, 0.5]
            if len(ags) < 2:
                continue
            a0, a1 = ags[0], ags[1]
            s0, s1 = float(scores[0]), float(scores[1])

            r0 = ratings[a0]
            r1 = ratings[a1]

            e0 = 1.0 / (1.0 + 10.0 ** ((r1 - r0) / 400.0))
            e1 = 1.0 - e0

            ratings[a0] = r0 + k_factor * (s0 - e0)
            ratings[a1] = r1 + k_factor * (s1 - e1)

    return ratings


def run_kagg_tournament(
    agents: Dict[str, str],
    seeds: Sequence[int],
    kagg_binary: str,
    workers: int = 4,
    output_dir: str = "tournaments/elo_tournament",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Execute round_robin tournament via kagg binary subprocess."""
    panel_specs = [{"name": name, "type": "python", "path": path} for name, path in agents.items()]

    cfg = {
        "name": "elo-round-robin",
        "schedule": "round_robin",
        "seats": "both",
        "panel": panel_specs,
        "worlds": {
            "strategy": "list",
            "seeds": list(seeds),
        },
        "workers": int(workers),
        "on_error": "forfeit",
        "output": {
            "dir": output_dir,
            "resume": False,
        },
        "python": {
            "exe": sys.executable,
            "path": [str(ROOT / "src"), str(ROOT)],
        },
    }

    temp_cfg = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        json.dump(cfg, temp_cfg, indent=2)
        temp_cfg.flush()
        temp_cfg_path = temp_cfg.name
    finally:
        temp_cfg.close()

    target_out = Path(output_dir) / "elo-round-robin"
    if target_out.exists():
        import shutil
        shutil.rmtree(target_out, ignore_errors=True)

    cmd = [kagg_binary, "tournament", temp_cfg_path, "--workers", str(workers)]
    logger.info("Executing: %s", " ".join(cmd))

    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)] + [run_env.get("PYTHONPATH", "")])
    run_env["KAGGSIM_PYTHON"] = sys.executable

    start_t = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=run_env)
    elapsed = time.perf_counter() - start_t

    try:
        os.unlink(temp_cfg_path)
    except OSError:
        pass

    if proc.returncode != 0:
        raise RuntimeError(f"kagg tournament failed ({proc.returncode}):\n{proc.stderr}")

    stdout_text = proc.stdout
    stderr_text = proc.stderr

    # Locate output directory from stderr
    results_dir = None
    for line in stderr_text.splitlines():
        if line.startswith("results: "):
            results_dir = line[len("results: "):].strip()

    if not results_dir or not Path(results_dir).exists():
        results_dir = os.path.join(output_dir, "elo-round-robin")

    summary_file = Path(results_dir) / "summary.json"
    results_file = Path(results_dir) / "results.jsonl"

    summary: Dict[str, Any] = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as fp:
            summary = json.load(fp)

    game_rows: List[Dict[str, Any]] = []
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8", errors="replace") as fp:
            for ln in fp:
                ln = ln.replace("\x00", "").strip()
                if ln and ln.startswith("{"):
                    try:
                        game_rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue

    return summary, game_rows, stdout_text


def parse_stdout_standings(stdout_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse Markdown standings table from kagg tournament stdout."""
    standings: Dict[str, Dict[str, Any]] = {}
    lines = stdout_text.splitlines()
    in_table = False

    for line in lines:
        if "| #" in line and "agent" in line and "W/D/L" in line:
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 9 and parts[0] != "#" and not parts[0].startswith("---"):
                agent_name = parts[1]
                try:
                    games = int(parts[2])
                    wdl = parts[3].split("/")
                    wins = int(wdl[0])
                    draws = int(wdl[1]) if len(wdl) > 1 else 0
                    losses = int(wdl[2]) if len(wdl) > 2 else 0
                    score = float(parts[4])
                    ci_str = parts[5]
                    ci_parts = [float(x.strip()) for x in ci_str.split("..")] if ".." in ci_str else [score, score]
                    margin = float(parts[6].replace(",", ""))
                    errors = int(parts[7])
                    ms_parts = parts[8].split("/")
                    mean_ms = float(ms_parts[0].strip()) if ms_parts[0].strip() not in ("-", "") else 0.0
                    max_ms = float(ms_parts[1].strip()) if len(ms_parts) > 1 and ms_parts[1].strip() not in ("-", "") else 0.0

                    standings[agent_name] = {
                        "games": games,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "score": score,
                        "ci95": (ci_parts[0], ci_parts[1]),
                        "mean_margin": margin,
                        "errors": errors,
                        "mean_act_ms": mean_ms,
                        "max_act_ms": max_ms,
                    }
                except (ValueError, IndexError):
                    continue

    return standings


def print_leaderboard(
    agent_results: List[AgentResult],
    top_agent: str,
    mcnemar_stats: Dict[str, Dict[str, Any]],
    latency_threshold_ms: float = 5.0,
) -> None:
    """Print beautifully formatted console output with Elo ratings and Profiling Gate."""
    print("\n" + "=" * 90)
    print(f"{BOLD}{CYAN}KAGGRICULTURE AGENT ELO LEADERBOARD & TOURNAMENT BENCHMARK{RESET}")
    print("=" * 90)

    # Header
    print(f"{BOLD}{'Rank':<5} {'Agent':<24} {'Elo':<8} {'Win Rate':<10} {'W/D/L':<12} {'Avg Cash':<12} {'ms/turn':<10} {'Profiling Gate':<18}{RESET}")
    print("-" * 90)

    violations: List[Tuple[str, float]] = []

    for rank, res in enumerate(agent_results, start=1):
        win_pct = f"{res.win_rate * 100:.1f}%"
        wdl = f"{res.wins}/{res.draws}/{res.losses}"
        cash_str = f"${res.avg_terminal_cash:,.0f}"
        ms_str = f"{res.mean_act_ms:.2f}ms"

        if res.latency_violation:
            gate_status = f"{RED}{BOLD}LATENCY VIOLATION{RESET}"
            agent_display = f"{RED}{BOLD}{res.name}{RESET}"
            ms_display = f"{RED}{BOLD}{ms_str}{RESET}"
            violations.append((res.name, res.mean_act_ms))
        else:
            gate_status = f"{GREEN}PASS (<{latency_threshold_ms}ms){RESET}"
            agent_display = f"{BOLD}{res.name}{RESET}" if rank <= 3 else res.name
            ms_display = ms_str

        print(f"{rank:<5} {agent_display:<33} {res.elo:<8.1f} {win_pct:<10} {wdl:<12} {cash_str:<12} {ms_display:<19} {gate_status}")

    print("=" * 90)

    # Profiling Gate Summary
    if violations:
        print(f"\n{RED}{BOLD}PROFILING GATE: {len(violations)} LATENCY VIOLATIONS DETECTED!{RESET}")
        for ag, ms in violations:
            print(f"  {RED}✖ Agent '{ag}' averaged {ms:.2f}ms per turn (strictly exceeds {latency_threshold_ms:.1f}ms limit){RESET}")
        print(f"{YELLOW}Warning: High latency risks consuming the 60s Kaggle overage bank.{RESET}\n")
    else:
        print(f"\n{GREEN}{BOLD}PROFILING GATE: ALL SUBMISSIONS COMPLIANT (< {latency_threshold_ms:.1f}ms/turn avg){RESET}\n")

    # McNemar A/B Confidence Intervals against Top Ranked Agent
    if top_agent and mcnemar_stats:
        print("-" * 90)
        print(f"{BOLD}PAIRED McNEMAR A/B TEST: VS TOP AGENT '{top_agent}'{RESET}")
        print(f"{'Opponent':<24} {'Games':<8} {'W - L vs Top':<14} {'McNemar p-value':<18} {'Cash Delta (95% CI)':<26}")
        print("-" * 90)
        for opp, stats in mcnemar_stats.items():
            if opp == top_agent:
                continue
            p_val = stats["p_value"]
            sig_str = f"{p_val:.4f}" + (" *" if p_val < 0.05 else " (ns)")
            record_str = f"{stats['better_a']} - {stats['better_b']}"
            ci = stats["ci95"]
            ci_str = f"${ci[0]:+,.0f} .. ${ci[1]:+,.0f}"
            print(f"{opp:<24} {stats['games']:<8} {record_str:<14} {sig_str:<18} {ci_str:<26}")
        print("=" * 90)


def run_elo_tournament(
    submissions_dir: Path | str = ROOT / "submissions",
    filter_agents: Optional[Sequence[str]] = None,
    num_seeds: int = 100,
    workers: int = 4,
    kagg_binary: Optional[str] = None,
    leaderboard_file: Path | str = ROOT / "data" / "elo_leaderboard.json",
    latency_threshold_ms: float = 5.0,
    k_factor: float = 32.0,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute round-robin Elo tournament and update leaderboard."""
    exe = find_kagg_binary(kagg_binary)
    agents = discover_submissions(submissions_dir, filter_agents=filter_agents, limit=limit)

    if len(agents) < 2:
        raise ValueError(f"Need at least 2 submissions for a round-robin tournament, found {len(agents)} in {submissions_dir}")

    seeds = list(range(num_seeds))
    logger.info("Starting Elo Tournament across %d agents over %d fixed seeds (%d matches planned)...",
                len(agents), len(seeds), len(agents) * (len(agents) - 1) * len(seeds))

    summary, game_rows, stdout_text = run_kagg_tournament(
        agents=agents,
        seeds=seeds,
        kagg_binary=exe,
        workers=workers,
    )

    # Parse stdout standings table
    table_standings = parse_stdout_standings(stdout_text)

    # Compute terminal cash aggregates per agent from results.jsonl
    cash_by_agent: Dict[str, List[float]] = defaultdict(list)
    act_ms_by_agent: Dict[str, List[float]] = defaultdict(list)
    max_ms_by_agent: Dict[str, float] = defaultdict(float)

    for g in game_rows:
        ags = g.get("agents") or []
        banks = g.get("banks") or [0, 0]
        act_means = g.get("act_ms_mean") or [0.0, 0.0]
        act_maxs = g.get("act_ms_max") or [0.0, 0.0]

        for s_idx, name in enumerate(ags):
            cash_by_agent[name].append(float(banks[s_idx]))
            act_ms_by_agent[name].append(float(act_means[s_idx]))
            if float(act_maxs[s_idx]) > max_ms_by_agent[name]:
                max_ms_by_agent[name] = float(act_maxs[s_idx])

    # Load existing Elo ratings if file exists
    lb_path = Path(leaderboard_file)
    existing_ratings: Dict[str, float] = {}
    if lb_path.exists():
        try:
            with open(lb_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for entry in data.get("leaderboard", []):
                    existing_ratings[entry["name"]] = float(entry["elo"])
        except Exception:
            pass

    # Compute Elo ratings
    updated_elos = update_elo_ratings(game_rows, initial_ratings=existing_ratings, k_factor=k_factor)

    # Build AgentResult objects
    results: List[AgentResult] = []
    for name, path in agents.items():
        st = table_standings.get(name, {})
        cash_list = cash_by_agent.get(name, [])
        ms_list = act_ms_by_agent.get(name, [])

        avg_cash = sum(cash_list) / len(cash_list) if cash_list else 0.0
        mean_ms = sum(ms_list) / len(ms_list) if ms_list else float(st.get("mean_act_ms", 0.0))
        max_ms = max_ms_by_agent.get(name, float(st.get("max_act_ms", 0.0)))

        games = st.get("games", len(cash_list))
        wins = st.get("wins", 0)
        draws = st.get("draws", 0)
        losses = st.get("losses", 0)
        win_rate = st.get("score", wins / games if games > 0 else 0.0)

        is_violation = (mean_ms > latency_threshold_ms)

        res = AgentResult(
            name=name,
            path=path,
            elo=updated_elos.get(name, 1500.0),
            games=games,
            wins=wins,
            draws=draws,
            losses=losses,
            win_rate=win_rate,
            win_rate_ci95=st.get("ci95", (0.0, 0.0)),
            avg_terminal_cash=avg_cash,
            mean_margin=float(st.get("mean_margin", 0.0)),
            mean_act_ms=mean_ms,
            max_act_ms=max_ms,
            errors=st.get("errors", 0),
            latency_violation=is_violation,
        )
        results.append(res)

    # Sort Leaderboard by Elo descending
    results.sort(key=lambda r: r.elo, reverse=True)
    top_agent = results[0].name if results else ""

    # Compute paired McNemar test for all agents vs top agent
    mcnemar_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r.name != top_agent:
            mcnemar_stats[r.name] = compute_paired_mcnemar(game_rows, top_agent, r.name)

    # Print Leaderboard & Console Output
    print_leaderboard(results, top_agent, mcnemar_stats, latency_threshold_ms=latency_threshold_ms)

    # Save to JSON leaderboard file
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_payload = {
        "timestamp": time.time(),
        "seeds": len(seeds),
        "total_games": len(game_rows),
        "top_agent": top_agent,
        "latency_threshold_ms": latency_threshold_ms,
        "leaderboard": [asdict(r) for r in results],
        "mcnemar_vs_top": mcnemar_stats,
    }
    with open(lb_path, "w", encoding="utf-8") as fp:
        json.dump(leaderboard_payload, fp, indent=2)

    logger.info("Saved Leaderboard to %s", lb_path)
    return leaderboard_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local round-robin Elo tournament across submissions via Rust kagg tournament."
    )
    parser.add_argument(
        "--submissions-dir",
        type=str,
        default=str(ROOT / "submissions"),
        help="Directory containing agent subdirectories with main.py.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated list of agent names to include (e.g. 'agent_final,care_mill,shop_opportunist').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all discovered submissions in the tournament.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=100,
        help="Number of fixed seeds for round-robin evaluation (default: 100).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker count for kagg tournament (default: 4).",
    )
    parser.add_argument(
        "--kagg",
        type=str,
        default=None,
        help="Path to kagg binary executable.",
    )
    parser.add_argument(
        "--leaderboard-file",
        type=str,
        default=str(ROOT / "data" / "elo_leaderboard.json"),
        help="Destination JSON path for Elo ratings and match stats.",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=5.0,
        help="Maximum allowed ms/turn average before raising Latency Violation (default: 5.0ms).",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        help="Elo K-factor for rating updates (default: 32.0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of agents to include (default: None).",
    )

    args = parser.parse_args()

    filter_list = None
    if args.agents:
        filter_list = [a.strip() for a in args.agents.split(",") if a.strip()]
    elif not args.all and args.limit is None:
        # Default to a representative set of top/benchmark submissions if neither --all nor --agents specified
        filter_list = ["agent_final", "care_mill", "shop_opportunist", "melon_rusher", "compound_expansion"]

    run_elo_tournament(
        submissions_dir=args.submissions_dir,
        filter_agents=filter_list,
        num_seeds=args.seeds,
        workers=args.workers,
        kagg_binary=args.kagg,
        leaderboard_file=args.leaderboard_file,
        latency_threshold_ms=args.latency_threshold_ms,
        k_factor=args.k_factor,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
