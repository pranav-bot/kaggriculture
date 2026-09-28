#!/usr/bin/env python3
"""Master evaluation: prove which agent version is superior.

Orchestrates a massive parallel round-robin via the Rust engine
(`kagg tournament` through Python's subprocess), over ALL submissions plus
static elite-ladder "tape" opponents, on a fixed seed set (default 500).
Extracts per-opponent win rates, average/worst-case terminal cash, paired
McNemar A/B statistics per pairing, latency profiles, and Elo ratings, then
emits a Markdown leaderboard recommending the final submission candidate.

Usage:
  .venv/bin/python scripts/find_best_agent.py --all
  .venv/bin/python scripts/find_best_agent.py --agents agent_final,care_mill --seeds 20
  .venv/bin/python scripts/find_best_agent.py --all --seeds 500 --workers 8 --no-tapes
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_elo_tournament import (  # noqa: E402  (reuses proven engine plumbing)
    compute_paired_mcnemar,
    discover_submissions,
    find_kagg_binary,
    update_elo_ratings,
)

LATENCY_THRESHOLD_MS = 5.0
DEFAULT_SEEDS = 500
TOP_TABLE_K = 8

# Elite ladder gauntlet: (league name, tape file). Tapes replay fixed
# single-seat elite play open-loop in Rust -- no Python-host cost.
DEFAULT_TAPES = [
    ("tape_boey_r1", "replays/tapes/112542379_opp_seat1.tape"),       # Boey, Rank 1
    ("tape_kaggledew_r1", "replays/tapes/112540075_opp_seat1.tape"),  # Kaggledew, Rank 1
    ("tape_decem_r2", "replays/tapes/112555622_opp_seat1.tape"),      # DECEM, Rank 2
    ("tape_clement_r1", "replays/tapes/112618133_opp_seat1.tape"),    # Clement Ling, Rank 1
]


def build_specs(agents: Dict[str, str],
                tapes: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """kagg agent specs: python submissions + static tape opponents."""
    specs = [{"name": n, "type": "python", "path": p} for n, p in agents.items()]
    for name, rel in tapes:
        tape = ROOT / rel
        if not tape.is_file():
            print(f"WARNING: tape missing, skipping gauntlet opponent {name} ({rel})",
                  file=sys.stderr)
            continue
        specs.append({"name": name, "type": "tape", "path": str(tape)})
    return specs


def run_mass_tournament(specs: List[Dict[str, Any]], seeds: Sequence[int],
                        kagg_binary: str, workers: int,
                        output_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Full round-robin via `kagg tournament` subprocess; parse engine outputs."""
    cfg = {
        "name": "best-agent-league",
        "schedule": "round_robin",
        "seats": "both",
        "panel": specs,
        "worlds": {"strategy": "list", "seeds": [int(s) for s in seeds]},
        "workers": int(workers),
        "on_error": "forfeit",
        "output": {"dir": output_dir, "resume": True},
        "python": {"exe": sys.executable,
                   "path": [str(ROOT / "src"), str(ROOT)],
                   "stderr": "null"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        cfg_path = fh.name
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
        env["KAGGSIM_PYTHON"] = sys.executable
        cmd = [kagg_binary, "tournament", cfg_path, "--workers", str(workers)]
        print(f"$ {' '.join(cmd)}  ({len(specs)} agents x {len(seeds)} seeds)",
              flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"kagg tournament failed ({proc.returncode}):\n"
                               f"{proc.stderr[-3000:]}")
        results_dir = None
        for line in proc.stderr.splitlines():
            if line.startswith("results: "):
                results_dir = line[len("results: "):].strip()
        if not results_dir or not Path(results_dir).exists():
            results_dir = os.path.join(output_dir, "best-agent-league")
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
    summary: Dict[str, Any] = {}
    sp = Path(results_dir) / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    rp = Path(results_dir) / "results.jsonl"
    if rp.exists():
        with open(rp, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.replace("\x00", "").strip()
                if ln.startswith("{"):
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue
    print(f"parsed {len(rows)} games from {results_dir} "
          f"(stdout {len(proc.stdout)}B / stderr {len(proc.stderr)}B)", flush=True)
    return summary, rows


def wilson_ci(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% interval for a win rate (draws count 0.5)."""
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate(rows: List[Dict[str, Any]], tape_names: Sequence[str],
              latency_ms: float = LATENCY_THRESHOLD_MS) -> Dict[str, Dict[str, Any]]:
    """Per-agent metrics: win rates, avg/worst cash, latency, per-opp records."""
    tape_set = set(tape_names)
    stats: Dict[str, Dict[str, Any]] = {}
    lat: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # name -> [sum, max]

    def entry(name: str) -> Dict[str, Any]:
        if name not in stats:
            stats[name] = {"name": name, "is_tape": name in tape_set,
                           "games": 0, "wins": 0.0, "w": 0, "d": 0, "l": 0,
                           "cash": [], "opp": defaultdict(lambda: [0.0, 0]),
                           "errors": 0}
        return stats[name]

    for g in rows:
        ags = g.get("agents") or []
        scores = g.get("scores") or [0.5] * len(ags)
        banks = g.get("banks") or [0.0] * len(ags)
        means = g.get("act_ms_mean") or []
        maxs = g.get("act_ms_max") or []
        for i, name in enumerate(ags):
            e = entry(name)
            s = float(scores[i]) if i < len(scores) else 0.5
            cash = float(banks[i]) if i < len(banks) else 0.0
            e["games"] += 1
            e["wins"] += s
            e["w"] += s == 1.0
            e["d"] += s == 0.5
            e["l"] += s == 0.0
            e["cash"].append(cash)
            if g.get("error") and i == 0:
                e["errors"] += 1
            for j, other in enumerate(ags):
                if j != i:
                    rec = e["opp"][other]
                    rec[0] += s
                    rec[1] += 1
        for i, name in enumerate(ags):
            if i < len(means) and means[i] is not None:
                lat[name][0] += float(means[i])
            if i < len(maxs) and maxs[i] is not None:
                lat[name][1] = max(lat[name][1], float(maxs[i]))

    for name, e in stats.items():
        n = e["games"]
        e["win_rate"] = e["wins"] / n if n else 0.0
        e["win_rate_ci95"] = wilson_ci(e["wins"], n)
        e["avg_cash"] = sum(e["cash"]) / n if n else 0.0
        e["worst_cash"] = min(e["cash"]) if e["cash"] else 0.0  # robustness floor
        e["best_cash"] = max(e["cash"]) if e["cash"] else 0.0
        e["mean_ms"] = lat[name][0] / n if n and name in lat else 0.0
        e["max_ms"] = lat[name][1] if name in lat else 0.0
        e["latency_flag"] = ("DANGEROUS: High Latency."
                             if e["mean_ms"] > latency_ms else "OK")
        e["opp"] = {o: {"win_rate": w / t if t else 0.0, "wins": w, "games": t}
                    for o, (w, t) in e["opp"].items()}
    return stats


def leaderboard_markdown(stats: Dict[str, Dict[str, Any]],
                         elos: Dict[str, float],
                         rows: List[Dict[str, Any]],
                         seeds: Sequence[int],
                         top_k: int = TOP_TABLE_K) -> Tuple[str, str]:
    """Elo-sorted Markdown leaderboard + top-candidate recommendation."""
    ranked = sorted(stats.values(), key=lambda e: -elos.get(e["name"], 1500.0))
    tapes = [e for e in ranked if e["is_tape"]]
    field = [e for e in ranked if not e["is_tape"]]

    lines = ["# Agent Leaderboard (round-robin, "
             f"{len(seeds)} fixed seeds, both seats)",
             "",
             "| Rank | Agent | Elo | W-D-L | Win rate (95% CI) | Avg cash | "
             "Worst cash | Mean ms | Latency |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, e in enumerate(ranked, 1):
        lo, hi = e["win_rate_ci95"]
        lines.append(
            f"| {i} | `{e['name']}` | {elos.get(e['name'], 1500.0):.0f} | "
            f"{e['w']}-{e['d']}-{e['l']} | {e['win_rate']:.3f} "
            f"({lo:.3f}..{hi:.3f}) | ${e['avg_cash']:,.0f} | "
            f"${e['worst_cash']:,.0f} | {e['mean_ms']:.2f} | {e['latency_flag']} |")

    # Elite gauntlet: candidate records vs static tapes.
    lines += ["", "## Elite-gauntlet record (vs tape opponents)", "",
              "| Agent | " + " | ".join(f"vs `{t['name']}`" for t in tapes) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in tapes) + " |")
    for e in field[:top_k * 2]:
        cells = []
        for t in tapes:
            rec = e["opp"].get(t["name"], {"win_rate": 0.0, "games": 0})
            cells.append(f"{rec['win_rate']:.2f} ({rec['games']}g)")
        lines.append(f"| `{e['name']}` | " + " | ".join(cells) + " |")

    # Pairwise McNemar among the top contenders.
    lines += ["", f"## Pairwise McNemar (top {top_k} by Elo)", "",
              "| A | B | games | better_A | better_B | p-value | "
              "mean cash diff (95% CI) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    top = [e["name"] for e in field[:top_k]]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            m = compute_paired_mcnemar(rows, top[i], top[j])
            lo, hi = m["ci95"]
            lines.append(
                f"| `{top[i]}` | `{top[j]}` | {m['games']} | {m['better_a']} | "
                f"{m['better_b']} | {m['p_value']:.4f} | "
                f"${m['mean_diff']:+,.0f} (${lo:+,.0f}..${hi:+,.0f}) |")

    # Recommendation: best Elo, latency-compliant, error-free candidate.
    pick = next((e for e in field
                 if e["latency_flag"] == "OK" and e["errors"] == 0), None)
    if pick is None:
        pick = field[0] if field else ranked[0]
        note = ("WARNING: no latency-compliant error-free candidate; "
                "recommendation is best-Elo regardless.")
    else:
        note = "Compliant pick."
    lines += ["", "## Recommendation",
              f"**Ship `{pick['name']}`** in the final `submission.tar.gz`. {note}",
              f"Elo {elos.get(pick['name'], 1500.0):.0f}, "
              f"win rate {pick['win_rate']:.3f}, avg cash ${pick['avg_cash']:,.0f}, "
              f"robustness floor ${pick['worst_cash']:,.0f}, "
              f"latency {pick['mean_ms']:.2f}ms/turn ({pick['latency_flag']})."]
    return "\n".join(lines) + "\n", pick["name"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Massive parallel agent evaluation.")
    ap.add_argument("--submissions-dir", default=str(ROOT / "submissions"))
    ap.add_argument("--agents", default=None,
                    help="Comma-separated subset (default: all).")
    ap.add_argument("--all", action="store_true", help="All submissions.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help=f"Fixed seed count (default: {DEFAULT_SEEDS}).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kagg", default=None)
    ap.add_argument("--tapes", action="store_true", default=True,
                    help="Inject elite tape opponents (default: on).")
    ap.add_argument("--no-tapes", dest="tapes", action="store_false")
    ap.add_argument("--k-factor", type=float, default=32.0)
    ap.add_argument("--latency-threshold-ms", type=float, default=LATENCY_THRESHOLD_MS)
    ap.add_argument("--output-dir", default="tournaments/full_league")
    ap.add_argument("--top-k", type=int, default=TOP_TABLE_K)
    args = ap.parse_args(argv)

    filt = [a.strip() for a in args.agents.split(",")] if args.agents else None
    if not args.all and filt is None and args.limit is None:
        ap.error("pass --all, --agents, or --limit")
    agents = discover_submissions(args.submissions_dir, filt, args.limit)
    if not agents:
        raise SystemExit("no agents discovered")
    tapes = list(DEFAULT_TAPES) if args.tapes else []
    specs = build_specs(agents, tapes)
    tape_names = [s["name"] for s in specs if s["type"] == "tape"]
    print(f"{len(agents)} candidates + {len(tape_names)} tapes: {tape_names}",
          flush=True)

    seeds = list(range(args.seeds))
    t0 = time.perf_counter()
    summary, rows = run_mass_tournament(
        specs, seeds, find_kagg_binary(args.kagg), args.workers,
        args.output_dir)
    print(f"tournament wall time: {time.perf_counter() - t0:.1f}s, "
          f"{len(rows)} games", flush=True)

    stats = aggregate(rows, tape_names, args.latency_threshold_ms)
    elos = update_elo_ratings(rows, k_factor=args.k_factor)
    md, pick = leaderboard_markdown(stats, elos, rows, seeds, args.top_k)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.md").write_text(md)
    (out / "metrics.json").write_text(json.dumps(
        {"agents": stats, "elos": elos, "pick": pick,
         "seeds": seeds, "games": len(rows)}, indent=1, default=str))
    print(md)
    print(f"wrote {out / 'leaderboard.md'} and metrics.json; ship `{pick}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
