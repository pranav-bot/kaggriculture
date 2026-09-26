#!/usr/bin/env python3
"""Full-season Kaggriculture standoff runner.

The selected submission is played against every other submission for 720 turns.
Powered by `kaggriculture-simulation` (Rust engine) with automatic fallback.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sim_engine import FastSimulation, is_kagg_available

SUBMISSIONS = ROOT / "submissions"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results.json"
TURN_LIMIT_SECONDS = 1.0
OVERAGE_BANK_SECONDS = 60.0
SEASON_STEPS = 720


def discover_agents() -> Dict[str, Path]:
    return {
        path.parent.name: path
        for path in sorted(SUBMISSIONS.glob("*/main.py"))
    }


def resolve_agent(name_or_path: str, agents: Dict[str, Path]) -> tuple[str, Path]:
    path = Path(name_or_path)
    if path.is_file():
        return path.parent.name, path.resolve()
    if name_or_path in agents:
        return name_or_path, agents[name_or_path]
    raise ValueError(
        f"Unknown agent {name_or_path!r}. Available agents: {', '.join(sorted(agents))}"
    )


def load_callable(path: Path, unique_name: str) -> Callable[[Dict[str, Any]], Any]:
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise TypeError(f"{path} does not expose a callable agent(obs)")
    return agent


class TimedAgent:
    """Measure turn latency and enforce the local overage budget."""

    def __init__(self, agent: Callable[[Dict[str, Any]], Any], label: str):
        self.agent = agent
        self.label = label
        self.turns = 0
        self.total_seconds = 0.0
        self.max_seconds = 0.0
        self.overage_seconds = 0.0
        self.timeout = False

    def __call__(self, obs: Dict[str, Any], _configuration: Any = None) -> Any:
        started = time.perf_counter()
        try:
            return self.agent(obs)
        finally:
            elapsed = time.perf_counter() - started
            self.turns += 1
            self.total_seconds += elapsed
            self.max_seconds = max(self.max_seconds, elapsed)
            self.overage_seconds += max(0.0, elapsed - TURN_LIMIT_SECONDS)
            if self.overage_seconds > OVERAGE_BANK_SECONDS:
                self.timeout = True
                raise TimeoutError(
                    f"{self.label} exhausted the 60-second overage bank "
                    f"after turn {self.turns}"
                )

    def metrics(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "total_seconds": round(self.total_seconds, 6),
            "mean_ms": round(self.total_seconds / max(1, self.turns) * 1000, 3),
            "max_ms": round(self.max_seconds * 1000, 3),
            "overage_seconds": round(self.overage_seconds, 6),
            "timed_out": self.timeout,
            "within_budget": not self.timeout and self.overage_seconds <= OVERAGE_BANK_SECONDS,
        }


def run_match(
    first_name: str,
    first_path: Path,
    second_name: str,
    second_path: Path,
    match_index: int,
    sim: FastSimulation | None = None,
) -> Dict[str, Any]:
    first = TimedAgent(load_callable(first_path, f"standoff_{match_index}_first"), first_name)
    second = TimedAgent(load_callable(second_path, f"standoff_{match_index}_second"), second_name)
    record: Dict[str, Any] = {
        "first": first_name,
        "second": second_name,
        "steps": 0,
        "status": "running",
    }
    try:
        if sim is not None:
            first_cash, second_cash, final_st = sim.run_match(first, second, seed=match_index)
            record["steps"] = int(final_st.get("step", SEASON_STEPS))
            record.update({
                "status": "completed",
                "first_cash": first_cash,
                "second_cash": second_cash,
                "winner": (
                    first_name if first_cash > second_cash
                    else second_name if second_cash > first_cash
                    else "tie"
                ),
            })
        else:
            from kaggle_environments import make
            env = make(
                "kaggriculture",
                configuration={"episodeSteps": SEASON_STEPS, "actTimeout": TURN_LIMIT_SECONDS, "seed": match_index},
                debug=False,
            )
            steps = env.run([first, second])
            record["steps"] = len(steps or [])
            if not steps:
                record["status"] = "no_steps"
            else:
                final = steps[-1]
                first_cash = float(final[0].observation["farms"][0]["money"])
                second_cash = float(final[1].observation["farms"][1]["money"])
                record.update({
                    "status": "completed",
                    "first_cash": first_cash,
                    "second_cash": second_cash,
                    "winner": (
                        first_name if first_cash > second_cash
                        else second_name if second_cash > first_cash
                        else "tie"
                    ),
                })
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    record["first_timing"] = first.metrics()
    record["second_timing"] = second.metrics()
    return record


def summarize(selected: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Dict[str, Any]] = {}
    for match in matches:
        opponent = match["second"] if match["first"] == selected else match["first"]
        row = summary.setdefault(opponent, {
            "matches": 0, "wins": 0, "losses": 0, "ties": 0,
            "selected_cash": [], "opponent_cash": [], "failed": 0,
            "max_overage_seconds": 0.0, "max_ms": 0.0,
        })
        row["matches"] += 1
        if match.get("status") != "completed":
            row["failed"] += 1
        elif match["winner"] == "tie":
            row["ties"] += 1
        elif match["winner"] == selected:
            row["wins"] += 1
        else:
            row["losses"] += 1
        if match["first"] == selected:
            selected_timing, opponent_timing = match["first_timing"], match["second_timing"]
            selected_cash = match.get("first_cash")
            opponent_cash = match.get("second_cash")
        else:
            selected_timing, opponent_timing = match["second_timing"], match["first_timing"]
            selected_cash = match.get("second_cash")
            opponent_cash = match.get("first_cash")
        if selected_cash is not None:
            row["selected_cash"].append(selected_cash)
            row["opponent_cash"].append(opponent_cash)
        row["max_overage_seconds"] = max(
            row["max_overage_seconds"],
            selected_timing["overage_seconds"],
            opponent_timing["overage_seconds"],
        )
        row["max_ms"] = max(row["max_ms"], selected_timing["max_ms"], opponent_timing["max_ms"])

    for row in summary.values():
        row["selected_cash_mean"] = round(
            sum(row.pop("selected_cash")) / max(1, row["matches"]), 2
        )
        row["opponent_cash_mean"] = round(
            sum(row.pop("opponent_cash")) / max(1, row["matches"]), 2
        )
        row["max_overage_seconds"] = round(row["max_overage_seconds"], 6)
        row["max_ms"] = round(row["max_ms"], 3)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full-season Kaggriculture standoff.")
    parser.add_argument("--agent", "-a", default="two_team_grandmaster",
                        help="Selected agent name or path (default: two_team_grandmaster).")
    parser.add_argument("--output", "--mode", choices=("summary", "detailed"), default="summary",
                        help="Console output format.")
    parser.add_argument("--results", "-o", type=Path, default=DEFAULT_RESULTS,
                        help="JSON file for persisted standoff results.")
    parser.add_argument("--no-swap", action="store_true",
                        help="Run one match per opponent instead of both starting positions.")
    parser.add_argument("--official", action="store_true",
                        help="Force official Python engine instead of fast Rust engine.")
    args = parser.parse_args()

    agents = discover_agents()
    selected_name, selected_path = resolve_agent(args.agent, agents)
    opponents = [(name, path) for name, path in agents.items() if name != selected_name]
    matches: List[Dict[str, Any]] = []
    match_index = 0

    use_fast = is_kagg_available() and not args.official
    engine_name = "kaggriculture-simulation (Rust)" if use_fast else "kaggle_environments (Python)"
    print(f"Engine: {engine_name} | Running standoff for {selected_name} against {len(opponents)} opponents...")

    t_start = time.perf_counter()

    def run_tournament(sim=None):
        nonlocal match_index
        for opponent_name, opponent_path in opponents:
            matches.append(run_match(selected_name, selected_path, opponent_name, opponent_path, match_index, sim=sim))
            match_index += 1
            if not args.no_swap:
                matches.append(run_match(opponent_name, opponent_path, selected_name, selected_path, match_index, sim=sim))
                match_index += 1

    if use_fast:
        with FastSimulation() as sim:
            run_tournament(sim)
    else:
        run_tournament(None)

    elapsed = time.perf_counter() - t_start

    payload = {
        "selected_agent": selected_name,
        "configuration": {
            "episode_steps": SEASON_STEPS,
            "act_timeout_seconds": TURN_LIMIT_SECONDS,
            "overage_bank_seconds": OVERAGE_BANK_SECONDS,
            "swapped_sides": not args.no_swap,
            "engine": engine_name,
            "elapsed_seconds": round(elapsed, 2),
        },
        "summary": summarize(selected_name, matches),
        "matches": matches,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2) + "\n")

    if args.output == "detailed":
        print(json.dumps(payload, indent=2))
        return
    print(f"Standoff: {selected_name} vs {len(opponents)} agents ({len(matches)} full 720-turn matches) | Time: {elapsed:.2f}s")
    for opponent, row in sorted(payload["summary"].items()):
        print(
            f"{selected_name} vs {opponent}: "
            f"{row['wins']}W/{row['losses']}L/{row['ties']}T, "
            f"cash {row['selected_cash_mean']:.0f}-{row['opponent_cash_mean']:.0f}, "
            f"max {row['max_ms']:.1f}ms, overage {row['max_overage_seconds']:.3f}s"
        )
    print(f"Results saved to {args.results}")


if __name__ == "__main__":
    main()
