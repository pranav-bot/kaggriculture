#!/usr/bin/env python3
"""Optuna search for the beam-search controller parameters.

The Rust tournament is deliberately run one seed at a time.  Besides making
the objective easy to test, this gives Optuna a useful checkpoint at every
seed so ``MedianPruner`` can stop bankrupt or clearly regressing trials.
Optuna remains an optional dependency: importing this module does not require
it, while running a search produces an actionable installation message.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SEEDS: tuple[int, ...] = tuple(range(30))
DAY15_LAG_FRACTION = 0.10
BANKRUPTCY_CASH = 0.0


def require_optuna() -> Any:
    """Import Optuna lazily and explain how to install the optional extra."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Optuna is required for beam tuning. Install it with "
            "`python -m pip install optuna` (or add it to the project environment)."
        ) from exc
    return optuna


def suggest_parameters(trial: Any) -> dict[str, float | int]:
    """Suggest one complete, serialisable tuning configuration."""
    return {
        "beam_width": trial.suggest_int("beam_width", 1, 5),
        "iql_weight": trial.suggest_float("iql_weight", 0.0, 1.0),
        "heuristic_weight": trial.suggest_float("heuristic_weight", 0.0, 1.0),
        "market_panic_threshold": trial.suggest_float(
            "market_panic_threshold", 0.7, 0.95
        ),
    }


def make_config(
    parameters: Mapping[str, float | int],
    *,
    seed: int,
    output_dir: Path,
    candidate_path: Path,
    baseline_path: Path,
    workers: int,
) -> dict[str, Any]:
    """Build the Rust tournament config for one fixed seed."""
    return {
        "name": f"beam-tune-{seed}",
        "schedule": "gauntlet",
        "seats": "both",
        "candidate": {"name": "tuned", "type": "python", "path": str(candidate_path)},
        "panel": [
            {"name": "agent_final", "type": "python", "path": str(baseline_path)}
        ],
        # `seeds` is the Rust tournament schema for list strategy.
        "worlds": {"strategy": "list", "seeds": [int(seed)]},
        "workers": int(workers),
        "on_error": "forfeit",
        "output": {"dir": str(output_dir), "resume": False},
        "python": {"exe": sys.executable, "path": [str(ROOT / "src"), str(ROOT)]},
        # The agent process inherits this path.  Keeping the tuning values in
        # the config also makes each trial reproducible and inspectable.
        "tuning": dict(parameters),
        "samples": {
            "features": ["none"],
            "labels": ["bank"],
            "seats": "all",
            "stride": 1,
            "steps": [360, 360],  # day 15 (24 turns/day)
        },
        "sinks": [
            {
                "type": "jsonl",
                "path": str(output_dir / "samples.jsonl"),
                "records": ["sample"],
            }
        ],
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Rust tournament did not write {path}")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_seed_result(
    result_path: Path,
    sample_path: Path,
    *,
    candidate: str = "tuned",
    baseline: str = "agent_final",
) -> tuple[float, float, float]:
    """Return candidate terminal mean, candidate day-15 mean, baseline day-15 mean."""
    games = [row for row in _rows(result_path) if row.get("record", "game") == "game"]
    candidate_cash = [
        float(row["banks"][seat])
        for row in games
        for seat, name in enumerate(row.get("agents", []))
        if name == candidate and row.get("banks") is not None
    ]
    if not candidate_cash:
        raise RuntimeError(f"No terminal cash found for {candidate!r} in {result_path}")

    day15: dict[str, list[float]] = {candidate: [], baseline: []}
    if sample_path.exists():
        for row in _rows(sample_path):
            if row.get("record") != "sample" or int(row.get("day", -1)) != 15:
                continue
            labels = row.get("labels", {})
            agent = row.get("agent") or row.get("agent_name")
            if agent in day15 and "bank" in labels:
                day15[agent].append(float(labels["bank"]))
    # Missing samples should not silently trigger pruning; terminal scoring
    # remains valid for older kagg binaries that do not support sinks.
    candidate_day15 = mean(day15[candidate]) if day15[candidate] else float("nan")
    baseline_day15 = mean(day15[baseline]) if day15[baseline] else float("nan")
    return mean(candidate_cash), candidate_day15, baseline_day15


def should_prune(
    *,
    terminal_cash: float,
    candidate_day15: float,
    baseline_day15: float,
    lag_fraction: float = DAY15_LAG_FRACTION,
) -> bool:
    """Whether a trial is bankrupt or materially behind at day 15."""
    if terminal_cash <= BANKRUPTCY_CASH:
        return True
    return (
        candidate_day15 == candidate_day15
        and baseline_day15 == baseline_day15
        and candidate_day15 < baseline_day15 * (1.0 - lag_fraction)
    )


def find_kagg_binary() -> str:
    candidates = [
        os.environ.get("KAGG_PATH"),
        str(ROOT / "kaggriculture-simulation/src-rust/target/release/kagg"),
        "kagg",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate).is_file() or __import__("shutil").which(candidate)):
            return candidate
    raise FileNotFoundError("Rust kagg binary not found; build kagg or set KAGG_PATH")


def objective(
    trial: Any,
    *,
    kagg_binary: str | None = None,
    candidate_path: Path = ROOT / "submissions/agent_final/main.py",
    baseline_path: Path = ROOT / "submissions/agent_final/main.py",
    workers: int = 2,
    runner: Callable[..., None] | None = None,
) -> float:
    """Evaluate exactly the 30 fixed seeds and maximise average terminal cash."""
    parameters = suggest_parameters(trial)
    kagg = kagg_binary or find_kagg_binary()
    runner = runner or subprocess.run
    (ROOT / ".optuna-tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="optuna-beam-", dir=ROOT / ".optuna-tmp"
    ) as temp_root:
        root = Path(temp_root)
        scores: list[float] = []
        for index, seed in enumerate(SEEDS):
            out_dir = root / f"seed-{seed}"
            out_dir.mkdir(parents=True)
            cfg = make_config(
                parameters,
                seed=seed,
                output_dir=out_dir,
                candidate_path=candidate_path,
                baseline_path=baseline_path,
                workers=workers,
            )
            cfg_path = root / f"config-{seed}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            env = os.environ.copy()
            env["KAGGRICULTURE_TUNE_CONFIG"] = str(cfg_path)
            completed = runner(
                [kagg, "tournament", str(cfg_path), "--workers", str(workers), "--quiet"],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"kagg tournament failed for seed {seed}: {completed.stderr}"
                )
            sub_dir = out_dir / f"beam-tune-{seed}"
            res_path = (sub_dir / "results.jsonl") if (sub_dir / "results.jsonl").is_file() else (out_dir / "results.jsonl")
            sample_path = (sub_dir / "samples.jsonl") if (sub_dir / "samples.jsonl").is_file() else (out_dir / "samples.jsonl")
            terminal, day15, baseline15 = parse_seed_result(res_path, sample_path)
            scores.append(terminal)
            trial.report(mean(scores), index)
            if trial.should_prune() or should_prune(
                terminal_cash=terminal,
                candidate_day15=day15,
                baseline_day15=baseline15,
            ):
                optuna = require_optuna()
                raise optuna.TrialPruned()
    return mean(scores)


def main() -> None:
    optuna = require_optuna()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--study", default="beam-width-tuning")
    args = parser.parse_args()
    (ROOT / ".optuna-tmp").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=args.study,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda trial: objective(trial, workers=args.workers), n_trials=args.trials)
    print(json.dumps({"value": study.best_value, "params": study.best_params}, indent=2))


if __name__ == "__main__":
    main()
