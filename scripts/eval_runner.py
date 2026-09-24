#!/usr/bin/env python3
"""Seed-matched candidate versus replay/surrogate evaluation.

The replay opponent uses the recorded action at ``obs.step + 1``. Missing or
malformed frames use a deterministic selling surrogate, then a safe PASS.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_utils import load_agent
from eval_metrics import EvaluationResult, write_markdown

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _step(obs: Any) -> int:
    value = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    return int(value)


def _observation(frame: Any) -> dict[str, Any]:
    raw = frame.observation if hasattr(frame, "observation") else frame
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "to_dict"):
        return raw.to_dict()
    raise TypeError(f"unsupported observation type: {type(raw).__name__}")


def _valid_action(action: Any) -> bool:
    return (
        isinstance(action, dict)
        and isinstance(action.get("farmer"), list)
        and isinstance(action.get("hands"), list)
        and isinstance(action.get("market"), list)
    )


def _surrogate(obs: Any) -> dict[str, Any]:
    """Conservative deterministic policy: sell one highest-priced shed item."""
    data = _observation(obs)
    private = data.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    prices = (data.get("market", {}) or {}).get("prices", {}) or {}
    candidates = [
        (float(prices.get(item, 0)), str(item))
        for item, quantity in shed.items()
        if isinstance(quantity, (int, float)) and quantity > 0
    ]
    if candidates:
        _, item = max(candidates, key=lambda pair: (pair[0], pair[1]))
        return {"farmer": ["PASS"], "hands": [], "market": [["SELL", item, 1]]}
    return dict(PASS)


def _cash(final: Any, player: int) -> float:
    observation = _observation(final[player])
    farms = observation.get("farms", [])
    return float(farms[player].get("money", 0))


def _load_replay(path: Path) -> tuple[Any, list[Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("info")
    steps = data.get("steps")
    if not isinstance(info, dict) or info.get("seed") is None:
        raise ValueError("replay has no authoritative info.seed")
    if not isinstance(steps, list) or not steps:
        raise ValueError("replay has no steps")
    return info["seed"], steps


def evaluate_replay(candidate_name: str, replay_path: Path, our_player: int = 0) -> EvaluationResult:
    result = EvaluationResult(replay=str(replay_path), seed=None)
    try:
        seed, steps = _load_replay(replay_path)
        result.seed = seed
        from kaggle_environments import make

        opponent = 1 - our_player
        actions = []
        for frame in steps:
            if isinstance(frame, list) and len(frame) > opponent and isinstance(frame[opponent], dict):
                actions.append(frame[opponent].get("action"))
            else:
                actions.append(None)
        result.total_opponent_turns = len(actions)
        tier_counts = {"exact": 0, "surrogate": 0, "noop": 0}
        reasons: list[str] = []

        def replay_agent(obs: Any) -> dict[str, Any]:
            index = _step(obs) + 1
            action = actions[index] if 0 <= index < len(actions) else None
            if _valid_action(action):
                tier_counts["exact"] += 1
                return action
            result.malformed_turns += 1 if action is not None else 0
            try:
                fallback = _surrogate(obs)
                if _valid_action(fallback) and fallback != PASS:
                    tier_counts["surrogate"] += 1
                    reasons.append(f"fallback at step {index}")
                    return fallback
            except Exception as exc:
                reasons.append(f"surrogate error at step {index}: {exc}")
            tier_counts["noop"] += 1
            reasons.append(f"safe no-op at step {index}")
            return dict(PASS)

        candidate = load_agent(candidate_name, module_suffix=f"_eval_{replay_path.stem}")
        env = make("kaggriculture", configuration={"episodeSteps": len(steps), "seed": seed})
        players = [candidate, replay_agent] if our_player == 0 else [replay_agent, candidate]
        final = env.run(players)
        result.candidate_cash = _cash(final[-1], our_player)
        result.replay_opponent_cash = _cash(final[-1], opponent)

        # Run the same seed/configuration with the deterministic surrogate in
        # the candidate's opposing seat to establish the counterfactual baseline.
        surrogate_env = make(
            "kaggriculture", configuration={"episodeSteps": len(steps), "seed": seed}
        )
        counterfactual_candidate = load_agent(
            candidate_name, module_suffix=f"_eval_cf_{replay_path.stem}"
        )
        surrogate_final = surrogate_env.run(
            [counterfactual_candidate, _surrogate]
            if our_player == 0
            else [_surrogate, counterfactual_candidate]
        )
        result.surrogate_cash = _cash(surrogate_final[-1], our_player)
        result.exact_turns = tier_counts["exact"]
        result.surrogate_turns = tier_counts["surrogate"]
        result.noop_turns = tier_counts["noop"]
        result.diverged_turns = result.surrogate_turns + result.noop_turns
        result.notes.extend(reasons[:10])
    except ImportError as exc:
        result.error = f"Kaggle dependency unavailable: {exc}"
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
        result.error = f"malformed replay: {exc}"
    except Exception as exc:
        result.error = f"simulator/evaluation failure: {type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="submission name, directory, or Python file")
    parser.add_argument("replays", nargs="*", help="replay paths or glob patterns")
    parser.add_argument("--output", default="eval_results.md")
    parser.add_argument("--our-player", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    paths: list[Path] = []
    for value in args.replays or ["replays/**/*.json"]:
        matches = glob.glob(value, recursive=True)
        paths.extend(Path(p) for p in (matches or [value]))
    paths = sorted(set(p for p in paths if p.is_file()))
    if not paths:
        print("No replay files matched.", file=sys.stderr)
        return 2
    results = []
    for path in paths:
        result = evaluate_replay(args.candidate, path, args.our_player)
        results.append(result)
        print(f"{path}: {result.error or 'ok'}")
    write_markdown(args.output, results)
    print(f"Wrote {args.output}")
    return 0 if any(r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
