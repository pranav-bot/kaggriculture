"""Compact, typed extraction of Kaggriculture replay JSON files.

The parser intentionally keeps one replay's decoded JSON in memory only long
enough to turn frames into compact records.  It does not print observations,
tiles, or other large payloads.  Use ``--summary`` for a corpus-level report.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]


def _number(value: Any, default: float = 0.0) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _delta(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, float]:
    before = previous or {}
    keys = set(current) | set(before)
    return {
        str(key): _number(current.get(key)) - _number(before.get(key))
        for key in sorted(keys)
        if _number(current.get(key)) != _number(before.get(key))
    }


def _inventory_sections(observation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    private = observation.get("private") or {}
    carried: dict[str, float] = {}
    for inventory in private.get("inventories") or []:
        if isinstance(inventory, Mapping):
            for item, quantity in inventory.items():
                carried[str(item)] = carried.get(str(item), 0.0) + _number(quantity)
    return {
        "carried": carried,
        "seeds": {str(k): _number(v) for k, v in (private.get("seeds") or {}).items()},
        "shed": {str(k): _number(v) for k, v in (private.get("shed") or {}).items()},
    }


@dataclass(frozen=True)
class ActionRecord:
    farmer: list[Any]
    hands: list[Any]
    market: list[Any]


@dataclass(frozen=True)
class ObservationRecord:
    player: int | None
    step: int | None
    day: int | None
    hour: int | None
    remaining_overage_time: float | None
    farm_cash: float | None
    farm_hands: int
    hires_today: int | None
    unlocked_quadrants: list[Any]
    inventory: dict[str, dict[str, float]]
    market: dict[str, Any]
    town: dict[str, Any]


@dataclass(frozen=True)
class TurnRecord:
    turn: int
    player: int
    action: ActionRecord
    observation: ObservationRecord
    cash_delta: float
    inventory_delta: dict[str, dict[str, float]]
    reward: float | None
    status: str | None


@dataclass(frozen=True)
class ReplayRecord:
    path: str
    replay_id: str | None
    info_seed: Any
    configuration_seed: Any
    seed: Any
    episode_steps: int | None
    players: int
    turns: list[TurnRecord]
    terminal_rewards: dict[str, float]
    errors: list[str]


def iter_replay_paths(root: Path = ROOT) -> Iterable[Path]:
    """Yield each JSON replay under both supported directory names."""
    seen: set[Path] = set()
    for directory in (root / "replay", root / "replays"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _action(value: Any) -> ActionRecord:
    value = value if isinstance(value, Mapping) else {}
    return ActionRecord(
        farmer=copy.deepcopy(value.get("farmer") or ["PASS"]),
        hands=copy.deepcopy(value.get("hands") or []),
        market=copy.deepcopy(value.get("market") or []),
    )


def _observation(value: Any) -> ObservationRecord:
    value = value if isinstance(value, Mapping) else {}
    farms = value.get("farms") or []
    player = value.get("player")
    farm = farms[player] if isinstance(player, int) and 0 <= player < len(farms) else {}
    farm = farm if isinstance(farm, Mapping) else {}
    return ObservationRecord(
        player=player if isinstance(player, int) else None,
        step=value.get("step") if isinstance(value.get("step"), int) else None,
        day=value.get("day") if isinstance(value.get("day"), int) else None,
        hour=value.get("hour") if isinstance(value.get("hour"), int) else None,
        remaining_overage_time=_number(value.get("remainingOverageTime"), None),
        farm_cash=_number(farm.get("money"), None),
        farm_hands=len(farm.get("hands") or []),
        hires_today=farm.get("hires_today") if isinstance(farm.get("hires_today"), int) else None,
        unlocked_quadrants=copy.deepcopy(farm.get("unlocked_quadrants") or []),
        inventory=_inventory_sections(value),
        # Market and town are small configuration/state summaries; tiles are
        # deliberately excluded because they dominate replay size.
        market=copy.deepcopy(value.get("market") or {}),
        town=copy.deepcopy(value.get("town") or {}),
    )


def parse_replay(path: str | Path, *, turn_start: int = 0, turn_stop: int | None = None) -> ReplayRecord:
    """Parse one replay into compact records, preserving frame order."""
    path = Path(path)
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return ReplayRecord(str(path), None, None, None, None, None, 0, [], {}, [str(exc)])

    info = document.get("info") or {}
    configuration = document.get("configuration") or {}
    info_seed = info.get("seed")
    configuration_seed = configuration.get("seed")
    steps = document.get("steps") or []
    turns: list[TurnRecord] = []
    previous: dict[int, tuple[float | None, dict[str, dict[str, float]]]] = {}
    stop = len(steps) if turn_stop is None else min(turn_stop, len(steps))

    for turn_number, frames in enumerate(steps[turn_start:stop], start=turn_start):
        if not isinstance(frames, list):
            errors.append(f"turn {turn_number}: expected player frame list")
            continue
        for player, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                errors.append(f"turn {turn_number} player {player}: invalid frame")
                continue
            observation = _observation(frame.get("observation"))
            current_inventory = observation.inventory
            current_cash = observation.farm_cash
            old_cash, old_inventory = previous.get(player, (None, {}))
            turns.append(
                TurnRecord(
                    turn=turn_number,
                    player=player,
                    action=_action(frame.get("action")),
                    observation=observation,
                    cash_delta=0.0 if old_cash is None or current_cash is None else current_cash - old_cash,
                    inventory_delta={
                        section: _delta(current_inventory[section], old_inventory.get(section))
                        for section in current_inventory
                    },
                    reward=_number(frame.get("reward"), None),
                    status=frame.get("status") if isinstance(frame.get("status"), str) else None,
                )
            )
            previous[player] = (current_cash, current_inventory)

    terminal_rewards: dict[str, float] = {}
    complete_parse = turn_start == 0 and stop == len(steps)
    for record in turns:
        if record.status == "DONE" or (complete_parse and record.turn == stop - 1):
            if record.reward is not None:
                terminal_rewards[str(record.player)] = record.reward
    if not terminal_rewards and complete_parse:
        rewards = document.get("rewards")
        if isinstance(rewards, list):
            terminal_rewards = {
                str(player): float(reward)
                for player, reward in enumerate(rewards)
                if isinstance(reward, (int, float))
            }

    return ReplayRecord(
        path=str(path),
        replay_id=document.get("id"),
        info_seed=info_seed,
        configuration_seed=configuration_seed,
        seed=info_seed if info_seed is not None else configuration_seed,
        episode_steps=configuration.get("episodeSteps") if isinstance(configuration.get("episodeSteps"), int) else None,
        players=max((record.player for record in turns), default=-1) + 1,
        turns=turns,
        terminal_rewards=terminal_rewards,
        errors=errors,
    )


def _summary(record: ReplayRecord) -> dict[str, Any]:
    return {
        "path": record.path,
        "seed": record.seed,
        "info_seed": record.info_seed,
        "configuration_seed": record.configuration_seed,
        "episode_steps": record.episode_steps,
        "players": record.players,
        "turn_records": len(record.turns),
        "terminal_rewards": record.terminal_rewards,
        "errors": record.errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="replay files or directories (default: replay/ and replays/)")
    parser.add_argument("--summary", action="store_true", help="explicitly request the compact JSON summary (the default)")
    parser.add_argument("--full", action="store_true", help="emit all compact turn records (still excludes tiles)")
    parser.add_argument("--limit", type=int, help="parse at most this many files")
    parser.add_argument("--turn-start", type=int, default=0)
    parser.add_argument("--turn-stop", type=int)
    args = parser.parse_args()

    paths = [p for root in args.paths for p in (root.rglob("*.json") if root.is_dir() else [root])]
    if not args.paths:
        paths = list(iter_replay_paths())
    for path in paths[: args.limit]:
        record = parse_replay(path, turn_start=args.turn_start, turn_stop=args.turn_stop)
        payload = asdict(record) if args.full else _summary(record)
        print(json.dumps(payload, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
