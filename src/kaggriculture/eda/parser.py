"""Parsing routines for Kaggriculture replay JSON and gzip streams."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping


def parse_episode(file_path: Path | str) -> dict[str, Any]:
    """Parse a single episode replay file (.json or .json.gz).

    Args:
        file_path: Path to the JSON or gzip-compressed replay file.

    Returns:
        Decoded episode dictionary.

    Raises:
        gzip.BadGzipFile: If gzip decompression fails.
        ValueError: If JSON decoding fails or file structure is corrupt.
        OSError: If file cannot be read.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Episode file not found: {path}")

    # Determine whether file is gzipped
    is_gzip = path.suffix == ".gz" or str(path).endswith(".json.gz")
    if is_gzip:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON from {path}: {exc}") from exc
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode JSON from {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dictionary in {path}, got {type(data).__name__}")
    if "steps" not in data:
        raise ValueError(f"Missing 'steps' key in {path}")

    return data


def extract_frame(steps: list[Any], t: int, seat: int) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Extract (observation, action, done) for a given seat at step t.

    In Kaggle Environments, step t contains the observation presented to the agent.
    The action the agent returned in response is stored in steps[t+1][seat]["action"].

    Args:
        steps: List of step frames from the episode.
        t: Timestep index (0..719).
        seat: Player seat (0 or 1).

    Returns:
        Tuple of (obs_t, action_t, done) where done indicates termination.
    """
    if t < 0 or t >= len(steps):
        raise IndexError(f"Timestep t={t} out of range [0, {len(steps) - 1}]")

    frame_t = steps[t]
    if not isinstance(frame_t, list) or seat >= len(frame_t):
        raise IndexError(f"Seat {seat} not present at step {t}")

    player_step_t = frame_t[seat]
    obs_t = player_step_t.get("observation") or {}
    if not isinstance(obs_t, dict):
        obs_t = dict(obs_t)

    # Ensure seat indicator is present
    if "player" not in obs_t:
        obs_t["player"] = seat

    done = (t >= len(steps) - 1) or (player_step_t.get("status") == "DONE")

    # Action taken in response to obs_t is in steps[t+1]
    if t + 1 < len(steps):
        next_frame = steps[t + 1]
        if isinstance(next_frame, list) and seat < len(next_frame):
            action_raw = next_frame[seat].get("action")
            if isinstance(action_raw, Mapping):
                action_t = dict(action_raw)
            else:
                action_t = {"farmer": ["PASS"], "hands": [], "market": []}
        else:
            action_t = {"farmer": ["PASS"], "hands": [], "market": []}
    else:
        action_t = {"farmer": ["PASS"], "hands": [], "market": []}

    return obs_t, action_t, done
