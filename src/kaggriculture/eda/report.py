"""Automated Exploratory Data Analysis report generator for Kaggriculture episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .parser import parse_episode


def generate_eda_report(
    episode_paths: Sequence[Path | str],
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Analyze a collection of episode replays and compile an EDA summary report.

    Args:
        episode_paths: Sequence of file paths to episode replays.
        output_path: Optional path to write the compiled JSON report.

    Returns:
        Dictionary containing EDA summary statistics.
    """
    total_episodes = 0
    episode_lengths = []
    tile_counts = {
        "EMPTY": 0,
        "LOCKED": 0,
        "WEED": 0,
        "PASTURE": 0,
        "PLANT": 0,
    }
    terminal_rewards = []
    versions: dict[str, int] = {}

    for p in episode_paths:
        try:
            data = parse_episode(p)
        except Exception:
            continue

        total_episodes += 1
        steps = data.get("steps", [])
        ep_len = len(steps)
        episode_lengths.append(ep_len)

        ver = data.get("module_version", "unknown")
        versions[ver] = versions.get(ver, 0) + 1

        # Check terminal rewards
        rewards = data.get("rewards")
        if isinstance(rewards, list) and len(rewards) == 2:
            terminal_rewards.extend([float(r or 0.0) for r in rewards])
        elif steps:
            last_frame = steps[-1]
            if isinstance(last_frame, list) and len(last_frame) == 2:
                for seat in (0, 1):
                    rew = last_frame[seat].get("reward")
                    if rew is not None:
                        terminal_rewards.append(float(rew))

        # Sample tile frequencies from steps (sample every 48 turns)
        sample_indices = range(0, ep_len, max(1, ep_len // 10))
        for idx in sample_indices:
            frame = steps[idx]
            if not isinstance(frame, list):
                continue
            for seat_frame in frame:
                obs = seat_frame.get("observation", {})
                farms = obs.get("farms", [])
                for farm in farms:
                    tiles = farm.get("tiles", [])
                    for row in tiles:
                        if not isinstance(row, list):
                            continue
                        for cell in row:
                            if cell is None:
                                tile_counts["EMPTY"] += 1
                            elif cell == "LOCKED":
                                tile_counts["LOCKED"] += 1
                            elif isinstance(cell, dict):
                                kind = cell.get("kind", "OTHER")
                                if kind in tile_counts:
                                    tile_counts[kind] += 1
                                else:
                                    tile_counts[kind] = tile_counts.get(kind, 0) + 1

    total_tiles = max(1, sum(tile_counts.values()))
    tile_frequencies = {k: v / total_tiles for k, v in tile_counts.items()}

    mean_length = float(sum(episode_lengths) / max(1, len(episode_lengths)))
    reward_summary = {
        "count": len(terminal_rewards),
        "min": float(min(terminal_rewards)) if terminal_rewards else 0.0,
        "max": float(max(terminal_rewards)) if terminal_rewards else 0.0,
        "mean": float(sum(terminal_rewards) / max(1, len(terminal_rewards))),
    }

    report = {
        "episode_count": total_episodes,
        "mean_episode_length": mean_length,
        "tile_frequencies": tile_frequencies,
        "tile_raw_counts": tile_counts,
        "reward_summary": reward_summary,
        "engine_versions": versions,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
