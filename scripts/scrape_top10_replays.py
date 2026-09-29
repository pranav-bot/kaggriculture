#!/usr/bin/env python3
"""Scrape Top 10 ladder match replays via Kaggle CLI into replays/live_top10/.

Traverses Kaggle competition leaderboard -> team submissions -> completed episodes
and downloads 20 unique match replays from current Top 10 leaderboard teams.
Handles pagination tokens ('Next Page Token = ...'), CLI instructional footers,
and provides graceful fallback to lower ranks (11-15) and local cached replays.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scrape_top10")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "replays" / "live_top10"
DEFAULT_KAGGLE_BIN = ROOT / ".venv" / "bin" / "kaggle"
FALLBACK_DIRS = [
    ROOT / "replays" / "other_agents",
    ROOT / "replays" / "live_losses",
    ROOT / "replays" / "live_wins",
]


def find_kaggle_cli(custom_path: Optional[str] = None) -> str:
    """Resolve Kaggle CLI binary path."""
    if custom_path and Path(custom_path).is_file():
        return str(Path(custom_path).resolve())
    if DEFAULT_KAGGLE_BIN.is_file():
        return str(DEFAULT_KAGGLE_BIN.resolve())
    which_kaggle = shutil.which("kaggle")
    if which_kaggle:
        return which_kaggle
    raise FileNotFoundError(
        f"Kaggle CLI not found at {DEFAULT_KAGGLE_BIN} or on PATH."
    )


def query_kaggle_json(
    cli_path: str,
    args: List[str],
    timeout_s: float = 30.0,
) -> Optional[Any]:
    """Execute Kaggle CLI command and safely parse JSON array/object.
    
    Handles Kaggle CLI output quirks:
    - 'Next Page Token = ...' prefixes on leaderboard queries
    - Instructional footers on episodes queries
    """
    cmd = [cli_path] + args + ["--format", "json"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        logger.warning("CLI invocation failed for %s: %s", " ".join(cmd), exc)
        return None

    if proc.returncode != 0:
        logger.warning(
            "CLI command exited with %d: %s\nStderr: %s",
            proc.returncode,
            " ".join(cmd),
            proc.stderr.strip()[:200],
        )
        return None

    out = proc.stdout
    # Slice strictly between first '[' and last ']' for array responses
    s_arr = out.find("[")
    e_arr = out.rfind("]")
    if s_arr != -1 and e_arr > s_arr:
        try:
            return json.loads(out[s_arr : e_arr + 1])
        except json.JSONDecodeError as exc:
            logger.warning("JSON decode failed for array slice: %s", exc)

    # Slice strictly between first '{' and last '}' for object responses
    s_obj = out.find("{")
    e_obj = out.rfind("}")
    if s_obj != -1 and e_obj > s_obj:
        try:
            return json.loads(out[s_obj : e_obj + 1])
        except json.JSONDecodeError as exc:
            logger.warning("JSON decode failed for object slice: %s", exc)

    return None


def fetch_leaderboard_teams(cli_path: str, competition: str = "kaggriculture") -> List[Dict[str, Any]]:
    """Fetch all ranked teams from the competition leaderboard."""
    logger.info("Fetching leaderboard for '%s'...", competition)
    data = query_kaggle_json(cli_path, ["competitions", "leaderboard", competition, "--show"])
    if not isinstance(data, list):
        logger.error("Failed to parse leaderboard from Kaggle CLI.")
        return []
    logger.info("Retrieved %d leaderboard teams.", len(data))
    return data


def fetch_team_submissions(cli_path: str, team_id: int) -> List[Dict[str, Any]]:
    """Fetch active submissions for a team."""
    data = query_kaggle_json(cli_path, ["competitions", "team-submissions", str(team_id)])
    if isinstance(data, list):
        return data
    return []


def fetch_submission_episodes(cli_path: str, submission_id: int) -> List[Dict[str, Any]]:
    """Fetch episodes played by a submission."""
    data = query_kaggle_json(cli_path, ["competitions", "episodes", str(submission_id)])
    if isinstance(data, list):
        return data
    return []


def download_episode_replay(
    cli_path: str,
    episode_id: int,
    output_dir: Path,
    timeout_s: float = 60.0,
) -> Optional[Path]:
    """Download replay for an episode into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check if already present
    expected_names = [
        output_dir / f"episode-{episode_id}-replay.json",
        output_dir / f"{episode_id}.json",
    ]
    for p in expected_names:
        if p.is_file() and p.stat().st_size > 1000:
            logger.info("Replay %d already exists at %s", episode_id, p.name)
            return p

    cmd = [cli_path, "competitions", "replay", str(episode_id), "-p", str(output_dir)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        logger.warning("Download command failed for episode %d: %s", episode_id, exc)
        return None

    if proc.returncode != 0:
        logger.warning(
            "Download failed for episode %d (code %d): %s",
            episode_id,
            proc.returncode,
            proc.stderr.strip()[:200],
        )
        return None

    for p in expected_names:
        if p.is_file() and p.stat().st_size > 1000:
            logger.info("Successfully downloaded episode %d -> %s", episode_id, p.name)
            return p

    # Check for any newly created file matching episode_id
    matches = list(output_dir.glob(f"*{episode_id}*.json"))
    if matches and matches[0].is_file() and matches[0].stat().st_size > 1000:
        return matches[0]

    logger.warning("Downloaded replay file for episode %d not found in %s", episode_id, output_dir)
    return None


def copy_cached_fallback_replays(output_dir: Path, needed: int) -> List[Path]:
    """Copy cached replays from fallback directories to satisfy target count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    seen_ids: Set[str] = {
        p.stem.replace("episode-", "").replace("-replay", "")
        for p in output_dir.glob("*.json")
    }

    for fallback in FALLBACK_DIRS:
        if not fallback.is_dir():
            continue
        for src in fallback.rglob("*.json"):
            if not src.is_file() or src.stat().st_size < 1000:
                continue
            # Ensure it is a valid match replay
            try:
                with open(src, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                if not isinstance(doc, dict) or "steps" not in doc:
                    continue
            except Exception:
                continue

            ep_id = src.stem.replace("episode-", "").replace("-replay", "")
            if ep_id in seen_ids:
                continue

            dest = output_dir / f"episode-{ep_id}-replay.json"
            shutil.copy2(src, dest)
            seen_ids.add(ep_id)
            copied.append(dest)
            logger.info("Copied fallback replay %s -> %s", src.name, dest.name)
            if len(copied) >= needed:
                return copied

    return copied


def scrape_top10_replays(
    target_count: int = 20,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    custom_cli: Optional[str] = None,
    competition: str = "kaggriculture",
) -> List[Path]:
    """Scrape target_count completed match replays from Top 10 leaderboard teams."""
    cli = find_kaggle_cli(custom_cli)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: List[Path] = []
    seen_episodes: Set[int] = set()

    # Register already present replays
    for p in output_dir.glob("*.json"):
        if p.is_file() and p.stat().st_size > 1000:
            ep_str = p.stem.replace("episode-", "").replace("-replay", "")
            if ep_str.isdigit():
                seen_episodes.add(int(ep_str))
                downloaded_paths.append(p)

    logger.info("Found %d existing replays in %s", len(downloaded_paths), output_dir)
    if len(downloaded_paths) >= target_count:
        logger.info("Target count of %d already reached.", target_count)
        return downloaded_paths[:target_count]

    # Query leaderboard
    teams = fetch_leaderboard_teams(cli, competition)
    if not teams:
        logger.warning("No leaderboard teams fetched. Falling back to cached replays.")
        needed = target_count - len(downloaded_paths)
        if needed > 0:
            downloaded_paths.extend(copy_cached_fallback_replays(output_dir, needed))
        return downloaded_paths[:target_count]

    metadata_records: List[Dict[str, Any]] = []

    # Target: 2 episodes per Top 10 team (with ranks 11-15 as fallback)
    teams_to_process = teams[:15]  # Top 10 + 5 fallback ranks
    episodes_per_team = max(2, (target_count + len(teams_to_process) - 1) // len(teams_to_process))

    for rank_idx, team in enumerate(teams_to_process, start=1):
        if len(downloaded_paths) >= target_count:
            break

        team_id = int(team["teamId"])
        team_name = str(team["teamName"])
        team_score = team.get("score")
        logger.info("Processing Rank %d: '%s' (teamId: %d, score: %s)", rank_idx, team_name, team_id, team_score)

        subs = fetch_team_submissions(cli, team_id)
        if not subs:
            logger.warning("Rank %d '%s' has 0 visible submissions, skipping to next.", rank_idx, team_name)
            continue

        team_downloaded = 0
        for sub in subs:
            if len(downloaded_paths) >= target_count or team_downloaded >= episodes_per_team:
                break
            sub_id = int(sub["id"])
            sub_score = sub.get("publicScore")
            episodes = fetch_submission_episodes(cli, sub_id)
            if not episodes:
                continue

            # Filter for completed public/competition episodes
            completed_eps = [
                e for e in episodes
                if str(e.get("state", "")).endswith("COMPLETED")
            ]

            for ep in completed_eps:
                if len(downloaded_paths) >= target_count or team_downloaded >= episodes_per_team:
                    break
                ep_id = int(ep["id"])
                if ep_id in seen_episodes:
                    continue

                logger.info(
                    "Downloading episode %d for team '%s' (sub %d, score %s)...",
                    ep_id, team_name, sub_id, sub_score
                )
                path = download_episode_replay(cli, ep_id, output_dir)
                if path:
                    seen_episodes.add(ep_id)
                    downloaded_paths.append(path)
                    team_downloaded += 1
                    metadata_records.append({
                        "episode_id": ep_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "rank": rank_idx,
                        "submission_id": sub_id,
                        "file_path": str(path.relative_to(ROOT)),
                    })

    # Save metadata index
    index_file = output_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)
    logger.info("Saved index metadata for %d replays to %s", len(metadata_records), index_file)

    # Fallback if still under target count
    if len(downloaded_paths) < target_count:
        needed = target_count - len(downloaded_paths)
        logger.warning(
            "Live scraping yielded %d replays, need %d more. Invoking cached fallback.",
            len(downloaded_paths), needed
        )
        fallback_paths = copy_cached_fallback_replays(output_dir, needed)
        downloaded_paths.extend(fallback_paths)

    logger.info("Final replay inventory in %s: %d files", output_dir, len(downloaded_paths))
    return downloaded_paths[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Scrape Top 10 ladder match replays into replays/live_top10/.")
    parser.add_argument("--count", "-n", type=int, default=20, help="Number of unique replays to fetch (default: 20)")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory")
    parser.add_argument("--kaggle-bin", type=str, default=None, help="Path to kaggle executable")
    parser.add_argument("--competition", type=str, default="kaggriculture", help="Kaggle competition name")
    args = parser.parse_args()

    replays = scrape_top10_replays(
        target_count=args.count,
        output_dir=args.output_dir,
        custom_cli=args.kaggle_bin,
        competition=args.competition,
    )
    print(f"Total replays acquired: {len(replays)}")
    for r in replays:
        print(f"  - {r.name} ({r.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
