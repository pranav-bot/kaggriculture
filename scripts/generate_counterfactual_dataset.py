#!/usr/bin/env python3
"""CLI Pipeline for Generating Counterfactual Market Datasets.

Iterates over historical replays, identifies market-critical states where a player sold >10 Milk/Wool,
injects HOLD counterfactual actions, executes fast heuristic rollouts in the Rust engine,
computes counterfactual_value labels, and writes JSONL dataset tuples via multiprocessing.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kaggriculture.counterfactual import run_counterfactual_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("counterfactual_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate counterfactual dataset using Rust simulator and replay data."
    )
    parser.add_argument(
        "--episodes-dir",
        type=str,
        default=str(ROOT / "datasets" / "il" / "episodes"),
        help="Directory containing .json or .json.gz episode replays.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "counterfactuals.jsonl"),
        help="Destination JSONL path for output tuples.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers (default: all CPU cores).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episode files to process.",
    )
    parser.add_argument(
        "--kagg",
        type=str,
        default=None,
        help="Path to kagg binary executable.",
    )

    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    pattern = str(episodes_dir / "**" / "*.json.gz")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern = str(episodes_dir / "**" / "*.json")
        files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        logger.error("No episode files found in %s", episodes_dir)
        sys.exit(1)

    logger.info("Found %d total episode files.", len(files))
    if args.max_episodes:
        files = files[: args.max_episodes]
        logger.info("Processing limited subset of %d episodes.", len(files))

    start_time = time.perf_counter()
    stats = run_counterfactual_pipeline(
        episodes=files,
        output_file=args.output,
        num_workers=args.workers,
        max_episodes=args.max_episodes,
        kagg_binary=args.kagg,
    )
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("Counterfactual Dataset Generation Summary")
    print("=" * 60)
    print(f"Output File:                      {stats['output_path']}")
    print(f"Episodes Processed:               {stats['episodes_processed']}")
    print(f"Episodes with Large Sells (>10):  {stats['episodes_with_large_sells']}")
    print(f"Total Counterfactual Tuples:      {stats['counterfactual_records_generated']}")
    print(f"Workers Utilized:                 {stats['num_workers']}")
    print(f"Elapsed Time:                     {elapsed:.2f}s")
    if elapsed > 0 and stats['counterfactual_records_generated'] > 0:
        rate = stats['counterfactual_records_generated'] / elapsed
        print(f"Throughput:                       {rate:.2f} tuples/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
