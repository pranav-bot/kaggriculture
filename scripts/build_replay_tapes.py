#!/usr/bin/env python3
"""Convert all Kaggle JSON replays into binary/line-oriented .tape files.

Generates:
  1. `<id>_opp_seat<N>.tape`: Single-seat opponent action tapes for direct Rust replay.
  2. `<id>_both.tape`: Two-seat interleaved episode tapes for `kagg episode` / `kagg bench`.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggriculture-simulation/src-python"))

from kaggsim.tape import (
    load_replay,
    replay_actions,
    replay_seed,
    replay_to_tape,
    write_episode_tape,
)


def main() -> None:
    out_dir = ROOT / "replays" / "tapes"
    out_dir.mkdir(parents=True, exist_ok=True)

    replays = sorted(glob.glob(str(ROOT / "replays/**/*.json"), recursive=True))
    print(f"Converting {len(replays)} replays to tapes in {out_dir}...")

    converted = 0
    for r in replays:
        try:
            rep = load_replay(r)
            stem = Path(r).stem
            info = rep.get("info", {})
            seed = replay_seed(rep)
            agents = [a.get("Name", f"P{i}") for i, a in enumerate(info.get("Agents", []))]

            # Determine opponent seat
            opp_seat = 0 if (len(agents) > 1 and "Pranav" in agents[1]) else 1

            # Save opponent single-seat tape
            opp_tape_path = out_dir / f"{stem}_opp_seat{opp_seat}.tape"
            replay_to_tape(rep, opp_seat, str(opp_tape_path))

            # Save two-seat episode tape
            both_tape_path = out_dir / f"{stem}_both.tape"
            a0 = replay_actions(rep, 0)
            a1 = replay_actions(rep, 1)
            write_episode_tape(str(both_tape_path), seed, a0, a1)
            converted += 1
        except Exception as e:
            print(f"Error converting {r}: {e}")

    tape_count = len(list(out_dir.glob("*.tape")))
    print(f"Finished: {converted} replays converted. Total tapes in {out_dir}: {tape_count}")


if __name__ == "__main__":
    main()
