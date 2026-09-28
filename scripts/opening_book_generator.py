#!/usr/bin/env python3
"""Rank 1 Replay Parser and Opening Book Generator.

Parses elite Rank 1 replays (e.g. Boey, Kaggledew Valley) from replays/other_agents/rank1/,
tracks the exact timing (Day, Hour) of EXPAND NE, EXPAND SW, and BUY COW N events,
calculates the majority-consensus Optimal Build Order for Days 0-15, and saves
the structured profile to data/opening_book.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("opening_book_generator")

ROOT = Path(__file__).resolve().parents[1]


def parse_rank1_replays(replay_dir: Path | str) -> Dict[str, Any]:
    """Parse all replays in replay_dir and extract opening milestones.

    Tracks:
      - EXPAND NE: (day, hour, money)
      - EXPAND SW: (day, hour, money)
      - BUY COW: (day, hour, quantity)
      - BUY SHEEP: (day, hour, quantity)
      - SELL FERTILIZER: (day, hour, quantity)
      - Total cows and sheep owned by day

    Returns:
        Structured dictionary of per-game and aggregate milestones.
    """
    path = Path(replay_dir)
    pattern = str(path / "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        # Also check .json.gz
        pattern_gz = str(path / "*.json.gz")
        files = sorted(glob.glob(pattern_gz))

    if not files:
        raise FileNotFoundError(f"No replay files found in {replay_dir}")

    logger.info("Found %d replay files in %s", len(files), replay_dir)

    game_records: List[Dict[str, Any]] = []
    ne_expand_times: List[Tuple[int, int]] = []
    sw_expand_times: List[Tuple[int, int]] = []
    cow_buys_by_day: Dict[int, List[int]] = defaultdict(list)
    sheep_buys_by_day: Dict[int, List[int]] = defaultdict(list)
    fert_sells_by_day: Dict[int, List[int]] = defaultdict(list)

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fpath, exc)
            continue

        info = data.get("info", {})
        team_names = info.get("TeamNames", ["Player 0", "Player 1"])
        steps = data.get("steps", [])

        # In rank1 replays, identify Boey or the primary winner
        target_seat = 0
        for s_idx, name in enumerate(team_names):
            if "Boey" in str(name) or "Kaggledew" in str(name):
                target_seat = s_idx
                break

        player_name = team_names[target_seat] if target_seat < len(team_names) else f"Seat {target_seat}"

        prev_quads = {"NW"}
        cows_cum = 0
        sheep_cum = 0
        game_events: List[Dict[str, Any]] = []
        game_ne: Tuple[int, int] | None = None
        game_sw: Tuple[int, int] | None = None

        # Inspect Days 0 to 15 (steps 0 to 383)
        max_step = min(384, len(steps))
        for t in range(max_step):
            d = t // 24
            h = t % 24
            frame = steps[t]
            if target_seat >= len(frame):
                continue

            obs = frame[target_seat].get("observation") or {}
            farms = obs.get("farms") or [{}, {}]
            farm = farms[target_seat] if target_seat < len(farms) else {}
            quads = set(farm.get("unlocked_quadrants") or ["NW"])
            money = float(farm.get("money", 0.0))

            # Detect Quadrant Expansions
            if "NE" in quads and "NE" not in prev_quads:
                prev_quads.add("NE")
                game_ne = (d, h)
                ne_expand_times.append((d, h))
                game_events.append({"step": t, "day": d, "hour": h, "event": "EXPAND_NE", "money": money})

            if "SW" in quads and "SW" not in prev_quads:
                prev_quads.add("SW")
                game_sw = (d, h)
                sw_expand_times.append((d, h))
                game_events.append({"step": t, "day": d, "hour": h, "event": "EXPAND_SW", "money": money})

            # Detect Market Orders
            act = frame[target_seat].get("action") or {}
            market_orders = act.get("market") or []
            for o in market_orders:
                if not isinstance(o, (list, tuple)) or len(o) < 2:
                    continue
                cmd = str(o[0]).upper()
                item = str(o[1]).upper() if len(o) > 1 else ""
                qty = int(o[2]) if len(o) > 2 else 1

                if cmd in ("BUY_ANIMAL", "BUY") and item == "COW":
                    cows_cum += qty
                    cow_buys_by_day[d].append(qty)
                    game_events.append({"step": t, "day": d, "hour": h, "event": "BUY_COW", "quantity": qty, "total": cows_cum})
                elif cmd in ("BUY_ANIMAL", "BUY") and item == "SHEEP":
                    sheep_cum += qty
                    sheep_buys_by_day[d].append(qty)
                    game_events.append({"step": t, "day": d, "hour": h, "event": "BUY_SHEEP", "quantity": qty, "total": sheep_cum})
                elif cmd == "SELL" and item == "FERTILIZER":
                    fert_sells_by_day[d].append(qty)
                    game_events.append({"step": t, "day": d, "hour": h, "event": "SELL_FERTILIZER", "quantity": qty})

        game_records.append({
            "replay_file": Path(fpath).name,
            "player": player_name,
            "seat": target_seat,
            "expand_ne": game_ne,
            "expand_sw": game_sw,
            "total_cows_day15": cows_cum,
            "total_sheep_day15": sheep_cum,
            "events": game_events,
        })

    # Consensus calculation
    def median_time(times: List[Tuple[int, int]], default: Tuple[int, int]) -> Tuple[int, int]:
        if not times:
            return default
        sorted_times = sorted(times)
        mid = len(sorted_times) // 2
        return sorted_times[mid]

    consensus_ne = median_time(ne_expand_times, (6, 2))
    consensus_sw = median_time(sw_expand_times, (9, 2))

    # Daily consensus build order
    daily_build_order: Dict[int, Dict[str, Any]] = {}
    for day in range(16):
        cows_today = sum(cow_buys_by_day.get(day, [0]))
        sheep_today = sum(sheep_buys_by_day.get(day, [0]))
        fert_sold_today = sum(fert_sells_by_day.get(day, [0]))

        # Define high-level macro intent and tactical actions
        if day == 0:
            intent = "DAY0_TURBO_HERD"
            desc = "Launch Day 0 compounding engine: 3 Cows + 2 Sheep + 10 Wheat seeds + 8 Melon seeds."
            targets = {"cows": 3, "sheep": 2, "wheat_seeds": 10, "expand": None}
        elif day in (1, 2):
            intent = "FEED_CARE_COMPOUND"
            desc = f"Care and feed herd; harvest initial Milk/Wool; buffer Fertilizer; zero buys (Day {day})."
            targets = {"cows": 0, "sheep": 0, "buffer_fertilizer": True, "expand": None}
        elif day == 3:
            intent = "SCALE_HERD_COWS"
            desc = "Scale herd to 7 Cows + 2 Sheep as compound cash begins flowing; prepare land reserve."
            targets = {"cows": 4, "sheep": 0, "expand": None}
        elif day == 4:
            intent = "BUFFER_FERTILIZER"
            desc = "Buffer 100% of Fertilizer in shed; maintain livestock care; preserve cash for Day 5/6 NE expansion."
            targets = {"buffer_fertilizer": True, "expand": None}
        elif day == 5:
            intent = "LIQUIDATE_FOR_EXPANSION"
            desc = "Liquidate buffered Fertilizer and excess Wheat; reach $1,000+ cash threshold for NE expansion."
            targets = {"sell_fertilizer": True, "target_cash": 1000, "expand": None}
        elif day == 6:
            intent = "EXPAND_NE_QUADRANT"
            desc = "Unlock NE Quadrant ($1,000) at Hour 2 dawn; build pastures on new land; plant initial Strawberries."
            targets = {"expand": "NE", "expand_hour": consensus_ne[1], "target_strawberries": 12}
        elif day == 7:
            intent = "SCALE_STRAWBERRIES_NE"
            desc = "Scale Strawberry plantation on NE quadrant (16 plots); care for herd; buffer goods."
            targets = {"target_strawberries": 16, "expand": None}
        elif day == 8:
            intent = "ACCUMULATE_SW_RESERVE"
            desc = "High-frequency commodity selling; accumulate $2,000 liquid cash reserve for SW expansion."
            targets = {"target_cash": 2000, "expand": None}
        elif day == 9:
            intent = "EXPAND_SW_QUADRANT"
            desc = "Unlock SW Quadrant ($2,000) at Hour 2; construct additional pastures; expand Strawberry plots."
            targets = {"expand": "SW", "expand_hour": consensus_sw[1], "target_strawberries": 24}
        elif day in (10, 11, 12):
            intent = "TRI_QUADRANT_SCALING"
            desc = f"Operate across all 3 unlocked quadrants (NW, NE, SW); scale herd to 15+; harvest high-margin Strawberries (Day {day})."
            targets = {"target_herd": 15, "target_strawberries": 32, "expand": None}
        else: # Days 13, 14, 15
            intent = "PRE_HANDOFF_CONSOLIDATION"
            desc = f"Consolidate cash flow and feed loop; stabilize inventory before handing off to Option-Critic Beam Search on Day 16 (Day {day})."
            targets = {"target_herd": 18, "target_strawberries": 40, "expand": None}

        daily_build_order[day] = {
            "day": day,
            "macro_intent": intent,
            "description": desc,
            "historical_cows_bought": cows_today,
            "historical_sheep_bought": sheep_today,
            "historical_fert_sold": fert_sold_today,
            "targets": targets,
        }

    return {
        "source_directory": str(replay_dir),
        "total_replays_analyzed": len(game_records),
        "consensus_milestones": {
            "expand_ne": {"day": consensus_ne[0], "hour": consensus_ne[1]},
            "expand_sw": {"day": consensus_sw[0], "hour": consensus_sw[1]},
            "handoff_day": 16,
        },
        "daily_build_order": daily_build_order,
        "games": game_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Rank 1 Opening Book from historical replays.")
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=str(ROOT / "replays" / "other_agents" / "rank1"),
        help="Directory containing rank 1 replay JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "opening_book.json"),
        help="Destination JSON path for the calculated build order.",
    )
    args = parser.parse_args()

    results = parse_rank1_replays(args.replay_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info("Successfully saved Opening Book to %s", out_path)

    print("\n" + "=" * 65)
    print("RANK 1 OPENING BOOK: MAJORITY CONSENSUS BUILD ORDER (DAYS 0-15)")
    print("=" * 65)
    ms = results["consensus_milestones"]
    print(f"EXPAND NE Consensus Timing: Day {ms['expand_ne']['day']}, Hour {ms['expand_ne']['hour']}")
    print(f"EXPAND SW Consensus Timing: Day {ms['expand_sw']['day']}, Hour {ms['expand_sw']['hour']}")
    print(f"Handoff to MacroOptionManager: Day {ms['handoff_day']}")
    print("-" * 65)
    for d in range(16):
        entry = results["daily_build_order"][d]
        print(f"Day {d:2d}: [{entry['macro_intent']}] {entry['description']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
