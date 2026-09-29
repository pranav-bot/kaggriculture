#!/usr/bin/env python3
"""Convert live match replays and autopsy records into executable Ghost Opponents.

Extracts opponent macro build orders from raw Kaggle match replays (or bridges
from autopsy loss_analysis schemas), and generates static executable submissions in
submissions/ladder_ghost_<EPISODE_ID>/main.py using scripts/ladder_ghost.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_ghosts")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from ladder_ghost import VALID_ACTIONS, generate_ghost, parse_build_order  # noqa: E402

DEFAULT_REPLAYS_DIR = ROOT / "replays" / "live_top10"
DEFAULT_SUBMISSIONS_DIR = ROOT / "submissions"


# =============================================================================
# 1. Autopsy Schema Bridging
# =============================================================================

def bridge_autopsy_to_build_order(autopsy_doc: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Bridges autopsy loss_analysis.json (opponent_build_order) to ladder_ghost schema."""
    ep_id = str(autopsy_doc.get("episode_id", "unknown"))
    opp_name = str(autopsy_doc.get("opponent_name", "UnknownOpponent"))

    # If already in ladder_ghost format
    if "build_order" in autopsy_doc and autopsy_doc["build_order"]:
        _, order = parse_build_order(autopsy_doc)
        return ep_id, opp_name, order

    raw_events = autopsy_doc.get("opponent_build_order", [])
    order: List[Dict[str, Any]] = []

    for ev in raw_events:
        day = int(ev.get("day", 0))
        event_name = str(ev.get("event", "")).upper()

        if event_name == "EXPAND_QUADRANT_NE":
            order.append({"day": day, "action": "EXPAND_NE"})
        elif event_name == "EXPAND_QUADRANT_SW":
            order.append({"day": day, "action": "EXPAND_SW"})
        elif event_name == "EXPAND_QUADRANT_SE":
            order.append({"day": day, "action": "EXPAND_SE"})
        elif event_name.startswith("BUY_") and event_name[4:] in ("COW", "SHEEP", "GOOSE", "PIG"):
            species = event_name[4:]
            # details e.g. "Purchased 3 COW(s)"
            details = str(ev.get("details", ""))
            count = 1
            for token in details.split():
                if token.isdigit():
                    count = int(token)
                    break
            order.append({"day": day, "action": "BUY_ANIMAL", "animal": species, "count": count})
        elif event_name.startswith("BUY_SEED_"):
            crop = event_name[9:]
            order.append({"day": day, "action": "BUY_SEED", "crop": crop, "count": 4})
        elif event_name == "HIRE_FARMHAND":
            order.append({"day": day, "action": "HIRE", "count": 1})

    # Add terminal liquidation default if not present
    if not any(d["action"] in ("SELL", "DUMP") for d in order):
        order.append({"day": 25, "action": "DUMP", "product": "MILK"})
        order.append({"day": 25, "action": "DUMP", "product": "WOOL"})

    order.sort(key=lambda s: s["day"])
    return ep_id, opp_name, order


# =============================================================================
# 2. Raw Replay Dissection & Macro Extraction
# =============================================================================

def resolve_target_seat(
    doc: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    preferred_team: Optional[str] = None,
) -> Tuple[int, str]:
    """Determine the seat index and name of the opponent to model."""
    info = doc.get("info", {})
    team_names = info.get("TeamNames", ["Player 0", "Player 1"])
    rewards = doc.get("rewards") or [0.0, 0.0]

    # 1. Match preferred team if given
    if preferred_team:
        for idx, name in enumerate(team_names):
            if preferred_team.lower() in str(name).lower():
                return idx, team_names[idx]

    # 2. Match from index metadata if available
    ep_id = str(info.get("EpisodeId", ""))
    if metadata and ep_id in metadata:
        meta_team = metadata[ep_id].get("team_name")
        if meta_team:
            for idx, name in enumerate(team_names):
                if meta_team.lower() in str(name).lower():
                    return idx, team_names[idx]

    # 3. Fallback: select the higher scoring team
    seat = 0 if float(rewards[0]) >= float(rewards[1]) else 1
    return seat, team_names[seat] if seat < len(team_names) else f"Player {seat}"


def extract_macro_build_order(
    replay_doc: Dict[str, Any],
    target_seat: int,
) -> List[Dict[str, Any]]:
    """Extract macro build order from a 720-step replay for a specific seat."""
    steps = replay_doc.get("steps", [])
    quadrant_unlocks: List[Tuple[int, int, str]] = []
    hires_by_day: Counter[int] = Counter()
    animals_by_day: Dict[int, Counter[str]] = {}
    crops_planted: Counter[str] = Counter()
    crop_first_day: Dict[str, int] = {}
    shed_products: Counter[str] = Counter()

    seen_quads = {"NW"}

    for t, step in enumerate(steps):
        day = t // 24
        hour = t % 24
        if target_seat >= len(step):
            continue
        frame = step[target_seat]
        act = frame.get("action") or {}
        obs = frame.get("observation") or (
            step[0].get("observation") if target_seat == 0 else step[1].get("observation")
        )
        farms = obs.get("farms", []) if obs else []

        # Check quadrant expansion
        if target_seat < len(farms):
            uq = farms[target_seat].get("unlocked_quadrants", [])
            for q in uq:
                if q not in seen_quads:
                    seen_quads.add(q)
                    quadrant_unlocks.append((day, hour, q))

        # Check market operations
        for m in act.get("market", []):
            if not isinstance(m, list) or not m:
                continue
            cmd = m[0]
            if cmd == "HIRE":
                hires_by_day[day] += 1
            elif cmd == "BUY_ANIMAL" and len(m) > 2:
                sp = str(m[1]).upper()
                cnt = int(m[2])
                animals_by_day.setdefault(day, Counter())[sp] += cnt
            elif cmd == "SELL" and len(m) > 2:
                prod = str(m[1]).upper()
                cnt = int(m[2])
                shed_products[prod] += cnt

        # Check unit planting actions
        units = [act.get("farmer", [])] + act.get("hands", [])
        for u in units:
            if isinstance(u, list) and len(u) > 1 and u[0] == "PLANT":
                crop = str(u[1]).upper()
                crops_planted[crop] += 1
                if crop not in crop_first_day:
                    crop_first_day[crop] = day

    order: List[Dict[str, Any]] = []

    # 1. Day 0 Opening Animals
    day0_animals = animals_by_day.get(0, Counter())
    for sp in ["COW", "SHEEP", "GOOSE", "PIG"]:
        cnt = day0_animals.get(sp, 0)
        if cnt > 0:
            order.append({"day": 0, "action": "BUY_ANIMAL", "animal": sp, "count": cnt})
    # If no day 0 animal buy recorded (e.g. bought on hour 0 of Day 1), provide standard opening
    if not order:
        order.append({"day": 0, "action": "BUY_ANIMAL", "animal": "COW", "count": 2})
        order.append({"day": 0, "action": "BUY_ANIMAL", "animal": "SHEEP", "count": 3})

    # 2. Day 0 Crops & Seeds
    order.append({"day": 0, "action": "PLANT", "crop": "WHEAT", "count": 8})

    # 3. Day 0 Opening Hires
    day0_hires = hires_by_day.get(0, 0)
    if day0_hires > 0:
        order.append({"day": 0, "action": "HIRE", "count": min(day0_hires, 5)})

    # 4. Quadrant Expansions & Major Hire Spikes
    ne_day = 6
    for day, hour, q in quadrant_unlocks:
        act = f"EXPAND_{q}"
        if act in VALID_ACTIONS:
            order.append({"day": day, "action": act})
            if q == "NE":
                ne_day = day
                # Major hire spike on expansion
                exp_hires = hires_by_day.get(day, 0)
                if exp_hires >= 3:
                    order.append({"day": day, "action": "HIRE", "count": min(exp_hires, 8)})

    # 5. Target Cash Crop Plantings
    if "STRAWBERRY" in crops_planted:
        straw_day = max(crop_first_day.get("STRAWBERRY", 6), ne_day)
        straw_cnt = min(max(crops_planted["STRAWBERRY"] // 2, 8), 16)
        order.append({"day": straw_day, "action": "PLANT", "crop": "STRAWBERRY", "count": straw_cnt})
    elif "MELON" in crops_planted:
        melon_cnt = min(max(crops_planted["MELON"], 6), 12)
        order.append({"day": 4, "action": "PLANT", "crop": "MELON", "count": melon_cnt})

    # 6. Terminal Liquidation / Shed Dumps
    if day0_animals.get("COW", 0) > 0 or shed_products.get("MILK", 0) > 0:
        order.append({"day": 25, "action": "DUMP", "product": "MILK"})
    if day0_animals.get("SHEEP", 0) > 0 or shed_products.get("WOOL", 0) > 0:
        order.append({"day": 25, "action": "DUMP", "product": "WOOL"})
    if shed_products.get("FERTILIZER", 0) > 0:
        order.append({"day": 25, "action": "DUMP", "product": "FERTILIZER"})
    if "STRAWBERRY" in crops_planted:
        order.append({"day": 27, "action": "DUMP", "product": "STRAWBERRY"})

    order.sort(key=lambda s: s["day"])
    return order


# =============================================================================
# 3. Replay to Ghost Conversion Driver
# =============================================================================

def convert_single_replay(
    replay_path: Path,
    submissions_dir: Path = DEFAULT_SUBMISSIONS_DIR,
    metadata: Optional[Dict[str, Any]] = None,
    preferred_team: Optional[str] = None,
) -> Path:
    """Convert a single replay (or autopsy file) into a ghost submission."""
    with open(replay_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Check if autopsy record or raw replay
    if "opponent_build_order" in doc:
        ep_id, opp_name, order = bridge_autopsy_to_build_order(doc)
    else:
        info = doc.get("info", {})
        ep_id = str(info.get("EpisodeId") or replay_path.stem.replace("episode-", "").replace("-replay", ""))
        seat, opp_name = resolve_target_seat(doc, metadata=metadata, preferred_team=preferred_team)
        order = extract_macro_build_order(doc, seat)

    # Validate schema
    _, validated_order = parse_build_order({"episode_id": ep_id, "build_order": order})

    dest = submissions_dir / f"ladder_ghost_{ep_id}"
    main_path = generate_ghost(ep_id, opp_name, validated_order, dest)
    logger.info("Generated ghost for %s (%s) -> %s", ep_id, opp_name, main_path)
    return dest


def convert_all_replays(
    replays_dir: Path = DEFAULT_REPLAYS_DIR,
    submissions_dir: Path = DEFAULT_SUBMISSIONS_DIR,
) -> List[Path]:
    """Convert all replay files in replays_dir into ghost submissions."""
    index_file = replays_dir / "index.json"
    metadata: Dict[str, Any] = {}
    if index_file.is_file():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    metadata[str(rec.get("episode_id", ""))] = rec
        except Exception as exc:
            logger.warning("Could not read index.json: %s", exc)

    replay_files = sorted([
        p for p in replays_dir.glob("*.json")
        if p.name != "index.json" and p.is_file() and p.stat().st_size > 1000
    ])

    logger.info("Found %d replay files to convert in %s", len(replay_files), replays_dir)
    generated_dirs: List[Path] = []

    for path in replay_files:
        try:
            dest = convert_single_replay(path, submissions_dir=submissions_dir, metadata=metadata)
            generated_dirs.append(dest)
        except Exception as exc:
            logger.error("Failed to convert replay %s: %s", path.name, exc, exc_info=True)

    logger.info("Successfully converted %d / %d replays to ghost submissions.", len(generated_dirs), len(replay_files))
    return generated_dirs


# =============================================================================
# 4. Execution Verification
# =============================================================================

def verify_ghost(ghost_dir: Path, steps: int = 720, seed: int = 42) -> Dict[str, Any]:
    """Verify that a ghost submission executes cleanly for steps turns."""
    from sim_engine import FastSimulation, is_kagg_available

    main_py = ghost_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"Missing main.py in {ghost_dir}")

    manifest_file = ghost_dir / "manifest.json"
    meta = {}
    if manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    ep_id = meta.get("episode_id", ghost_dir.name)
    opp = meta.get("opponent", "Unknown")

    t0 = time.perf_counter()
    with FastSimulation() as sim:
        p1_money, p2_money, final_st = sim.run_match(
            str(main_py),
            "random",
            seed=seed,
        )
    total_time = time.perf_counter() - t0
    completed_steps = int(final_st.get("step", 0)) + 1
    ms_per_turn = (total_time / completed_steps) * 1000.0

    return {
        "ghost": ghost_dir.name,
        "episode_id": ep_id,
        "opponent": opp,
        "completed_steps": completed_steps,
        "final_cash": p1_money,
        "opp_cash": p2_money,
        "total_time_s": total_time,
        "ms_per_turn": ms_per_turn,
        "clean": completed_steps == steps and p1_money >= 0,
    }


def verify_all_ghosts(ghost_dirs: List[Path], steps: int = 720) -> List[Dict[str, Any]]:
    """Verify execution across all generated ghost submissions."""
    results = []
    logger.info("Beginning verification of %d ghost submissions across %d turns...", len(ghost_dirs), steps)
    for g_dir in ghost_dirs:
        res = verify_ghost(g_dir, steps=steps)
        status_str = "PASS" if res["clean"] else "FAIL"
        logger.info(
            "[%s] %s (%s): %d/%d steps, cash: $%.2f, time: %.2fs (%.3fms/turn)",
            status_str,
            res["ghost"],
            res["opponent"],
            res["completed_steps"],
            steps,
            res["final_cash"],
            res["total_time_s"],
            res["ms_per_turn"],
        )
        results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert live match replays to Ghost Opponent submissions.")
    parser.add_argument("--replays-dir", "-r", type=Path, default=DEFAULT_REPLAYS_DIR, help="Directory containing replays")
    parser.add_argument("--submissions-dir", "-s", type=Path, default=DEFAULT_SUBMISSIONS_DIR, help="Submissions directory")
    parser.add_argument("--verify", "-v", action="store_true", help="Run 720-step verification on all generated ghosts")
    parser.add_argument("--single-replay", type=Path, default=None, help="Convert a single replay file")
    args = parser.parse_args()

    if args.single_replay:
        dest = convert_single_replay(args.single_replay, submissions_dir=args.submissions_dir)
        if args.verify:
            res = verify_ghost(dest)
            print(json.dumps(res, indent=2))
        return

    ghost_dirs = convert_all_replays(replays_dir=args.replays_dir, submissions_dir=args.submissions_dir)
    print(f"Generated {len(ghost_dirs)} ghost submissions.")

    if args.verify:
        results = verify_all_ghosts(ghost_dirs)
        clean_count = sum(1 for r in results if r["clean"])
        print(f"\nVerification summary: {clean_count} / {len(results)} passed cleanly across 720 steps.")
        for r in results:
            print(f"  {r['ghost']} ({r['opponent']}): {r['ms_per_turn']:.2f}ms/turn, ${r['final_cash']:,.2f} cash")


if __name__ == "__main__":
    main()
