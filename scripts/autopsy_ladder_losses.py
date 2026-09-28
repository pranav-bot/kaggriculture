#!/usr/bin/env python3
"""Ladder Loss Autopsy & Opponent Reverse-Engineering Pipeline.

Automatically:
1. Queries the database from our Leaderboard Tracker (`data/submission_ratings.db`)
   to identify matches where our agent lost (terminal score < opponent's score).
2. Downloads match replays via the Kaggle CLI:
   `kaggle competitions replay <EPISODE_ID> -p ./replays/live_losses/`
3. Parses the match frames using `scripts/replay_parser.py`.
4. Extracts winning opponent macro-milestones:
   - Days and hours they unlocked the NE / SW quadrants.
   - Peak herd size and livestock breakdown.
   - Whether they triggered a quadratic market crash on Wool.
5. Computes net worth and projected return-to-go (return_to_go_t) turn-by-turn to pinpoint
   the EXACT turn our agent fell permanently behind.
6. Outputs a structured autopsy file `loss_analysis_<EPISODE_ID>.json`.

Usage:
    .venv/bin/python scripts/autopsy_ladder_losses.py
    .venv/bin/python scripts/autopsy_ladder_losses.py --episode 114793445
    .venv/bin/python scripts/autopsy_ladder_losses.py --all-losses
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggriculture.features.returns import compute_net_worth, compute_reward, compute_return_to_go
from replay_parser import parse_replay, ReplayRecord

DEFAULT_DB_PATH = "data/submission_ratings.db"
DEFAULT_REPLAYS_DIR = "replays/live_losses"
DEFAULT_PLAYER = "Pranav"


def find_kaggle_cli() -> str:
    """Locates the kaggle CLI executable in PATH or virtualenv."""
    candidate = shutil.which("kaggle")
    if candidate:
        return candidate

    venv_bin = Path(sys.executable).parent / "kaggle"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)

    root_venv = ROOT / ".venv" / "bin" / "kaggle"
    if root_venv.is_file() and os.access(root_venv, os.X_OK):
        return str(root_venv)

    return "kaggle"


# =============================================================================
# 1. Database Query: Identify Loss Episodes
# =============================================================================

def ensure_database_schema(conn: sqlite3.Connection) -> None:
    """Ensures reward and loss tracking columns exist on the episodes table."""
    with conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()]
        if "my_reward" not in cols:
            conn.execute("ALTER TABLE episodes ADD COLUMN my_reward REAL")
        if "opp_reward" not in cols:
            conn.execute("ALTER TABLE episodes ADD COLUMN opp_reward REAL")
        if "is_loss" not in cols:
            conn.execute("ALTER TABLE episodes ADD COLUMN is_loss INTEGER DEFAULT 0")


def query_loss_episodes(
    conn: sqlite3.Connection,
    submission_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Queries DB for episodes where our agent's terminal score was lower than the opponent's."""
    ensure_database_schema(conn)

    query = """
        SELECT submission_id, episode_id, create_time, my_reward, opp_reward, is_loss
        FROM episodes
        WHERE is_loss = 1 OR (my_reward IS NOT NULL AND opp_reward IS NOT NULL AND my_reward < opp_reward)
    """
    params: List[Any] = []
    if submission_id:
        query += " AND submission_id = ?"
        params.append(str(submission_id))
    query += " ORDER BY create_time DESC, episode_id DESC"

    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# 2. Replay Downloader via Kaggle CLI
# =============================================================================

def download_replay(
    episode_id: Union[str, int],
    download_dir: Union[str, Path] = DEFAULT_REPLAYS_DIR,
    cli_path: Optional[str] = None,
) -> Path:
    """Downloads match replay via `kaggle competitions replay <EPISODE_ID>` if not cached."""
    d_dir = Path(download_dir)
    d_dir.mkdir(parents=True, exist_ok=True)

    # Check existing cached files
    patterns = [
        d_dir / f"episode-{episode_id}-replay.json",
        d_dir / f"{episode_id}.json",
        ROOT / "replays" / "my_agents" / "psro_leauge_pick" / f"{episode_id}.json",
    ]
    for p in patterns:
        if p.is_file():
            return p.resolve()

    cli = cli_path or find_kaggle_cli()
    cmd = [cli, "competitions", "replay", str(episode_id), "-p", str(d_dir), "-q"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to download replay for episode {episode_id}:\n{res.stderr.strip()}")

    # Find the downloaded file
    expected = d_dir / f"episode-{episode_id}-replay.json"
    if expected.is_file():
        return expected.resolve()

    matches = list(d_dir.glob(f"*{episode_id}*.json"))
    if matches and matches[0].is_file():
        return matches[0].resolve()

    raise FileNotFoundError(f"Replay file for episode {episode_id} not found in {d_dir}.")


# =============================================================================
# 3. Macro-Milestone & Crossover Autopsy Analyzer
# =============================================================================

def analyze_loss_match(
    replay_path: Union[str, Path],
    player_query: str = DEFAULT_PLAYER,
) -> Dict[str, Any]:
    """Dissects a loss replay to extract opponent milestones, crash triggers, and crossover turn."""
    path = Path(replay_path)
    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)

    # Pass into typed replay_parser
    parsed_record = parse_replay(path)

    steps = document.get("steps", [])
    info = document.get("info", {})
    team_names = info.get("TeamNames", ["Player 0", "Player 1"])
    episode_id = str(info.get("EpisodeId") or parsed_record.replay_id or path.stem.replace("episode-", "").replace("-replay", ""))

    # Determine seat indices
    my_seat = 0
    for idx, name in enumerate(team_names):
        if player_query.lower() in str(name).lower():
            my_seat = idx
            break
    opp_seat = 1 - my_seat

    my_name = team_names[my_seat] if my_seat < len(team_names) else f"Player {my_seat}"
    opp_name = team_names[opp_seat] if opp_seat < len(team_names) else f"Player {opp_seat}"

    terminal_rewards = document.get("rewards") or [0.0, 0.0]
    my_final_reward = float(terminal_rewards[my_seat])
    opp_final_reward = float(terminal_rewards[opp_seat])
    score_deficit = opp_final_reward - my_final_reward

    # 1. Macro-Milestone Trackers
    quadrant_unlocks: Dict[str, Dict[str, Any]] = {}
    animals_bought = Counter()
    crops_planted = Counter()
    opp_build_order: List[Dict[str, Any]] = []

    peak_herd_size = 0
    peak_herd_day = 0
    peak_herd_hour = 0

    # 2. Wool Market Crash Trackers
    wool_prices: List[Tuple[int, int, int, float]] = []  # (step, day, hour, price)
    opp_wool_sales: List[Dict[str, Any]] = []
    total_wool_sold_by_opp = 0

    # 3. Turn-by-Turn Net Worth and Return-to-Go Trackers
    our_net_worths: List[float] = []
    opp_net_worths: List[float] = []

    for t, step_data in enumerate(steps):
        day = t // 24
        hour = t % 24

        obs_turn = step_data[0].get("observation", {})
        farms = obs_turn.get("farms", [])
        mkt = obs_turn.get("market", {})
        prices = mkt.get("prices", {})

        # Compute net worth for both seats
        nw_our = compute_net_worth(obs_turn, my_seat)
        nw_opp = compute_net_worth(obs_turn, opp_seat)
        our_net_worths.append(nw_our)
        opp_net_worths.append(nw_opp)

        # Track Wool market price
        if "WOOL" in prices:
            w_price = float(prices["WOOL"])
            wool_prices.append((t, day, hour, w_price))

        # Opponent actions
        opp_frame = step_data[opp_seat]
        opp_act = opp_frame.get("action", {})
        if not isinstance(opp_act, dict):
            continue

        opp_market = opp_act.get("market", [])
        opp_farmer = opp_act.get("farmer", [])
        opp_hands = opp_act.get("hands", [])

        # Check market operations
        for order in opp_market:
            if not isinstance(order, list) or not order:
                continue
            cmd = order[0]

            if cmd == "BUY_LAND":
                # In Kaggle environment, order is: 1st=NE, 2nd=SW, 3rd=SE
                q_count = len(quadrant_unlocks)
                q_name = "NE" if q_count == 0 else ("SW" if q_count == 1 else "SE")
                quadrant_unlocks[q_name] = {
                    "day": day,
                    "hour": hour,
                    "turn": t,
                    "cost": 1000 if q_count == 0 else (2000 if q_count == 1 else 4000),
                }
                opp_build_order.append({
                    "turn": t,
                    "day": day,
                    "hour": hour,
                    "event": f"EXPAND_QUADRANT_{q_name}",
                    "details": f"Unlocked {q_name} quadrant",
                })

            elif cmd == "BUY_ANIMAL" and len(order) > 2:
                species = str(order[1]).upper()
                qty = int(order[2])
                animals_bought[species] += qty
                opp_build_order.append({
                    "turn": t,
                    "day": day,
                    "hour": hour,
                    "event": f"BUY_{species}",
                    "details": f"Purchased {qty} {species}(s)",
                })

            elif cmd == "SELL" and len(order) > 2:
                item = str(order[1]).upper()
                qty = int(order[2])
                if item == "WOOL":
                    total_wool_sold_by_opp += qty
                    opp_wool_sales.append({
                        "turn": t,
                        "day": day,
                        "hour": hour,
                        "quantity": qty,
                        "market_price": float(prices.get("WOOL", 200.0)),
                    })

            elif cmd == "BUY_SEED" and len(order) > 2:
                crop = str(order[1]).upper()
                qty = int(order[2])
                if day == 0:
                    opp_build_order.append({
                        "turn": t,
                        "day": day,
                        "hour": hour,
                        "event": f"BUY_SEED_{crop}",
                        "details": f"Bought {qty} {crop} seeds",
                    })

            elif cmd == "HIRE":
                if day == 0 and hour <= 1:
                    opp_build_order.append({
                        "turn": t,
                        "day": day,
                        "hour": hour,
                        "event": "HIRE_FARMHAND",
                        "details": "Hired farmhand on Day 0",
                    })

        # Check crops planted by opponent
        all_opp_units = [opp_farmer] + opp_hands
        for u_act in all_opp_units:
            if isinstance(u_act, list) and len(u_act) > 1 and u_act[0] == "PLANT":
                crop = str(u_act[1]).upper()
                crops_planted[crop] += 1

        # Check opponent livestock herd on tiles
        if opp_seat < len(farms):
            opp_farm = farms[opp_seat]
            current_herd = 0
            for r in opp_farm.get("tiles", []):
                if not isinstance(r, list):
                    continue
                for c in r:
                    if isinstance(c, dict) and c.get("kind") == "PASTURE" and c.get("animal"):
                        current_herd += 1
            if current_herd > peak_herd_size:
                peak_herd_size = current_herd
                peak_herd_day = day
                peak_herd_hour = hour

    # 4. Wool Market Crash Analysis
    min_wool_price = min((p[3] for p in wool_prices), default=200.0)
    wool_crashed = (min_wool_price <= 50.0)
    wool_crash_day: Optional[int] = None
    wool_crash_turn: Optional[int] = None

    if wool_crashed:
        for t, day, hour, price in wool_prices:
            if price <= 50.0:
                wool_crash_day = day
                wool_crash_turn = t
                break

    # Determine if opponent triggered the crash
    wool_sold_before_crash = 0
    if wool_crash_turn is not None:
        for sale in opp_wool_sales:
            if sale["turn"] <= wool_crash_turn:
                wool_sold_before_crash += sale["quantity"]
    else:
        wool_sold_before_crash = total_wool_sold_by_opp

    opp_triggered_crash = wool_crashed and (wool_sold_before_crash >= 15)

    # 5. Return-to-Go and Crossover Turn Calculation
    # Compute undiscounted return-to-go for each player
    our_rewards = [compute_reward(steps[t][0]["observation"], steps[t+1][0]["observation"], my_seat) for t in range(len(steps)-1)]
    opp_rewards = [compute_reward(steps[t][0]["observation"], steps[t+1][0]["observation"], opp_seat) for t in range(len(steps)-1)]
    our_rtg = compute_return_to_go(our_rewards)
    opp_rtg = compute_return_to_go(opp_rewards)

    # Find the permanent crossover turn where our agent's projected trajectory / net worth fell behind
    crossover_turn = 0
    for t in range(len(our_net_worths)):
        if opp_net_worths[t] > our_net_worths[t]:
            # Verify if opponent remained ahead until terminal end
            if all(opp_net_worths[k] >= our_net_worths[k] for k in range(t, len(our_net_worths))):
                crossover_turn = t
                break

    crossover_day = crossover_turn // 24
    crossover_hour = crossover_turn % 24
    our_nw_at_crossover = our_net_worths[crossover_turn]
    opp_nw_at_crossover = opp_net_worths[crossover_turn]
    nw_deficit = opp_nw_at_crossover - our_nw_at_crossover

    # Forensic summary of catalyst around crossover
    catalyst_events = []
    if "NE" in quadrant_unlocks and abs(quadrant_unlocks["NE"]["turn"] - crossover_turn) <= 12:
        catalyst_events.append(f"Opponent unlocked NE quadrant on Day {quadrant_unlocks['NE']['day']} (Turn {quadrant_unlocks['NE']['turn']})")
    if "SW" in quadrant_unlocks and abs(quadrant_unlocks["SW"]["turn"] - crossover_turn) <= 12:
        catalyst_events.append(f"Opponent unlocked SW quadrant on Day {quadrant_unlocks['SW']['day']} (Turn {quadrant_unlocks['SW']['turn']})")
    if wool_crashed and wool_crash_turn is not None and abs(wool_crash_turn - crossover_turn) <= 24:
        catalyst_events.append(f"Wool market crash occurred on Day {wool_crash_day}")

    crossover_catalyst = (
        "; ".join(catalyst_events)
        if catalyst_events
        else f"Opponent compound net worth surpassed ours on Day {crossover_day} Hour {crossover_hour} via superior asset compounding."
    )

    # Sample trajectory at key intervals
    trajectory_sample = []
    for d in [0, 3, 6, 10, 15, 20, 25, 29]:
        idx = min(d * 24, len(our_net_worths) - 1)
        trajectory_sample.append({
            "day": d,
            "turn": idx,
            "our_net_worth": round(our_net_worths[idx], 1),
            "opp_net_worth": round(opp_net_worths[idx], 1),
            "our_rtg": round(float(our_rtg[idx]), 1) if idx < len(our_rtg) else 0.0,
            "opp_rtg": round(float(opp_rtg[idx]), 1) if idx < len(opp_rtg) else 0.0,
            "lead": "OPPONENT" if opp_net_worths[idx] > our_net_worths[idx] else "OURS",
        })

    # Assemble structured autopsy record
    analysis = {
        "episode_id": episode_id,
        "match_metadata": {
            "our_agent_name": my_name,
            "opponent_name": opp_name,
            "our_terminal_score": my_final_reward,
            "opponent_terminal_score": opp_final_reward,
            "score_deficit": score_deficit,
            "match_result": "LOSS",
            "total_turns": len(steps),
        },
        "opponent_macro_milestones": {
            "ne_quadrant_unlock": quadrant_unlocks.get("NE"),
            "sw_quadrant_unlock": quadrant_unlocks.get("SW"),
            "se_quadrant_unlock": quadrant_unlocks.get("SE"),
            "peak_herd_size": peak_herd_size,
            "peak_herd_timing": {"day": peak_herd_day, "hour": peak_herd_hour},
            "animals_purchased": dict(animals_bought),
            "crops_planted": dict(crops_planted),
            "wool_market_crash": {
                "triggered": wool_crashed,
                "min_wool_price": min_wool_price,
                "crash_day": wool_crash_day,
                "crash_turn": wool_crash_turn,
                "opponent_wool_sold_total": total_wool_sold_by_opp,
                "opponent_wool_sold_before_crash": wool_sold_before_crash,
                "did_opponent_trigger_crash": opp_triggered_crash,
            },
        },
        "crossover_analysis": {
            "crossover_turn": crossover_turn,
            "crossover_day": crossover_day,
            "crossover_hour": crossover_hour,
            "our_net_worth_at_crossover": round(our_nw_at_crossover, 1),
            "opp_net_worth_at_crossover": round(opp_nw_at_crossover, 1),
            "net_worth_deficit_at_crossover": round(nw_deficit, 1),
            "crossover_catalyst": crossover_catalyst,
            "trajectory_timeline": trajectory_sample,
        },
        "opponent_build_order": opp_build_order[:30],  # Key initial and expansion milestones
    }

    return analysis


# =============================================================================
# 4. Master Autopsy Pipeline
# =============================================================================

def run_autopsy_pipeline(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    replays_dir: Union[str, Path] = DEFAULT_REPLAYS_DIR,
    output_dir: Union[str, Path] = ".",
    episode_id: Optional[str] = None,
    player_query: str = DEFAULT_PLAYER,
    auto_download: bool = True,
) -> List[Dict[str, Any]]:
    """Runs the complete autopsy pipeline across loss episodes."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_database_schema(conn)

    target_episodes: List[str] = []
    if episode_id:
        target_episodes = [str(episode_id)]
    else:
        db_losses = query_loss_episodes(conn)
        target_episodes = [str(r["episode_id"]) for r in db_losses]

        # If DB had no recorded losses, look for cached replays in replays_dir
        if not target_episodes:
            cached = glob.glob(f"{replays_dir}/*.json")
            for c in cached:
                stem = Path(c).stem.replace("episode-", "").replace("-replay", "")
                if stem.isdigit():
                    target_episodes.append(stem)

    conn.close()

    if not target_episodes:
        print("ℹ️  No loss episodes found to analyze.")
        return []

    print("=" * 88)
    print("🔬 Kaggriculture Match Autopsy: Opponent Reverse-Engineering Pipeline")
    print(f"   Target Loss Episodes: {len(target_episodes)} | Replays Dir: {replays_dir}")
    print("=" * 88)

    completed_analyses: List[Dict[str, Any]] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_id in target_episodes:
        print(f"\n📂 Investigating Episode {ep_id}...")
        try:
            # 1. Download or find replay
            replay_file = download_replay(ep_id, download_dir=replays_dir)
            print(f"   Downloaded / Found Replay: {replay_file.name}")

            # 2. Run deep autopsy
            analysis = analyze_loss_match(replay_file, player_query=player_query)

            # 3. Save structured JSON
            out_file = out_dir / f"loss_analysis_{ep_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2)
            print(f"   💾 Saved structured autopsy: {out_file.resolve()}")

            # 4. Print clean terminal summary
            meta = analysis["match_metadata"]
            miles = analysis["opponent_macro_milestones"]
            cross = analysis["crossover_analysis"]

            print(f"   Matchup: {meta['our_agent_name']} (${meta['our_terminal_score']:,.0f}) vs {meta['opponent_name']} (${meta['opponent_terminal_score']:,.0f})")
            print(f"   Score Deficit: -${meta['score_deficit']:,.0f}")

            ne_info = miles['ne_quadrant_unlock']
            sw_info = miles['sw_quadrant_unlock']
            ne_str = f"Day {ne_info['day']} Hr {ne_info['hour']}" if ne_info else "Never"
            sw_str = f"Day {sw_info['day']} Hr {sw_info['hour']}" if sw_info else "Never"
            print(f"   Opponent Quadrant Unlocks: NE={ne_str} | SW={sw_str}")
            print(f"   Opponent Peak Herd: {miles['peak_herd_size']} livestock (Day {miles['peak_herd_timing']['day']}) | Animals Bought: {miles['animals_purchased']}")

            wool = miles["wool_market_crash"]
            crash_icon = "💥 CRASH TRIGGERED" if wool["triggered"] else "STABLE"
            print(f"   Wool Market: {crash_icon} (Min Price: ${wool['min_wool_price']:.1f}, Opponent Sold: {wool['opponent_wool_sold_total']} units)")

            print(
                f"   ⚡ Crossover Turn: Turn {cross['crossover_turn']} (Day {cross['crossover_day']}, Hour {cross['crossover_hour']})\n"
                f"      Our NW: ${cross['our_net_worth_at_crossover']:,.0f} | Opp NW: ${cross['opp_net_worth_at_crossover']:,.0f} (Deficit: -${cross['net_worth_deficit_at_crossover']:,.0f})\n"
                f"      Catalyst: {cross['crossover_catalyst']}"
            )

            completed_analyses.append(analysis)

        except Exception as e:
            print(f"   ❌ Failed to analyze Episode {ep_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 88)
    print(f"✅ AUTOPSY COMPLETE: Dissected {len(completed_analyses)} loss matches.")
    print("=" * 88)

    return completed_analyses


def main() -> None:
    parser = argparse.ArgumentParser(description="Autopsy pipeline for Kaggriculture ladder losses.")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--replays-dir", type=str, default=DEFAULT_REPLAYS_DIR, help="Replays storage directory")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save loss_analysis_<ID>.json")
    parser.add_argument("--episode", "-e", type=str, default=None, help="Specific episode ID to analyze")
    parser.add_argument("--all-losses", action="store_true", help="Analyze all loss episodes in the database")
    parser.add_argument("--player", type=str, default=DEFAULT_PLAYER, help="Player name query for agent seat")
    args = parser.parse_args()

    run_autopsy_pipeline(
        db_path=args.db_path,
        replays_dir=args.replays_dir,
        output_dir=args.output_dir,
        episode_id=args.episode,
        player_query=args.player,
    )


if __name__ == "__main__":
    main()
