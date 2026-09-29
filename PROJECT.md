# Project: Master Integration Gauntlet

## Architecture
The Master Integration Gauntlet simulates live Kaggle competition conditions to validate and benchmark the autonomous quantitative policy:
1. **Candidate Agent (`submissions/hybrid_grandmaster_v2/main.py`)**:
   - Two-Team Division of Labor (Livestock lifecycle vs Field/Expansion crops).
   - Strategic Stuttering (MacroOptionManager beam search at Hours 0/12, Kuhn-Munkres worker routing during intervening hours).
   - Poisoned Well Market Trap (deceptive planting & market crash exploitation).
   - Strict `time.perf_counter()` Execution Watchdog (1.0s soft turn limit, 60.0s cumulative overage bank, 5.0s emergency fallback).
   - 100% self-contained, <100 MiB, <6.5 GiB RAM, entrypoint callable at end of file.
2. **Ghost Opponent Fleet (`submissions/ladder_ghost_<EPISODE_ID>/main.py`)**:
   - 20 match replays scraped from live Top 10 Kaggle leaderboard teams via Kaggle CLI.
   - Distilled into macro build orders with continuous seed/shed provisioning and Kuhn-Munkres field routing.
   - Exception-shielded, crash-proof execution across all 720 turns.
3. **Master Tournament Runner (Rust `kagg tournament`)**:
   - High-throughput native Rust simulation engine.
   - 1,000 fixed seeds pitting candidate against the 20 Top 10 ghost opponents.
   - Dual time limit enforcement: `"time_limits": {"act_s": 1.0, "overage_s": 60.0}`.
   - Outputs: `results.jsonl`, `summary.json`, `summary.md`.
4. **Diagnostic Autopsy & Optuna HPO Loop**:
   - `scripts/autopsy_ladder_losses.py` computes turn-by-turn net worth, RTG, and loss catalysts.
   - `scripts/ladder_ghost.py` / Optuna optimizes 7 Beam Search heuristic weights if benchmark assertions are breached.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Candidate Packaging | Package `scratch_grandmaster.py` into self-contained `submissions/hybrid_grandmaster_v2/main.py` (<100 MiB, <6.5 GiB) | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Packaging Bug Fixes | Fix callable ordering (place `agent` last) and multi-day expansion window logic (Days 3-8 NE, Days 6-18 SW) | M1 | Survey Explorer 1 |
| 3 | Execution Watchdog | Wrap agent in strict `time.perf_counter()` watchdog (1.0s/turn, 60.0s cumulative overage, 5.0s circuit breaker) | M1 | ORIGINAL_REQUEST §R3 |
| 4 | Top 10 Replay Scraping | Scrape 20 match replays from Top 10 leaderboard teams into `replays/live_top10/` using Kaggle CLI | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Ghost Opponent Generation | Convert 20 replays via macro extraction and `ladder_ghost.py` into executable `submissions/ladder_ghost_*` | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Ghost Zero-Crash Execution | Ensure all 20 ghost opponents execute 720 turns cleanly with zero illegal moves or unhandled exceptions | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Master Tournament Orchestration | Run 1,000-seed tournament via Rust `kagg tournament` pitting candidate against 20 ghost opponents | M3 | ORIGINAL_REQUEST §R4 |
| 8 | Tournament Performance Gate | Assert aggregate win rate ≥ 60.0% and average terminal cash ≥ $130,000 across the tournament | M3 | ORIGINAL_REQUEST §R5 |
| 9 | Diagnostic Autopsy Pipeline | Analyze losses via `autopsy_ladder_losses.py` (crossover turn, net worth deficit, opponent catalysts) | M4 | ORIGINAL_REQUEST §R5 |
| 10 | Automated Optuna HPO Loop | Re-tune 7 Beam Search heuristic weights via Optuna if win rate < 60% or cash < $130,000 | M4 | ORIGINAL_REQUEST §R5 |
| 11 | E2E Test Suite | Comprehensive opaque-box and compliance test suite covering packaging, ghost fleet, watchdog, and tournament | E2E | Dual Track |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Package Candidate & Watchdog | Create `submissions/hybrid_grandmaster_v2/main.py` with inlined watchdog, fixed expansion logic, and `test_submission.py` validation | none | IN_PROGRESS |
| M2 | Scrape Replays & Generate Ghost Fleet | Scrape 20 Top 10 replays into `replays/live_top10/` and convert to 20 functional ghost submissions | none | PLANNED |
| M3 | Master 1,000-Seed Tournament | Configure and execute Rust `kagg tournament`, assert ≥60% win rate and ≥$130k cash | M1, M2 | PLANNED |
| M4 | Diagnostic Autopsy & HPO Retuning | Run loss autopsy and automated Optuna tuning loop if criteria are unmet | M3 | PLANNED |

## Interface Contracts
### Candidate Submission (`submissions/hybrid_grandmaster_v2/main.py`)
- Callable entry point: `agent(obs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]` strictly at the bottom of the file.
- Action dictionary schema: `{"farmer": [direction], "hands": [[direction], ...], "market": [[order_type, ...], ...]}`.
- Latency contract: `time.perf_counter()` duration < 1.0s per turn; cumulative duration over 1.0s drawn from 60.0s overage bank.

### Ghost Opponent (`submissions/ladder_ghost_<EPISODE_ID>/main.py`)
- Callable entry point: `agent(obs)` with top-level try/except safety net returning PASS fallback on exception.
- Action format identical to candidate agent.
- Turn duration: < 1.0ms per turn.

### Tournament Configuration (`tournaments/gauntlet_1000/config.json`)
- Candidate: `{"name": "hybrid_grandmaster_v2", "type": "python", "path": "submissions/hybrid_grandmaster_v2/main.py"}`.
- Panel: 20 ghost entries `[{"name": "ladder_ghost_<EPISODE_ID>", "type": "python", "path": "submissions/ladder_ghost_<EPISODE_ID>/main.py"}, ...]`.
- Limits: `"time_limits": {"act_s": 1.0, "overage_s": 60.0}`.
- Python: `{"exe": "/Users/pranav/dev/kaggriculture/.venv/bin/python"}`.

## Code Layout
- `submissions/hybrid_grandmaster_v2/main.py`: Self-contained candidate agent.
- `replays/live_top10/`: Scraped match replays (20 JSON files).
- `submissions/ladder_ghost_<EPISODE_ID>/`: 20 static ghost submissions (`main.py`, `manifest.json`).
- `scripts/scrape_top10_replays.py`: Automation script to scrape 20 Top 10 replays from Kaggle CLI.
- `scripts/convert_replays_to_ghosts.py`: Automation script to convert scraped replays into ghost submissions.
- `tournaments/gauntlet_1000/`: Tournament configuration and run artifacts (`results.jsonl`, `summary.json`, `summary.md`).
- `tests/test_gauntlet_e2e.py`: Integration test suite verifying acceptance criteria.
